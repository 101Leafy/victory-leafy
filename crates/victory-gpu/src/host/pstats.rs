//! The card's own counters around each noted command (`VICTORY_GPU_STATS=1`,
//! diagnostics builds): a pipeline statistics query from one note to the
//! next, so each command's vertices, primitives and shader invocations are
//! counted by the card. Answers whether the indirect commands and small
//! dispatches have work in them at all, which their times cannot.
//!
//! The query runs from a note to the next one, so it holds the command and
//! the setup recorded before the next (barriers, clamps, copies), which
//! invoke no shaders. Ending a statistics query drains the card, so a run
//! with the switch on is for the counts, not for times or FPS.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// Queries a list keeps; commands past them are counted as missed.
const KEPT: u32 = 1024;
/// No query open.
const NONE: u32 = 0;
/// The card's threads at once on an RTX 5080 (84 SMs of 1536): a command
/// under it cannot fill the card alone.
const FULL: u64 = 84 * 1536;

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_STATS"))
}

/// Whether the card counts mesh and amplification shaders (the second kind
/// of statistics query), asked once.
fn with_mesh(device: &ID3D12Device) -> bool {
    static THERE: OnceLock<bool> = OnceLock::new();
    *THERE.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
        // SAFETY: writes `o` of the given size.
        let ok = unsafe {
            device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, (&raw mut o).cast(), size_of_val(&o) as u32).is_ok()
        };
        let mesh = ok && o.MeshShaderPipelineStatsSupported.as_bool();
        tracing::info!(mesh, "pipeline statistics per command (VICTORY_GPU_STATS)");
        mesh
    })
}

/// A list's statistics queries.
#[derive(Default)]
pub(crate) struct Stats {
    /// The query heap, its read-back buffer, where that is mapped, and
    /// whether it is the kind that counts mesh shaders.
    made: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource, usize, bool)>>,
    /// The query open now, plus one.
    open: AtomicU32,
    /// Queries begun since the reset.
    begun: AtomicU32,
    /// Notes past [`KEPT`].
    missed: AtomicU32,
}

impl Stats {
    fn made(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource, usize, bool)> {
        self.made
            .get_or_init(|| {
                let device = &crate::host::host()?.device;
                let mesh = with_mesh(device);
                let size = if mesh {
                    size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>()
                } else {
                    size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>()
                } as u64;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(KEPT) * size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                let kind = if mesh {
                    D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1
                } else {
                    D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS
                };
                let mut data = std::ptr::null_mut();
                // SAFETY: plain object creation; a read-back buffer stays
                // mapped while it lives.
                unsafe {
                    device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: kind, Count: KEPT, NodeMask: 0 }, &mut heap).ok()?;
                    device
                        .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
                        .ok()?;
                    let back = back?;
                    back.Map(0, None, Some(&mut data)).ok()?;
                    Some((heap?, back, data as usize, mesh))
                }
            })
            .as_ref()
    }

    fn kind(mesh: bool) -> D3D12_QUERY_TYPE {
        if mesh { D3D12_QUERY_TYPE_PIPELINE_STATISTICS1 } else { D3D12_QUERY_TYPE_PIPELINE_STATISTICS }
    }

    /// Note `at` is about to be recorded: the last query ends, its own
    /// begins.
    pub(crate) fn at_note(&self, l: &ID3D12GraphicsCommandList, at: u32) {
        let Some((heap, _, _, mesh)) = self.made() else {
            return;
        };
        let open = self.open.swap(NONE, Relaxed);
        // SAFETY: an open list and our heap; only a begun query is ended.
        unsafe {
            if open != NONE {
                l.EndQuery(heap, Self::kind(*mesh), open - 1);
            }
            if at < KEPT {
                l.BeginQuery(heap, Self::kind(*mesh), at);
                self.open.store(at + 1, Relaxed);
                self.begun.store(at + 1, Relaxed);
            } else {
                self.missed.fetch_add(1, Relaxed);
            }
        }
    }

    /// At the list's close: the open query ends and all go to the
    /// read-back buffer.
    pub(crate) fn resolve(&self, l: &ID3D12GraphicsCommandList) {
        let Some((heap, back, _, mesh)) = self.made.get().and_then(Option::as_ref) else {
            return;
        };
        let open = self.open.swap(NONE, Relaxed);
        let n = self.begun.load(Relaxed);
        // SAFETY: an open list, our heap and buffer; every query resolved
        // was begun and ended in this list.
        unsafe {
            if open != NONE {
                l.EndQuery(heap, Self::kind(*mesh), open - 1);
            }
            if n > 0 {
                l.ResolveQueryData(heap, Self::kind(*mesh), 0, n, back, 0);
            }
        }
    }

    /// Once the list ran, before its notes are cleared: each command's
    /// counts to the sums. A list closed and never run reads what the last
    /// run left; `begun` is cleared either way.
    pub(crate) fn read(&self, notes: &[(String, Option<(u64, u64)>)]) {
        let n = self.begun.swap(0, Relaxed) as usize;
        self.open.store(NONE, Relaxed);
        let missed = self.missed.swap(0, Relaxed);
        let Some((_, _, at, mesh)) = self.made.get().and_then(Option::as_ref) else {
            return;
        };
        let words = if *mesh { 14 } else { 11 };
        // SAFETY: the mapped read-back buffer, which the finished list
        // resolved into and nothing writes now.
        let all = unsafe { std::slice::from_raw_parts(*at as *const u64, n * words) };
        let mut sums = SUMS.lock().unwrap_or_else(|e| e.into_inner());
        let sums = sums.get_or_insert_with(Sums::default);
        sums.missed += u64::from(missed);
        for (i, c) in all.chunks_exact(words).enumerate() {
            let Some((text, _)) = notes.get(i) else {
                break;
            };
            sums.add(text, Counts::from(c));
        }
    }
}

/// One command's counts.
#[derive(Default, Clone, Copy)]
struct Counts {
    vertices: u64,
    primitives: u64,
    vs: u64,
    clipped_in: u64,
    clipped_out: u64,
    ps: u64,
    cs: u64,
    amplify: u64,
    ms: u64,
    ms_primitives: u64,
}

impl Counts {
    fn from(c: &[u64]) -> Self {
        let at = |i: usize| c.get(i).copied().unwrap_or(0);
        Self {
            vertices: at(0),
            primitives: at(1),
            vs: at(2),
            clipped_in: at(5),
            clipped_out: at(6),
            ps: at(7),
            cs: at(10),
            amplify: at(11),
            ms: at(12),
            ms_primitives: at(13),
        }
    }

    /// The shader threads it ran.
    fn threads(&self) -> u64 {
        self.vs + self.ps + self.cs + self.amplify + self.ms
    }

    fn plus(&mut self, o: &Self) {
        self.vertices += o.vertices;
        self.primitives += o.primitives;
        self.vs += o.vs;
        self.clipped_in += o.clipped_in;
        self.clipped_out += o.clipped_out;
        self.ps += o.ps;
        self.cs += o.cs;
        self.amplify += o.amplify;
        self.ms += o.ms;
        self.ms_primitives += o.ms_primitives;
    }
}

/// Sizes by shader threads: none, a warp or less, a few SMs, under the
/// whole card, more.
const SIZES: [&str; 5] = ["empty", "1-64", "65-4096", "under the card", "card or more"];

fn size(threads: u64) -> usize {
    match threads {
        0 => 0,
        1..=64 => 1,
        65..=4096 => 2,
        t if t < FULL => 3,
        _ => 4,
    }
}

#[derive(Default)]
struct Sums {
    /// By first word: commands by size, and the counts.
    kinds: HashMap<String, ([u64; 5], Counts)>,
    /// By pipeline key: commands, empty ones, the counts.
    keys: HashMap<String, (u64, u64, Counts)>,
    missed: u64,
}

impl Sums {
    fn add(&mut self, text: &str, c: Counts) {
        let word = text.split(' ').next().unwrap_or("").to_owned();
        let k = self.kinds.entry(word).or_default();
        k.0[size(c.threads())] += 1;
        k.1.plus(&c);
        let p = self.keys.entry(key(text)).or_default();
        p.0 += 1;
        p.1 += u64::from(c.threads() == 0);
        p.2.plus(&c);
    }
}

/// As the timing run sums them: the first word with the pipeline in
/// brackets, or the first two words.
fn key(op: &str) -> String {
    let head = op.split(" | ").next().unwrap_or(op);
    let word = head.split(' ').next().unwrap_or("");
    match (head.find('['), head.find(']')) {
        (Some(a), Some(b)) if a < b => format!("{word} {}", &head[a..=b]),
        _ => head.split(' ').take(2).collect::<Vec<_>>().join(" "),
    }
}

static SUMS: Mutex<Option<Sums>> = Mutex::new(None);

/// With the counter line: the sums per frame since the last one.
pub(crate) fn report(frames: f64) {
    let Some(sums) = SUMS.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let f = frames.max(1.0);
    let mut kinds: Vec<_> = sums.kinds.into_iter().collect();
    kinds.sort_by_key(|k| std::cmp::Reverse(k.1.0.iter().sum::<u64>()));
    for (word, (by_size, c)) in &kinds {
        let all: u64 = by_size.iter().sum();
        let sizes: Vec<String> =
            SIZES.iter().zip(by_size).map(|(name, n)| format!("{name} {:.0}", *n as f64 / f)).collect();
        tracing::info!(
            "  card counts by command (VICTORY_GPU_STATS), {word}: {:.0} a frame, by shader threads: {}; \
             a frame: {:.0}k compute threads, {:.0}k vertex, {:.0}k mesh + {:.0}k amplification, {:.0}k pixel; \
             {:.0}k vertices in, {:.0}k primitives in, {:.0}k to the clipper, {:.0}k out of it, {:.0}k mesh primitives",
            all as f64 / f,
            sizes.join(", "),
            c.cs as f64 / f / 1e3,
            c.vs as f64 / f / 1e3,
            c.ms as f64 / f / 1e3,
            c.amplify as f64 / f / 1e3,
            c.ps as f64 / f / 1e3,
            c.vertices as f64 / f / 1e3,
            c.primitives as f64 / f / 1e3,
            c.clipped_in as f64 / f / 1e3,
            c.clipped_out as f64 / f / 1e3,
            c.ms_primitives as f64 / f / 1e3,
        );
    }
    let mut keys: Vec<_> = sums.keys.into_iter().collect();
    keys.sort_by_key(|k| std::cmp::Reverse(k.1.1));
    for (what, (n, empty, _)) in keys.iter().filter(|k| k.1.1 > 0).take(20) {
        tracing::info!(
            "  card counts, empty commands: {:.1} of {:.1} a frame ran no shader: {what}",
            *empty as f64 / f,
            *n as f64 / f
        );
    }
    keys.sort_by_key(|k| std::cmp::Reverse(k.1.2.ps));
    for (what, (n, _, c)) in keys.iter().take(15) {
        tracing::info!(
            "  card counts, pixel shading: {:.2} screens (1280x720) in {:.1} commands a frame: {what}",
            c.ps as f64 / f / (1280.0 * 720.0),
            *n as f64 / f
        );
    }
    keys.sort_by_key(|k| k.1.2.threads() / k.1.0.max(1));
    keys.retain(|k| k.1.0 as f64 / f >= 1.0);
    for (what, (n, _, c)) in keys.iter().take(15) {
        tracing::info!(
            "  card counts, smallest commands run every frame: {:.0} threads each, {:.1} a frame: {what}",
            c.threads() as f64 / (*n).max(1) as f64,
            *n as f64 / f
        );
    }
    if sums.missed > 0 {
        tracing::info!("  card counts: {:.0} commands a frame not counted (past {KEPT} in a list)", sums.missed as f64 / f);
    }
}
