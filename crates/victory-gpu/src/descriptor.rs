//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;
#[cfg_attr(not(windows), allow(dead_code))]

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (the title) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// Changed pages of a placed shader-visible heap. Some titles copy
    /// descriptor records into it as bytes, without a descriptor API call.
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The handle of descriptor `index` of this heap.
    #[cfg(windows)]
    pub(crate) fn handle(&self, index: u32) -> Option<u64> {
        let step = u64::from(increment(self.desc.kind)?);
        (index < self.desc.count).then(|| self.start + u64::from(index) * step)
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
///
/// Held shared while a heap found here is used (by every thread that writes
/// descriptors, and by [`sync_heaps`]), alone only where heaps come and go.
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` locked for the duration
/// so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let Some(&(_, _, at)) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))
    else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Whether [`sync_heaps`] is done (`VICTORY_HEAP_SYNC=0`: not).
#[cfg(windows)]
fn sync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "0"))
}

/// Whether a submission's [`sync_heaps`] is done by a thread of its own
/// beside the rest of the preparing (`VICTORY_HEAP_SYNC_ASIDE=0`: in front of
/// it, as before).
///
/// It took 13 ms a frame of the submitting thread (session play-1125-0203:
/// 230 changed heap pages and 470,000 slots in 30 submissions). What it makes
/// are the host heap's descriptors, which the card reads: it has to be done
/// before the lists go to the queue, and nothing of the preparing looks at
/// them (what a draw reads through a table it made itself, when recorded).
#[cfg(windows)]
fn sync_aside() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| sync_on() && std::env::var_os("VICTORY_HEAP_SYNC_ASIDE").is_none_or(|v| v != "0"))
}

/// How long that thread waits for a submission before it makes a round of its
/// own (`VICTORY_HEAP_SYNC_EARLY=<ms>`, 4 if not given; `0`: none). The title
/// copies its records while it records, and the first submission of a frame
/// found all of them: 8 ms waited for (run cpu-2). A round made meanwhile
/// leaves it those written since.
#[cfg(windows)]
fn sync_early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = std::env::var("VICTORY_HEAP_SYNC_EARLY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// The thread's work: rounds asked for, rounds done.
#[cfg(windows)]
static SYNC_ROUNDS: (std::sync::Mutex<(u64, u64)>, std::sync::Condvar) = (std::sync::Mutex::new((0, 0)), std::sync::Condvar::new());

/// Rounds of the thread ended, asked for or early.
#[cfg(windows)]
static SYNC_ENDED: (std::sync::Mutex<u64>, std::sync::Condvar) = (std::sync::Mutex::new(0), std::sync::Condvar::new());

/// Waits until a round of the heap sync thread ends after `seen` of them
/// had, or for `most`; the rounds ended by then.
#[cfg(windows)]
pub(crate) fn sync_round_after(seen: u64, most: std::time::Duration) -> u64 {
    let ended = SYNC_ENDED.0.lock().unwrap_or_else(|e| e.into_inner());
    let (ended, _) = SYNC_ENDED.1.wait_timeout_while(ended, most, |n| *n <= seen).unwrap_or_else(|e| e.into_inner());
    *ended
}

/// [`sync_heaps`], begun; done when it is dropped (with [`sync_aside`]), or
/// when this returns.
#[cfg(windows)]
pub(crate) struct SyncedHeaps(Option<u64>);

#[cfg(windows)]
pub(crate) fn begin_sync_heaps() -> SyncedHeaps {
    if !sync_aside() {
        sync_heaps();
        return SyncedHeaps(None);
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    static THREAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    STARTED.call_once(|| {
        let made = std::thread::Builder::new().name("victory-heap-sync".into()).spawn(|| {
            crate::cores::keep_on_fast_cores("victory-heap-sync");
            // (The submitting thread waits for its rounds.)
            let mut priority = crate::cores::Priority::new();
            let rounds = || SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
            /// A round is over, however it ends.
            struct Over;
            impl Drop for Over {
                fn drop(&mut self) {
                    SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
                    SYNC_ROUNDS.1.notify_all();
                }
            }
            // Between the rounds asked for, while submissions come.
            let mut last = std::time::Instant::now();
            loop {
                let mut r = rounds();
                let mut asked = r.0 != r.1;
                while !asked {
                    let Some(every) = sync_early().filter(|_| last.elapsed().as_millis() < 500) else {
                        r = SYNC_ROUNDS.1.wait(r).unwrap_or_else(|e| e.into_inner());
                        asked = r.0 != r.1;
                        continue;
                    };
                    let (back, over) = SYNC_ROUNDS.1.wait_timeout(r, every).unwrap_or_else(|e| e.into_inner());
                    r = back;
                    asked = r.0 != r.1;
                    if over.timed_out() {
                        break;
                    }
                }
                drop(r);
                priority.keep();
                let _over = asked.then(|| {
                    last = std::time::Instant::now();
                    Over
                });
                let _round =
                    crate::timeline::span(if asked { "the heaps' records, for a submission" } else { "the heaps' records, early" });
                let _ = std::panic::catch_unwind(sync_heaps);
                *SYNC_ENDED.0.lock().unwrap_or_else(|e| e.into_inner()) += 1;
                SYNC_ENDED.1.notify_all();
            }
        });
        THREAD.store(made.is_ok(), std::sync::atomic::Ordering::Release);
    });
    if !THREAD.load(std::sync::atomic::Ordering::Acquire) {
        sync_heaps();
        return SyncedHeaps(None);
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    rounds.0 += 1;
    let mine = rounds.0;
    drop(rounds);
    SYNC_ROUNDS.1.notify_all();
    SyncedHeaps(Some(mine))
}

/// Renderer port #7 (`crate::port_journal`, from renderer patch #4): [`begin_sync_heaps`] for a
/// submission that needs round `mark` (one that began after what it waits
/// for was written, `crate::port_journal::round_after`): asked for if it has not
/// been. 0: as [`begin_sync_heaps`].
#[cfg(windows)]
pub(crate) fn begin_sync_heaps_after(mark: u64) -> SyncedHeaps {
    if mark == 0 || !sync_aside() {
        return begin_sync_heaps();
    }
    // (The thread is started by the first `begin_sync_heaps`.)
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        drop(rounds);
        return begin_sync_heaps();
    }
    let asked = rounds.0 < mark;
    rounds.0 = rounds.0.max(mark);
    drop(rounds);
    if asked {
        SYNC_ROUNDS.1.notify_all();
        crate::port_journal::STATS[crate::port_journal::LATE_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        crate::port_journal::STATS[crate::port_journal::EARLY_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    SyncedHeaps(Some(mark))
}

/// Renderer port #7: the number of a round of the heaps' records that
/// begins after this call, asked for if none is waiting to begin (at the
/// title's submit: the round runs while the submission waits in the line).
/// 0 if the records are not synced by a thread of their own (or it has not
/// started yet).
#[cfg(windows)]
pub(crate) fn round_after_now() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        return 0;
    }
    let (round, ask) = crate::port_journal::round_after(rounds.0, rounds.1);
    if ask {
        rounds.0 = round;
        drop(rounds);
        SYNC_ROUNDS.1.notify_all();
        crate::port_journal::STATS[crate::port_journal::ROUNDS_ASKED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    round
}

/// Renderer port #7: the number of the first round that begins after this
/// call, not asked for ([`begin_sync_heaps_after`] asks for it if nobody did).
#[cfg(windows)]
pub(crate) fn round_mark() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        return 0;
    }
    rounds.0 + 1
}

/// Whether `[at, at + len)` touches the title memory of a placed
/// shader-visible heap of descriptors.
#[cfg(windows)]
pub(crate) fn in_shader_heap(at: usize, len: usize) -> bool {
    let (from, to) = (at as u64, at as u64 + len as u64);
    SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner()).iter().any(|&(start, end, _)| from < end && start < to)
}

#[cfg(windows)]
impl Drop for SyncedHeaps {
    fn drop(&mut self) {
        let Some(mine) = self.0 else {
            return;
        };
        let began = std::time::Instant::now();
        let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
        while rounds.1 < mine {
            rounds = SYNC_ROUNDS.1.wait(rounds).unwrap_or_else(|e| e.into_inner());
        }
        drop(rounds);
        SYNCED[6].fetch_add(began.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Mirrors records copied directly into placed shader-visible heaps. Calls
/// to `Create*View` are already mirrored by [`mirror`]; raw CPU copies leave
/// only write-watch pages to reveal which slots changed.
///
/// Upstream VIctory 86a5ed6 does the same and counts 1,300 to 14,400 such
/// slots a frame in the world of the title this fork works on, against a
/// few hundred views made by calls; without it those host slots stayed
/// null, or went on showing what they showed before.
/// `VICTORY_HEAP_SYNC=0`: not done, as before. [`SYNCED`] counts.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    use std::sync::atomic::Ordering::Relaxed;
    if !sync_on() {
        return;
    }
    let started = std::time::Instant::now();
    let (mut pages, mut looked, mut mirrored, mut tabled, mut later) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut views, mut bytes) = (0u64, 0u64);
    let mut spent = [0u64; 3];
    let since = |at: std::time::Instant| at.elapsed().as_nanos() as u64;
    let mut kinds = [0u64; 5];
    let mut short = 0u64;
    let mut wait: Vec<u32> = Vec::new();
    let frame = crate::frame::current();
    let (pages_a_job, records_a_job) = sync_jobs();
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, end, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // Capped host heaps have no counterparts for the title slots reserved
        // for per-draw tables; raw writes there must not be mirrored.
        let mirrored_count = heap.desc.count.min(host.count());
        // The records of some frames ago that no draw read through a table
        // since: a shader may pick them by index.
        let before = host.copied_before(frame);
        let of_tables = std::sync::atomic::AtomicU64::new(0);
        crate::land::each_view(true, before.len().div_ceil(records_a_job), &|n| {
            for &index in before.chunks(records_a_job).nth(n).unwrap_or_default() {
                if host.copied_tabled(index, frame) {
                    of_tables.fetch_add(1, Relaxed);
                } else {
                    host.mirror(index, start + u64::from(index) * step, false);
                }
            }
        });
        let of_tables = of_tables.into_inner();
        tabled += of_tables;
        mirrored += before.len() as u64 - of_tables;
        kinds[4] += before.len() as u64 - of_tables;
        // Those of the changed pages, made together at the end.
        let mut make: Vec<u32> = Vec::new();
        let began = std::time::Instant::now();
        let taken = written.take(start as usize, end as usize);
        spent[0] += since(began);
        let mut parts: Vec<(usize, usize)> = Vec::new();
        for (range_from, range_end) in taken {
            let mut page = range_from & !(crate::watch::PAGE - 1);
            while page < range_end {
                parts.push((page.max(range_from), (page + crate::watch::PAGE).min(range_end)));
                page += crate::watch::PAGE;
            }
        }
        // A few pages a thread at a time: read, compared with what was last
        // seen there and sorted out, beside one another (14 ms a frame of one
        // thread at 160 pages, run faults-6).
        let began = std::time::Instant::now();
        let copied = host.copied_begin(step as usize);
        let samplers = heap.samplers();
        let in_place = sync_in_place() && step == 32 && crate::safecopy::on();
        let found: Vec<std::sync::Mutex<SyncPart>> = parts.chunks(pages_a_job).map(|_| Default::default()).collect();
        crate::land::each_view(true, found.len(), &|n| {
            let mut out = SyncPart::default();
            let mut page_bytes: Vec<u8> = Vec::new();
            let mut changed: Vec<u32> = Vec::new();
            let mut bits: Vec<u64> = Vec::new();
            for &(from, to) in parts.chunks(pages_a_job).nth(n).unwrap_or_default() {
                let slots = changed_slots(from, to, start as usize, step, mirrored_count);
                // A record that straddles the range's ends is left
                // to the range that holds its first byte.
                let first = ((from - start as usize) as u64).div_ceil(step).max(slots.start);
                let lead = (start + first * step) as usize;
                let records: &[u8] = if in_place {
                    // Read where they are, each record once, and compared
                    // as it is read (`host::Copied::changes_in_place`).
                    let count = (to.saturating_sub(lead) as u64 / step).min(slots.end.saturating_sub(first));
                    changed.clear();
                    // SAFETY: as below; title memory is read by a copy that
                    // fails where a page is not there.
                    let whole =
                        unsafe { copied.changes_in_place(first as u32, lead as *const u8, count as usize, &mut bits, &mut changed) };
                    out.pages += 1;
                    out.views += u64::from(crate::watch::guarded(from, to));
                    out.bytes += (to - from) as u64;
                    out.looked += count;
                    out.short += u64::from(!whole);
                    // What was read of the changed records is in the records
                    // last seen now.
                    // SAFETY: this part's records alone, which nothing else
                    // writes meanwhile.
                    unsafe { copied.seen(first as u32, count as usize) }
                } else {
                    // The page's records are read through the kernel, into a
                    // copy: a page the title unmaps meanwhile fails the read
                    // where 2,048 reads in place would fault.
                    page_bytes.resize(to - from, 0);
                    if !crate::host::read_into(from as *const u8, &mut page_bytes) {
                        continue;
                    }
                    out.pages += 1;
                    out.views += u64::from(crate::watch::guarded(from, to));
                    out.bytes += (to - from) as u64;
                    let Some(records) = page_bytes.get(lead.wrapping_sub(from)..) else {
                        continue;
                    };
                    let count = (records.len() as u64 / step).min(slots.end.saturating_sub(first));
                    out.looked += count;
                    // Only the records that are not the ones last seen
                    // there (`host::Copied::changes`).
                    changed.clear();
                    // SAFETY: the parts are ranges of the heap apart from one
                    // another, and only records that lie whole in one are its.
                    unsafe { copied.changes(first as u32, &records[..(count * step) as usize], &mut changed) };
                    records
                };
                for &index in &changed {
                    let at = (u64::from(index) - first) as usize * step as usize;
                    let word = |n: usize| {
                        records
                            .get(at + 4 * n..at + 4 * n + 4)
                            .and_then(|b| b.try_into().ok())
                            .map_or(0, u32::from_le_bytes)
                    };
                    if !samplers && word(0) == VIEW_TAG {
                        // A draw's own table: the draw makes what it
                        // reads of it (`draw.rs` `tables`).
                        if host.copied_tabled(index, frame) {
                            out.tabled += 1;
                            continue;
                        }
                        // Buffer views and placed textures get no host
                        // slot from this anyway, they wait for a draw
                        // or a pass: noted some frames on, unless a
                        // draw's table turns out to hold them.
                        if matches!(word(2) >> KIND_SHIFT, CONSTANT | PLACED) && host.copied_waits() {
                            out.wait.push(index);
                            continue;
                        }
                    }
                    // No view, in a slot that shows none: nothing to make or
                    // unmake (32,000 of 41,000 a frame on the water, run
                    // b2-water).
                    if word(0) != VIEW_TAG && host.shows_none(index) {
                        out.nothing += 1;
                        continue;
                    }
                    out.make.push(index);
                    out.kinds[if samplers {
                        3
                    } else if word(0) != VIEW_TAG {
                        0
                    } else if word(2) >> KIND_SHIFT == 0 {
                        1
                    } else {
                        2
                    }] += 1;
                }
            }
            *found[n].lock().unwrap_or_else(|e| e.into_inner()) = out;
        });
        drop(copied);
        for part in found {
            let mut part = part.into_inner().unwrap_or_else(|e| e.into_inner());
            pages += part.pages;
            views += part.views;
            bytes += part.bytes;
            looked += part.looked;
            short += part.short;
            tabled += part.tabled;
            kinds[0] += part.nothing;
            later += part.wait.len() as u64;
            mirrored += part.make.len() as u64;
            wait.append(&mut part.wait);
            make.append(&mut part.make);
            for (kind, n) in kinds.iter_mut().zip(part.kinds) {
                *kind += n;
            }
        }
        spent[1] += since(began);
        let began = std::time::Instant::now();
        crate::land::each_view(true, make.len().div_ceil(records_a_job), &|n| {
            for &index in make.chunks(records_a_job).nth(n).unwrap_or_default() {
                host.mirror(index, start + u64::from(index) * step, false);
            }
        });
        spent[2] += since(began);
        if !wait.is_empty() {
            host.copied_wait(frame, std::mem::take(&mut wait));
        }
    }
    drop(heaps);
    let micros = started.elapsed().as_micros() as u64;
    for (count, n) in SYNCED.iter().zip([pages, looked, mirrored, micros, tabled, later]) {
        count.fetch_add(n, Relaxed);
    }
    let [taking, reading, making] = spent;
    for (count, n) in SYNC_SPLIT.iter().zip([views, bytes, taking, reading, making]) {
        count.fetch_add(n, Relaxed);
    }
    for (count, n) in SYNCED_KINDS.iter().zip(kinds) {
        count.fetch_add(n, Relaxed);
    }
    SYNC_SHORT.fetch_add(short, Relaxed);
}

/// Pages [`sync_heaps`] read in place of which a part could not be read
/// ([`sync_in_place`]), since the last frame line.
#[cfg(windows)]
static SYNC_SHORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether [`sync_heaps`] reads a changed page's records where they are, each
/// once, and compares each with the one last seen as it reads it, 32 bytes at
/// a time (`VICTORY_SYNC_IN_PLACE=1`; off by default: into a copy of the page
/// first, through the kernel, then compared, as before).
///
/// Ported from Belobug/VIctory-Belobug-AMD 15ab94e (measured there: the copy
/// and the compare were each a pass over all of the page, 7.4 and 5.3 ms a
/// frame of the six threads for 35 MB, run heapsync-ab1; on by default there
/// since 8822ed4). Here the round's "reading and comparing" was 2.6-3.1 ms a
/// frame (release run 233354), and the submitting thread waits for the
/// round. A page the title unmaps meanwhile fails where the copy failed; the
/// records before the place are taken as they were read.
#[cfg(windows)]
fn sync_in_place() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SYNC_IN_PLACE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_IN_PLACE").is_some_and(|v| v != "0"))
}

/// Of those mirrored, since the last frame line: no view, a view of an object,
/// another, a sampler; waited and no draw came.
#[cfg(windows)]
static SYNCED_KINDS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What a thread found on its pages of a heap, for [`sync_heaps`].
#[cfg(windows)]
#[derive(Default)]
struct SyncPart {
    pages: u64,
    views: u64,
    bytes: u64,
    looked: u64,
    tabled: u64,
    nothing: u64,
    make: Vec<u32>,
    wait: Vec<u32>,
    kinds: [u64; 5],
    /// Pages read in place not all there ([`SYNC_SHORT`]).
    short: u64,
}

/// Pages [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_PAGES: usize = 8;

/// Records [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_AT_A_TIME: usize = 256;

/// With `VICTORY_SYNC_FINE=1`: pages and records [`sync_heaps`] gives a thread
/// at a time.
#[cfg(windows)]
const SYNC_PAGES_FINE: usize = 2;
#[cfg(windows)]
const SYNC_AT_A_TIME_FINE: usize = 64;

/// Pages and records [`sync_heaps`] gives a thread at a time
/// (`VICTORY_SYNC_FINE=1`: [`SYNC_PAGES_FINE`] and [`SYNC_AT_A_TIME_FINE`];
/// off by default: [`SYNC_PAGES`] and [`SYNC_AT_A_TIME`], as before).
///
/// Ported from Belobug/VIctory-Belobug-AMD a284ca7: a round for a submission
/// finds a dozen pages or so; by eight a job that was two jobs, on two of the
/// six threads, while the submission waited for the round (2 to 3.4 ms at a
/// time, their run city4).
#[cfg(windows)]
fn sync_jobs() -> (usize, usize) {
    #[cfg(feature = "diagnostics")]
    if let Some(fine) = crate::ab::pick("VICTORY_SYNC_FINE") {
        return if fine { (SYNC_PAGES_FINE, SYNC_AT_A_TIME_FINE) } else { (SYNC_PAGES, SYNC_AT_A_TIME) };
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_FINE").is_some_and(|v| v != "0")) {
        (SYNC_PAGES_FINE, SYNC_AT_A_TIME_FINE)
    } else {
        (SYNC_PAGES, SYNC_AT_A_TIME)
    }
}

/// Since the last frame line, of [`sync_heaps`]: changed pages of placed
/// shader-visible heaps looked at, slots on them, slots whose record was
/// not the one last seen and was mirrored, the microseconds it took; then
/// new records left to the draws that read them through a table, and new
/// records that waited for one ([`host::HostHeap::copied_wait`]); and the
/// microseconds submissions waited for it ([`sync_aside`]).
#[cfg(windows)]
pub(crate) static SYNCED: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// The same, of [`SYNCED`]'s pages: those in view memory that was guarded;
/// bytes read; nanoseconds asking what was written, reading it and comparing
/// it with what was last seen, making views.
#[cfg(windows)]
static SYNC_SPLIT: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// With the frame line: what [`sync_heaps`] did per frame over `frames`.
/// Nothing while no heap page changed.
#[cfg(windows)]
pub(crate) fn sync_line(frames: f64) {
    let [twins, moved, lost] = [0, 1, 2].map(|i| host::TWINS[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    if twins > 0 {
        tracing::info!(
            "raw twins of buffer views made per frame (VICTORY_RAW_VIEWS=0: none): {:.0}, of them {:.1} that do not begin on 16 bytes ({:.1} null)",
            twins as f64 / frames,
            moved as f64 / frames,
            lost as f64 / frames
        );
    }
    let [pages, looked, mirrored, micros, tabled, later, waited] = [0, 1, 2, 3, 4, 5, 6].map(|i| SYNCED[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let split = [0, 1, 2, 3, 4].map(|i| SYNC_SPLIT[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64);
    let kinds = [0, 1, 2, 3, 4].map(|i| SYNCED_KINDS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    let short = SYNC_SHORT.swap(0, std::sync::atomic::Ordering::Relaxed);
    let quick = [0, 1].map(|i| host::SPANS_QUICK[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    if quick[0] > 0.0 || sync_jobs().0 == SYNC_PAGES_FINE {
        tracing::info!(
            "writes of view slots per frame (VICTORY_SPANS_QUICK=1): {:.0} leaving the placed views' spans alone, {:.0} under \
             their lock; the heaps' records in jobs of {} pages (VICTORY_SYNC_FINE)",
            quick[0],
            quick[1],
            sync_jobs().0
        );
    }
    if pages == 0 {
        return;
    }
    if sync_in_place() {
        tracing::info!(
            "the heaps' records read where they are (VICTORY_SYNC_IN_PLACE=1): {:.1} pages per frame, {short} of them not all \
             there since the last line",
            pages as f64 / frames
        );
    }
    tracing::info!(
        "descriptor records the title copied into its heap as bytes (VICTORY_HEAP_SYNC=0: not looked for): per frame \
         {:.1} changed heap pages, {:.0} slots looked at, {:.1} whose record was new and was mirrored, {:.1} new ones of \
         draws' tables left to the draws, {:.1} that waited for a draw first, {:.2} ms ({:.2} ms waited for); of the mirrored \
         {:.0} no views (with those of slots that show none), {:.0} of objects, {:.0} others, {:.0} samplers, {:.0} that waited and no draw came; \
         {:.1} of the pages guarded views, {:.1} MB read; {:.2} ms asking, {:.2} reading and comparing, {:.2} making",
        pages as f64 / frames,
        looked as f64 / frames,
        mirrored as f64 / frames,
        tabled as f64 / frames,
        later as f64 / frames,
        micros as f64 / 1000.0 / frames,
        waited as f64 / 1000.0 / frames,
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3],
        kinds[4],
        split[0] / frames,
        split[1] / (1 << 20) as f64 / frames,
        split[2] / 1e6 / frames,
        split[3] / 1e6 / frames,
        split[4] / 1e6 / frames
    );
}

/// Title descriptor slots in a changed byte range that have host counterparts.
/// `range_end` is exclusive.
#[cfg(windows)]
fn changed_slots(
    range_from: usize,
    range_end: usize,
    heap_start: usize,
    step: u64,
    mirrored_count: u32,
) -> std::ops::Range<u64> {
    let from = (range_from - heap_start) as u64;
    let to = (range_end - heap_start) as u64;
    let first = from / step;
    let last = to.div_ceil(step).min(u64::from(mirrored_count));
    first.min(last)..last
}

/// The title's view heap that shaders of a list without one get their views
/// from (`draw.rs`, `bind`): the title's structure builder binds root views
/// only, and the host shows those, and memory by address, through heap slots.
/// The biggest there is. `VICTORY_LENT_HEAP=0`: none, such a dispatch is
/// dropped, as before.
#[cfg(windows)]
pub(crate) fn lent_heap() -> Option<usize> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_LENT_HEAP").is_none_or(|v| v != "0")) {
        return None;
    }
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, h)| (h, unsafe { &*(h as *const Heap) }))
        .filter(|(_, heap)| !heap.samplers() && matches!(heap.host.get(), Some(Some(_))))
        .max_by_key(|(_, heap)| heap.desc.count)
        .map(|(h, _)| h)
}

/// `f` of the title heap at `heap`, if that heap still exists; it stays
/// until `f` is done.
#[cfg(windows)]
pub(crate) fn with_heap<T>(heap: usize, f: impl FnOnce(&Heap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    // SAFETY: registered heaps are live (removed when dropped).
    heaps.iter().any(|&(_, _, h)| h == heap).then(|| f(unsafe { &*(heap as *const Heap) }))
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap is still there and `id` says it is the one that
/// took them ([`host::HostHeap::id`]): the title can release the heap a
/// list's slots came from, and a later heap can be made at the same address
/// before the list is reset.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, id: u64, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(id, first, n);
    }
}

/// Runs `f` with the host heap of the title's largest view heap, if there is
/// one (nothing is waited for).
#[cfg(windows)]
pub(crate) fn with_view_heap<T>(f: impl FnOnce(&host::HostHeap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.try_read().ok()?;
    let host = heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, at)| unsafe { &*(at as *const Heap) })
        .filter(|heap| !heap.samplers())
        .filter_map(|heap| heap.host.get()?.as_ref())
        .max_by_key(|host| host.count())?;
    Some(f(host))
}

/// Makes dynamic slot `index` of the host heap behind the title heap at
/// `heap`, if that heap still exists, the constant buffer view of `size`
/// bytes of `buffer` at `gpu`.
#[cfg(windows)]
pub(crate) fn constant_slot(
    heap: usize,
    index: u32,
    buffer: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    gpu: u64,
    size: u32,
) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.constant_at(index, buffer, gpu, size);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        {
            let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
            heaps.retain(|&(_, _, h)| h != at);
        }
        if let Some(Some(host)) = self.host.get() {
            crate::memory::release_heap(std::ptr::from_ref(host) as usize);
        }
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// All the view it records is made from: records with equal keys have
    /// equal views (of the same host texture).
    #[cfg(windows)]
    pub(crate) fn key(&self) -> (usize, [u32; 7]) {
        let mut words = [self.described; 7];
        words[1..].copy_from_slice(&self.desc);
        (self.resource as usize, words)
    }

    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // A target or depth view: of a target, in its stand-in format.
        words[0] = crate::resource::target_format(crate::resource::pc_format(words[0]));
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
///
/// Views without a resource object have nothing that releases them, and a
/// title copies descriptors as bytes, so nothing says when the last
/// descriptor naming an entry is gone: such entries were kept for the run.
/// The title writes some 14,000 constant buffer views a frame,
/// each at the address its constants have that frame: 2,450,139 entries
/// after four minutes of play, 1,047 MB with the index, and 11,000 more
/// every second (run 65). So the two kinds that carry an address are not
/// entries any more: their records name them whole ([`constant_words`],
/// [`placed_words`]), and the table holds only what is bounded: views of
/// resource objects (freed with the resource), samplers (as many as there
/// are sampler states), and the shapes of placed views (their descriptions
/// without the address: as many as the title has kinds of textures and
/// buffers, however often it moves them).
/// `VICTORY_VIEW_INLINE=0`: every view is an entry, as before.
///
/// The shapes were not bounded after all. A record numbers at most
/// [`SHAPES`] of them, and play-g reached that in about an hour; from
/// then on each placed view of a shape not seen before was a table entry,
/// which nothing frees: 5,404,074 of them at the end, and the table with
/// its indexes 3,301 MB. Each view and shape was kept twice, in the list
/// and again as a key of a tree index, whose nodes were millions of small
/// blocks. The indexes now hold a digest of each ([`digest`]) and the
/// number of its entry or shape, and the list is checked for the view
/// itself; a view whose digest another view already has is kept in a
/// small tree of its own.
///
/// What is still an entry without a resource (a placed view once the shapes
/// are full, or one past 1 TiB; a constant buffer view that does not fit
/// its record) is kept up to [`ownerless_cap`] entries, oldest freed first.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    /// The entry of each view by its digest, but those in `collided`.
    index: std::collections::HashMap<u64, u32, std::hash::BuildHasherDefault<Digested>>,
    /// Entries whose view's digest was taken by another view's entry.
    collided: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// The shapes of views of placed memory, and where each is (by digest,
    /// as the entries).
    shapes: Vec<Shape>,
    shape_index: std::collections::HashMap<u64, u32, std::hash::BuildHasherDefault<Digested>>,
    shapes_collided: std::collections::BTreeMap<Shape, u32>,
    /// For the memory line: the last new shape of each kind (unordered or
    /// not, resource dimension, view dimension), how often each word of a
    /// new shape differed from it ([`Views::note_new`]), and the placed
    /// views of title memory made table entries because the shapes were
    /// full.
    last_shapes: std::collections::BTreeMap<(bool, u32, u32), Shape>,
    varied: [u64; 28],
    past_shapes: u64,
    /// Entries no resource owns (constant buffer views and views of placed
    /// memory), oldest first, with the generation each was made at: past
    /// [`ownerless_cap`] the oldest is freed, as nothing else frees them.
    ownerless: std::collections::VecDeque<(u32, u32)>,
    /// Entries of [`ownerless`](Self::ownerless) freed under the cap since
    /// the start.
    ownerless_freed: u64,
}

/// How many table entries no resource owns are kept ([`Views::ownerless`]).
/// Nothing else frees them, and once the shapes of placed views are full
/// every placed view the title writes is one: TaynanGT/Vice-Tay saw the
/// 4,194,304 shapes full after half an hour of play, then some 15,000 such
/// entries a second, 34.9 million of them and 16.7 GB after 70 minutes, and
/// the frame rate halved. That many come from views written anew every
/// frame or few, so the oldest of a million are long out of use; a record
/// that still names a freed entry reads as stale (no view).
/// `VICTORY_OWNERLESS_VIEWS=<n>` keeps `n` (0: all, as before). By default
/// 1,048,576; all with `VICTORY_VIEW_INLINE=0`, where every constant buffer
/// view is such an entry (10,000 a second) and a million would last under
/// two minutes, so that switch stays as before.
fn ownerless_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_OWNERLESS_VIEWS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(if inline() { 1 << 20 } else { 0 })
    })
}

impl Views {
    const fn new() -> Views {
        Views {
            list: Vec::new(),
            index: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()),
            collided: std::collections::BTreeMap::new(),
            by_resource: std::collections::BTreeMap::new(),
            free: Vec::new(),
            shapes: Vec::new(),
            shape_index: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()),
            shapes_collided: std::collections::BTreeMap::new(),
            last_shapes: std::collections::BTreeMap::new(),
            varied: [0; 28],
            past_shapes: 0,
            ownerless: std::collections::VecDeque::new(),
            ownerless_freed: 0,
        }
    }
}

/// A view's or a shape's digest, which the indexes of [`Views`] keep in
/// place of the value.
fn digest<T: std::hash::Hash>(value: &T) -> u64 {
    use std::hash::BuildHasher;
    std::hash::BuildHasherDefault::<std::hash::DefaultHasher>::default().hash_one(value)
}

/// The hash of the indexes of [`Views`]: their keys are digests, spread
/// already.
#[derive(Default)]
struct Digested(u64);

impl std::hash::Hasher for Digested {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

/// Read far more often than written: every slot a draw reads through a
/// table is looked up here, on several recording threads at once in the
/// world. Readers share it; only a view not seen before (a new table entry
/// or shape) and a released resource take it alone.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views::new());
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// A record is the tag and two words, `i` and `generation` (the host keeps
/// them as one 64-bit key per slot). The top two bits of `generation` say
/// what the other 62 bits are: 0, entry `i` of the table at that
/// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
/// view of placed memory: its address and the number of its shape.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATIONS: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPES: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]). On by default; `VICTORY_VIEW_INLINE=0`
/// makes every view a table entry, as before.
fn inline() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_INLINE").is_none_or(|v| v != "0"))
}

/// Views named whole in their records since the start: constant buffer
/// views, views of placed memory (for the memory line).
static NAMED_WHOLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATIONS) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address. (The view comes without the words
/// the title leaves unwritten: [`placed_resource_words`],
/// [`placed_view_words`].)
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    (
        (unordered, *resource, rest),
        u64::from(desc[10]) | u64::from(desc[11]) << 32,
    )
}

/// Whether the filler of placed views' descriptions is kept as the title
/// left it ([`placed_resource_words`], [`placed_view_words`]). Off by
/// default; `VICTORY_PLACED_FILLER=1` keeps it, as before.
fn keep_placed_filler() -> bool {
    static KEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEPT.get_or_init(|| std::env::var_os("VICTORY_PLACED_FILLER").is_some_and(|v| v == "1"))
}

/// A placed view's resource description as words, its filler zeroed: word 1
/// (after the dimension, before the 8-byte alignment) and word 13 (after the
/// flags, to the 8-byte end). The title leaves there what its stack held
/// (`crate::resource::texture_at` keeps them out of its key for the same
/// reason); kept, they made a new shape for most views, 1.5 million shapes
/// by the end of play-g.
fn placed_resource_words(desc: *const [u32; 14]) -> Option<[u32; 14]> {
    let mut words = words(desc)?;
    if !keep_placed_filler() {
        (words[1], words[13]) = (0, 0);
    }
    Some(words)
}

/// By D3D12_SRV_DIMENSION (unknown, buffer, 1D, 1D array, 2D, 2D array, 2D
/// multisample, 2D multisample array, 3D, cube, cube array, acceleration
/// structure): the words of D3D12_SHADER_RESOURCE_VIEW_DESC's union that
/// the dimension's member takes, up to the end of its last field.
/// - Buffer (D3D12_BUFFER_SRV): first element (two words), count, stride,
///   flags: 5; the sixth is the member's padding to 8 bytes.
/// - 1D: most detailed mip, mips, LOD clamp: 3; 1D array: most detailed
///   mip, mips, first slice, slices, LOD clamp: 5.
/// - 2D: most detailed mip, mips, plane, LOD clamp: 4; 2D array: most
///   detailed mip, mips, first slice, slices, plane, LOD clamp: 6.
/// - 2D multisample: one field D3D12 does not read
///   (`UnusedField_NothingToDefine`): 0; its array: first slice, slices: 2.
/// - 3D and cube: most detailed mip, mips, LOD clamp: 3; cube array: most
///   detailed mip, mips, first face, cubes, LOD clamp: 5.
/// - Acceleration structure: its address (two words): 2.
const SRV_UNION_WORDS: [u8; 12] = [0, 5, 3, 5, 4, 6, 0, 2, 3, 3, 5, 2];

/// As [`SRV_UNION_WORDS`], of D3D12_UNORDERED_ACCESS_VIEW_DESC, by
/// D3D12_UAV_DIMENSION (unknown, buffer, 1D, 1D array, 2D, 2D array, 2D
/// multisample, 2D multisample array, 3D).
/// - Buffer (D3D12_BUFFER_UAV): first element (two words), count, stride,
///   counter offset (two words), flags: 7; the eighth is padding.
/// - 1D: mip: 1; 1D array: mip, first slice, slices: 3.
/// - 2D: mip, plane: 2; 2D array: mip, first slice, slices, plane: 4.
/// - 2D multisample: the unused field: 0; its array: first slice, slices: 2.
/// - 3D: mip, first depth slice, depth slices: 3.
const UAV_UNION_WORDS: [u8; 9] = [0, 7, 1, 3, 2, 4, 0, 2, 3];

/// Which of the sixteen words of a placed view's description tell it apart
/// (as bits) for view dimension `dimension`; `None` for a dimension D3D12
/// has not. Words 0 to 9 are the PC's view description: for a shader
/// resource view the format, the dimension and the component mapping, a
/// word of padding (the union holds 8-byte fields, so it starts at byte 16)
/// and the union's six words; for an unordered access view the format, the
/// dimension and the union's eight. Each dimension takes the head of the
/// union only. Words 10 and 11 are the address and 12 to 15 the console's
/// own: kept.
fn placed_view_used(unordered: bool, dimension: u32) -> Option<u16> {
    let (head, union, by_dimension): (u16, u32, &[u8]) = if unordered {
        (0b11, 2, &UAV_UNION_WORDS)
    } else {
        (0b111, 4, &SRV_UNION_WORDS)
    };
    if dimension == 0 {
        return None;
    }
    let n = *by_dimension.get(dimension as usize)?;
    Some(head | ((1u16 << n) - 1) << union | 0xfc00)
}

/// A placed view's description as words, those its dimension does not take
/// zeroed ([`placed_view_used`]). The title leaves them unwritten:
/// TaynanGT/Vice-Tay counted 131,392 values in word 3 and 36,413 in word 9
/// over a million shapes, each a new shape for the same view. A shader
/// resource view's word 3 is padding whatever its dimension and always
/// goes. Kept whole otherwise: a dimension D3D12 has not, and a view of a
/// buffer that names another dimension (the host reads the buffer member of
/// its words whatever the dimension says: `host::view_stride`,
/// `host::make`).
fn placed_view_words(unordered: bool, resource: &[u32; 14], desc: *const [u32; 16]) -> Option<[u32; 16]> {
    const BUFFER: u32 = 1;
    let mut words = words(desc)?;
    if keep_placed_filler() {
        return Some(words);
    }
    if !unordered {
        words[3] = 0;
    }
    if resource[0] == BUFFER && words[1] != BUFFER {
        return Some(words);
    }
    if let Some(used) = placed_view_used(unordered, words[1]) {
        for (k, word) in words.iter_mut().enumerate() {
            if used >> k & 1 == 0 {
                *word = 0;
            }
        }
    }
    Some(words)
}

/// The record words that name the view of placed memory at `address` with
/// shape number `shape`.
fn placed_words(shape: u32, address: u64) -> [u32; 2] {
    kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address)
}

impl Views {
    /// The number of `shape`, of digest `digest`, if it has one.
    fn shape_at(&self, shape: &Shape, digest: u64) -> Option<u32> {
        match self.shape_index.get(&digest) {
            Some(&at) if self.shapes[at as usize] == *shape => Some(at),
            _ if self.shapes_collided.is_empty() => None,
            _ => self.shapes_collided.get(shape).copied(),
        }
    }

    /// Whether every number a record has for a shape is taken.
    fn shapes_full(&self) -> bool {
        self.shapes.len() >= SHAPES
    }

    /// The number of `shape`, of digest `digest`, a new one if it is not
    /// there yet; `None` when the shapes are full (the view is then a table
    /// entry).
    fn shape(&mut self, shape: Shape, digest: u64) -> Option<u32> {
        if let Some(at) = self.shape_at(&shape, digest) {
            return Some(at);
        }
        if self.shapes_full() {
            return None;
        }
        let at = self.shapes.len() as u32;
        self.note_new(&shape);
        match self.shape_index.entry(digest) {
            std::collections::hash_map::Entry::Vacant(free) => {
                free.insert(at);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                self.shapes_collided.insert(shape, at);
            }
        }
        self.shapes.push(shape);
        Some(at)
    }

    /// Counts the words of the new shape `shape` that differ from the last
    /// new shape of its kind: the words that make shapes new.
    fn note_new(&mut self, shape: &Shape) {
        let kind = (shape.0, shape.1[0], shape.2[1]);
        let Some(last) = self.last_shapes.insert(kind, *shape) else {
            return;
        };
        let words = |s: &Shape| s.1.into_iter().chain(s.2);
        for (k, (a, b)) in words(&last).zip(words(shape)).enumerate() {
            if a != b {
                self.varied[k] += 1;
            }
        }
    }

    /// The view of placed memory [`placed_words`] named.
    fn placed_of(&self, i: u32, generation: u32) -> Option<View> {
        let payload = payload(i, generation);
        let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
        let &(unordered, resource, rest) = self.shapes.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&rest[..10]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        desc[12..].copy_from_slice(&rest[10..]);
        Some(View::Placed(unordered, resource, desc))
    }

    /// The entry of `view`, of digest `digest`, if it has one.
    fn entry_at(&self, view: &View, digest: u64) -> Option<u32> {
        match self.index.get(&digest) {
            Some(&i) if self.list[i as usize].0 == *view => Some(i),
            _ if self.collided.is_empty() => None,
            _ => self.collided.get(view).copied(),
        }
    }

    /// The record words of `view` as a table entry, if it has one.
    fn entry_of(&self, view: &View, digest: u64) -> Option<[u32; 2]> {
        let i = self.entry_at(view, digest)?;
        Some([i, self.list[i as usize].1])
    }

    /// The record words of `view` as a table entry: its entry (a new or a
    /// freed one if the view is not there yet) and that entry's generation.
    fn entry(&mut self, view: View, digest: u64) -> [u32; 2] {
        self.entry_kept(view, digest, ownerless_cap())
    }

    /// [`entry`](Self::entry), keeping at most `cap` entries no resource
    /// owns (0: all).
    fn entry_kept(&mut self, view: View, digest: u64, cap: usize) -> [u32; 2] {
        if let Some(words) = self.entry_of(&view, digest) {
            return words;
        }
        let ownerless = matches!(view, View::Constant(..) | View::Placed(..));
        let words = self.entry_new(view, digest);
        if ownerless {
            self.ownerless.push_back((words[0], words[1]));
            while cap != 0 && self.ownerless.len() > cap {
                let Some((i, generation)) = self.ownerless.pop_front() else {
                    break;
                };
                self.free_ownerless(i, generation);
            }
        }
        words
    }

    /// Frees entry `i` if it still holds the view it had at `generation`,
    /// as [`release`](Self::release) frees the entries of a released
    /// resource.
    fn free_ownerless(&mut self, i: u32, generation: u32) {
        let Some((view, current)) = self.list.get(i as usize).cloned() else {
            return;
        };
        let digest = digest(&view);
        if current != generation || self.entry_at(&view, digest) != Some(i) {
            return;
        }
        if self.index.get(&digest) == Some(&i) {
            self.index.remove(&digest);
        } else {
            self.collided.remove(&view);
        }
        // (Below the bits that say a record is no table entry.)
        self.list[i as usize].1 = ((generation + 1) & GENERATIONS).max(1);
        self.free.push(i);
        self.ownerless_freed += 1;
        // Slots whose record names it read nothing now: the pass over the
        // heap's buffer views looks at them again.
        note_forgotten(i);
    }

    /// A new entry for `view`, which has none.
    fn entry_new(&mut self, view: View, digest: u64) -> [u32; 2] {
        if let View::Placed(unordered, resource, desc) = &view
            && self.shapes_full()
        {
            let (shape, address) = shape_of(*unordered, resource, desc);
            if address >> PLACED_ADDRESS_BITS == 0 {
                self.past_shapes += 1;
                self.note_new(&shape);
            }
        }
        let i = match self.free.pop() {
            Some(i) => {
                self.list[i as usize].0 = view.clone();
                i
            }
            None => {
                self.list.push((view.clone(), 1));
                self.list.len() as u32 - 1
            }
        };
        for r in view.resources().into_iter().filter(|&r| r != 0) {
            self.by_resource.entry(r).or_default().push(i);
        }
        match self.index.entry(digest) {
            std::collections::hash_map::Entry::Vacant(free) => {
                free.insert(i);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                self.collided.insert(view, i);
            }
        }
        [i, self.list[i as usize].1]
    }

    /// Frees the entries of views naming `resource`, under a new
    /// generation each; the entries freed.
    fn release(&mut self, resource: usize) -> Vec<u32> {
        let Some(entries) = self.by_resource.remove(&resource) else {
            return Vec::new();
        };
        let mut freed = Vec::with_capacity(entries.len());
        for i in entries {
            let (view, generation) = self.list[i as usize].clone();
            // Entries freed through a view's other resource may hold another
            // view by now.
            let digest = digest(&view);
            if self.entry_at(&view, digest) != Some(i) || !view.resources().contains(&resource) {
                continue;
            }
            if self.index.get(&digest) == Some(&i) {
                self.index.remove(&digest);
            } else {
                self.collided.remove(&view);
            }
            // (Below the bits that say a record is no table entry.)
            self.list[i as usize].1 = ((generation + 1) & GENERATIONS).max(1);
            // The view's other resource (a counter) keeps a stale entry, which
            // `forget` of that resource skips.
            self.free.push(i);
            freed.push(i);
        }
        freed
    }
}

/// The two words that name `view` in a record: the view itself where its
/// kind allows ([`Views`]), else its table entry.
fn words_for(view: View) -> [u32; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    if inline() {
        match &view {
            View::Constant(address, size) => {
                if let Some(words) = constant_words(*address, *size) {
                    NAMED_WHOLE[0].fetch_add(1, Relaxed);
                    return words;
                }
            }
            View::Placed(unordered, resource, desc) => {
                let (shape, address) = shape_of(*unordered, resource, desc);
                if address >> PLACED_ADDRESS_BITS == 0 {
                    // Nearly every placed view has a shape seen before. Once
                    // the shapes are full a new one takes no write lock for
                    // its shape, only for its entry.
                    let digest = digest(&shape);
                    let (known, full) = {
                        let views = views_read();
                        (views.shape_at(&shape, digest), views.shapes_full())
                    };
                    let shape = known.or_else(|| if full { None } else { views_write().shape(shape, digest) });
                    if let Some(shape) = shape {
                        NAMED_WHOLE[1].fetch_add(1, Relaxed);
                        return placed_words(shape, address);
                    }
                }
            }
            _ => {}
        }
    }
    // A view written before keeps its entry and generation.
    let digest = digest(&view);
    if let Some(words) = views_read().entry_of(&view, digest) {
        return words;
    }
    views_write().entry(view, digest)
}

fn views_read() -> std::sync::RwLockReadGuard<'static, Views> {
    VIEWS.read().unwrap_or_else(|e| e.into_inner())
}

fn views_write() -> std::sync::RwLockWriteGuard<'static, Views> {
    VIEWS.write().unwrap_or_else(|e| e.into_inner())
}

/// For the memory line: the distinct views kept ([`Views`]), those of them
/// free for reuse, and the bytes the table and its index hold for them (an
/// estimate: an entry of the table, a place in the index for its digest and
/// number, and a tree entry for each view whose digest was taken; the
/// shapes likewise); then the entries that are constant
/// buffer views and views of placed memory (those that did not fit a
/// record, or all of them with `VICTORY_VIEW_INLINE=0`), the shapes of
/// placed views kept, and the views named whole in their records since the
/// start: constant buffer views, views of placed memory.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn views_kept() -> [u64; 8] {
    use std::sync::atomic::Ordering::Relaxed;
    let views = views_read();
    let each = size_of::<(View, u32)>() as u64;
    let shape = size_of::<Shape>() as u64;
    let (mut constants, mut placed) = (0, 0);
    for (view, _) in &views.list {
        match view {
            View::Constant(..) => constants += 1,
            View::Placed(..) => placed += 1,
            _ => {}
        }
    }
    [
        views.list.len() as u64,
        views.free.len() as u64,
        views.list.capacity() as u64 * each
            + views.index.capacity() as u64 * 17 * 8 / 7
            + views.collided.len() as u64 * each * 3 / 2
            + views.shapes.capacity() as u64 * shape
            + views.shape_index.capacity() as u64 * 17 * 8 / 7
            + views.shapes_collided.len() as u64 * shape * 3 / 2,
        constants,
        placed,
        views.shapes.len() as u64,
        NAMED_WHOLE[0].load(Relaxed),
        NAMED_WHOLE[1].load(Relaxed),
    ]
}

/// For the memory line: the placed views of title memory made table
/// entries because the shapes were full, the views and shapes kept by
/// themselves because another had their digest, and how often each word
/// of a new shape differed from the last new shape of its kind (the 14
/// words of the resource description, then the view's 0 to 9 and 12 to
/// 15).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn shapes_new() -> (u64, u64, [u64; 28]) {
    let views = views_read();
    (
        views.past_shapes,
        (views.collided.len() + views.shapes_collided.len()) as u64,
        views.varied,
    )
}

/// For the memory line: the entries no resource owns ([`Views::ownerless`]),
/// the most kept ([`ownerless_cap`], 0: all), and how many were freed under
/// it since the start.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn ownerless_kept() -> [u64; 3] {
    let views = views_read();
    [views.ownerless.len() as u64, ownerless_cap() as u64, views.ownerless_freed]
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }

    /// The texture object the view is of, if that texture has no host copy
    /// yet (`crate::resource::host_waits`): a slot written with such a
    /// view is left null until the texture is first used.
    #[cfg(windows)]
    fn waiting_texture(&self) -> Option<usize> {
        match *self {
            View::Shader(r, _) | View::Unordered(r, _, _)
                if crate::resource::host_waits(r as This) =>
            {
                Some(r)
            }
            _ => None,
        }
    }
}

/// Title slots (their handles) written with a view of a texture object
/// that had no host copy then, by texture. A title texture gets its host
/// copy on first use (`resource.rs` `Hosted`), and the title
/// writes each texture's views into its view heap when it creates the
/// texture: making those host slots would make every host copy at once.
/// They are made when the texture is: translated shaders index the heap
/// directly, so a slot is not always made again by the draw that reads it.
#[cfg(windows)]
static WAITING: std::sync::Mutex<std::collections::BTreeMap<usize, Vec<u64>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Textures given their host copy since [`make_waiting`] last ran.
#[cfg(windows)]
static MADE_LATE: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Title texture `resource` was given its host copy (called under that
/// texture's own lock: nothing else is locked here).
#[cfg(windows)]
pub(crate) fn texture_made(resource: usize) {
    MADE_LATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(resource);
}

/// The slot at `handle` was written with a view of texture `resource`,
/// which has no host copy: it waits for one.
#[cfg(windows)]
fn wait_for(resource: usize, handle: u64) {
    /// A texture with more slots waiting than this is in use for all we
    /// know (its views are being copied about): its host copy is made.
    const MOST: usize = 64;
    let many = {
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        let slots = waiting.entry(resource).or_default();
        if !slots.contains(&handle) {
            slots.push(handle);
        }
        slots.len() > MOST
    };
    if many {
        // Makes it, and notes it made.
        let _ = crate::resource::host_copy(resource as This);
    } else if !crate::resource::host_waits(resource as This) {
        // Made by another thread since the slot was looked at: that making
        // may have been taken by `make_waiting` before this slot was noted.
        texture_made(resource);
    }
}

/// Makes the host slots that waited for a texture's host copy, for the
/// textures made since the last call. Called before a submission runs, by
/// the thread that submits: a slot is read when a list runs.
#[cfg(windows)]
pub(crate) fn make_waiting() {
    let made = std::mem::take(&mut *MADE_LATE.lock().unwrap_or_else(|e| e.into_inner()));
    for resource in made {
        let handles = WAITING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&resource);
        for handle in handles.into_iter().flatten() {
            // Only if the slot still holds a view of that texture: the
            // title may have written it since (and a view of another
            // texture without a host copy must go on waiting).
            // (Looked up first: the handle is read only while its heap is
            // there.)
            // (The heap stays while the lock is held: `mirror`.)
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            let Some(&(_, _, at)) = heaps.iter().find(|&&(start, end, _)| (start..end).contains(&handle)) else {
                continue;
            };
            // SAFETY: a registered heap, live while the lock is held.
            let heap = unsafe { &*(at as *const Heap) };
            let Some(index) = heap.index(handle) else {
                continue;
            };
            let same = match view(handle) {
                Some(View::Shader(r, _) | View::Unordered(r, _, _)) => r == resource,
                _ => false,
            };
            if same && let Some(Some(host)) = heap.host.get() {
                host.mirror(index, handle, true);
            }
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let [i, generation] = words_for(view);
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    #[cfg(windows)]
    mirror(handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    #[cfg(windows)]
    WAITING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&resource);
    if !views_read().by_resource.contains_key(&resource) {
        return;
    }
    let freed = views_write().release(resource);
    for i in freed {
        note_forgotten(i);
    }
}

/// Table entries freed since the pass over the heap's buffer views last
/// asked ([`take_forgotten`]): records that name them read nothing now,
/// though no slot was written. Kept only once a pass has asked.
static FORGOTTEN: std::sync::Mutex<(bool, bool, Vec<u32>)> = std::sync::Mutex::new((false, false, Vec::new()));

/// More entries than this between two asks: the list is dropped, and the
/// next ask is told it does not know.
const FORGOTTEN_MOST: usize = 1 << 20;

fn note_forgotten(i: u32) {
    let mut forgotten = FORGOTTEN.lock().unwrap_or_else(|e| e.into_inner());
    let (asked, overflowed, list) = &mut *forgotten;
    if !*asked || *overflowed {
        return;
    }
    if list.len() >= FORGOTTEN_MOST {
        *overflowed = true;
        *list = Vec::new();
        return;
    }
    list.push(i);
}

/// The table entries freed since the last call (in any order, maybe
/// twice); `None` on the first call and when too many were to be kept.
pub(crate) fn take_forgotten() -> Option<Vec<u32>> {
    let mut forgotten = FORGOTTEN.lock().unwrap_or_else(|e| e.into_inner());
    let (asked, overflowed, list) = &mut *forgotten;
    let known = *asked && !*overflowed;
    *asked = true;
    *overflowed = false;
    let list = std::mem::take(list);
    known.then_some(list)
}

/// The record at `handle` (a descriptor slot in title memory): its tag and
/// two words, read as they are.
pub(crate) fn record_words(handle: u64) -> [u32; 3] {
    // SAFETY: the handle is a descriptor slot in title memory, at least 16
    // bytes.
    unsafe { (handle as *const [u32; 3]).read_unaligned() }
}

/// The view table entry a record names, if it names one (not a view named
/// whole in its record, not a record of no view).
pub(crate) fn record_entry(record: [u32; 3]) -> Option<u32> {
    let [tag, i, generation] = record;
    (tag == VIEW_TAG && generation >> KIND_SHIFT == 0).then_some(i)
}

/// The view the record words `i` and `generation` name: the one they hold
/// whole, or the view of entry `i` if `generation` is still its own.
fn view_at(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        0 => {
            let views = views_read();
            let (view, current) = views.list.get(i as usize)?;
            (*current == generation).then(|| view.clone())
        }
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => views_read().placed_of(i, generation),
        _ => None,
    }
}

/// The words of the record at `handle`, if they are a view's.
pub(crate) fn record_at(handle: u64) -> Option<[u32; 3]> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let words = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    (words[0] == VIEW_TAG).then_some(words)
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    view_at(i, generation)
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
///
/// A null `desc` writes a null CBV (`Constant(0, 0)`), matching D3D12's
/// `CreateConstantBufferView` with a null description. Leaving the slot
/// untouched would keep a stale view active.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    let view = match words(desc) {
        Some([lo, hi, size, _]) => {
            let address = u64::from(lo) | u64::from(hi) << 32;
            View::Constant(address, size)
        }
        None => View::Constant(0, 0),
    };
    record(handle, view);
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`. What the title leaves unwritten in both descriptions is left
/// out ([`placed_resource_words`], [`placed_view_words`]): one texture's
/// views are one shape however often it writes them.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    let Some(resource) = placed_resource_words(resource) else {
        return;
    };
    if let Some(desc) = placed_view_words(false, &resource, desc) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    let Some(resource) = placed_resource_words(resource) else {
        return;
    };
    if let Some(desc) = placed_view_words(true, &resource, desc) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        #[cfg(windows)]
        unsafe {
            crate::titlewrite::put(dst as *mut u8, src as *const u8, len as usize)
        };
        // SAFETY: as above.
        #[cfg(not(windows))]
        unsafe {
            std::ptr::copy(src as *const u8, dst as *mut u8, len as usize)
        };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(all(test, windows))]
mod safety_tests {
    use super::*;
    use crate::object::S_OK;

    #[test]
    fn released_view_heap_evicts_memory_heap_entry() {
        #[repr(C)]
        struct Extended(HeapDesc, Placement);

        struct HeapGuard(This);
        impl HeapGuard {
            fn release_now(&mut self) -> u32 {
                let ptr = std::mem::replace(&mut self.0, std::ptr::null());
                if ptr.is_null() { 0 } else { release(ptr) }
            }
        }
        impl Drop for HeapGuard {
            fn drop(&mut self) {
                let _ = self.release_now();
            }
        }

        let Some(host) = crate::host::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let extended = Extended(
            HeapDesc {
                kind: VIEWS_KIND,
                count: 64,
                flags: SHADER_VISIBLE,
                nodes: 1,
            },
            Placement {
                address: 0,
                size: 0,
            },
        );
        let mut raw = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], &mut raw),
            S_OK
        );
        let mut h = HeapGuard(raw);
        let Some(host_heap) = heap(h.0).host() else {
            return;
        };
        let key = std::ptr::from_ref(host_heap) as usize;
        if crate::memory::ready(
            &list,
            host_heap,
            &[],
            &[],
            &[],
            crate::memory::Use::Reads,
            None,
        )
        .is_none()
        {
            return;
        }
        assert!(crate::memory::has_heap(key));
        assert_eq!(h.release_now(), 0);
        assert!(!crate::memory::has_heap(key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[cfg(windows)]
    #[test]
    fn raw_heap_sync_stops_before_dynamic_host_slots() {
        let start = 0x1000usize;
        let step = 32;
        let capped = 5;
        assert_eq!(
            changed_slots(
                start + 3 * step + 4,
                start + 4 * step + 8,
                start,
                step as u64,
                capped
            ),
            3..5
        );
        assert_eq!(
            changed_slots(
                start + 6 * step,
                start + 9 * step,
                start,
                step as u64,
                capped
            ),
            5..5
        );
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A descriptor copied into a placed heap as bytes is mirrored before
    /// the next submission, even though no descriptor method saw the copy.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 2];
            let source = elsewhere.as_mut_ptr() as u64;
            let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
            create_sampler(std::ptr::null(), &sampler, source);
            // SAFETY: the heap slot and source both hold 16 descriptor bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source as *const u8,
                    memory.cast::<u8>().add(5 * 16),
                    16,
                )
            };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5));
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes
    /// up to 4 MiB; what does not fit says so.
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(generation >> KIND_SHIFT, CONSTANT);
                assert_eq!(view_at(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
    }

    /// Views of one placed texture whose resource descriptions differ only
    /// in the filler words keep one shape.
    #[test]
    fn placed_views_that_differ_in_filler_share_a_shape() {
        if !inline() || keep_placed_filler() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        let resource: [u32; 14] = std::array::from_fn(|k| 0x5ef0_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a80_0000 + k as u32);
        (desc[10], desc[11]) = (0x4940_0000, 0xd);
        let mut blank = resource;
        (blank[1], blank[13]) = (0, 0);
        let shapes = || VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == blank).count();
        let mut blank_desc = desc;
        blank_desc[3] = 0;
        for n in 0..100u32 {
            let (mut stack, mut stack_desc) = (resource, desc);
            (stack[1], stack[13], stack_desc[3]) = (n.wrapping_mul(0x9e37_79b9), !n, n ^ 0xb717_5a38);
            create_placed_srv(null, &stack, &stack_desc, at);
            // The view comes back without its filler.
            assert_eq!(view(at), Some(View::Placed(false, blank, blank_desc)));
        }
        assert_eq!(shapes(), 1, "one shape for 100 fillers");
        // A buffer view's word 9, shader resource and unordered access alike.
        (desc[1], blank_desc[1]) = (1, 1);
        blank_desc[9] = 0;
        for n in 0..100u32 {
            let mut stack_desc = desc;
            (stack_desc[3], stack_desc[9]) = (n, n.wrapping_mul(0x9e37_79b9));
            create_placed_srv(null, &blank, &stack_desc, at);
            assert_eq!(view(at), Some(View::Placed(false, blank, blank_desc)));
            create_placed_uav(null, &blank, &stack_desc, at);
            let mut unordered = blank_desc;
            unordered[3] = n;
            assert_eq!(view(at), Some(View::Placed(true, blank, unordered)));
        }
        assert_eq!(shapes(), 102, "a buffer shape for views, one for each word 3 of unordered ones");
    }

    /// The leak of run 65: a title that writes constant buffer views and
    /// views of placed memory at ever new addresses adds nothing to the
    /// table; one shape stands for all the placed views alike.
    #[test]
    fn views_at_new_addresses_add_no_entries() {
        if !inline() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        // A resource description and a view no other test uses, with no
        // filler (words 1 and 13 and view word 3 leave the shape: the test
        // above).
        let resource: [u32; 14] = std::array::from_fn(|k| if k == 1 || k == 13 { 0 } else { 0x5ee0_0000 + k as u32 });
        let mut desc: [u32; 16] = std::array::from_fn(|k| if k == 3 { 0 } else { 0x7a70_0000 + k as u32 });
        let mine = |v: &View| match v {
            View::Constant(address, _) => (0x6_0000_0000..0x6_1000_0000).contains(address),
            View::Placed(_, r, _) => *r == resource,
            _ => false,
        };
        let entries = || VIEWS.read().unwrap().list.iter().filter(|(v, _)| mine(v)).count();
        let shapes = || VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == resource).count();
        for n in 0..20_000u64 {
            let address = 0x6_0000_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            let placed = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (placed as u32, (placed >> 32) as u32);
            create_placed_srv(null, &resource, &desc, at);
            assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        }
        assert_eq!(entries(), 0, "no table entries");
        assert_eq!(shapes(), 1, "one shape for 20,000 placed views");
        // The same shape, unordered: another shape.
        create_placed_uav(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(true, resource, desc)));
        assert_eq!(shapes(), 2);
        // Placed memory past 1 TiB (not title memory) is a table entry.
        (desc[10], desc[11]) = (0, 0x100);
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        assert_eq!(entries(), 1);
        // A record's words are the same for the same view (the host knows
        // a slot's view by them), and differ with the address.
        // SAFETY: both handles are slots of 32 bytes above.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        let other = [0u64; 4];
        create_cbv(null, &[0x40, 6, 512, 0], at);
        create_cbv(null, &[0x40, 6, 512, 0], other.as_ptr() as u64);
        assert_eq!(words(at), words(other.as_ptr() as u64));
        create_cbv(null, &[0x140, 6, 512, 0], other.as_ptr() as u64);
        assert_ne!(words(at), words(other.as_ptr() as u64));
    }

    /// The words [`placed_view_used`] keeps are D3D12's, as the windows
    /// crate lays its view descriptions out: the head, and of the union
    /// the member of the view's dimension up to the end of its last field.
    #[cfg(windows)]
    #[test]
    fn placed_view_words_follow_the_d3d12_layout() {
        use std::mem::offset_of;
        use windows::Win32::Graphics::Direct3D12::*;
        fn size_of_field<M, F>(_: impl Fn(&M) -> &F) -> usize {
            size_of::<F>()
        }
        // Words of `$t` up to the end of its field `$last`.
        macro_rules! words_to {
            ($t:ty, $last:ident) => {
                (offset_of!($t, $last) + size_of_field(|m: &$t| &m.$last)) / 4
            };
        }
        let srv_head = words_to!(D3D12_SHADER_RESOURCE_VIEW_DESC, Shader4ComponentMapping);
        let uav_head = words_to!(D3D12_UNORDERED_ACCESS_VIEW_DESC, ViewDimension);
        let srv_union = offset_of!(D3D12_SHADER_RESOURCE_VIEW_DESC, Anonymous) / 4;
        let uav_union = offset_of!(D3D12_UNORDERED_ACCESS_VIEW_DESC, Anonymous) / 4;
        // Ten words each; the shader resource view's word 3 is padding.
        assert_eq!((srv_head, srv_union, uav_head, uav_union), (3, 4, 2, 2));
        assert_eq!(size_of::<D3D12_SHADER_RESOURCE_VIEW_DESC>(), 40);
        assert_eq!(size_of::<D3D12_UNORDERED_ACCESS_VIEW_DESC>(), 40);
        // The multisample members hold one field, which D3D12 does not
        // read (`UnusedField_NothingToDefine`): no word.
        assert_eq!(size_of::<D3D12_TEX2DMS_SRV>(), 4);
        assert_eq!(size_of::<D3D12_TEX2DMS_UAV>(), 4);
        let srv = [
            (D3D12_SRV_DIMENSION_BUFFER, words_to!(D3D12_BUFFER_SRV, Flags)),
            (D3D12_SRV_DIMENSION_TEXTURE1D, words_to!(D3D12_TEX1D_SRV, ResourceMinLODClamp)),
            (D3D12_SRV_DIMENSION_TEXTURE1DARRAY, words_to!(D3D12_TEX1D_ARRAY_SRV, ResourceMinLODClamp)),
            (D3D12_SRV_DIMENSION_TEXTURE2D, words_to!(D3D12_TEX2D_SRV, ResourceMinLODClamp)),
            (D3D12_SRV_DIMENSION_TEXTURE2DARRAY, words_to!(D3D12_TEX2D_ARRAY_SRV, ResourceMinLODClamp)),
            (D3D12_SRV_DIMENSION_TEXTURE2DMS, 0),
            (D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY, words_to!(D3D12_TEX2DMS_ARRAY_SRV, ArraySize)),
            (D3D12_SRV_DIMENSION_TEXTURE3D, words_to!(D3D12_TEX3D_SRV, ResourceMinLODClamp)),
            (D3D12_SRV_DIMENSION_TEXTURECUBE, words_to!(D3D12_TEXCUBE_SRV, ResourceMinLODClamp)),
            (D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, words_to!(D3D12_TEXCUBE_ARRAY_SRV, ResourceMinLODClamp)),
            (
                D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                words_to!(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV, Location),
            ),
        ];
        let uav = [
            (D3D12_UAV_DIMENSION_BUFFER, words_to!(D3D12_BUFFER_UAV, Flags)),
            (D3D12_UAV_DIMENSION_TEXTURE1D, words_to!(D3D12_TEX1D_UAV, MipSlice)),
            (D3D12_UAV_DIMENSION_TEXTURE1DARRAY, words_to!(D3D12_TEX1D_ARRAY_UAV, ArraySize)),
            (D3D12_UAV_DIMENSION_TEXTURE2D, words_to!(D3D12_TEX2D_UAV, PlaneSlice)),
            (D3D12_UAV_DIMENSION_TEXTURE2DARRAY, words_to!(D3D12_TEX2D_ARRAY_UAV, PlaneSlice)),
            (D3D12_UAV_DIMENSION_TEXTURE2DMS, 0),
            (D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY, words_to!(D3D12_TEX2DMS_ARRAY_UAV, ArraySize)),
            (D3D12_UAV_DIMENSION_TEXTURE3D, words_to!(D3D12_TEX3D_UAV, WSize)),
        ];
        // Every dimension D3D12 has, in order: 1 to 11, and 1 to 8.
        assert_eq!(srv.map(|(d, _)| d.0), std::array::from_fn(|k| k as i32 + 1));
        assert_eq!(uav.map(|(d, _)| d.0), std::array::from_fn(|k| k as i32 + 1));
        let mask = |head: usize, union: usize, n: usize| ((1u16 << head) - 1) | ((1u16 << n) - 1) << union | 0xfc00;
        for (dimension, n) in srv {
            assert!(n <= 6, "inside the union");
            let expected = mask(srv_head, srv_union, n);
            assert_eq!(placed_view_used(false, dimension.0 as u32), Some(expected), "SRV dimension {}", dimension.0);
        }
        for (dimension, n) in uav {
            assert!(n <= 8, "inside the union");
            let expected = mask(uav_head, uav_union, n);
            assert_eq!(placed_view_used(true, dimension.0 as u32), Some(expected), "UAV dimension {}", dimension.0);
        }
        // Unknown and past the last: every word counts.
        for (unordered, past) in [(false, 12), (true, 9)] {
            assert_eq!(placed_view_used(unordered, 0), None);
            assert_eq!(placed_view_used(unordered, past), None);
        }
    }

    /// Placed views of one texture are one shape whatever the words their
    /// dimension does not take and the resource description's filler hold,
    /// and two when a word that is taken differs: for every dimension of
    /// both kinds, every word (TaynanGT/Vice-Tay: 4,194,304 shapes full in
    /// half an hour of play, then a table entry kept for the run per view).
    #[test]
    fn placed_views_of_one_texture_share_a_shape_whatever_their_filler() {
        if !inline() || keep_placed_filler() {
            return;
        }
        let null = std::ptr::null();
        let slot = [0u64; 4];
        let at = slot.as_ptr() as u64;
        // Title memory (below 1 TiB): words 10 and 11, not in the shape.
        let address = 0x3_5ad0_0000u64;
        let kinds = [
            (false, create_placed_srv as extern "system" fn(_, _, _, _), 0..=12u32),
            (true, create_placed_uav, 0..=9),
        ];
        for (unordered, create, dimensions) in kinds {
            for dimension in dimensions {
                // Words no other test uses; a buffer view of a buffer, the
                // others of a 2D texture.
                let tag = 0x5ad0_0000 | u32::from(unordered) << 12 | dimension << 6;
                let mut resource: [u32; 14] = std::array::from_fn(|k| tag | k as u32);
                resource[0] = if dimension == 1 { 1 } else { 3 };
                let mut desc: [u32; 16] = std::array::from_fn(|k| tag | 0x20 | k as u32);
                desc[1] = dimension;
                (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
                // A shader resource view's word 3 is padding whatever the
                // dimension.
                let used = placed_view_used(unordered, dimension);
                let taken = |k: usize| !(k == 3 && !unordered) && used.is_none_or(|u| u >> k & 1 != 0);
                let mut kept = desc;
                for (k, word) in kept.iter_mut().enumerate() {
                    if !taken(k) {
                        *word = 0;
                    }
                }
                let mut stored = resource;
                (stored[1], stored[13]) = (0, 0);
                create(null, &resource, &desc, at);
                let first = record_words(at);
                let what = format!("unordered {unordered}, dimension {dimension}");
                assert_eq!(first[2] >> KIND_SHIFT, PLACED, "{what}: named whole");
                assert_eq!(view(at), Some(View::Placed(unordered, stored, kept)), "{what}");
                for k in (0..16).filter(|k| !(10..12).contains(k)) {
                    let mut other = desc;
                    other[k] ^= 0x4000_0000;
                    create(null, &resource, &other, at);
                    if taken(k) {
                        assert_ne!(record_words(at), first, "{what}: view word {k} is taken, two shapes");
                    } else {
                        assert_eq!(record_words(at), first, "{what}: view word {k} is not, one shape");
                    }
                }
                for k in 0..14 {
                    let mut other = resource;
                    other[k] ^= 0x4000_0000;
                    create(null, &other, &desc, at);
                    if k == 1 || k == 13 {
                        assert_eq!(record_words(at), first, "{what}: resource filler word {k}, one shape");
                    } else {
                        assert_ne!(record_words(at), first, "{what}: resource word {k}, two shapes");
                    }
                }
            }
        }
        // A view of a buffer that names a texture dimension is kept whole
        // (but its word 3): the host reads its words as a buffer view's
        // whatever it names.
        let mut resource: [u32; 14] = std::array::from_fn(|k| 0x5ad1_0000 | k as u32);
        resource[0] = 1;
        let mut desc = [0u32; 16];
        (desc[1], desc[10], desc[11]) = (4, address as u32, (address >> 32) as u32);
        create_placed_srv(null, &resource, &desc, at);
        let first = record_words(at);
        // A buffer view's flags; past a 2D view's member.
        desc[8] = 1;
        create_placed_srv(null, &resource, &desc, at);
        assert_ne!(record_words(at), first);
    }

    /// Entries no resource owns stay under their cap: the oldest go stale
    /// (and the heap pass hears of it), the newest still name their views,
    /// entries are reused rather than added, and views of resources are
    /// never dropped.
    #[test]
    fn ownerless_entries_are_capped() {
        let mut views = Views::new();
        let placed = |n: u32| View::Placed(false, [n; 14], [n; 16]);
        let named = |views: &Views, words: [u32; 2], view: &View| {
            views.list.get(words[0] as usize).is_some_and(|(v, g)| *g == words[1] && v == view)
        };
        let kept = |views: &mut Views, view: View, cap: usize| {
            let digest = digest(&view);
            views.entry_kept(view, digest, cap)
        };
        // From here the entries freed are noted (another test's may be too).
        let _ = take_forgotten();
        let words: Vec<[u32; 2]> = (0..10).map(|n| kept(&mut views, placed(n), 4)).collect();
        assert_eq!(views.index.len() + views.collided.len(), 4, "four kept");
        assert_eq!((views.ownerless.len(), views.ownerless_freed), (4, 6));
        assert!(views.list.len() <= 5, "entries reused, not added: {}", views.list.len());
        for n in 0..6 {
            assert!(!named(&views, words[n], &placed(n as u32)), "entry {n} stale");
        }
        for n in 6..10 {
            assert!(named(&views, words[n], &placed(n as u32)), "entry {n} kept");
        }
        let forgotten = take_forgotten().expect("asked before");
        for n in 0..6 {
            assert!(forgotten.contains(&words[n][0]), "entry {n} noted for the heap pass");
        }
        // The same view written again keeps its words; a constant buffer
        // view counts as well.
        assert_eq!(kept(&mut views, placed(9), 4), words[9]);
        let constant = View::Constant(0x7_0000_0000, 1 << 22);
        let kept_constant = kept(&mut views, constant.clone(), 4);
        assert!(named(&views, kept_constant, &constant));
        assert!(!named(&views, words[6], &placed(6)), "pushed out by the constant view");
        // A view of a resource is freed with the resource only.
        let shader = View::Shader(0x1234, None);
        let kept_shader = kept(&mut views, shader.clone(), 4);
        for n in 10..40 {
            kept(&mut views, placed(n), 4);
        }
        assert!(named(&views, kept_shader, &shader));
        assert!(views.list.len() <= 6);
        assert_eq!(views.ownerless.len(), 4);
        // 0 keeps them all.
        for n in 40..60 {
            kept(&mut views, placed(n), 0);
        }
        assert_eq!(views.index.len() + views.collided.len(), 1 + 4 + 20);
        assert_eq!(views.ownerless.len(), 4 + 20);
    }

    /// The indexes keep digests: a view whose digest another view's entry
    /// has is found all the same, until its resource goes, and so is the
    /// other; shapes alike. New shapes count the words that made them new.
    #[test]
    fn views_of_a_taken_digest_are_kept_by_themselves() {
        let mut views = Views::new();
        let (a, b) = (View::Shader(0x10, None), View::Shader(0x20, None));
        let [i, generation] = views.entry(a.clone(), digest(&a));
        assert_eq!(views.entry_of(&a, digest(&a)), Some([i, generation]));
        assert_eq!(views.entry_of(&b, digest(&b)), None);
        // `b` asked for under the digest of `a`, as if the two had one.
        let [j, _] = views.entry(b.clone(), digest(&a));
        assert_ne!(i, j);
        assert_eq!(views.collided.len(), 1);
        assert_eq!(views.entry_at(&a, digest(&a)), Some(i));
        assert_eq!(views.entry_at(&b, digest(&a)), Some(j));
        assert_eq!(views.entry(b.clone(), digest(&a))[0], j, "no second entry");
        // `a` goes; `b` is still found, then goes too.
        assert_eq!(views.release(0x10), vec![i]);
        assert_eq!(views.entry_at(&a, digest(&a)), None);
        assert_eq!(views.entry_at(&b, digest(&a)), Some(j));
        assert_eq!(views.release(0x20), vec![j]);
        assert_eq!(views.entry_at(&b, digest(&a)), None);
        assert!(views.collided.is_empty() && views.index.is_empty());
        // A freed entry is used again under a new generation.
        let [k, again] = views.entry(a.clone(), digest(&a));
        assert!(k == i || k == j);
        assert_ne!(again, generation);

        let resource = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        let mut rest = [0u32; 14];
        rest[1] = 4;
        let first: Shape = (false, resource, rest);
        let mut wider = first;
        wider.1[4] = 128;
        assert_eq!(views.shape(first, digest(&first)), Some(0));
        assert_eq!(views.shape(wider, digest(&first)), Some(1));
        assert_eq!(views.shapes_collided.len(), 1);
        assert_eq!(views.shape_at(&first, digest(&first)), Some(0));
        assert_eq!(views.shape_at(&wider, digest(&first)), Some(1));
        assert_eq!(views.shape(wider, digest(&first)), Some(1), "no second shape");
        let mut higher = wider;
        higher.1[6] = 128;
        assert_eq!(views.shape(higher, digest(&higher)), Some(2));
        let mut varied = [0; 28];
        (varied[4], varied[6]) = (1, 1);
        assert_eq!(views.varied, varied);
        assert_eq!(views.past_shapes, 0);
    }

    /// A null CBV description overwrites the slot (D3D12 null CBV), so a
    /// later read does not see the previous view.
    #[test]
    fn null_cbv_clears_a_stale_view() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let a = slots.as_ptr() as u64;
        create_cbv(null, &[0x1000, 1, 256, 0], a);
        assert_eq!(view(a), Some(View::Constant(0x1_0000_1000, 256)));
        create_cbv(null, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Constant(0, 0)));
        create_srv(null, null, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0, None)));
        create_cbv(null, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Constant(0, 0)));
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// The title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }
}
