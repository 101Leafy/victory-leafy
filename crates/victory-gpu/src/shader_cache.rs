//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the translator
//! and its settings, so a changed translator never serves an old
//! translation. The translator in the key is its hand-kept number
//! (`victory_shader::emit::VERSION`) with what it is built from beside its
//! source ([`Check`]). Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too, and [`FORMAT`] raised.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType, ShaderError};

/// The form of an entry. This file is not part of the translator's
/// fingerprint (`build.rs`), so a change to `encode` or `decode` that old
/// entries do not read back right must raise it.
pub(crate) const FORMAT: u32 = 1;

static CACHE: OnceLock<Cache> = OnceLock::new();
static CHECK: OnceLock<Arc<Check>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
/// `VICTORY_SHADER_CACHE_DAYS=n`: entries not used for `n` days are removed,
/// on a thread of their own (a folder kept every translation each translator
/// change made: 77,082 entries, 2.7 GB).
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir.clone()) {
        Ok(cache) => {
            let days = std::env::var("VICTORY_SHADER_CACHE_DAYS").ok().and_then(|v| v.parse::<u64>().ok());
            if let Some(days) = days.filter(|&d| d > 0) {
                let cache = cache.clone();
                let max_age = std::time::Duration::from_secs(days * 24 * 60 * 60);
                let _ = std::thread::Builder::new()
                    .name("shader cache prune".into())
                    .spawn(move || {
                        cache.prune(max_age);
                    });
            }
            let kept = format!("v{} {}", victory_shader::emit::VERSION, env!("VICTORY_TRANSLATOR_DEPS"));
            let _ = CHECK.set(Check::open(Some(dir.join(RECORD)), kept, env!("VICTORY_TRANSLATOR").to_owned()));
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies a translation beside its container: the translator
/// (`translator`, [`Check::key`]), the settings it runs with, and what it
/// was told of the host. Not our executable: a rebuild of anything else
/// used to translate every shader again.
fn build(translator: &str) -> [u8; 32] {
    // The view heap's null slot differs by host GPU: a translation cached
    // without the clamp must not be served (`host_settings` below).
    crate::descriptor::host::probe_view_limit();
    static FIXED: OnceLock<Vec<u8>> = OnceLock::new();
    let fixed = FIXED.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        let mut bytes = FORMAT.to_le_bytes().to_vec();
        bytes.extend_from_slice(&limit.to_le_bytes());
        // Which group barriers wait is part of the translation too.
        bytes.push(victory_shader::emit::group_sync());
        // So are the switches that pick the form of a wave64 program:
        // without them here a run with `VICTORY_WAVE_WIDE=0` was
        // served the translations an earlier run of the same
        // executable had cached with the switch unset.
        for name in [
            "VICTORY_WAVE_WIDE",
            "VICTORY_WAVE_DEEP_SKIPS",
            "VICTORY_WAVE_OWN_BITS",
            "VICTORY_WAVE_LOCAL",
            "VICTORY_F16_NEAREST",
            "VICTORY_HELPER_OWN",
            "VICTORY_FIRSTLANE_OWN",
            "VICTORY_FIRST_HALF",
            "VICTORY_STRICT_JOINS",
            "VICTORY_MESH_OUTPUT_STRICT",
            "VICTORY_BVH_FENCES",
            "VICTORY_BVH_BOUNDS",
            "VICTORY_WAVE_PAIRS",
            "VICTORY_LAUNCHED_EXEC",
            "VICTORY_GUARD_FLOAT",
            "VICTORY_GUARD_RUNS",
        ] {
            append_switch(&mut bytes, name, std::env::var_os(name).as_deref());
        }
        // And which clean-up passes ran over the translated code.
        bytes.extend_from_slice(&victory_shader::emit::shader_opt().to_le_bytes());
        bytes
    });
    let mut hash = blake3::Hasher::new();
    hash.update(translator.as_bytes());
    hash.update(&[0]);
    hash.update(fixed);
    for value in victory_shader::emit::host_settings() {
        hash.update(&value.to_le_bytes());
    }
    *hash.finalize().as_bytes()
}

/// What tells one set of translations from another ([`build`]).
pub(crate) fn translation() -> [u8; 32] {
    build(check().key())
}

fn check() -> &'static Arc<Check> {
    CHECK.get_or_init(|| Check::open(None, String::new(), String::new()))
}

/// The file in the cache folder that keeps what [`Check`] found, a line
/// per translator: `v<VERSION> <dependencies> <source> same|differs|unchecked`.
/// `unchecked` is the first build seen with its number: it owns the
/// number's entries and was compared with nothing, so its output may differ
/// from an earlier build's on another number (play-g's `same` read as
/// proof that its translations matched play-f's).
const RECORD: &str = "translator-checked.txt";

/// How many hits a translator whose source changed without a new
/// `emit::VERSION` translates again before its output counts as the same.
const CHECKED: u32 = 64;

const SAME: u8 = 0;
const PENDING: u8 = 1;
const DIFFERS: u8 = 2;

/// Whether this build of the translator may use the entries kept under its
/// number. The first build seen with a number (and its dependencies) owns
/// it. A build with other source and the same number translates its first
/// [`CHECKED`] hits again and compares the entries byte for byte: all the
/// same, and it shares them; one differs (a raise forgotten), and it keys
/// on its source for good, as before the number. Until then it reads the
/// number's entries but keeps what it translates under its source, so an
/// unchecked build never writes there.
struct Check {
    record: Option<PathBuf>,
    /// `v<VERSION> <dependencies>`.
    kept: String,
    /// The BLAKE3 of the translator's source (`build.rs`).
    source: String,
    state: AtomicU8,
    left: AtomicU32,
    queue: Mutex<Option<mpsc::Sender<(String, Sample)>>>,
}

/// Translates a hit again: whether it gives the entry's bytes.
type Sample = Box<dyn FnOnce() -> bool + Send>;

impl Check {
    fn open(record: Option<PathBuf>, kept: String, source: String) -> Arc<Self> {
        let line = format!("{kept} {source}");
        let text = record.as_ref().and_then(|r| std::fs::read_to_string(r).ok()).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        let noted = |what: &str| lines.contains(&format!("{line} {what}").as_str());
        let state = if noted("differs") {
            DIFFERS
        } else if noted("same") || noted("unchecked") || record.is_none() || !lines.iter().any(|l| l.starts_with(&format!("{kept} "))) {
            SAME
        } else {
            PENDING
        };
        let check = Arc::new(Self {
            record,
            kept,
            source,
            state: AtomicU8::new(state),
            left: AtomicU32::new(CHECKED),
            queue: Mutex::new(None),
        });
        if state == SAME && !noted("same") && !noted("unchecked") {
            check.note("unchecked");
        }
        match state {
            DIFFERS => tracing::info!("shader cache keyed by the translator's source: its output differed from its number's"),
            PENDING => tracing::info!(shaders = CHECKED, "translator source changed without a new number; comparing its output"),
            _ => {}
        }
        check
    }

    fn state(&self) -> u8 {
        self.state.load(Ordering::Relaxed)
    }

    /// The translator as it goes into the key of entries read and kept.
    fn key(&self) -> &str {
        if self.state() == DIFFERS { &self.source } else { &self.kept }
    }

    fn note(&self, what: &str) {
        use std::io::Write;
        let Some(record) = &self.record else { return };
        if let Ok(mut file) = std::fs::File::options().create(true).append(true).open(record) {
            let _ = writeln!(file, "{} {} {what}", self.kept, self.source);
        }
    }

    /// Has the hit of the shader `shader` (its 16-digit hash) translated
    /// again on the check's own thread, while samples are still wanted.
    fn offer(self: &Arc<Self>, shader: &str, same: impl FnOnce() -> bool + Send + 'static) {
        if self.state() != PENDING
            || self.left.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1)).is_err()
        {
            return;
        }
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let send = queue.get_or_insert_with(|| {
            let (send, samples) = mpsc::channel();
            let check = Arc::clone(self);
            let _ = std::thread::Builder::new()
                .name("translator check".into())
                .spawn(move || check.run(samples));
            send
        });
        let _ = send.send((shader.to_owned(), Box::new(same)));
    }

    fn run(&self, samples: mpsc::Receiver<(String, Sample)>) {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
            // SAFETY: this thread's own.
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) };
        }
        let mut same = 0;
        for (shader, sample) in samples {
            if !std::panic::catch_unwind(std::panic::AssertUnwindSafe(sample)).unwrap_or(false) {
                self.state.store(DIFFERS, Ordering::Relaxed);
                self.note("differs");
                tracing::warn!(
                    shader,
                    "translator output changed without a new emit::VERSION; shaders are keyed by its source from now on"
                );
                return;
            }
            same += 1;
            if same == CHECKED {
                self.state.store(SAME, Ordering::Relaxed);
                self.note("same");
                tracing::info!(shaders = same, "translator output unchanged; its number's entries are shared");
                return;
            }
        }
    }
}

/// Puts the translation of `container` into the cache before any title asks
/// for it (`victory shaders`): whether it was there already, or why it does
/// not translate. The host is opened first, as a run does: what the card
/// can do decides the form of some programs.
pub fn precompile(container: &[u8]) -> Result<bool, String> {
    let _ = crate::host::host();
    // Told by this call, not by `MISSES` around it: with more than one
    // worker another thread's miss would count here.
    let (_, translated) = host_shader_by(container, 0)?;
    Ok(!translated)
}

/// Translations the pipeline workers run at once: the host's processors
/// less the title's (an Xbox Series title has 7 cores, 14 threads), at
/// least one. In play-i 744 shaders that did not translate took 2,761,450
/// ms of worker time, and from 200 s on the title got too little of the
/// processor. The title's own threads never wait here: a draw that asks
/// for a shader first translates it at once.
/// `VICTORY_TRANSLATE_JOBS=n`: n at once.
const TITLE_THREADS: usize = 14;

struct Turns {
    limit: usize,
    busy: Mutex<usize>,
    free: Condvar,
}

struct Turn<'a>(&'a Turns);

impl Turns {
    fn new(limit: usize) -> Self {
        Self { limit: limit.max(1), busy: Mutex::new(0), free: Condvar::new() }
    }

    fn take(&self) -> Turn<'_> {
        let mut busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        while *busy >= self.limit {
            busy = self.free.wait(busy).unwrap_or_else(|e| e.into_inner());
        }
        *busy += 1;
        Turn(self)
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        self.0.free.notify_one();
    }
}

/// A turn to translate when this is a pipeline worker (`pipeline::make`).
fn worker_turn() -> Option<Turn<'static>> {
    static TURNS: OnceLock<Turns> = OnceLock::new();
    if !std::thread::current().name().is_some_and(|n| n.starts_with("victory-pipelines-")) {
        return None;
    }
    let turns = TURNS.get_or_init(|| {
        let asked = std::env::var("VICTORY_TRANSLATE_JOBS").ok().and_then(|v| v.trim().parse().ok());
        let processors = std::thread::available_parallelism().map_or(2, |n| n.get());
        let limit = asked.unwrap_or(processors.saturating_sub(TITLE_THREADS));
        tracing::info!(limit = limit.max(1), "translations at once on the pipeline workers");
        Turns::new(limit)
    });
    Some(turns.take())
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    host_shader_finite(container, 0)
}

/// One switch in the key: `NAME=value`, `NAME=` when unset, and `NAME`
/// alone when set and empty, so a switch read only for whether it is set
/// (`VICTORY_STRICT_JOINS`) set to nothing does not share the unset key.
fn append_switch(bytes: &mut Vec<u8>, name: &str, value: Option<&std::ffi::OsStr>) {
    bytes.extend_from_slice(name.as_bytes());
    if value.is_none_or(|v| !v.is_empty()) {
        bytes.push(b'=');
        bytes.extend_from_slice(value.unwrap_or_default().as_encoded_bytes());
    }
    bytes.push(0);
}

/// [`host_shader`] with the float colour exports to the targets in `mask`
/// held finite (`emit::with_finite_targets`), cached apart from the plain
/// translation; a mask of 0 is the plain one.
pub(crate) fn host_shader_finite(container: &[u8], mask: u8) -> Result<HostShader, String> {
    host_shader_by(container, mask).map(|(shader, _)| shader)
}

/// [`host_shader_finite`], and whether this call translated it (rather than
/// read it from the cache).
fn host_shader_by(container: &[u8], mask: u8) -> Result<(HostShader, bool), String> {
    let hash = blake3::hash(container).to_hex();
    let (wide, own) = (wave_wide(&hash), firstlane_own(&hash));
    let translate = move |c: &[u8]| {
        let _turn = worker_turn();
        let plain = || {
            victory_shader::emit::with_wave_wide(wide, || {
                victory_shader::emit::with_firstlane_own(own, || victory_shader::emit::translate(c))
            })
        };
        if mask == 0 { plain() } else { victory_shader::emit::with_finite_targets(mask, plain) }
    };
    match CACHE.get() {
        // The forms a list allows are part of what the entry was made from. A
        // container on no list keeps the key it had, so a change to the lists
        // translates only the containers it names again.
        Some(cache) => {
            let salted = |translator: &str| {
                let mut build = build(translator);
                if wide {
                    build[0] ^= 0x57;
                }
                if own {
                    build[1] ^= 0x4f;
                }
                if mask != 0 {
                    let mut key = blake3::Hasher::new();
                    key.update(&build);
                    key.update(b"finite targets");
                    key.update(&[mask]);
                    build = *key.finalize().as_bytes();
                }
                build
            };
            keyed(cache, check(), salted, &hash[..16], container, translate)
        }
        None => translate(container).map(|shader| (shader, true)).map_err(|e| e.to_string()),
    }
}

/// [`cached`] under the key [`Check`] allows; `build` makes the key of a
/// translator, `shader` is the container's 16-digit hash.
fn keyed(
    cache: &Cache,
    check: &Arc<Check>,
    build: impl Fn(&str) -> [u8; 32],
    shader: &str,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, ShaderError> + Send + Sync + 'static,
) -> Result<(HostShader, bool), String> {
    if check.state() == PENDING {
        if let Some((found, entry)) = lookup(cache, &build(&check.kept), container) {
            let container = container.to_vec();
            check.offer(shader, move || translate(&container).is_ok_and(|s| encode(&s) == entry));
            return Ok((found, false));
        }
        return cached_by(cache, &build(&check.source), container, translate);
    }
    cached_by(cache, &build(check.key()), container, translate)
}

/// Console wave64 compute programs (by the 16-digit hash the log names
/// shaders with) that a 32-lane host refuses as translated (they ask for
/// 64-lane waves) and that run in the wave-wide branches form: the water's
/// transmission volumetrics (`_VolumePortalsOnly`, `Submerged`) and
/// `sr_tracing`'s `CS_Tiling_Generate_Opaque`. Left out: the particles'
/// `CS_Visibility` (369cbe906ea72764); with it on, a run on an RTX 5060 Ti
/// stopped as the first world frame was made. Also left out: the
/// volumetrics' `_Tiled` and `_Tiled_LocalLights` (5dc3b25986622949,
/// 6f225b641c5b0708); with them on, black patches lay on the water around
/// the boat in the first boat mission (2026-11-25).
/// `VICTORY_WAVE_WIDE_ONLY` adds hashes (comma-separated, read by
/// `victory_shader::emit::wave_wide_only`), `VICTORY_WAVE_WIDE_SEEN=0` leaves
/// these three out.
const WAVE_WIDE_SEEN: [&str; 3] = ["a10179ab7eda0dc3", "926c46bab3b86bd6", "2cc4c197ee29e291"];

/// Whether the container whose hash (in hex) is `hash` takes the wave-wide
/// form ([`WAVE_WIDE_SEEN`]).
fn wave_wide(hash: &str) -> bool {
    static SEEN_ON: OnceLock<bool> = OnceLock::new();
    let seen_on = *SEEN_ON.get_or_init(|| std::env::var_os("VICTORY_WAVE_WIDE_SEEN").is_none_or(|v| v != "0"));
    victory_shader::emit::wave_wide_only(&hash[..16]) || (seen_on && WAVE_WIDE_SEEN.iter().any(|h| hash.starts_with(h)))
}

/// Whether the container whose hash (in hex) is `hash` reads a first lane
/// where not every lane runs from the host wave's own first lane
/// (`victory_shader::emit::with_firstlane_own`): `VICTORY_FIRSTLANE_OWN_ONLY`,
/// hash prefixes separated by commas. The ray-tracing structure builder
/// (`XboxBvhBuilder.dll`) needs it for three programs to translate for
/// 32-lane waves (39321a95223001ed, 77360df21e12ea43, 9fc111d3be79364f as
/// the title passes them; dc78c5de69c7c155, 322ed4ad832bdea6,
/// d441b097704ab616 as `victory translate` finds them in the DLL), and its
/// other programs run only with them. Not on by default until a run shows
/// the builder working on this card.
fn firstlane_own(hash: &str) -> bool {
    static ONLY: OnceLock<Vec<String>> = OnceLock::new();
    let only = ONLY.get_or_init(|| {
        std::env::var("VICTORY_FIRSTLANE_OWN_ONLY")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_lowercase())
            .filter(|h| !h.is_empty())
            .collect()
    });
    only.iter().any(|h| hash.starts_with(h.as_str()))
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit; `made` keeps a
/// miss's translation or refusal, so the entry just written is not read
/// back (the first read of a new file costs about as much as a small
/// translation where a virus scanner looks at it: 0.5-3 ms).
///
/// Code the translator does not handle (`ShaderError::Unsupported`) is
/// refused the same way every time under one build, so the refusal is an
/// entry too and the next run does not try again. Other errors are not
/// kept.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    made: Mutex<Option<Result<HostShader, String>>>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, ShaderError> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let (bytes, made) = match (self.translate)(&input[32..]) {
            Ok(shader) => (encode(&shader), Ok(shader)),
            Err(e @ ShaderError::Unsupported { .. }) => {
                let reason = e.to_string();
                (refusal(&reason), Err(reason))
            }
            Err(e) => return Err(e.into()),
        };
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, ShaderError> + Send + Sync,
) -> Result<HostShader, String> {
    cached_by(cache, build, container, translate).map(|(shader, _)| shader)
}

/// [`cached`], and whether this call translated it (rather than read it
/// from the cache).
fn cached_by(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, ShaderError> + Send + Sync,
) -> Result<(HostShader, bool), String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let made = || converter.made.lock().unwrap_or_else(|e| e.into_inner()).take();
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            // A translation whose entry could not be written is handed back
            // as it was made.
            Err(e) => {
                let made = made().ok_or_else(|| format!("{e:#}"))?;
                MISSES.fetch_add(1, Ordering::Relaxed);
                return made.map(|shader| (shader, true));
            }
        };
        if let Some(made) = made() {
            MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(entry = %entry_name(&path), "shader translated and cached");
            return made.map(|shader| (shader, true));
        }
        if let Some(entry) = std::fs::read(&path).ok().and_then(|b| decode_entry(&b)) {
            let ran = converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if ran {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(entry = %entry_name(&path), "shader {what}");
            return entry.map(|shader| (shader, ran));
        }
        tracing::debug!(entry = %entry_name(&path), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    made()
        .unwrap_or_else(|| (converter.translate)(container).map_err(|e| e.to_string()))
        .map(|shader| (shader, true))
}

/// The entry for `container` under `build`, read and not translated: its
/// translation and its bytes.
fn lookup(cache: &Cache, build: &[u8; 32], container: &[u8]) -> Option<(HostShader, Vec<u8>)> {
    let input = [&build[..], container].concat();
    let absent = Translate {
        translate: |_: &[u8]| Err::<HostShader, _>(ShaderError::NotDxbc),
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let path = cache.get_or_convert(&absent, &input).ok()?;
    let bytes = std::fs::read(&path).ok()?;
    let shader = match decode_entry(&bytes) {
        Some(Ok(shader)) => shader,
        Some(Err(_)) => return None,
        None => {
            let _ = std::fs::remove_file(&path);
            return None;
        }
    };
    HITS.fetch_add(1, Ordering::Relaxed);
    Some((shader, bytes))
}

/// An entry's name without the folders above it (they are the user's).
fn entry_name(path: &std::path::Path) -> String {
    let parent = path.parent().and_then(|p| p.file_name()).unwrap_or_default();
    format!("<path>/{}/{}", parent.to_string_lossy(), path.file_name().unwrap_or_default().to_string_lossy())
}

const MAGIC: &[u8; 4] = b"VSH1";
const REFUSED: &[u8; 4] = b"VSR1";

/// The translator's source and what it is built from (`build.rs`), in each
/// refusal. A refusal can come from a translator bug, and a fix to it may
/// leave every translation the same: then [`Check`] finds the output
/// unchanged and goes on keying by `emit::VERSION`, so the key alone would
/// keep serving the refusal.
const MADE_BY: &str = env!("VICTORY_TRANSLATOR");

/// A refusal's entry: [`REFUSED`], [`MADE_BY`], the reason's length (u32)
/// and the reason. Builds from before refusals were kept do not read it and
/// translate again.
fn refusal(reason: &str) -> Vec<u8> {
    [REFUSED.as_slice(), MADE_BY.as_bytes(), &(reason.len() as u32).to_le_bytes(), reason.as_bytes()].concat()
}

/// An entry read back: a translation, a refusal's reason, or `None` when it
/// is damaged or a refusal by another translator source, which is
/// translated again.
fn decode_entry(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    let Some(rest) = bytes.strip_prefix(REFUSED) else {
        return decode(bytes).map(Ok);
    };
    let (len, reason) = rest.strip_prefix(MADE_BY.as_bytes())?.split_first_chunk::<4>()?;
    if reason.is_empty() || reason.len() != u32::from_le_bytes(*len) as usize {
        return None;
    }
    String::from_utf8(reason.to_vec()).ok().map(Err)
}
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures other than a refusal are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>(ShaderError::Malformed("test input"));
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    /// A refusal under the same build is read back on the next run, not
    /// translated again; another build or another container tries again.
    #[test]
    fn refusals_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let runs = AtomicU64::new(0);
        let refuse = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>(ShaderError::Unsupported { offset: 4, what: "test opcode" })
        };
        let want = Err("GFX10 code at 0x4: test opcode is not supported".to_owned());
        assert_eq!(cached(&Cache::open(dir.path()).unwrap(), &[7; 32], b"a container", refuse), want);
        let cache = Cache::open(dir.path()).unwrap();
        assert_eq!(cached(&cache, &[7; 32], b"a container", refuse), want);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(cached(&cache, &[8; 32], b"a container", refuse), want);
        assert_eq!(cached(&cache, &[7; 32], b"another container", refuse), want);
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Not a translation to offer, and kept.
        assert!(lookup(&cache, &[7; 32], b"a container").is_none());
        assert_eq!(cached(&cache, &[7; 32], b"a container", refuse), want);
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn other_failures_are_tried_again() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let flaky = |_: &[u8]| {
            if runs.fetch_add(1, Ordering::Relaxed) == 0 {
                Err(ShaderError::Malformed("test input"))
            } else {
                Ok(shader())
            }
        };
        assert!(cached(&cache, &[7; 32], b"a container", flaky).is_err());
        assert_eq!(cached(&cache, &[7; 32], b"a container", flaky), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
    }

    /// Set and empty is a key of its own; unset and set keep the bytes
    /// entries were made under before.
    #[test]
    fn an_empty_switch_is_not_an_unset_one() {
        let key = |value: Option<&str>| {
            let mut bytes = Vec::new();
            append_switch(&mut bytes, "VICTORY_STRICT_JOINS", value.map(std::ffi::OsStr::new));
            bytes
        };
        assert_eq!(key(None), b"VICTORY_STRICT_JOINS=\0");
        assert_eq!(key(Some("1")), b"VICTORY_STRICT_JOINS=1\0");
        assert_eq!(key(Some("")), b"VICTORY_STRICT_JOINS\0");
    }

    /// A translation whose entry cannot be written (the cache folder is
    /// a file now) is handed back as it was made, not made again.
    #[test]
    fn a_translation_is_made_once_when_its_entry_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::write(&root, b"not a folder").unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &[7; 32], b"a container", translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    /// No more than the limit translate at once, and every one gets a turn.
    #[test]
    fn turns_hold_the_limit() {
        let turns = &Turns::new(3);
        let (now, most) = (&AtomicU32::new(0), &AtomicU32::new(0));
        std::thread::scope(|s| {
            for _ in 0..12 {
                s.spawn(move || {
                    for _ in 0..20 {
                        let _turn = turns.take();
                        most.fetch_max(now.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                        std::thread::yield_now();
                        now.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        assert!((1..=3).contains(&most.load(Ordering::SeqCst)));
        assert_eq!(*turns.busy.lock().unwrap(), 0);
        assert_eq!(Turns::new(0).limit, 1);
    }

    /// Whether a call translated is its own, not read off `MISSES`: the
    /// warm-up (`precompile`) runs several workers at once.
    #[test]
    fn each_call_tells_its_own_hit_from_its_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = &Cache::open(dir.path()).unwrap();
        let build = [11u8; 32];
        let translate = |_: &[u8]| Ok(shader());
        let hit = b"a container already in the cache".as_slice();
        assert_eq!(cached_by(cache, &build, hit, translate), Ok((shader(), true)));
        let misses: Vec<Vec<u8>> = (0..64).map(|i| format!("miss {i}").into_bytes()).collect();
        std::thread::scope(|s| {
            for c in &misses {
                s.spawn(move || {
                    assert_eq!(cached_by(cache, &build, c, translate), Ok((shader(), true)));
                });
            }
            for _ in 0..64 {
                assert_eq!(cached_by(cache, &build, hit, translate), Ok((shader(), false)));
            }
        });
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    fn key(translator: &str) -> [u8; 32] {
        *blake3::hash(translator.as_bytes()).as_bytes()
    }

    fn settled(check: &Check) -> u8 {
        let start = std::time::Instant::now();
        while check.state() == PENDING && start.elapsed().as_secs() < 30 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check.state()
    }

    #[test]
    fn a_changed_translation_without_a_new_number_keys_on_the_source() {
        static RUNS: AtomicU64 = AtomicU64::new(0);
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path().join("cache")).unwrap();
        let record = dir.path().join(RECORD);
        let open = |source: &str| Check::open(Some(record.clone()), "v1 deps".into(), source.into());
        let containers: Vec<Vec<u8>> = (0..CHECKED + 4).map(|n| format!("container {n}").into_bytes()).collect();
        let same = |_: &[u8]| {
            RUNS.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let changed = |_: &[u8]| {
            RUNS.fetch_add(1, Ordering::Relaxed);
            let mut s = shader();
            s.dxil.push(0);
            Ok(s)
        };

        // The first build with a number owns it.
        let first = open("a");
        assert_eq!(first.state(), SAME);
        for c in &containers {
            assert_eq!(keyed(&cache, &first, key, "0123456789abcdef", c, same).map(|(s, _)| s), Ok(shader()));
        }
        assert_eq!(RUNS.load(Ordering::Relaxed), u64::from(CHECKED) + 4);

        // Other source with the same output: served the number's entries
        // while its hits are translated again, then shares them.
        let second = open("b");
        assert_eq!(second.state(), PENDING);
        RUNS.store(0, Ordering::Relaxed);
        for c in &containers {
            assert_eq!(keyed(&cache, &second, key, "0123456789abcdef", c, same).map(|(s, _)| s), Ok(shader()));
        }
        assert_eq!(settled(&second), SAME);
        assert_eq!(RUNS.load(Ordering::Relaxed), u64::from(CHECKED), "only the samples ran");
        assert_eq!(open("b").state(), SAME, "a run after the check does not check again");

        // A miss while the check runs is kept under the source, not the number.
        let third = open("c");
        assert_eq!(keyed(&cache, &third, key, "0123456789abcdef", b"new", same).map(|(s, _)| s), Ok(shader()));
        assert!(lookup(&cache, &key("v1 deps"), b"new").is_none());
        assert!(lookup(&cache, &key("c"), b"new").is_some());

        // Other source whose output differs: the first sample trips it, and
        // from then on it translates under its source.
        let fourth = open("d");
        RUNS.store(0, Ordering::Relaxed);
        assert_eq!(keyed(&cache, &fourth, key, "0123456789abcdef", &containers[0], changed).map(|(s, _)| s), Ok(shader()));
        assert_eq!(settled(&fourth), DIFFERS);
        assert_eq!(fourth.key(), "d");
        let mut want = shader();
        want.dxil.push(0);
        assert_eq!(keyed(&cache, &fourth, key, "0123456789abcdef", &containers[0], changed).map(|(s, _)| s), Ok(want));
        assert_eq!(RUNS.load(Ordering::Relaxed), 2);
        assert_eq!(open("d").state(), DIFFERS, "and so does its next run");

        let text = std::fs::read_to_string(&record).unwrap();
        assert_eq!(text, "v1 deps a unchecked\nv1 deps b same\nv1 deps d differs\n");
        assert_eq!(open("a").state(), SAME, "the owner does not check itself");

        // A new number (or new dependencies) owns its entries again.
        let raised = Check::open(Some(record.clone()), "v2 deps".into(), "d".into());
        assert_eq!(raised.state(), SAME);
    }

    /// The check compares translations byte for byte, so the translator
    /// must give the same bytes for a container every time, on any thread.
    /// `VICTORY_SHADER_SAMPLES=<file>`: the shader containers in that file
    /// (a title's executable), up to 200 spread over it. Skipped without it.
    #[test]
    fn translations_are_the_same_every_time() {
        let Some(path) = std::env::var_os("VICTORY_SHADER_SAMPLES") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let mut containers = Vec::new();
        let mut at = 0;
        while let Some(found) = data[at..].windows(4).position(|w| w == b"DXBC") {
            let start = at + found;
            at = start + 4;
            let size = data.get(start + 24..start + 28).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
            if let Some(container) = size.and_then(|n| data.get(start..start + n))
                && victory_shader::Container::parse(container).is_ok()
            {
                containers.push(container);
                at = start + container.len();
            }
        }
        assert!(!containers.is_empty(), "no containers in the file");
        let step = containers.len().div_ceil(200);
        let picked: Vec<&[u8]> = containers.iter().step_by(step).copied().collect();
        let translate = |c: &[u8]| victory_shader::emit::translate(c).map(|s| encode(&s)).map_err(|e| e.to_string());
        let here: Vec<_> = picked.iter().map(|c| translate(c)).collect();
        let again: Vec<_> = picked.iter().map(|c| translate(c)).collect();
        let there: Vec<_> = std::thread::scope(|s| s.spawn(|| picked.iter().map(|c| translate(c)).collect()).join().unwrap());
        let translated = here.iter().filter(|r| r.is_ok()).count();
        eprintln!("{} containers, {} compared, {translated} translated", containers.len(), picked.len());
        for (n, c) in picked.iter().enumerate() {
            let name = &blake3::hash(c).to_hex()[..16];
            assert!(here[n] == again[n], "{name} translated twice gave other bytes");
            assert!(here[n] == there[n], "{name} on another thread gave other bytes");
        }
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        assert_eq!(decode_entry(&bytes), Some(Ok(shader())));
        assert_eq!(decode_entry(&refusal("why")), Some(Err("why".to_owned())));
        let refused = |rest: &[u8]| [REFUSED.as_slice(), MADE_BY.as_bytes(), rest].concat();
        assert_eq!(decode_entry(&refused(b"\x03\0\0\0why")), Some(Err("why".to_owned())));
        assert_eq!(decode_entry(&refused(b"\x04\0\0\0cut")), None);
        assert_eq!(decode_entry(&refused(b"\x01\0\0\0\xff")), None);
        assert_eq!(decode_entry(&refused(b"\0\0\0\0")), None);
        assert_eq!(decode_entry(b"VSR1\x03\0\0\0why"), None);
    }

    /// A refusal by another translator source is translated again, even
    /// where the key is the same (a fix that changed no translation).
    #[test]
    fn a_refusal_by_another_translator_is_tried_again() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let refuse = |_: &[u8]| Err::<HostShader, _>(ShaderError::Unsupported { offset: 4, what: "test opcode" });
        assert!(cached(&cache, &[7; 32], b"a container", refuse).is_err());
        // Entries sit in folders; the root holds the cache's own marker.
        let entries: Vec<_> = walk(dir.path()).into_iter().filter(|e| e.parent() != Some(dir.path())).collect();
        assert_eq!(entries.len(), 1);
        for e in entries {
            let mut bytes = std::fs::read(&e).unwrap();
            assert!(bytes.starts_with(REFUSED));
            bytes[REFUSED.len()] ^= 1;
            assert_eq!(decode_entry(&bytes), None);
            std::fs::write(e, bytes).unwrap();
        }
        assert_eq!(cached(&cache, &[7; 32], b"a container", |_: &[u8]| Ok(shader())), Ok(shader()));
    }
}
