//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline, TargetFormats};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    /// Taken by a worker from the resolves' own line ([`Queue::resolves`]).
    as_resolve: AtomicBool,
    /// When it was first queued, and the list a worker took it from.
    queued: OnceLock<Instant>,
    taken_from: AtomicU8,
    /// Started at the title's creation of it and not made yet ([`early`]).
    started_early: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]), or with
    /// `VICTORY_SCENE_INPUT=1` one of the opening scene's producers.
    screen: bool,
    /// One of the opening scene's lighting, tiling or bake producers, made
    /// at creation even on a cold cache ([`scene_input`]).
    scene_input: bool,
    /// Why it is never made ([`left_out`]).
    left_out: Option<&'static str>,
    /// A compute program known to take seconds to make ([`long_compute`]).
    long: bool,
    /// With `VICTORY_RESOLVE_NO_WAIT=1`, a per-material ray-tracing resolve
    /// (`*:CS_RaytracingResolve`): while it is being made its dispatches are
    /// dropped at once, never waited for. There are about a thousand of these
    /// programs and the card's driver takes seconds for one. Run 403 (ray
    /// tracing on): for the 3 minutes some were being made, the threads that
    /// translate the title's lists slept the frame's wait budget on them and
    /// dropped the dispatch anyway: 279 ms a frame (20 readings) against
    /// 54 ms (5 readings) once none was being made, at the same 3,000-4,000
    /// draws.
    no_wait: bool,
    /// Whether draws already waited [`GIVE_UP`] for it in vain: later
    /// draws do not wait for it (see [`wait`]).
    given_up: AtomicBool,
    /// What draws waited for it in vain so far, in microseconds.
    waited_us: AtomicU64,
    /// The title frame in which draws first waited for it (`u64::MAX`: none
    /// yet). Draws of later frames do not wait for it ([`claim_wait`]).
    waited_frame: AtomicU64,
    /// The last title frame a draw asked for it and found it not made
    /// ([`newest`]).
    asked_frame: AtomicU64,
    /// The formats its state declares for what a draw draws into (zeros
    /// without a graphics state).
    pub(crate) formats: TargetFormats,
    /// The same shaders and state for other formats, bound at a draw
    /// ([`for_formats`]): a chain from the title's pipeline, one link per
    /// set of formats met, never shortened.
    variant: OnceLock<Arc<Pending>>,
}

/// Dispatches dropped because their resolve was still being made
/// ([`Pending::no_wait`]), since the start.
static NOT_WAITED: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_RESOLVE_NO_WAIT=1`: see [`Pending::no_wait`] and
/// [`Queue::resolves`]. Off by default.
fn resolve_no_wait() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RESOLVE_NO_WAIT").is_some_and(|v| v == "1"))
}

fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders
        .iter()
        .any(|s| s.stage == "compute" && s.container.windows(name.len()).any(|w| w == name))
}

/// Ray-tracing programs left out without the structure builder: with
/// `VICTORY_RT_RESOLVE=0` the ~1,000 `CS_RaytracingResolve` pipelines,
/// which shade no hits without it and clog a cold compiler queue; with
/// `VICTORY_ACTIVATE_RAYS=0` `CS_ScreenProbes_ActivateRays`.
fn left_out(shaders: &[Arc<Shader>]) -> Option<&'static str> {
    static SKIP_RT_RESOLVE: OnceLock<bool> = OnceLock::new();
    let skip_rt = *SKIP_RT_RESOLVE.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        !builder && std::env::var_os("VICTORY_RT_RESOLVE").is_some_and(|v| v == "0")
    });
    if skip_rt && named(shaders, b"CS_RaytracingResolve") {
        return Some("shades ray-tracing hits without BVH, left out");
    }
    static SKIP_ACTIVATE_RAYS: OnceLock<bool> = OnceLock::new();
    let skip_activate =
        *SKIP_ACTIVATE_RAYS.get_or_init(|| std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0"));
    if skip_activate && named(shaders, b"CS_ScreenProbes_ActivateRays") {
        return Some("ray activation shader without BVH, left out");
    }
    None
}

/// The compute programs that take 7 to 30 s each to make on a cold cache:
/// run 164 still had CS_DepthCopy (3 ms) asked for 81 s while they held
/// every compute place, the one past the cap too, before the slow list knew
/// any of them.
fn long_compute(shaders: &[Arc<Shader>]) -> bool {
    named(shaders, b"CS_RaytracingResolve") || named(shaders, b"CS_GatherRaymarchedLocalLights")
}

/// Whether the opening scene's producers are made first
/// (`VICTORY_SCENE_INPUT=1`; off): see [`scene_input`].
fn scene_inputs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCENE_INPUT").is_ok_and(|v| v.trim() == "1"))
}

/// The opening scene's lighting, tiling and bake programs, so cutscene
/// lighting and bakes never miss their first frame; and passes over the
/// screen that use a mesh shader are no longer counted as screen passes.
fn scene_input(shaders: &[Arc<Shader>]) -> bool {
    scene_inputs()
        && shaders.iter().any(|s| {
            matches!(
                s.hash.as_str(),
                "970c4fbb3b017c0e"
                    | "068c90badc86f1fd"
                    | "e4a7e088c7780ce8"
                    | "ebe4e53f031f5249"
                    | "2cc4c197ee29e291"
                    | "369cbe906ea72764"
            ) || crate::draw::BAKE_PIXEL_SHADERS.contains(&s.hash.as_str())
        })
}

/// Shader hashes whose pipelines are left out in every build
/// (`VICTORY_LEAVE_OUT=<16 hex digits>[,..]`; `VICTORY_SKIP_PIPELINES` does
/// the same in diagnostics builds only). The decal heap's
/// `CS_DefragHeap_ComputeRelocations` (02384a60fda6f2d0), translatable for
/// 32-lane hosts through host waves that meet by spinning, took 11.85 ms of
/// the card a frame in the first gameplay area (timing run, 2026-11-26).
/// Such a pipeline is never made: it starts out refused, and its draws and
/// dispatches are dropped as those of a pipeline that does not translate.
fn leave_out() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        std::env::var("VICTORY_LEAVE_OUT")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_ascii_lowercase())
            .filter(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
            .collect()
    })
}

fn leave_out_hash(shaders: &[Arc<Shader>]) -> bool {
    !leave_out().is_empty() && shaders.iter().any(|s| leave_out().contains(&s.hash))
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        let scene_input = scene_input(&shaders);
        let screen = (graphics.as_ref().is_some_and(|g| g.screen_pass())
            && !(scene_inputs() && shaders.iter().any(|s| s.stage == "mesh")))
            || scene_input;
        let left_out = left_out(&shaders);
        let long = long_compute(&shaders);
        let no_wait = resolve_no_wait() && named(&shaders, b"CS_RaytracingResolve");
        let made = if leave_out_hash(&shaders) { OnceLock::from(None) } else { OnceLock::new() };
        Pending {
            screen,
            scene_input,
            left_out,
            long,
            no_wait,
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            as_resolve: AtomicBool::new(false),
            queued: OnceLock::new(),
            taken_from: AtomicU8::new(TAKEN_BY_TITLE),
            started_early: AtomicBool::new(false),
            made,
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            given_up: AtomicBool::new(false),
            waited_us: AtomicU64::new(0),
            waited_frame: AtomicU64::new(u64::MAX),
            asked_frame: AtomicU64::new(0),
        }
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.make_with(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
    }

    fn make_with(&self, build: impl FnOnce() -> Option<HostPipeline>) {
        let mut came = None;
        self.made.get_or_init(|| {
            let run = super::spent::begin();
            let began = Instant::now();
            // A failed translation must complete this queue entry and wake its waiters.
            let (made, how) = super::library::came_of(|| match catch_unwind(AssertUnwindSafe(build)) {
                Ok(pipeline) => pipeline,
                Err(_) => {
                    tracing::warn!(shaders = ?self.shaders, "host pipeline build panicked");
                    None
                }
            });
            super::library::note_time(&self.key, began.elapsed().as_millis() as u64, how);
            came = Some(how);
            let from = ["title", "screen", "requested", "early", "slow"][self.taken_from.load(Ordering::Relaxed) as usize];
            let queued = self.queued.get().map(|at| began.saturating_duration_since(*at));
            super::spent::end(run, &self.key, queued, from, made.is_some(), queues);
            made
        });
        if let Some(came) = came {
            self.early_done(came);
        }
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }

    /// An early pipeline ([`early`]) is made (or given up): counted, and the
    /// line logged once none is left.
    fn early_done(&self, came: super::library::Came) {
        if !self.started_early.swap(false, Ordering::Relaxed) {
            return;
        }
        EARLY_CAME[early_came_at(came)].fetch_add(1, Ordering::Relaxed);
        if EARLY_LEFT.fetch_sub(1, Ordering::Relaxed) == 1 {
            early_drained();
        }
    }
}

/// Where in [`EARLY_CAME`] an early pipeline that came about as `came` is
/// counted.
fn early_came_at(came: super::library::Came) -> usize {
    use super::library::Came;
    match came {
        Came::Loaded => 0,
        Came::Compiled => 1,
        Came::Nothing => 2,
    }
}

/// Early pipelines ([`early`]) not made yet; made since the start: loaded
/// from the library, compiled, neither (refused, left out); when the first
/// was started (`crate::host::now_ms`; `u64::MAX`: none yet).
static EARLY_LEFT: AtomicU64 = AtomicU64::new(0);
static EARLY_CAME: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static EARLY_BEGAN: AtomicU64 = AtomicU64::new(u64::MAX);
/// Early pipelines put in [`Queue::slow`] (being on the slow list).
static SLOW_QUEUED: AtomicU64 = AtomicU64::new(0);
/// The early pipelines made when the line was last logged.
static EARLY_LOGGED: AtomicU64 = AtomicU64::new(0);

/// Whether "early pipelines all made" is logged with `total` made, `logged`
/// of them made at the last line: after 10 or more since.
fn early_line_due(total: u64, logged: u64) -> bool {
    total >= logged + 10
}

/// No early pipeline is left to make: logged after 10 or more since the
/// last line.
fn early_drained() {
    use super::library::{SLOW_ADDED, SLOW_REMOVED};
    let came: [u64; 3] = std::array::from_fn(|i| EARLY_CAME[i].load(Ordering::Relaxed));
    let total = came.iter().sum::<u64>();
    if !early_line_due(total, EARLY_LOGGED.load(Ordering::Relaxed)) {
        return;
    }
    EARLY_LOGGED.store(total, Ordering::Relaxed);
    let began = EARLY_BEGAN.load(Ordering::Relaxed);
    tracing::info!(
        loaded = came[0],
        compiled = came[1],
        neither = came[2],
        slow_queued = SLOW_QUEUED.load(Ordering::Relaxed),
        slow_list_added = SLOW_ADDED.load(Ordering::Relaxed),
        slow_list_removed = SLOW_REMOVED.load(Ordering::Relaxed),
        ms_since_first_early = crate::host::now_ms().saturating_sub(began),
        "early pipelines all made"
    );
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) if leave_out_hash(&p.shaders) => Some("left out (VICTORY_LEAVE_OUT)".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Why a draw or dispatch did not run (diagnostics): [`missing`]'s reason,
/// or, for a pipeline that has a host pipeline, a binding or buffer the
/// host could not give it (`State::prepare`, `State::bind`). Those drops
/// were counted in the frame line but named in no per-reason line.
#[cfg(feature = "diagnostics")]
pub(crate) fn dropped_why(missing: Option<String>, hashes: &str) -> String {
    missing.unwrap_or_else(|| {
        format!("has a host pipeline; a binding or buffer could not be given: {hashes}")
    })
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Waits that ran out of the frame's budget with the pipeline still not
/// there, for the one line that says what a run like that is.
static GAVE_UP: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        let frame = crate::frame::current();
        p.asked_frame.store(frame, Ordering::Relaxed);
        start(p, REQUESTED);
        if p.no_wait {
            let n = NOT_WAITED.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    being_made = WAITING.load(Ordering::Relaxed),
                    "ray-tracing resolves dropped while their pipeline is being made, not waited for"
                );
            }
            dropped(p, frame);
        } else if p.graphics.is_none() && compute_waits() && compute_wait_left(frame) {
            // Made here, or waited for while a worker makes it.
            let began = Instant::now();
            p.make();
            compute_waited(frame, began.elapsed());
        } else {
            wait(p, frame);
            if p.made.get().is_none() {
                dropped(p, frame);
            }
        }
    }
    p.made.get()?.as_ref()
}

/// Whether a dispatch waits for its compute pipeline until it is made,
/// whatever the frame's budget ([`wait_budget`]); a draw still waits within
/// the budget and is dropped while its pipeline is not there.
///
/// A dropped draw is drawn again in the next frame. A dropped dispatch may
/// not be: the title runs some compute passes once, when a scene changes
/// (a buffer reset or cleared), and what they leave is read for as long as
/// the scene lasts. In the prison corridor a compute program met for the
/// first time took 4.4 s to make on a 16 GB machine; its one dispatch was
/// dropped after the half second of budget, and the windows of the room
/// before stayed blended over the whole picture for minutes (run 195).
/// A wait of seconds once (the pipeline library has it on the next run)
/// against a picture wrong until the scene ends.
/// Off by default until a cold and a warm run show what it costs;
/// `VICTORY_PIPELINE_COMPUTE_WAIT=1`: on.
fn compute_waits() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_COMPUTE_WAIT").is_some_and(|v| v == "1"))
}

/// What a frame may spend making compute pipelines for its dispatches
/// (`VICTORY_PIPELINE_COMPUTE_WAIT_MS`, 5 s; 0: no cap). A cold library can
/// meet dozens of new compute programs in one frame, and the title's
/// watchdog fires at 30 s. Once a frame has spent it, its dispatches wait
/// within the frame's budget as draws do; the one pipeline being made when
/// the cap runs out still finishes.
fn compute_wait_cap() -> Duration {
    static CAP: OnceLock<Duration> = OnceLock::new();
    *CAP.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_COMPUTE_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(5_000);
        Duration::from_millis(millis)
    })
}

/// Frame and time spent in it making compute pipelines for dispatches.
static COMPUTE_SPENT: Mutex<(u64, Duration)> = Mutex::new((u64::MAX, Duration::ZERO));

fn compute_wait_left(frame: u64) -> bool {
    let cap = compute_wait_cap();
    if cap.is_zero() {
        return true;
    }
    let mut spent = COMPUTE_SPENT.lock().unwrap_or_else(|e| e.into_inner());
    if spent.0 != frame {
        *spent = (frame, Duration::ZERO);
    }
    spent.1 < cap
}

fn compute_waited(frame: u64, took: Duration) {
    let cap = compute_wait_cap();
    let mut spent = COMPUTE_SPENT.lock().unwrap_or_else(|e| e.into_inner());
    if spent.0 != frame {
        *spent = (frame, Duration::ZERO);
    }
    let before = spent.1;
    spent.1 += took;
    if !cap.is_zero() && before < cap && spent.1 >= cap {
        tracing::info!(
            frame,
            ms = spent.1.as_millis() as u64,
            "a frame spent its compute wait: its other dispatches wait within the frame's budget (VICTORY_PIPELINE_COMPUTE_WAIT_MS)"
        );
    }
}

/// The pipelines a frame's draws were dropped for, logged when the next
/// frame drops one (at most 4 lines a second, `VICTORY_PIPELINE_DROP_LOG=0`:
/// none). play-g's binoculars dropped 550-930 draws a frame for 30 frames
/// and the zoom came and went; with a warm cache these say which ones.
#[derive(Default)]
struct Dropped {
    frame: u64,
    draws: u64,
    pipelines: std::collections::HashSet<usize>,
    keys: Vec<String>,
    given_up: usize,
    logged_at: Option<Instant>,
    lines: u32,
    skipped: u64,
}

fn dropped(p: &Arc<Pending>, frame: u64) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_DROP_LOG").is_none_or(|v| v != "0")) {
        return;
    }
    static DROPPED: Mutex<Option<Dropped>> = Mutex::new(None);
    let mut d = DROPPED.lock().unwrap_or_else(|e| e.into_inner());
    let d = d.get_or_insert_with(Dropped::default);
    if d.frame != frame {
        if d.draws > 0 {
            let now = Instant::now();
            if d.logged_at.is_none_or(|at| now - at >= Duration::from_secs(1)) {
                d.logged_at = Some(now);
                d.lines = 0;
            }
            if d.lines < 4 {
                d.lines += 1;
                tracing::info!(
                    frame = d.frame,
                    draws = d.draws,
                    pipelines = d.pipelines.len(),
                    given_up = d.given_up,
                    frames_not_logged = d.skipped,
                    keys = %d.keys.join(" "),
                    "draws dropped for pipelines not made yet"
                );
                d.skipped = 0;
            } else {
                d.skipped += 1;
            }
        }
        d.frame = frame;
        d.draws = 0;
        d.pipelines.clear();
        d.keys.clear();
        d.given_up = 0;
    }
    d.draws += 1;
    if d.pipelines.insert(Arc::as_ptr(p) as usize) {
        if p.given_up.load(Ordering::Relaxed) {
            d.given_up += 1;
        }
        if d.keys.len() < 16 {
            d.keys.push(p.key.clone());
        }
    }
}

/// Pipelines made again for the formats bound at a draw, since the start.
static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Whether a draw into views of other formats than its pipeline declares
/// gets a host pipeline for those formats ([`for_formats`]). On by
/// default; `VICTORY_TARGET_VARIANTS=0` sends such a draw to the host with
/// the title's pipeline, as before.
fn variants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TARGET_VARIANTS").is_none_or(|v| v != "0"))
}

/// The host pipeline for a draw with `p` into views of the formats
/// `bound`, which `p` does not declare (`host::formats_differ`): that of
/// the same shaders and state for those formats. It is a pipeline like the
/// title's own: queued at its first draw, made by a worker, waited for
/// within the frame's budget, and the draw is dropped while it is not
/// there ([`ready`]).
#[cold]
pub(crate) fn for_formats<'a>(p: &'a Arc<Pending>, bound: &TargetFormats) -> Option<&'a HostPipeline> {
    match p.graphics.as_deref() {
        Some(g) if variants_on() => ready(variant(p, g, &super::host::formats_for(&p.formats, bound))),
        _ => ready(p),
    }
}

/// The pipeline of `p`'s shaders and its state `g` for the formats
/// `bound`: found in the chain that starts at `p`, or added to it (logged
/// then: once per pipeline and set of formats).
fn variant<'a>(p: &'a Arc<Pending>, g: &GraphicsState, bound: &TargetFormats) -> &'a Arc<Pending> {
    let mut at = p;
    // (`bound` is what `host::formats_for` gives, and a state made for it
    // declares exactly it, so the walk ends at the link made for it; the
    // count is a bound all the same: past it the title's own pipeline.)
    for _ in 0..64 {
        let next = at.variant.get_or_init(|| {
            let so_far = VARIANTS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                pipeline = ?p.shaders,
                declared = ?p.formats,
                bound = ?bound,
                so_far,
                "a pipeline is made again for the render target formats bound at its draw (eight render targets, then depth; VICTORY_TARGET_VARIANTS=0: the draw goes out with the title's pipeline)"
            );
            Arc::new(Pending::new(p.shaders.clone(), Some(Box::new(g.with_formats(bound)))))
        });
        // (A state made for `bound` declares exactly `bound`.)
        if next.formats == *bound {
            return next;
        }
        at = next;
    }
    p
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    if p.left_out.is_some() {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// Starts making `p` now, without waiting for it.
pub(crate) fn begin(p: &Arc<Pending>) {
    if !p.shaders.is_empty() {
        start(p, REQUESTED);
    }
}

/// Whether `p`'s host pipeline is made (`Some(true)`), refused or not
/// translated (`Some(false)`), or not yet (`None`); never waits.
pub(crate) fn state(p: &Pending) -> Option<bool> {
    p.made.get().map(Option::is_some)
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it),
/// or if it is one of the opening scene's producers.
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_some() {
        return;
    }
    if !p.shaders.is_empty() && (p.scene_input || super::library::wanted_early(&p.key)) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        let _ = EARLY_BEGAN.compare_exchange(u64::MAX, crate::host::now_ms(), Ordering::Relaxed, Ordering::Relaxed);
        if !p.started_early.swap(true, Ordering::Relaxed) {
            EARLY_LEFT.fetch_add(1, Ordering::Relaxed);
        }
        start(p, if p.scene_input { REQUESTED } else { EARLY });
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// How long draws may wait for a pipeline in vain, all frames together,
/// before later draws stop waiting for it. A pipeline the driver takes seconds for otherwise
/// costs every frame the whole budget: in play-g's drive into the world
/// the frames waited 470-500 ms each for three minutes while pipelines
/// took 6-16 s each. Given up, it is still made in the background and its
/// draws are dropped until then, as they are once the budget is spent.
/// Counted over frames: in play-g's minute 27 the binoculars' pipelines
/// took the 500 ms of 30 frames in a row, most frames sharing it among
/// several, so no single wait reached 100 ms. Passes over the screen keep
/// waiting. `VICTORY_PIPELINE_GIVE_UP=0` waits as before.
const GIVE_UP: Duration = Duration::from_millis(100);

fn give_up() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_GIVE_UP").is_none_or(|v| v != "0"))
}

/// Whether a pipeline is waited for in one frame only: one the worker has
/// not made within that frame's budget (a slow or stuck driver compile) has
/// its draws dropped in later frames until it is made, instead of taking
/// what is left of the budget again in every frame that draws with it
/// (fxceless/victory-fork #1234, for its #1181: 500 ms waited in every world
/// frame, felt as a freeze while the player moves). [`GIVE_UP`] stops the
/// waits once they add up to 100 ms; this stops them after the first frame,
/// however short its wait. Passes over the screen keep waiting.
/// `VICTORY_PIPELINE_WAIT_EVERY_FRAME=1`: as before.
fn wait_every_frame() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_EVERY_FRAME").is_some_and(|v| v == "1"))
}

/// Whether a draw waits only for a pipeline a worker can make now: one
/// being made, or one queued while a worker is free. In main-own-r1's
/// world entry (game 41-66 s) every worker was in a driver compile of
/// about 10 s, 250 requested pipelines queued behind them, and each new
/// one met took the frame's whole 500 ms before its draws were dropped:
/// 2 to 3 fps for 25 s. Its draws are dropped from the first frame on,
/// as they were after the wait. Passes over the screen keep waiting.
/// `VICTORY_PIPELINE_WAIT_FREE=1`: on.
fn wait_free_only() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_FREE").is_some_and(|v| v == "1"))
}

/// Claims `frame` as the one in which draws wait for `p`: whether it is (or
/// already was) that frame.
fn claim_wait(p: &Pending, frame: u64) -> bool {
    if wait_every_frame() {
        return true;
    }
    match p.waited_frame.compare_exchange(u64::MAX, frame, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => true,
        Err(first) => first == frame,
    }
}

/// When a frame last reached our window, in microseconds from [`EPOCH`]
/// plus one (0: none yet), for [`held_room`].
static PRESENTED_US: AtomicU64 = AtomicU64::new(0);
static EPOCH: OnceLock<Instant> = OnceLock::new();
/// The present whose held frame was told of (one line per held frame).
static HELD_TOLD: AtomicU64 = AtomicU64::new(0);

fn since_epoch_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// A frame reached our window.
pub(crate) fn presented() {
    held_limit();
    PRESENTED_US.store(since_epoch_us() + 1, Ordering::Relaxed);
}

/// How long the title may go without a present before pipeline waits stop
/// (`VICTORY_HELD_FRAME_MS`, off by default). Its GPU watchdog (thread at
/// image+0x2215a60, b99f876b74d4a77f) counts 30 one-second looks at the
/// frame tick and then writes to 0 (+0x23cf683). Budgets are per title
/// frame, so on a cold library one held frame can still add a pass over
/// the screen's 10 s to other waits: ff-4386 r91 waited 7.1, 6.5 and 3.5 s
/// for three passes while frame 1288 was held for 32 s, and was killed.
fn held_limit() -> Option<Duration> {
    static LIMIT: OnceLock<Option<Duration>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let raw = std::env::var("VICTORY_HELD_FRAME_MS").unwrap_or_default();
        let limit = raw.trim().parse::<u64>().ok().filter(|&m| m > 0).map(Duration::from_millis);
        match limit {
            Some(l) => eprintln!("VICTORY_HELD_FRAME_MS={raw:?}: pipeline waits stop after {} ms without a present", l.as_millis()),
            None => eprintln!("VICTORY_HELD_FRAME_MS={raw:?}: off"),
        }
        limit
    })
}

/// What is left of [`held_limit`] since the last present (`None`: no limit,
/// or nothing presented yet). Past it, says so once for that present.
fn held_room() -> Option<Duration> {
    let limit = held_limit()?;
    let at = PRESENTED_US.load(Ordering::Relaxed);
    if at == 0 {
        return None;
    }
    let held = Duration::from_micros(since_epoch_us().saturating_sub(at - 1));
    let room = limit.saturating_sub(held);
    if room.is_zero() && HELD_TOLD.swap(at, Ordering::Relaxed) != at {
        tracing::warn!(
            held_ms = held.as_millis() as u64,
            still_being_made = waiting(),
            "a frame held past VICTORY_HELD_FRAME_MS: its draws and passes do not wait for pipelines until the next present"
        );
    }
    Some(room)
}

/// Waits for `p` while frame `frame` has budget left, unless an earlier
/// wait for it gave up or an earlier frame waited for it.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    if !screen && p.given_up.load(Ordering::Relaxed) {
        return;
    }
    let budget = if screen { screen_budget() } else { wait_budget() };
    let room = held_room();
    if budget.is_zero() || room.is_some_and(|r| r.is_zero()) {
        return;
    }
    if !screen
        && wait_free_only()
        && p.queue_state.load(Ordering::Relaxed) != MAKING
        && GENERAL_BUSY.load(Ordering::Relaxed) >= workers_wanted() as u64
    {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let shared = shared_budget();
    let left = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1).min(room.unwrap_or(Duration::MAX));
        // Claimed only with time left, so a pipeline first drawn with once
        // the frame's budget is spent keeps its one wait for a later frame.
        if left.is_zero() || (!screen && !claim_wait(p, frame)) {
            return;
        }
        if shared {
            spent.1 += left;
        }
        left
    };
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    // The frame's share of the budget is gone and the pipeline is still not
    // there: this is the shape of a first run that stalls its way through the
    // world while pipelines are made as they are drawn, and of a run that
    // would have been quick had it been warmed. Said once per process, with
    // the two things that fix it, because nothing else in the log says the
    // difference between a slow run and a cold one.
    if p.made.get().is_none() && GAVE_UP.fetch_add(1, Ordering::Relaxed) == 0 {
        tracing::warn!(
            still_being_made = waiting(),
            budget_ms = budget.as_millis() as u64,
            "a frame's pipelines did not arrive within its budget (victory warm --used makes them ahead of a run; VICTORY_PIPELINE_WORKERS makes them sooner)"
        );
    }
    let waited = started.elapsed().min(left);
    if !screen && p.made.get().is_none() && give_up() && waited_in_vain(p, waited) {
        tracing::info!(
            ms = p.waited_us.load(Ordering::Relaxed) / 1000,
            key = %p.key,
            "a pipeline waited for in vain: later draws do not wait for it (VICTORY_PIPELINE_GIVE_UP=0: they do)"
        );
    }
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

/// Lists for [`Pending::taken_from`].
const TAKEN_BY_TITLE: u8 = 0;
/// Adds `waited` to what draws waited for `p` in vain: whether that has now
/// reached [`GIVE_UP`] (true once).
fn waited_in_vain(p: &Pending, waited: Duration) -> bool {
    let us = waited.as_micros() as u64;
    let total = p.waited_us.fetch_add(us, Ordering::Relaxed) + us;
    total >= GIVE_UP.as_micros() as u64 && !p.given_up.swap(true, Ordering::Relaxed)
}

const TAKEN_SCREEN: u8 = 1;
const TAKEN_REQUESTED: u8 = 2;
const TAKEN_EARLY: u8 = 3;
const TAKEN_SLOW: u8 = 4;

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            (workers_wanted() / 4).max(1)
        }
    })
}

/// `VICTORY_PIPELINE_KEEP_FREE=n`: early and slow pipelines, which no draw
/// has asked for, take at most all workers but `n`, so a pipeline a draw asks
/// for finds a free worker. In main-own-r1 (cold driver cache) early ones
/// held all 24 workers in the minute before game 41, 6 s each, and the requested ones then
/// queued behind them: from game 41 to 66 every frame waited its 500 ms in
/// vain. Off (0) by default.
fn keep_free() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_PIPELINE_KEEP_FREE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        n.min(workers_wanted().saturating_sub(1))
    })
}

/// How long nothing but known slow pipelines must have been queued before
/// [`quiet_slow_limit`] holds.
const QUIET: Duration = Duration::from_secs(2);

/// [`slow_limit`] while the title queues nothing else: every worker. In
/// play-g the early list ran out at 06:28 and for four minutes 3 to 5 of the
/// 16 workers were busy, all on the slow list (854 waiting), while the rest
/// idled; the drive at 06:32 then asked for 519 of them at once, and draws
/// waited 500 ms a frame. On the first run after a translator VERSION
/// (play-cold, 24 workers) the title loaded with nothing but the slow list
/// queued, 868 of them, and with three quarters 15 to 18 workers were busy
/// for 140 s while the other 6 idled. Cold starts with every worker against
/// three quarters (sc2 and sc8 against sc1 and sc7): 51k and 65k draws
/// dropped in the first 340 s against 89k and 87k, and in sc8 the slow list
/// was made by 245 s (create_ms summed 8871 s against 24291 s, 22 made in
/// vain against 260) while sc7 still had 230 being made at 565 s.
/// `VICTORY_PIPELINE_SLOW_QUIET=3/4`: three quarters, as before. `=0`: a
/// quarter at all times.
fn quiet_slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        quiet_limit(std::env::var("VICTORY_PIPELINE_SLOW_QUIET").ok().as_deref(), workers_wanted(), slow_limit())
    })
}

fn quiet_limit(asked: Option<&str>, workers: usize, slow: usize) -> usize {
    match asked.map(str::trim) {
        Some("0") => slow,
        Some("3/4") => (workers * 3 / 4).max(slow),
        _ => workers.max(slow),
    }
}

/// `VICTORY_PIPELINE_SLOW_FIRST=1`: the slow list's longest first (by the
/// longest each took before), and those of [`LONG_MS`] or more made at
/// normal priority. In play-cold one compute program took 130 s to
/// translate: taken 11 s into the run behind shorter ones, made at 141 s,
/// the world started at 150 s. It had a processor 94% of the time, but
/// went from processor 7 to 30.
fn longest_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_SLOW_FIRST").is_some_and(|v| v == "1"))
}

const LONG_MS: u64 = 10_000;

/// The unclaimed entry of `list` that took longest by `ms` (the first among
/// equals); claimed ones are dropped from it.
fn longest(list: &mut VecDeque<Arc<Pending>>, ms: impl Fn(&str) -> u64) -> Option<Arc<Pending>> {
    list.retain(|p| p.queue_state.load(Ordering::Relaxed) == EARLY);
    let at = (0..list.len()).max_by_key(|&i| (ms(&list[i].key), std::cmp::Reverse(i)))?;
    list.remove(at)
}

/// The unclaimed entry of `list` that took least by `ms` (the first among
/// equals), for [`super::library::hits_first`]; claimed ones are dropped.
fn shortest(list: &mut VecDeque<Arc<Pending>>, ms: impl Fn(&str) -> u64) -> Option<Arc<Pending>> {
    list.retain(|p| p.queue_state.load(Ordering::Relaxed) == EARLY);
    let at = (0..list.len()).min_by_key(|&i| (ms(&list[i].key), i))?;
    list.remove(at)
}

/// Graphics pipelines are made before compute ones, and compute ones never
/// hold every worker (upstream #3025). In the world some 180 compute
/// pipelines take the driver 3.7 s each on average (up to 24 s); taken in
/// order of first use they held all workers for minutes while the scene's
/// graphics pipelines, made in milliseconds, waited behind them and their
/// draws were dropped (a 360 s run: ~870 queued, 500 to 1,000 draws a frame
/// dropped; with graphics first, 7 queued after three minutes and ~370
/// dropped). `VICTORY_PIPELINE_GRAPHICS_FIRST=0`: one list for both, as
/// before.
fn graphics_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_GRAPHICS_FIRST").is_none_or(|v| v != "0"))
}

/// `VICTORY_COMPUTE_FIRST=1`: compute pipelines draws have asked for go
/// ahead of every graphics request, and one the slow list does not name
/// goes past the compute cap. Run 139: `CS_DepthCopy` (a full-screen depth
/// copy, 1 ms to make) waited 245 s behind 7 to 25 s ray-tracing creates
/// that held every compute place, and the world was black until it was made.
fn compute_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMPUTE_FIRST").is_some_and(|v| v == "1"))
}

/// For [`compute_first`]: the quickest asked-for compute pipeline while
/// compute has places, else, in one place past them, the one asked for last
/// that neither the slow list names nor [`long_compute`] is. A cold run has no slow list, and taking
/// the oldest went to the 15-30 s CS_RaytracingResolve variants asked for
/// first (run 139: about 120 of them ahead of CS_DepthCopy).
fn compute_ahead(list: &mut VecDeque<Arc<Pending>>, compute_free: bool, one_past: bool) -> Option<Arc<Pending>> {
    if compute_free {
        // Under COMPUTE_AHEAD the places are few, and a cold run has no
        // slow list to put the long resolves last: they wait for the
        // places graphics leave.
        if compute_ahead_places().is_some() {
            let at = (0..list.len()).filter(|&i| !list[i].long).min_by_key(|&i| super::library::slow_ms(&list[i].key))?;
            return list.remove(at);
        }
        return quickest(list);
    }
    if !one_past {
        return None;
    }
    let at = list.iter().rposition(|p| !p.long && !super::library::slow(&p.key))?;
    list.remove(at)
}

/// `VICTORY_COMPUTE_AHEAD=N`: [`compute_first`] takes compute ahead of the
/// graphics requests only while fewer than N compute pipelines are being
/// made; past that the graphics requests go first and compute takes the
/// places they leave. Cold ff-fixc r164: from game 44 on the workers made
/// 377 compute pipelines and 26 graphics ones, 600 to 770 graphics requests
/// waited, and scene A drew 280 of its 1700 draws for 14 s. Off by default.
fn compute_ahead_places() -> Option<usize> {
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_COMPUTE_AHEAD").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0))
}

/// Of the compute pipelines draws have asked for, the one the slow list says
/// is made soonest (one not on it first; in the order asked among equals).
/// In the 980073cbe run draws asked for 323: 107 of them took the driver
/// 20 s or more each (2,989 s in all) and the other 216 616 s in all, but in
/// the order asked they waited behind the long ones, half of them 160 s or
/// more, their dispatches dropped all the while.
/// `VICTORY_PIPELINE_QUICK_FIRST=0`: in the order asked, as before.
/// Compute workers that early and slow ones leave for compute pipelines a
/// draw waits on. In ff-held r139 the world stayed black for 245 s: early
/// compute creates of 7 to 25 s held every compute worker while
/// CS_DepthCopy, a 1 ms create its draws asked for, waited behind them
/// until game 65. `VICTORY_COMPUTE_KEEP=N` keeps N (at least one compute
/// worker stays open to early ones); off (0) by default.
fn compute_kept() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_COMPUTE_KEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

fn quickest(list: &mut VecDeque<Arc<Pending>>) -> Option<Arc<Pending>> {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_QUICK_FIRST").is_none_or(|v| v != "0")) {
        return list.pop_front();
    }
    let at = (0..list.len()).min_by_key(|&i| super::library::slow_ms(&list[i].key))?;
    list.remove(at)
}

/// `VICTORY_PIPELINE_NOW_FIRST=1`: of the requested graphics pipelines, the
/// first one a draw of this frame or the one before asked for, else the
/// oldest. In the order asked alone, r107 and r111 (cold library) had the
/// world's requests waiting 10 to 175 s, longer the older the run, behind
/// requests of scenes the title had left; the frame on screen got none of
/// its own until they were made. Off: in the order asked, as before.
fn now_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_NOW_FIRST").is_some_and(|v| v == "1"))
}

fn newest(list: &mut VecDeque<Arc<Pending>>) -> Option<Arc<Pending>> {
    if !now_first() {
        return list.pop_front();
    }
    let since = crate::frame::current().saturating_sub(1);
    match list.iter().position(|p| p.asked_frame.load(Ordering::Relaxed) >= since) {
        Some(at) => list.remove(at),
        None => list.pop_front(),
    }
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
    /// Compute ones, after the graphics ones of the same list
    /// ([`graphics_first`]).
    requested_compute: VecDeque<Arc<Pending>>,
    early_compute: VecDeque<Arc<Pending>>,
    /// Compute pipelines being made, and how many may be: all workers but
    /// one, so one is always free for graphics.
    compute_now: usize,
    compute_cap: usize,
    /// Of those, how many only requested ones may take ([`compute_kept`]).
    compute_kept: usize,
    /// Requested per-material ray-tracing resolves ([`Pending::no_wait`]),
    /// how many are being made, and how many may be: all workers but a
    /// quarter, which stay free for the pipelines draws ask for. They are
    /// hundreds at once and take seconds each: in the one line of requested
    /// pipelines every other pipeline a draw asked for stood behind them, and
    /// the frame's wait budget went on that one each frame (run 405: 300 ms
    /// frames with the resolves not waited for, 90 to 530 draws a frame
    /// dropped). Taken after the other requested ones and within the compute
    /// places, so neither [`compute_first`] nor [`compute_kept`] counts them
    /// twice.
    resolves: VecDeque<Arc<Pending>>,
    resolves_now: usize,
    resolve_cap: usize,
    /// Early and slow pipelines being made ([`keep_free`]).
    unasked_now: usize,
    /// When a pipeline last went into a list other than the slow one.
    others_at: Option<Instant>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
            requested_compute: VecDeque::new(),
            early_compute: VecDeque::new(),
            compute_now: 0,
            compute_cap: usize::MAX,
            compute_kept: 0,
            resolves: VecDeque::new(),
            resolves_now: 0,
            resolve_cap: usize::MAX,
            unasked_now: 0,
            others_at: None,
        }
    }

    /// How many slow pipelines may be made at once now.
    fn slow_cap(&self) -> usize {
        if self.others_at.is_none_or(|at| at.elapsed() >= QUIET) { quiet_slow_limit() } else { slow_limit() }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let slow = slow_limit() != usize::MAX && super::library::slow(&p.key);
        let pushed = self.push_as(p, priority, slow);
        if slow && priority == EARLY && pushed.is_some() {
            SLOW_QUEUED.fetch_add(1, Ordering::Relaxed);
        }
        pushed
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        p.queued.get_or_init(Instant::now);
        if priority == REQUESTED || !slow {
            self.others_at = Some(Instant::now());
        }
        let compute = p.graphics.is_none() && graphics_first();
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED && p.no_wait {
            self.resolves.push_back(p.clone());
        } else if priority == REQUESTED && compute {
            self.requested_compute.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else if compute {
            self.early_compute.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let compute_free = self.compute_now < self.compute_cap;
        let early_compute_free = self.compute_now + self.compute_kept < self.compute_cap;
        let mut p = self.screen.pop_front();
        let mut from = TAKEN_SCREEN;
        if p.is_none() && !screens_only {
            if compute_first() && compute_ahead_places().is_none_or(|n| self.compute_now < n) {
                p = compute_ahead(&mut self.requested_compute, compute_free, self.compute_now <= self.compute_cap);
            }
            if p.is_none() {
                p = newest(&mut self.requested);
            }
            from = TAKEN_REQUESTED;
            if p.is_none() && compute_free {
                p = quickest(&mut self.requested_compute);
            }
            if p.is_none() && compute_free && self.resolves_now < self.resolve_cap {
                p = self.resolves.pop_front();
                if let Some(p) = &p {
                    p.as_resolve.store(true, Ordering::Relaxed);
                    self.resolves_now += 1;
                }
            }
            let unasked_free = self.unasked_now + keep_free() < workers_wanted();
            if p.is_none() && unasked_free {
                p = unclaimed(&mut self.early);
                from = TAKEN_EARLY;
            }
            if p.is_none() && early_compute_free && unasked_free {
                p = unclaimed(&mut self.early_compute);
            }
            if p.is_none() && self.slow_now < slow_limit && unasked_free {
                p = if longest_first() {
                    longest(&mut self.slow, super::library::slow_ms)
                } else if super::library::hits_first() {
                    shortest(&mut self.slow, super::library::slow_ms)
                } else {
                    unclaimed(&mut self.slow)
                };
                // A compute one waits its turn while compute holds all it may.
                if let Some(q) = p.take_if(|q| q.graphics.is_none() && !early_compute_free) {
                    self.slow.push_front(q);
                }
                from = TAKEN_SLOW;
                if let Some(p) = &p {
                    p.was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
        }
        let p = p?;
        if p.graphics.is_none() {
            self.compute_now += 1;
        }
        if from >= TAKEN_EARLY && keep_free() > 0 {
            self.unasked_now += 1;
        }
        p.taken_from.store(from, Ordering::Relaxed);
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        let cap = queue.slow_cap();
        if let Some(p) = queue.pop(screens_only, cap) {
            return p;
        }
        // (Woken again after QUIET: the slow list's limit can have risen.)
        queue = AVAILABLE.wait_timeout(queue, QUIET).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else three quarters of the logical processors, 2 to 24. They run below
/// the title's own threads, and a cold cache queues hundreds at once. The
/// driver compiles each pipeline on one thread: the first run after a change
/// of the translator (980073cbe) had 7.5 ks of compute pipelines to make, at
/// least 470 s on 16 workers; with a warm library they wait idle (two thirds,
/// at most 16, before).
fn worker_count(asked: Option<usize>, processors: usize) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None => (processors * 3 / 4).clamp(2, 24),
    }
}

/// [`worker_count`] here: the slow list's limits follow it too, so that
/// `VICTORY_PIPELINE_WORKERS` changes them as well (they took 16 whatever it
/// asked).
fn workers_wanted() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        worker_count(
            std::env::var("VICTORY_PIPELINE_WORKERS").ok().and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
        )
    })
}

/// Stack of each pipeline worker: room for the driver's shader compiler,
/// which runs on it (reserved address space; committed only as used).
const PIPELINE_STACK: usize = 64 << 20;

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = workers_wanted();
        tracing::info!(workers, "pipeline workers");
        if graphics_first() {
            // With one worker only, compute has to take it, else none is made.
            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
            queue.compute_cap = workers.max(2) - 1;
            queue.compute_kept = compute_kept().min(queue.compute_cap - 1);
        }
        QUEUE.lock().unwrap_or_else(|e| e.into_inner()).resolve_cap = (workers - workers / 4).max(1);
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            let screens_only = i >= workers;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                // The driver compiles pipelines on this thread, recursively:
                // NVIDIA's compiler (nvgpucomp64) overflowed the default 2 MB
                // on a world compute program and refused it as out of memory.
                .stack_size(PIPELINE_STACK)
                .spawn(move || {
                    // Below the title's own threads (loading, recording),
                    // and the streaming one, which is below normal itself
                    // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
                    use windows::Win32::System::Threading::*;
                    let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
                    let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                    // SAFETY: a plain call on our own thread.
                    let set = |priority| unsafe {
                        let _ = SetThreadPriority(GetCurrentThread(), priority);
                    };
                    set(priority);
                    loop {
                        let p = next(screens_only);
                        let raise = longest_first();
                        let hold = crate::cores::hold_long_on_fast_cores();
                        let long = (raise || hold)
                            && p.was_slow.load(Ordering::Relaxed)
                            && super::library::slow_ms(&p.key) >= LONG_MS;
                        if long && raise {
                            set(THREAD_PRIORITY_NORMAL);
                        }
                        if long && hold {
                            crate::cores::hold_on_fast_cores(true);
                        }
                        BUSY.fetch_add(1, Ordering::Relaxed);
                        if !screens_only {
                            GENERAL_BUSY.fetch_add(1, Ordering::Relaxed);
                        }
                        {
                            let _slot = ComputeSlot(p.graphics.is_none());
                            p.make();
                        }
                        BUSY.fetch_sub(1, Ordering::Relaxed);
                        if !screens_only {
                            GENERAL_BUSY.fetch_sub(1, Ordering::Relaxed);
                        }
                        if long && raise {
                            set(priority);
                        }
                        if long && hold {
                            crate::cores::hold_on_fast_cores(false);
                        }
                        let slow = p.was_slow.load(Ordering::Relaxed);
                        let unasked = keep_free() > 0 && p.taken_from.load(Ordering::Relaxed) >= TAKEN_EARLY;
                        if slow || unasked {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            if slow {
                                queue.slow_now -= 1;
                            }
                            if unasked {
                                queue.unasked_now -= 1;
                            }
                            AVAILABLE.notify_all();
                        }
                        if p.as_resolve.load(Ordering::Relaxed) {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            queue.resolves_now -= 1;
                            AVAILABLE.notify_all();
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save_while_busy();
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        // The workers save after a pipeline (at most every 20 s), so the
        // last of a burst waited for the next pipeline, and with none made
        // until the window was closed they were lost: the binoculars' in
        // play-g. Saves nothing when nothing was added. While pipelines are
        // still queued it keeps to the workers' burst interval, or it would
        // save the whole library every 20 s in the middle of a burst anyway.
        let saver = std::thread::Builder::new().name("victory-pipelines-save".into()).spawn(|| loop {
            std::thread::sleep(Duration::from_secs(20));
            if WAITING.load(Ordering::Relaxed) > 0 {
                super::library::save_while_busy();
            } else {
                super::library::save(false);
            }
        });
        if let Err(e) = saver {
            tracing::warn!(error = %e, "no pipeline library saver");
        }
        running
    })
}

/// A compute pipeline's place in [`Queue::compute_now`], given back once it
/// is made, also when the driver's compiler panics (it would be kept for
/// good otherwise).
struct ComputeSlot(bool);

impl Drop for ComputeSlot {
    fn drop(&mut self) {
        if self.0 {
            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
            queue.compute_now -= 1;
            AVAILABLE.notify_all();
        }
    }
}

/// Workers making a pipeline now.
static BUSY: AtomicU64 = AtomicU64::new(0);

/// Of them, workers that take any pipeline (not passes over the screen alone).
static GENERAL_BUSY: AtomicU64 = AtomicU64::new(0);

/// The lists' lengths and the busy workers, for the workers' summary line.
fn queues() -> super::spent::Queues {
    let queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    super::spent::Queues {
        lists: [
            queue.screen.len(),
            queue.requested.len() + queue.requested_compute.len() + queue.resolves.len(),
            queue.early.len() + queue.early_compute.len(),
            queue.slow.len(),
        ],
        busy: BUSY.load(Ordering::Relaxed) as usize,
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::refused::summary(false);
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panicking_pipeline_is_refused_and_the_worker_continues() {
        let bad = Arc::new(Pending::new(Vec::new(), None));
        let good = Arc::new(Pending::new(Vec::new(), None));
        let failed = bad.clone();
        let next = good.clone();
        let worker = std::thread::spawn(move || {
            failed.make_with(|| panic!("synthetic pipeline build panic"));
            next.make_with(|| None);
        });

        worker.join().expect("the pipeline worker should survive");
        assert!(matches!(bad.made.get(), Some(None)));
        assert!(matches!(good.made.get(), Some(None)));

        bad.make_with(|| panic!("the refused pipeline should stay complete"));
    }

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count(None, 1), 2);
        assert_eq!(worker_count(None, 24), 18);
        assert_eq!(worker_count(None, 32), 24);
        assert_eq!(worker_count(None, 64), 24);
        assert_eq!(worker_count(Some(0), 24), 1);
        assert_eq!(worker_count(Some(20), 24), 20);
    }

    /// A pipeline one wait gave up on is not waited for again: nothing makes
    /// it here, so the first draw waits the frame's budget for it, and a
    /// draw of a later frame not at all.
    #[test]
    fn a_pipeline_waited_for_in_vain_is_not_waited_for_again() {
        let p = Pending::new(Vec::new(), None);
        // Frames of its own, so the first has its whole budget.
        let frame = u64::MAX - 7;
        let started = Instant::now();
        wait(&p, frame);
        assert!(started.elapsed() >= GIVE_UP, "{:?}", started.elapsed());
        assert!(p.given_up.load(Ordering::Relaxed));
        let again = Instant::now();
        wait(&p, frame - 1);
        assert!(again.elapsed() < GIVE_UP / 10, "{:?}", again.elapsed());
    }

    /// A pipeline an earlier frame waited for is not waited for again, even
    /// if that wait was short (fxceless/victory-fork #1234).
    #[test]
    fn a_pipeline_is_waited_for_in_one_frame_only() {
        let p = Pending::new(Vec::new(), None);
        let frame = u64::MAX - 17;
        p.waited_frame.store(frame - 1, Ordering::Relaxed);
        let started = Instant::now();
        wait(&p, frame);
        assert!(started.elapsed() < GIVE_UP / 10, "{:?}", started.elapsed());
        assert!(!p.given_up.load(Ordering::Relaxed));
    }

    /// The first frame keeps the claim, and its other draws may wait too.
    #[test]
    fn the_first_frame_claims_the_wait() {
        let p = Pending::new(Vec::new(), None);
        assert!(claim_wait(&p, 7));
        assert!(claim_wait(&p, 7));
        assert!(!claim_wait(&p, 8));
        assert_eq!(p.waited_frame.load(Ordering::Relaxed), 7);
    }

    /// Short waits in vain add up over frames until draws stop waiting.
    #[test]
    fn short_waits_in_vain_add_up() {
        let p = Pending::new(Vec::new(), None);
        let short = GIVE_UP / 3;
        assert!(!waited_in_vain(&p, short));
        assert!(!waited_in_vain(&p, short));
        assert!(!p.given_up.load(Ordering::Relaxed));
        assert!(waited_in_vain(&p, short * 2));
        assert!(p.given_up.load(Ordering::Relaxed));
        assert!(!waited_in_vain(&p, short));
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// Early pipelines are counted by how they came about, each once, and
    /// the line is logged after ten or more since the last.
    #[test]
    fn early_pipelines_are_counted_once_by_how_they_came() {
        use super::super::library::Came;
        assert_eq!(early_came_at(Came::Loaded), 0);
        assert_eq!(early_came_at(Came::Compiled), 1);
        assert_eq!(early_came_at(Came::Nothing), 2);
        assert!(!early_line_due(9, 0));
        assert!(early_line_due(10, 0));
        assert!(!early_line_due(15, 10));

        let p = Pending::new(Vec::new(), None);
        // Not started early: not counted.
        let before = EARLY_CAME[1].load(Ordering::Relaxed);
        p.early_done(Came::Compiled);
        assert_eq!(EARLY_CAME[1].load(Ordering::Relaxed), before);
        p.started_early.store(true, Ordering::Relaxed);
        EARLY_LEFT.fetch_add(1, Ordering::Relaxed);
        p.early_done(Came::Compiled);
        assert!(!p.started_early.load(Ordering::Relaxed));
        assert!(EARLY_CAME[1].load(Ordering::Relaxed) > before);
    }

    /// A draw into other formats than its pipeline declares finds the
    /// pipeline for them, which exists once per set of formats and is
    /// queued by its first draw like any other.
    #[test]
    fn a_pipeline_for_bound_formats_is_made_once() {
        // The title's pipeline declares one render target, format 24.
        let p = Arc::new(Pending::new(Vec::new(), Some(Box::new(GraphicsState::declaring(24)))));
        assert_eq!(p.formats, [24, 0, 0, 0, 0, 0, 0, 0, 0]);
        let g = p.graphics.as_deref().unwrap();
        let (a, b) = ([26, 0, 0, 0, 0, 0, 0, 0, 0], [28, 0, 0, 0, 0, 0, 0, 0, 0]);
        let first = variant(&p, g, &a);
        assert_eq!(first.formats, a);
        assert_ne!(first.key, p.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        let second = variant(&p, g, &b);
        assert_eq!(second.formats, b);
        assert!(!Arc::ptr_eq(first, second));
        assert_ne!(second.key, first.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        assert!(Arc::ptr_eq(second, variant(&p, g, &b)));
        assert_eq!(first.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
    }

    /// Every drop gets a reason: one with a host pipeline (a failed binding
    /// or a buffer without a host copy) is named, not left out of the split.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn every_dropped_draw_has_a_reason() {
        assert_eq!(
            dropped_why(Some("being made".into()), "aa+bb"),
            "being made"
        );
        assert_eq!(
            dropped_why(None, "aa+bb"),
            "has a host pipeline; a binding or buffer could not be given: aa+bb"
        );
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn slow_pipelines_get_more_workers_while_nothing_else_is_queued() {
        let mut queue = Queue::new();
        assert_eq!(queue.slow_cap(), quiet_slow_limit());
        let long = Arc::new(Pending::new(Vec::new(), None));
        assert_eq!(queue.push_as(&long, EARLY, true), Some(true));
        assert_eq!(queue.slow_cap(), quiet_slow_limit());
        let quick = Arc::new(Pending::new(Vec::new(), None));
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        assert_eq!(queue.slow_cap(), slow_limit());
        queue.others_at = Instant::now().checked_sub(QUIET);
        assert_eq!(queue.slow_cap(), quiet_slow_limit());
        assert!(quiet_slow_limit() >= slow_limit());
    }

    #[test]
    fn the_quiet_slow_limit_follows_its_switch() {
        assert_eq!(quiet_limit(None, 24, 6), 24);
        assert_eq!(quiet_limit(Some("3/4"), 24, 6), 18);
        assert_eq!(quiet_limit(Some("3/4"), 4, 2), 3);
        assert_eq!(quiet_limit(Some("0"), 24, 6), 6);
        assert_eq!(quiet_limit(Some("all"), 24, 6), 24);
        assert_eq!(quiet_limit(Some("all"), 1, 1), 1);
    }

    #[test]
    fn a_frame_stops_making_compute_pipelines_at_its_cap() {
        let cap = compute_wait_cap();
        if cap.is_zero() {
            return;
        }
        let frame = u64::MAX - 7;
        assert!(compute_wait_left(frame));
        compute_waited(frame, cap / 2);
        assert!(compute_wait_left(frame));
        compute_waited(frame, cap);
        assert!(!compute_wait_left(frame));
        assert!(compute_wait_left(frame + 1));
    }

    #[test]
    fn the_longest_unclaimed_is_taken_first() {
        let one = |key: &str| {
            let mut p = Pending::new(Vec::new(), None);
            p.key = key.to_owned();
            p.queue_state.store(EARLY, Ordering::Relaxed);
            Arc::new(p)
        };
        let ms = |key: &str| match key {
            "a" => 5_000,
            "b" | "c" | "d" => 130_000,
            _ => 0,
        };
        let mut list: VecDeque<_> = ["a", "b", "c", "d", "e"].map(one).into();
        list[1].queue_state.store(MAKING, Ordering::Relaxed);
        let taken: Vec<String> = std::iter::from_fn(|| longest(&mut list, ms)).map(|p| p.key.clone()).collect();
        assert_eq!(taken, ["c", "d", "a", "e"]);
    }

    #[test]
    fn the_shortest_unclaimed_is_taken_first() {
        let one = |key: &str| {
            let mut p = Pending::new(Vec::new(), None);
            p.key = key.to_owned();
            p.queue_state.store(EARLY, Ordering::Relaxed);
            Arc::new(p)
        };
        let ms = |key: &str| match key {
            "a" => 5_000,
            "b" | "c" => 1_200,
            "d" => 130_000,
            _ => 0,
        };
        let mut list: VecDeque<_> = ["a", "b", "c", "d", "e"].map(one).into();
        list[1].queue_state.store(MAKING, Ordering::Relaxed);
        let taken: Vec<String> = std::iter::from_fn(|| shortest(&mut list, ms)).map(|p| p.key.clone()).collect();
        assert_eq!(taken, ["e", "c", "a", "d"]);
    }

    /// Graphics pipelines go before compute ones of the same list, and
    /// compute never takes the last worker (upstream #3025).
    #[test]
    fn graphics_go_first_and_compute_leaves_a_worker() {
        let compute = || Arc::new(Pending::new(Vec::new(), None));
        let graphics = || Arc::new(Pending::new(Vec::new(), Some(Box::default())));
        let (asked_c, asked_g, early_c, early_g, long_c) = (compute(), graphics(), compute(), graphics(), compute());
        let mut queue = Queue::new();
        queue.compute_cap = 1;
        assert_eq!(queue.push_as(&early_c, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&long_c, EARLY, true), Some(true));
        assert_eq!(queue.push_as(&early_g, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&asked_c, REQUESTED, false), Some(true));
        assert_eq!(queue.push_as(&asked_g, REQUESTED, false), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked_g));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked_c));
        // One compute is being made: the others wait, graphics do not.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &early_g));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.slow.len(), 1, "the slow compute one is left for later");
        queue.compute_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &early_c));
        queue.compute_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long_c));
    }

    /// Kept compute workers go to requested compute ones only.
    #[test]
    fn kept_compute_workers_wait_for_requested_ones() {
        let compute = || Arc::new(Pending::new(Vec::new(), None));
        let (early_a, early_b, long_c, asked) = (compute(), compute(), compute(), compute());
        let mut queue = Queue::new();
        queue.compute_cap = 2;
        queue.compute_kept = 1;
        queue.push_as(&early_a, EARLY, false);
        queue.push_as(&early_b, EARLY, false);
        queue.push_as(&long_c, EARLY, true);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &early_a));
        assert!(queue.pop(false, 1).is_none(), "the kept worker is not for early or slow ones");
        queue.push_as(&asked, REQUESTED, false);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        queue.compute_now -= 2;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &early_b));
    }

    /// Resolves go after the other requested pipelines, within their own
    /// limit and the compute places.
    #[test]
    fn resolves_wait_behind_other_requests_and_their_limit() {
        let resolve = || {
            let mut p = Pending::new(Vec::new(), None);
            p.no_wait = true;
            Arc::new(p)
        };
        let compute = || Arc::new(Pending::new(Vec::new(), None));
        let (r1, r2, asked, early) = (resolve(), resolve(), compute(), compute());
        let mut queue = Queue::new();
        queue.compute_cap = 3;
        queue.resolve_cap = 1;
        queue.push_as(&r1, REQUESTED, false);
        queue.push_as(&r2, REQUESTED, false);
        queue.push_as(&early, EARLY, false);
        queue.push_as(&asked, REQUESTED, false);
        assert_eq!(queue.resolves.len(), 2);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &r1));
        assert!(r1.as_resolve.load(Ordering::Relaxed));
        // One resolve at a time: the early one goes before the second.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &early));
        assert!(queue.pop(false, 1).is_none());
        queue.resolves_now -= 1;
        queue.compute_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &r2));
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }
}
