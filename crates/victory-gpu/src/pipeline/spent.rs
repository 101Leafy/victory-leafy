//! Where the workers' time goes. Each pipeline the driver compiles (and
//! each that takes a quarter of a second or more anyway) gets one line, and
//! every 10 s one line sums up all of them: time queued, the worker
//! thread's own processor time against wall time, translation (its own, or
//! waited for while another thread translated the same shader), the
//! library's load, the driver's compile, and the store (waited for while a
//! save serialised the library, then the store itself), with the logical
//! processors the worker began and ended on.

use std::cell::Cell;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default)]
pub(crate) struct Parts {
    pub(crate) translate: Duration,
    pub(crate) translated: u32,
    pub(crate) shared: Duration,
    pub(crate) load: Duration,
    pub(crate) create: Duration,
    pub(crate) store_wait: Duration,
    pub(crate) store: Duration,
    pub(crate) made: u32,
    pub(crate) loaded: u32,
}

thread_local! {
    static PARTS: Cell<Parts> = Cell::new(Parts::default());
}

pub(crate) fn add(f: impl FnOnce(&mut Parts)) {
    PARTS.with(|p| {
        let mut parts = p.get();
        f(&mut parts);
        p.set(parts);
    });
}

/// This thread's processor time, user and kernel.
fn thread_cpu() -> Duration {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own thread's pseudo handle and four out pointers.
    if unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
        return Duration::ZERO;
    }
    let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    Duration::from_nanos((ticks(kernel) + ticks(user)) * 100)
}

fn processor() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() }
}

pub(crate) struct Run {
    began: Instant,
    cpu: Duration,
    first: u32,
}

pub(crate) fn begin() -> Run {
    PARTS.with(|p| p.set(Parts::default()));
    Run { began: Instant::now(), cpu: thread_cpu(), first: processor() }
}

/// What the workers looked like at a summary: pipelines waiting in each
/// list (screen, requested, early, slow) and workers making one.
pub(crate) struct Queues {
    pub(crate) lists: [usize; 4],
    pub(crate) busy: usize,
}

#[derive(Default)]
struct Total {
    since: Option<Instant>,
    runs: u32,
    made: u32,
    loaded: u32,
    failed: u32,
    wall: Duration,
    cpu: Duration,
    queued: Duration,
    parts: Parts,
    first: Vec<u32>,
    last: Vec<u32>,
}

static TOTAL: Mutex<Total> = Mutex::new(Total {
    since: None,
    runs: 0,
    made: 0,
    loaded: 0,
    failed: 0,
    wall: Duration::ZERO,
    cpu: Duration::ZERO,
    queued: Duration::ZERO,
    parts: Parts {
        translate: Duration::ZERO,
        translated: 0,
        shared: Duration::ZERO,
        load: Duration::ZERO,
        create: Duration::ZERO,
        store_wait: Duration::ZERO,
        store: Duration::ZERO,
        made: 0,
        loaded: 0,
    },
    first: Vec::new(),
    last: Vec::new(),
});

const EVERY: Duration = Duration::from_secs(10);

fn ms(d: Duration) -> u64 {
    d.as_millis() as u64
}

/// Ends `run`, a make of the pipeline `key` that waited `queued` in the
/// list `from`; `ok`: it has a host pipeline.
pub(crate) fn end(run: Run, key: &str, queued: Option<Duration>, from: &str, ok: bool, queues: impl FnOnce() -> Queues) {
    let wall = run.began.elapsed();
    let cpu = thread_cpu().saturating_sub(run.cpu);
    let last = processor();
    let parts = PARTS.with(Cell::get);
    let queued = queued.unwrap_or_default();
    if parts.made > 0 || wall >= Duration::from_millis(250) {
        tracing::info!(
            key,
            from,
            ok,
            queued_ms = ms(queued),
            wall_ms = ms(wall),
            cpu_ms = ms(cpu),
            translate_ms = ms(parts.translate),
            translated = parts.translated,
            shared_wait_ms = ms(parts.shared),
            load_ms = ms(parts.load),
            create_ms = ms(parts.create),
            store_wait_ms = ms(parts.store_wait),
            store_ms = ms(parts.store),
            cpu_first = run.first,
            cpu_last = last,
            "pipeline worker made"
        );
    }
    let mut total = TOTAL.lock().unwrap_or_else(|e| e.into_inner());
    let t = &mut *total;
    let since = *t.since.get_or_insert_with(Instant::now);
    t.runs += 1;
    t.made += u32::from(parts.made > 0);
    t.loaded += u32::from(parts.made == 0 && parts.loaded > 0);
    t.failed += u32::from(!ok);
    t.wall += wall;
    t.cpu += cpu;
    t.queued += queued;
    let p = &mut t.parts;
    p.translate += parts.translate;
    p.translated += parts.translated;
    p.shared += parts.shared;
    p.load += parts.load;
    p.create += parts.create;
    p.store_wait += parts.store_wait;
    p.store += parts.store;
    for (counts, cpu) in [(&mut t.first, run.first), (&mut t.last, last)] {
        let cpu = cpu as usize;
        if counts.len() <= cpu {
            counts.resize(cpu + 1, 0);
        }
        counts[cpu] += 1;
    }
    let span = since.elapsed();
    if span < EVERY {
        return;
    }
    let t = std::mem::take(t);
    drop(total);
    let q = queues();
    let list = |counts: &[u32]| {
        let used: Vec<String> = counts.iter().enumerate().filter(|(_, n)| **n > 0).map(|(i, n)| format!("{i}:{n}")).collect();
        used.join(" ")
    };
    let cores = |d: Duration| format!("{:.2}", d.as_secs_f64() / span.as_secs_f64());
    tracing::info!(
        span_ms = ms(span),
        runs = t.runs,
        made = t.made,
        loaded = t.loaded,
        failed = t.failed,
        wall_cores = %cores(t.wall),
        cpu_cores = %cores(t.cpu),
        queued_ms = ms(t.queued),
        translate_ms = ms(t.parts.translate),
        translated = t.parts.translated,
        shared_wait_ms = ms(t.parts.shared),
        load_ms = ms(t.parts.load),
        create_ms = ms(t.parts.create),
        store_wait_ms = ms(t.parts.store_wait),
        store_ms = ms(t.parts.store),
        screen = q.lists[0],
        requested = q.lists[1],
        early = q.lists[2],
        slow = q.lists[3],
        busy = q.busy,
        cpus_first = %list(&t.first),
        cpus_last = %list(&t.last),
        "pipeline workers"
    );
}
