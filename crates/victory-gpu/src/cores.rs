//! Which processors our own threads run on, where a processor has two kinds
//! of cores. `cpu_sets` and `keep_on_fast_cores` are fxceless/victory-fork's
//! (`framelog.rs`, commit 0b0074e0, GPL-3.0-or-later, as this).

/// The system's processor sets of group 0: each one's id, its efficiency
/// class (higher is faster), and its logical processor.
fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.EfficiencyClass, set.LogicalProcessorIndex));
        }
        at += info.Size as usize;
    }
    sets
}

/// The sets of the fastest class from logical processor `from` on, if there
/// are slower ones too.
fn fast_of(sets: &[(u32, u8, u8)], from: u8) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    if sets.iter().all(|c| c.1 == best) {
        return Vec::new();
    }
    sets.iter().filter(|c| c.1 == best && c.2 >= from).map(|c| c.0).collect()
}

/// `VICTORY_FAST_CORES`: 0: no thread asks; 1: the threads a frame waits for
/// one by one; 2 (the default): the threads they share their work with too;
/// 3: those that work beside the rest as well (lists translated, descriptors
/// made): more threads by then than the fast cores run at a time, while the
/// slow ones stood idle.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| std::env::var("VICTORY_FAST_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. That fork's submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.) The title's threads are not asked for.
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

fn ask(what: &str, from: u32) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    if stage() < from {
        return;
    }
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    let (fast, of) = FAST.get_or_init(|| {
        let sets = cpu_sets();
        (fast_of(&sets, 0), sets.len())
    });
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() {
        return;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), fast.as_ptr(), fast.len() as u32) };
    if ok == 0 {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
    } else {
        tracing::info!(what, fast = fast.len(), of, "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
    }
}

/// `VICTORY_PIPELINE_FAST_CORES=1`: a pipeline worker making one that took
/// 10 s or more before (the slow list's time) is kept on the fast cores past
/// the first four logical processors while it makes it, and let go after.
/// The XCL stream guard keeps processors 0 to 3 for OBS. In play-cold a
/// 130 s compute program went from processor 7 to 30 (a slow core).
pub(crate) fn hold_long_on_fast_cores() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_FAST_CORES").is_some_and(|v| v == "1"))
}

/// Keeps the calling thread on those cores (`hold`), or lets it go.
pub(crate) fn hold_on_fast_cores(hold: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    static FAST: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    let fast = FAST.get_or_init(|| {
        let fast = fast_of(&cpu_sets(), 4);
        tracing::info!(fast = fast.len(), "long pipelines are kept on the fast cores past processor 3");
        fast
    });
    if fast.is_empty() {
        return;
    }
    let (list, n) = if hold { (fast.as_ptr(), fast.len() as u32) } else { (std::ptr::null(), 0) };
    // SAFETY: our own thread and a list of the system's own set ids (or none,
    // which clears the selection).
    if unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), list, n) } == 0 {
        tracing::warn!(hold, error = %std::io::Error::last_os_error(), "the pipeline worker's processors could not be set");
    }
}

/// `VICTORY_CPU_LOG=1`: once a second, for the title memory copies and the
/// heap pass, the time each thread spent at them by the logical processor it
/// was on when a piece ended (`GetCurrentProcessorNumber`), and the share of
/// it on the fast cores. Off by default.
pub(crate) mod log {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, LazyLock, Mutex};

    pub(crate) const COPIES: usize = 0;
    pub(crate) const HEAP: usize = 1;
    const WHAT: [&str; 2] = ["copies", "heap pass"];
    const CPUS: usize = 64;

    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("VICTORY_CPU_LOG").is_some_and(|v| v == "1"));

    struct Thread {
        name: String,
        micros: [[AtomicU64; CPUS]; 2],
    }

    static THREADS: Mutex<Vec<Arc<Thread>>> = Mutex::new(Vec::new());

    thread_local! {
        static THIS: Arc<Thread> = {
            let this = Arc::new(Thread {
                name: std::thread::current().name().unwrap_or("title").to_owned(),
                micros: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
            });
            THREADS.lock().unwrap_or_else(|e| e.into_inner()).push(this.clone());
            this
        };
    }

    pub(crate) fn on() -> bool {
        *ON
    }

    /// `us` spent at `what` by the calling thread.
    pub(crate) fn note(what: usize, us: u64) {
        if !*ON {
            return;
        }
        static WATCHER: std::sync::Once = std::sync::Once::new();
        WATCHER.call_once(|| {
            std::thread::Builder::new().name("victory-cpu-log".into()).spawn(watch).ok();
        });
        // SAFETY: no arguments; the calling thread's processor number.
        let cpu = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as usize;
        THIS.with(|t| t.micros[what][cpu.min(CPUS - 1)].fetch_add(us, Ordering::Relaxed));
    }

    fn watch() {
        let sets = super::cpu_sets();
        let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
        let fast = |cpu: usize| sets.iter().any(|c| c.2 as usize == cpu && c.1 == best);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let threads = THREADS.lock().unwrap_or_else(|e| e.into_inner()).clone();
            for (what, name) in WHAT.iter().enumerate() {
                let (mut all, mut on_fast) = (0u64, 0u64);
                let mut by = std::collections::BTreeMap::<&str, ([u64; CPUS], usize)>::new();
                for t in &threads {
                    let taken: Vec<u64> = t.micros[what].iter().map(|m| m.swap(0, Ordering::Relaxed)).collect();
                    if taken.iter().all(|&us| us == 0) {
                        continue;
                    }
                    let entry = by.entry(&t.name).or_insert(([0; CPUS], 0));
                    entry.1 += 1;
                    for (cpu, &us) in taken.iter().enumerate() {
                        entry.0[cpu] += us;
                        all += us;
                        if fast(cpu) {
                            on_fast += us;
                        }
                    }
                }
                if all == 0 {
                    continue;
                }
                let threads: Vec<String> = by
                    .iter()
                    .map(|(name, (cpus, n))| {
                        let at: Vec<String> = cpus
                            .iter()
                            .enumerate()
                            .filter(|&(_, &us)| us >= 100)
                            .map(|(cpu, &us)| format!("{cpu}:{:.1}", us as f64 / 1e3))
                            .collect();
                        format!("{name} x{n} {:.1} ms [{}]", cpus.iter().sum::<u64>() as f64 / 1e3, at.join(" "))
                    })
                    .collect();
                tracing::info!(
                    "cpu {name} {:.1} ms, {:.0}% on the fast cores: {}",
                    all as f64 / 1e3,
                    on_fast as f64 * 100.0 / all as f64,
                    threads.join("; ")
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_of(&[(256, 1, 0), (257, 1, 1), (258, 0, 2)], 0), [256, 257]);
        assert_eq!(fast_of(&[(256, 1, 0), (257, 1, 4), (258, 0, 16)], 4), [257]);
        assert!(fast_of(&[(256, 0, 0), (257, 0, 1)], 0).is_empty());
        assert!(fast_of(&[], 0).is_empty());
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test")).join().unwrap();
        std::thread::spawn(|| {
            hold_on_fast_cores(true);
            hold_on_fast_cores(false);
        })
        .join()
        .unwrap();
    }
}
