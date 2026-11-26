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
