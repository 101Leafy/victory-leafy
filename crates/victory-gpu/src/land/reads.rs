//! Whether the CPU reads what a landing put into title memory before shaders
//! write it again (`VICTORY_LAND_READS=1`, off by default): the most that
//! leaving it on the card until something reads it could save.
//!
//! In run overlap-census every piece landed held a new byte, and 5,819 of its
//! 8,868 pages of 4 KiB did, so leaving out bytes that did not change saves
//! little more. What could go is the merge of pages nothing reads before the
//! next landing. Here pages put through the title's own address are made
//! no-access once they are in; the first touch opens them again and is
//! counted, by whose code touched (ours or other) and how many frames after.
//! A page landed in again before that counts as never touched. Pages beside
//! the guards are left as they are, their protection is the guards'
//! (`crate::watch`), and so are pages the guards hold read-only and the first
//! page of every 64 KiB block: the guards take what that page has for the
//! whole block, and would give a block found no-access back as it was.
//! A block they guard later takes our pages with it (counted as changed by
//! something else).
//!
//! Not seen: reads through another view of the same pages, the card's, and
//! the kernel's (a call given an address there fails instead of faulting).
//! It costs a protection change per run of pages and a fault per page
//! touched, so a run's frame times say nothing about landing.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualProtect, VirtualQuery,
};

const PAGE: usize = 4096;

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_READS").is_some_and(|v| v == "1"))
}

/// Pages made no-access, by number: the protection they had, the frame they
/// were landed in.
static KEPT: Mutex<BTreeMap<usize, (u32, u64)>> = Mutex::new(BTreeMap::new());

/// Since the last frame-rate line: pages made no-access; first touches by
/// other code than ours, read and written, and by ours; pages landed in
/// again untouched; pages whose protection something else changed first;
/// frames from landing to first touch, summed.
mod n {
    pub(super) const WATCHED: usize = 0;
    pub(super) const READ: usize = 1;
    pub(super) const WRITTEN: usize = 2;
    pub(super) const OURS_READ: usize = 3;
    pub(super) const OURS_WRITTEN: usize = 4;
    pub(super) const AGAIN: usize = 5;
    pub(super) const LOST: usize = 6;
    pub(super) const FRAMES: usize = 7;
    pub(super) const COUNT: usize = 8;
}

static COUNTS: [AtomicU64; n::COUNT] = [const { AtomicU64::new(0) }; n::COUNT];

/// The same since the start, by MiB of title memory: pages made no-access,
/// landed in again untouched, first touched by other code (read, written),
/// first touched by ours. Ranges whose pages are only ever landed in again
/// are the ones a landing could leave on the card. Locked after `KEPT`.
static REGIONS: Mutex<BTreeMap<usize, [u64; 5]>> = Mutex::new(BTreeMap::new());

const REGION: usize = 1 << 20;

fn region(page: usize, which: usize, by: u64) {
    let mut r = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    r.entry(page * PAGE / REGION).or_default()[which] += by;
}

fn count(which: usize, by: u64) {
    COUNTS[which].fetch_add(by, Ordering::Relaxed);
}

fn kept() -> std::sync::MutexGuard<'static, BTreeMap<usize, (u32, u64)>> {
    KEPT.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a query says of the pages from `at`: the protection, and the bytes
/// from `at` that have it; none if they are not committed.
fn query(at: usize) -> Option<(u32, usize)> {
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let got = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if got == 0 || info.State != MEM_COMMIT {
        return None;
    }
    Some((info.Protect, info.BaseAddress as usize + info.RegionSize - at))
}

/// Page `page` gets protection `old` back if it is no-access still; whether
/// it was.
fn open(page: usize, old: u32) -> bool {
    let at = page * PAGE;
    if query(at).is_none_or(|q| q.0 != PAGE_NOACCESS) {
        return false;
    }
    let mut was = 0;
    // SAFETY: a committed page of title memory, given back what it had.
    unsafe { VirtualProtect(at as *const _, PAGE, old, &mut was) != 0 }
}

/// Pages that a landing put new bytes into through the title's own address,
/// `(address, bytes)` within one page each, in order, are made no-access.
pub(super) fn watch(put: &[(usize, usize)]) {
    let mut pages: Vec<usize> = put.iter().map(|&(at, _)| at / PAGE).collect();
    pages.dedup();
    pages.retain(|&p| p % (super::BLOCK / PAGE) != 0 && !crate::watch::guarded(p * PAGE, p * PAGE + PAGE));
    if pages.is_empty() {
        return;
    }
    install();
    let frame = crate::frame::current();
    let mut kept = kept();
    let mut i = 0;
    while i < pages.len() {
        let first = pages[i];
        let mut end = i + 1;
        while end < pages.len() && pages[end] == pages[end - 1] + 1 {
            end += 1;
        }
        // One call for pages that have one protection now.
        let Some((old, len)) = query(first * PAGE) else {
            i = end;
            continue;
        };
        let end = end.min(i + len / PAGE).max(i + 1);
        let n = end - i;
        let mut was = 0;
        // SAFETY: committed pages of title memory; the fault handler below
        // gives each its protection back when it is touched.
        if old != PAGE_NOACCESS && unsafe { VirtualProtect((first * PAGE) as *const _, n * PAGE, PAGE_NOACCESS, &mut was) } != 0 {
            for page in first..first + n {
                kept.insert(page, (old, frame));
                region(page, 0, 1);
            }
            count(n::WATCHED, n as u64);
        }
        i = end;
    }
}

/// Pages of `[at, at + len)` still no-access are about to be landed in again:
/// opened, as never touched.
pub(super) fn again(at: usize, len: usize) {
    let mut kept = kept();
    let pages: Vec<(usize, u32)> = kept.range(at / PAGE..(at + len).div_ceil(PAGE)).map(|(&p, &(old, _))| (p, old)).collect();
    for (page, old) in pages {
        kept.remove(&page);
        if open(page, old) {
            count(n::AGAIN, 1);
            region(page, 1, 1);
        } else {
            count(n::LOST, 1);
        }
    }
}

/// Where our own code lies: from, to.
fn ours() -> (usize, usize) {
    static OURS: OnceLock<(usize, usize)> = OnceLock::new();
    *OURS.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleHandleExW,
        };
        const FLAGS: u32 = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        let mut module = std::ptr::null_mut();
        // SAFETY: an address of ours, and room for the answer.
        if unsafe { GetModuleHandleExW(FLAGS, ours as *const u16, &mut module) } == 0 {
            return (0, 0);
        }
        let base = module as usize;
        // SAFETY: a loaded image's headers: e_lfanew at 0x3c, and in the
        // optional header (24 bytes after the signature) SizeOfImage at 56.
        let size = unsafe {
            let nt = base + std::ptr::read_unaligned((base + 0x3c) as *const u32) as usize;
            std::ptr::read_unaligned((nt + 24 + 56) as *const u32) as usize
        };
        (base, base + size)
    })
}

fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        ours();
        // SAFETY: the handler only looks at the pages kept here.
        unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
    });
}

unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes the exception being dispatched.
    let r = unsafe { &*(*info).ExceptionRecord };
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (write, at) = (r.ExceptionInformation[0] == 1, r.ExceptionInformation[1]);
    let mut kept = kept();
    let Some((old, frame)) = kept.remove(&(at / PAGE)) else {
        drop(kept);
        // Another thread opened it first: a read goes on. A write is the
        // guards' to look at, if it is not writable now.
        let readable = query(at).is_some_and(|q| q.0 != PAGE_NOACCESS);
        return if !write && readable { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH };
    };
    let opened = open(at / PAGE, old);
    drop(kept);
    if !opened {
        count(n::LOST, 1);
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (from, to) = ours();
    let by_us = (from..to).contains(&(r.ExceptionAddress as usize));
    count(
        match (by_us, write) {
            (false, false) => n::READ,
            (false, true) => n::WRITTEN,
            (true, false) => n::OURS_READ,
            (true, true) => n::OURS_WRITTEN,
        },
        1,
    );
    count(n::FRAMES, crate::frame::current().saturating_sub(frame));
    region(at / PAGE, if by_us { 4 } else if write { 3 } else { 2 }, 1);
    EXCEPTION_CONTINUE_EXECUTION
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let c: Vec<u64> = COUNTS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    let touched = c[n::READ] + c[n::WRITTEN] + c[n::OURS_READ] + c[n::OURS_WRITTEN];
    let f = |i: usize| c[i] as f64 / frames;
    tracing::info!(
        "landed pages watched for a first touch per frame (VICTORY_LAND_READS=1): {:.0} made no-access ({:.1} MB); first touched \
         by other code than ours {:.0} read, {:.0} written, by ours {:.0} read, {:.0} written; landed in again untouched {:.0}; \
         changed by something else first {:.0}; {} waiting now; first touches {:.1} frames after their landing",
        f(n::WATCHED),
        f(n::WATCHED) * PAGE as f64 / (1 << 20) as f64,
        f(n::READ),
        f(n::WRITTEN),
        f(n::OURS_READ),
        f(n::OURS_WRITTEN),
        f(n::AGAIN),
        f(n::LOST),
        kept().len(),
        if touched == 0 { 0.0 } else { c[n::FRAMES] as f64 / touched as f64 },
    );
    let r: Vec<(usize, [u64; 5])> = REGIONS.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(&k, &v)| (k, v)).collect();
    // Runs of MiB next to each other with no touch by other code count as one.
    let mut never: Vec<(usize, usize, u64, u64)> = Vec::new();
    for &(k, v) in &r {
        if v[2] + v[3] != 0 || v[1] == 0 {
            continue;
        }
        match never.last_mut() {
            Some(last) if last.1 == k => {
                last.1 = k + 1;
                last.2 += v[0];
                last.3 += v[1];
            }
            _ => never.push((k, k + 1, v[0], v[1])),
        }
    }
    let all: u64 = r.iter().map(|(_, v)| v[0]).sum();
    let left: u64 = never.iter().map(|n| n.2).sum();
    never.sort_by_key(|n| std::cmp::Reverse(n.2));
    let list: Vec<String> =
        never.iter().take(16).map(|n| format!("{:#x}+{:#x} {}/{}", n.0 * REGION, (n.1 - n.0) * REGION, n.3, n.2)).collect();
    let mut read: Vec<&(usize, [u64; 5])> = r.iter().filter(|(_, v)| v[2] + v[3] != 0).collect();
    read.sort_by_key(|(_, v)| std::cmp::Reverse(v[0]));
    let read: Vec<String> =
        read.iter().take(8).map(|(k, v)| format!("{:#x} {} read {} written {} again {}", k * REGION, v[0], v[2], v[3], v[1])).collect();
    tracing::info!(
        "landed MiB never touched by other code since the start (VICTORY_LAND_READS=1): {} of {} pages watched ({:.0}%),          biggest (range, landed in again untouched/watched): {}; most watched of the touched: {}",
        left,
        all,
        if all == 0 { 0.0 } else { left as f64 * 100.0 / all as f64 },
        list.join(", "),
        read.join(", "),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};

    fn snapshot() -> Vec<u64> {
        COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    #[test]
    fn a_landed_page_is_counted_when_read_and_when_landed_in_again() {
        // SAFETY: fresh memory of our own, never let go.
        let block = unsafe { VirtualAlloc(std::ptr::null(), 5 * PAGE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(block, 0);
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(block as *mut u8, 7, 5 * PAGE) };
        let at = block + PAGE;
        let before = snapshot();
        watch(&[(block, 8), (at, 16), (at + PAGE + 8, 8), (at + 2 * PAGE, PAGE)]);
        assert_eq!(query(block).unwrap().0, PAGE_READWRITE);
        assert_eq!(query(at).unwrap().0, PAGE_NOACCESS);
        assert_eq!(query(at + 3 * PAGE).unwrap().0, PAGE_READWRITE);
        // SAFETY: a page of ours, opened by the handler on the way.
        assert_eq!(unsafe { std::ptr::read_volatile((at + 5) as *const u8) }, 7);
        assert_eq!(query(at).unwrap().0, PAGE_READWRITE);
        // SAFETY: as above.
        unsafe { std::ptr::write_volatile((at + PAGE) as *mut u8, 9) };
        again(at, 3 * PAGE);
        assert_eq!(query(at + 2 * PAGE).unwrap().0, PAGE_READWRITE);
        let after = snapshot();
        let d = |i: usize| after[i] - before[i];
        assert_eq!(d(n::WATCHED), 3);
        assert_eq!(d(n::OURS_READ), 1);
        assert_eq!(d(n::OURS_WRITTEN), 1);
        assert_eq!(d(n::AGAIN), 1);
        assert_eq!(d(n::LOST), 0);
        assert!(kept().range(block / PAGE..block / PAGE + 5).next().is_none());
    }
}
