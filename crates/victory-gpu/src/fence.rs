//! Fences: a 64-bit value and what waits for it, the events the title names
//! and (no event) the title's own thread, which `SetEventOnCompletion`
//! blocks. Nothing runs on a GPU yet, so only signals (the title's, later our
//! queues') move one.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::{Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 5] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    // ID3D12Fence1.
    guid(
        0x433685fe,
        0xe22b,
        0x4ca0,
        [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers),
    /// and what a thread blocked on the fence wakes on.
    waiters: Mutex<Vec<(u64, usize)>>,
    /// Told whenever the value moves, for a thread blocked without an
    /// event. Its lock is `waiters`, so a value stored between a blocked
    /// thread's look and its wait is never missed.
    reached: Condvar,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
        reached: Condvar::new(),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
/// The title's `GetCompletedValue`.
extern "system" fn completed_called(this: This) -> u64 {
    let value = completed(this);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::polled(this as usize, value);
    value
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::told("an event asked for", this as usize, value);
    if !event.is_null() {
        waiters.push((value, event as usize));
        return S_OK;
    }
    // Does the title ever wait through a null event at all? Said once per
    // process, because the answer is what tells a run that exercises this
    // path from one that cannot: a test that reports "no regression" and a
    // run where the wait happens fifty times look the same in a log that
    // never mentions it.
    static NULL_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if NULL_WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        tracing::info!(
            value,
            "a fence is waited for through a null event: the title's thread blocks until the value is reached"
        );
    }
    // A null event means "block until reached". With submissions sent to the
    // card and not waited for (`crate::host::late`) the signal may be behind
    // them: they are settled first, which sets it, so the usual wait finds
    // its value and returns at once.
    drop(waiters);
    #[cfg(windows)]
    if crate::host::late() {
        crate::queue::before_sync();
        crate::host::settle_for_fence();
    }
    if !blocks() {
        tracing::warn!(value, "blocking fence wait not implemented (VICTORY_FENCE_WAIT=0)");
        return E_NOTIMPL;
    }
    block(fence, value)
}

/// How long a blocked wait goes unreported, in seconds.
const STALL_REPORT_SECS: u64 = 10;

/// Whether a null event blocks the title's thread until its value is
/// reached, as `SetEventOnCompletion` says it does. `VICTORY_FENCE_WAIT=0`
/// answers `E_NOTIMPL` instead, for a run that shows a wait nothing
/// satisfies.
fn blocks() -> bool {
    static BLOCKS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BLOCKS.get_or_init(|| std::env::var("VICTORY_FENCE_WAIT").map_or(true, |v| v != "0"))
}

/// Blocks the calling thread until `value` is reached, as a null event asks.
/// `set` moves the value and wakes it; events waiting beside it are left
/// alone. A wait still unmet says so every [`STALL_REPORT_SECS`], so one
/// nothing will satisfy leaves a line in the log and not only a title that
/// stopped.
fn block(fence: &Fence, value: u64) -> HRESULT {
    let started = std::time::Instant::now();
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if fence.value.load(Ordering::Acquire) >= value {
            return S_OK;
        }
        let (guard, waited) = fence
            .reached
            .wait_timeout(waiters, Duration::from_secs(STALL_REPORT_SECS))
            .unwrap_or_else(|e| e.into_inner());
        waiters = guard;
        let reached = fence.value.load(Ordering::Acquire);
        // A signal below the value is news, not a stall: only a wait that
        // went the whole time without reaching it is reported.
        if waited.timed_out() && reached < value {
            tracing::warn!(
                value,
                reached,
                waited_ms = started.elapsed().as_millis() as u64,
                "a fence the title blocks on is unmet (VICTORY_FENCE_WAIT=0: fail the wait instead)"
            );
        }
    }
}
/// Moves the fence to `value` and sets the events that waited for it.
pub(crate) fn set(this: This, value: u64) {
    let fence = fence(this);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::told("a fence set", this as usize, value);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    // A thread blocked without an event looks at the value again.
    fence.reached.notify_all();
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let _called = crate::timeline::span("a fence set");
    set(this, value);
    // A queue may have waited for it: what it held runs now.
    #[cfg(windows)]
    crate::queue::release_held(false);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed_called as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // A null event blocks; the value is already reached here.
        assert_eq!(wait(f, 5, std::ptr::null_mut()), S_OK);
        // The title reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    #[test]
    fn fence1_interface_is_created_and_queried() {
        const FENCE1_IID: Iid = guid(
            0x433685fe,
            0xe22b,
            0x4ca0,
            [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
        );
        type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
        type Flags = extern "system" fn(This) -> u32;
        type Release = extern "system" fn(This) -> u32;

        // Direct creation as ID3D12Fence1, and GetCreationFlags at slot 0x58.
        let mut f1 = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 3, 0x5, &FENCE1_IID, &mut f1), S_OK);
        assert!(!f1.is_null());
        assert_eq!(slot::<Flags>(f1, 0x58)(f1), 0x5);
        assert_eq!(slot::<Release>(f1, 0x10)(f1), 0);

        // QueryInterface from ID3D12Fence to ID3D12Fence1 returns the same object.
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), 0, 0x2, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        let mut queried = std::ptr::null();
        assert_eq!(slot::<Query>(f, 0x00)(f, &FENCE1_IID, &mut queried), S_OK);
        assert_eq!(queried, f);
        assert_eq!(slot::<Flags>(queried, 0x58)(queried), 0x2);
        assert_eq!(slot::<Release>(queried, 0x10)(queried), 1);
        assert_eq!(slot::<Release>(f, 0x10)(f), 0);
    }

    /// A null event blocks its thread until the value is reached, and
    /// `Signal` wakes it, once it is at or past the value. Runs on every
    /// host: no event handle is made.
    #[test]
    fn a_null_event_blocks_until_the_value_is_reached() {
        use std::sync::mpsc;
        type Wait = extern "system" fn(This, u64, *mut c_void) -> HRESULT;
        type Signal = extern "system" fn(This, u64) -> HRESULT;
        type Release = extern "system" fn(This) -> u32;

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let wait = slot::<Wait>(f, 0x48);
        let signal = slot::<Signal>(f, 0x50);
        let release = slot::<Release>(f, 0x10);
        let at = f as usize;
        let (done, finished) = mpsc::channel();
        let blocking = std::thread::spawn(move || {
            // The fence is released below, after this thread is joined; a
            // null event asks for a blocking wait.
            let hr = wait(at as This, 5, std::ptr::null_mut());
            done.send(hr).expect("the test is still listening");
        });
        // Unmet, so the call is still in its wait.
        assert!(
            finished.recv_timeout(Duration::from_millis(200)).is_err(),
            "a null event returned before its value was reached"
        );
        // A value below the one waited for is news, but not enough.
        assert_eq!(signal(f, 4), S_OK);
        assert!(
            finished.recv_timeout(Duration::from_millis(200)).is_err(),
            "a signal below the wanted value woke the wait"
        );
        assert_eq!(signal(f, 5), S_OK);
        assert_eq!(
            finished.recv_timeout(Duration::from_secs(10)),
            Ok(S_OK),
            "the signal that reached the value did not wake the wait"
        );
        blocking.join().expect("the waiting thread panicked");
        assert_eq!(release(f), 0);
    }
}
