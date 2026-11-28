//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap
//! (unless `VICTORY_LATE_WAIT=1`, see [`late`]);
//! overlap them with per-submission fences once more than clears run.
//! `VICTORY_SETTLE_ROOM=1|alt` gives flights more room before a wait (off by
//! default, see [`settle_room`]).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Context;
use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

use crate::memory::BuilderMemoryUse;

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_start: usize,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_start: usize,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
    /// Heaps of one null view and one sampler, bound where a list has no
    /// heap of that kind: root signatures that index the heaps directly
    /// need both set ([`Host::fallback_view_heap`]).
    fallback_view: OnceLock<Option<ID3D12DescriptorHeap>>,
    fallback_sampler: OnceLock<Option<ID3D12DescriptorHeap>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
const RETIRED_SHARDS: usize = 16;
static RETIRED: [Mutex<std::collections::VecDeque<(u64, Retired)>>; RETIRED_SHARDS] =
    [const { Mutex::new(std::collections::VecDeque::new()) }; RETIRED_SHARDS];

fn retired_shard() -> &'static Mutex<std::collections::VecDeque<(u64, Retired)>> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    thread_local! {
        static INDEX: usize = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) & (RETIRED_SHARDS - 1);
    }
    &RETIRED[INDEX.with(|&i| i)]
}
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A retired object, dropped on whatever thread finishes the submission.
#[allow(dead_code)]
enum Retired {
    Any(Box<dyn std::any::Any>),
    /// What a descriptor slot showed: 20,000 a frame, without a box each.
    Resource(ID3D12Resource),
}
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// [`now_ms`] to the system's tick (16 ms): a few nanoseconds, where the
/// performance counter's are 25. For what is noted tens of thousands of
/// times a frame and read in tenths of a second.
pub(crate) fn coarse_ms() -> u64 {
    use windows_sys::Win32::System::SystemInformation::GetTickCount64;
    // The same beginning as `now_ms`.
    static BEGAN: OnceLock<u64> = OnceLock::new();
    // SAFETY: a plain query.
    let now = unsafe { GetTickCount64() };
    now.saturating_sub(*BEGAN.get_or_init(|| now.saturating_sub(now_ms())))
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((coarse_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// Physical RAM of the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn total_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullTotalPhys)
}

/// How long host copies of textures the title fills must have been idle to
/// be let go, in milliseconds, and how many bytes of them go at most (the
/// longest idle first): none but those of ten minutes while what is in use
/// of the card (`used`: without what our own blocks have free, with what is
/// kept free for others) is under 88 % of its budget, else enough to come
/// down to 85 %. `VICTORY_TEXTURE_KEEP=0`: [`vram_idle_ms`]' older rule,
/// all that are idle; `=1`: that rule with its two longer times.
///
/// (By those times everything idle for 2.5 s went from 75 % of the budget
/// on: 61,000 of them in the first 300 s of session play-1125-0203, at 78 to
/// 80 %, of which 13,000 were made and uploaded again.)
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn vram_keep(budget: u64, used: u64) -> (u64, u64) {
    vram_keep_below(budget, used, budget / 100 * 90)
}

/// [`vram_keep`] around the texture queue's line `line` (`crate::resource::room_line`)
/// instead of 90 % of the budget: none go while `used` is 2 % of the budget
/// under it or more, those idle 5 s go down to 5 % under it while it is
/// under 5 % over it, those idle 1.5 s above that. (At a line of 90 %: 88,
/// 85 and 95 %, as [`vram_keep`] was.)
pub(crate) fn vram_keep_below(budget: u64, used: u64, line: u64) -> (u64, u64) {
    const KEPT_MS: u64 = 600_000;
    const IDLE_MS: u64 = 5_000;
    const SHORT_MS: u64 = 1_500;
    if budget == 0 {
        return (IDLE_MS, u64::MAX);
    }
    let step = budget / 100;
    let aim = line.saturating_sub(5 * step);
    if used < line.saturating_sub(2 * step) {
        (KEPT_MS, u64::MAX)
    } else if used < line + 5 * step {
        (IDLE_MS, used.saturating_sub(aim))
    } else {
        (SHORT_MS, used.saturating_sub(aim))
    }
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
///
/// With room on the card nothing has to go: in session play-1125-0127 88,000
/// of them (31 GB) went after 5 s with 7 GB of the card free, and 24,700
/// (12 GB) had to be made and uploaded again.
/// `VICTORY_TEXTURE_KEEP=0`: 5 s there too, as before.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    static KEEP: OnceLock<bool> = OnceLock::new();
    let keep = *KEEP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_KEEP").is_none_or(|v| v != "0"));
    vram_idle_for(video_memory, keep)
}

fn vram_idle_for(video_memory: Option<[u64; 4]>, keep: bool) -> u64 {
    const KEPT_MS: u64 = 600_000;
    const LONG_MS: u64 = 30_000;
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else if !keep {
        NORMAL_MS
    } else if used_pct >= 68 {
        LONG_MS
    } else {
        KEPT_MS
    }
}

/// Whether new memory is made resident, and waited for, before anything can
/// name it (on by default; `VICTORY_WAIT_RESIDENT=0`: as before). A heap or
/// resource of the default kind is resident once it is made, but its
/// mapping into the card's address space is work of the system's that runs
/// later: only lists sent after it wait for it. Lists sent before run on,
/// and read our descriptor heaps as they are: a slot that shows a texture
/// in a new block is a page fault for them until the block is mapped. The
/// device losses of the night of 2026-11-25 were that (run n08: the fault
/// lay in a texture made in the same tenth of a second, in a new block).
fn wait_resident() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_RESIDENT").is_none_or(|v| v != "0"))
}

/// Whether plain buffers are waited for as well ([`wait_resident`]).
/// `VICTORY_WAIT_RESIDENT=textures` waits for texture blocks and textures
/// only: the device losses above lay in texture blocks, and the waits for
/// buffers are most of them (an RTX 4090 run: 59,000 waits, 18.1 s, the
/// longest 1.2 s; on an RTX 3080, 212,000 waits, 60.8 s, Belobug issue #346).
fn wait_resident_buffers() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_RESIDENT").is_none_or(|v| v != "0" && v != "textures"))
}

/// The flag memory is made with that [`resident`] waits for.
pub(crate) fn not_resident() -> D3D12_HEAP_FLAGS {
    if wait_resident() { D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT } else { D3D12_HEAP_FLAG_NONE }
}

/// How much new memory is made without zeros (`VICTORY_NOT_ZEROED`, default
/// 0: all of it zeroed). 1: texture blocks and committed textures the title
/// fills, whose every texel is uploaded before anything reads it. 2: render
/// targets and depth buffers too, which the title clears before use. Never
/// unordered-access textures or buffers. The system zeroes memory it hands
/// to a process unless it was that process's own before; in Leafy's play
/// run of 2026-11-27 making new memory resident took 2.4 s of the first
/// world frame (29k calls, 23.9 s over the run).
fn zero_level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_NOT_ZEROED").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// [`not_resident`], plus not zeroed for memory of `level` (see
/// [`zero_level`]: 1 title-filled textures and their blocks, 2 targets and
/// depth).
pub(crate) fn fresh(level: u8) -> D3D12_HEAP_FLAGS {
    if level != 0 && zero_level() >= level {
        not_resident() | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
    } else {
        not_resident()
    }
}

/// [`not_resident`], [`resident`] and [`mapped`] for a plain buffer
/// ([`wait_resident_buffers`]).
fn buffer_not_resident() -> D3D12_HEAP_FLAGS {
    if wait_resident_buffers() { not_resident() } else { D3D12_HEAP_FLAG_NONE }
}

fn buffer_resident(device: &ID3D12Device, object: &impl Interface) -> windows::core::Result<()> {
    if wait_resident_buffers() { resident(device, object) } else { Ok(()) }
}

fn buffer_mapped(device: &ID3D12Device) -> windows::core::Result<()> {
    if wait_resident_buffers() { mapped(device) } else { Ok(()) }
}

/// A queue of its own that new memory is waited for with ([`mapped`]).
struct Aside {
    queue: ID3D12CommandQueue,
    list: ID3D12CommandList,
    fence: ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    value: u64,
    _keep: (ID3D12CommandAllocator, [ID3D12Resource; 2]),
}
// SAFETY: D3D12 objects are free-threaded, the event is waited on by one
// thread at a time (the lock).
unsafe impl Send for Aside {}

/// New memory of the card is mapped into its address space when this
/// returns: a copy of a few bytes is sent to a queue of ours after the
/// memory was made, which the system starts only when all mappings asked
/// for so far are done, and waited for. (`MakeResident` came back before
/// that: run n11 was lost as n08 was with it alone.) Best done off a hot
/// path: a tenth of a millisecond to some.
pub(crate) fn mapped(device: &ID3D12Device) -> windows::core::Result<()> {
    static ASIDE: Mutex<Option<Aside>> = Mutex::new(None);
    if !wait_resident() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let mut aside = ASIDE.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: plain object creation; the objects the list names are kept.
    unsafe {
        if aside.is_none() {
            let buffer = |kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| -> windows::core::Result<ID3D12Resource> {
                let mut made: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut made,
                )?;
                made.ok_or_else(windows::core::Error::empty)
            };
            let from = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let to = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST)?;
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.CopyBufferRegion(&to, 0, &from, 0, 16);
            list.Close()?;
            *aside = Some(Aside {
                queue,
                list: list.cast()?,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                event: CreateEventW(None, false, false, None)?,
                value: 0,
                _keep: (allocator, [from, to]),
            });
        }
        let Some(a) = aside.as_mut() else { return Ok(()) };
        a.queue.ExecuteCommandLists(&[Some(a.list.clone())]);
        a.value += 1;
        a.queue.Signal(&a.fence, a.value)?;
        if a.fence.GetCompletedValue() < a.value {
            a.fence.SetEventOnCompletion(a.value, a.event)?;
            WaitForSingleObject(a.event, INFINITE);
        }
    }
    let us = started.elapsed().as_micros() as u64;
    MAPPED[0].fetch_add(1, Ordering::Relaxed);
    MAPPED[1].fetch_add(us, Ordering::Relaxed);
    MAPPED[2].fetch_max(us, Ordering::Relaxed);
    let n = MAPPED[0].load(Ordering::Relaxed);
    if n == 1 || n.is_multiple_of(200) {
        tracing::info!(
            "new card memory waited for until it is mapped (VICTORY_WAIT_RESIDENT=0: not), since the start: {n} times, {:.1} ms, the longest {:.1} ms",
            MAPPED[1].load(Ordering::Relaxed) as f64 / 1000.0,
            MAPPED[2].load(Ordering::Relaxed) as f64 / 1000.0
        );
    }
    Ok(())
}

/// Times, microseconds and the longest of [`mapped`].
static MAPPED: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// New memory (made with [`not_resident`]) is resident when this returns.
pub(crate) fn resident(device: &ID3D12Device, object: &impl Interface) -> windows::core::Result<()> {
    if !wait_resident() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let object: ID3D12Pageable = object.cast()?;
    // SAFETY: a live heap or resource of the device's.
    let made = unsafe { device.MakeResident(&[Some(object)]) };
    let us = started.elapsed().as_micros() as u64;
    RESIDENT[0].fetch_add(1, Ordering::Relaxed);
    RESIDENT[1].fetch_add(us, Ordering::Relaxed);
    RESIDENT[2].fetch_max(us, Ordering::Relaxed);
    static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = RESIDENT[0].load(Ordering::Relaxed);
    if n >= 1000 * (SAID.load(Ordering::Relaxed) + 1) {
        SAID.store(n / 1000, Ordering::Relaxed);
        tracing::info!(
            "new memory waited for (VICTORY_WAIT_RESIDENT=0: not), since the start: {n} times, {:.1} ms, the longest {:.1} ms",
            RESIDENT[1].load(Ordering::Relaxed) as f64 / 1000.0,
            RESIDENT[2].load(Ordering::Relaxed) as f64 / 1000.0
        );
    }
    made
}

/// Whether new memory is only asked to be made resident where it is made
/// (`VICTORY_RESIDENT_LATER=1`; off by default), with
/// `EnqueueMakeResident`: the system pages it in while we go on and signals
/// a fence of ours. What could name it waits then: a slot of our descriptor
/// heap before it is written ([`settle`]: on the CPU, then [`mapped`] once for
/// all asked so far, since lists already sent read the heap as it is), and a
/// queue before lists are sent to it ([`before_send`]: on the card). Memory the
/// CPU maps where it is made is still waited for there. An intro frame of
/// Leafy's play run waited 20 times, 16.5 ms in all, plus 4.5 ms in
/// [`mapped`].
fn resident_later() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RESIDENT_LATER").is_some_and(|v| v == "1"))
}

/// The fence [`resident_soon`] asks with, and the device that asks.
static LATER: OnceLock<Option<(ID3D12Device3, ID3D12Fence)>> = OnceLock::new();
/// The last value asked for, and the last one [`settle`] waited for.
static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SETTLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Asked, microseconds asking, settles that waited, microseconds of them
/// (with [`mapped`]), the longest, queue waits.
static LATER_STATS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// New memory (made with [`not_resident`]) is resident when this returns,
/// and mapped too with `map`; with [`resident_later`], asked to be.
pub(crate) fn resident_soon(device: &ID3D12Device, object: &impl Interface, map: bool) -> windows::core::Result<()> {
    if !wait_resident() {
        return Ok(());
    }
    if resident_later()
        && let Some((asker, fence)) = LATER
            .get_or_init(|| {
                // SAFETY: plain creation.
                let made = device.cast::<ID3D12Device3>().and_then(|d| Ok((d, unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? })));
                made.inspect_err(|e| tracing::warn!(error = %e, "no EnqueueMakeResident: new memory is waited for where it is made")).ok()
            })
            .as_ref()
    {
        static ASKING: Mutex<()> = Mutex::new(());
        let started = std::time::Instant::now();
        let object: ID3D12Pageable = object.cast()?;
        let _one = ASKING.lock().unwrap_or_else(|e| e.into_inner());
        let value = ASKED.load(Ordering::Relaxed) + 1;
        // SAFETY: a live heap or resource of the device's, our fence.
        unsafe { asker.EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, &[Some(object)], fence, value)? };
        ASKED.store(value, Ordering::Release);
        LATER_STATS[0].fetch_add(1, Ordering::Relaxed);
        LATER_STATS[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        return Ok(());
    }
    resident(device, object)?;
    if map { mapped(device) } else { Ok(()) }
}

/// What [`resident_soon`] asked for is resident and mapped when this
/// returns. Called before a slot of our descriptor heap is written.
pub(crate) fn settle() {
    let asked = ASKED.load(Ordering::Acquire);
    if asked != SETTLED.load(Ordering::Acquire) {
        wait_asked(asked);
    }
}

#[cold]
fn wait_asked(asked: u64) {
    // (The event's handle as a number: a handle is not Send.)
    static WAITING: Mutex<Option<usize>> = Mutex::new(None);
    let Some(Some((asker, fence))) = LATER.get() else { return };
    let started = std::time::Instant::now();
    let mut event = WAITING.lock().unwrap_or_else(|e| e.into_inner());
    if SETTLED.load(Ordering::Acquire) >= asked {
        return;
    }
    // SAFETY: our fence, an event only this lock's holder waits on.
    unsafe {
        if fence.GetCompletedValue() < asked {
            if event.is_none() {
                *event = CreateEventW(None, false, false, None).ok().map(|e| e.0 as usize);
            }
            match event.map(|e| HANDLE(e as *mut _)) {
                Some(e) if fence.SetEventOnCompletion(asked, e).is_ok() => {
                    WaitForSingleObject(e, INFINITE);
                }
                _ => {
                    while fence.GetCompletedValue() < asked {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }
    if let Ok(device) = asker.cast::<ID3D12Device>()
        && let Err(e) = mapped(&device)
    {
        tracing::warn!(error = %e, "new memory not waited for until it is mapped");
    }
    SETTLED.fetch_max(asked, Ordering::Release);
    let us = started.elapsed().as_micros() as u64;
    let n = LATER_STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
    LATER_STATS[3].fetch_add(us, Ordering::Relaxed);
    LATER_STATS[4].fetch_max(us, Ordering::Relaxed);
    if n == 1 || n.is_multiple_of(200) {
        let ms = |i: usize| LATER_STATS[i].load(Ordering::Relaxed) as f64 / 1000.0;
        tracing::info!(
            "new memory asked to be resident where it is made (VICTORY_RESIDENT_LATER=1), since the start: {} times, {:.1} ms asking; waited for before a descriptor slot {n} times, {:.1} ms with the mapping, the longest {:.1} ms; queues waited on the card {} times",
            LATER_STATS[0].load(Ordering::Relaxed),
            ms(1),
            ms(3),
            ms(4),
            LATER_STATS[5].load(Ordering::Relaxed)
        );
    }
}

/// Lists sent to `queue` after this wait on the card for what
/// [`resident_soon`] asked for.
pub(crate) fn before_send(queue: &ID3D12CommandQueue) {
    let asked = ASKED.load(Ordering::Acquire);
    if asked == 0 {
        return;
    }
    let Some(Some((_, fence))) = LATER.get() else { return };
    // SAFETY: our fence and a live queue.
    unsafe {
        if fence.GetCompletedValue() < asked {
            let _ = queue.Wait(fence, asked);
            LATER_STATS[5].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Whether memory of the card that shaders use every frame is made with the
/// highest residency priority (`VICTORY_RESIDENCY_HIGH=1`; off by default):
/// card buffers (card sides of title buffers and windows, hash buffers) and
/// render targets, depth, and unordered-access textures and their twins.
/// Plain textures keep the normal priority. When the card's budget runs short
/// the system moves memory into the PC's, and with this it moves plain
/// textures first. Same-settings runs step once into frames 40% slower at
/// 167-240 s and stay there, the card busy at low power, as if something it
/// reads every frame went to the PC's memory (test-cycle, 2026-11-27).
fn residency_high() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RESIDENCY_HIGH").is_some_and(|v| v == "1"))
}

/// Gives new card memory shaders use every frame the highest residency
/// priority, with [`residency_high`].
pub(crate) fn hot(device: &ID3D12Device, object: &impl Interface) {
    if !residency_high() {
        return;
    }
    let (Ok(device), Ok(object)) = (device.cast::<ID3D12Device1>(), object.cast::<ID3D12Pageable>()) else {
        return;
    };
    // SAFETY: a live heap or resource of the device's, one object and one priority.
    let set = unsafe { device.SetResidencyPriority(1, &Some(object), &D3D12_RESIDENCY_PRIORITY_MAXIMUM) };
    let n = HOT[0].fetch_add(1, Ordering::Relaxed) + 1;
    if set.is_err() {
        HOT[1].fetch_add(1, Ordering::Relaxed);
    }
    if n == 1 || n.is_multiple_of(1000) {
        tracing::info!(
            "card memory shaders use every frame kept on the card before textures (VICTORY_RESIDENCY_HIGH), since the start: {n} allocations, {} refused",
            HOT[1].load(Ordering::Relaxed)
        );
    }
}

/// Allocations [`hot`] gave the highest priority, and those the system refused.
static HOT: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Times, microseconds and the longest of [`resident`].
static RESIDENT: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether memory the card was given stays with it while there is room (on
/// by default; `VICTORY_KEEP_FREED=0`: it is given back as before). A shader
/// that reads past a view (a raw load through a typed, structured or
/// constant buffer view has no end on AMD: `indirect::tests::raw_loads_past_views`)
/// or through a slot that shows wrong data reads whatever lies there, and
/// nothing happens; where memory was given back it is a page fault, and the
/// device is lost. Both losses of 2026-11-25 were reads in memory given back
/// seconds before (a block of textures; a list's scratch and a window of
/// title memory). Kept are: empty blocks of textures ([`crate::pool`]) below
/// 85 % of the card's budget, windows of title memory and staging buffers
/// for five minutes with 12 GiB of the PC's memory free, and 1 GiB of
/// scratch chunks.
pub(crate) fn keep_freed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_FREED").is_none_or(|v| v != "0"))
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
/// With plenty of it five minutes ([`keep_freed`]), unless this process
/// is already near the PC's commit limit ([`near_commit_limit`]).
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    ram_idle_for(avail_phys, keep_freed() && !near_commit_limit())
}

/// Whether this process's private bytes have reached 70 % of the smaller
/// of the PC's RAM and its commit limit (RAM plus page files). Free RAM
/// alone does not show it: on a 32 GB PC the process grows to 35-43 GB of
/// commit while 12 GB of RAM can still look free, and freed copies kept for
/// five minutes then push it into the page file. The commit limit alone
/// does not either: with a 48 GB page file it is 80 GB, far past where
/// paging starts. Past the mark the shorter idle times by free RAM apply.
/// A failed query changes nothing.
pub(crate) fn near_commit_limit() -> bool {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return false;
    }
    private_near_limit(private_bytes(), m.ullTotalPhys, m.ullTotalPageFile)
}

fn private_near_limit(private: u64, total_physical: u64, commit_limit: u64) -> bool {
    let room = match (total_physical, commit_limit) {
        (0, 0) => return false,
        (0, b) | (b, 0) => b,
        (a, b) => a.min(b),
    };
    private.saturating_mul(10) >= room.saturating_mul(7)
}

fn ram_idle_for(avail_phys: Option<u64>, keep: bool) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const KEPT_MS: u64 = 300_000;
    if keep && avail_phys.is_some_and(|avail| avail >= 12 * GIB) {
        return KEPT_MS;
    }
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::book::check();
    trim_spare_uploads();
    crate::pool::room(host().and_then(|h| h.video_memory()));
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
}

/// The number of the newest submission begun, and of the newest of which all
/// before it have finished too.
pub(crate) fn submissions() -> (u64, u64) {
    (STARTED.load(Ordering::Acquire), FINISHED.load(Ordering::Acquire))
}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    // (Those the title made and the host has not begun get the next numbers.)
    let queued = crate::queue::queued();
    let started = STARTED.load(Ordering::Acquire) + queued;
    if bury_lag() == 0 && started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    retired_shard()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((started, Retired::Any(Box::new(object))));
}

/// [`retire`] for a resource.
pub(crate) fn retire_resource(object: ID3D12Resource) {
    let queued = crate::queue::queued();
    let started = STARTED.load(Ordering::Acquire) + queued;
    if bury_lag() == 0 && started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    retired_shard()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((started, Retired::Resource(object)));
}

/// How many submissions later than it could (the last one that may use it
/// has finished) an object let go is dropped, and a host list that has run
/// is reset, about two frames: the device losses of the night of 2026-11-25
/// were page faults in memory nothing of ours named any more, at no command
/// in particular. Objects let go while nothing runs wait as well: lists
/// recorded and not yet sent may name them. `VICTORY_BURY_LAG=0`: not.
pub(crate) fn bury_lag() -> u64 {
    static LAG: OnceLock<u64> = OnceLock::new();
    *LAG.get_or_init(|| std::env::var("VICTORY_BURY_LAG").ok().and_then(|v| v.parse().ok()).unwrap_or(64))
}

/// Whether a flight's lists give their dynamic descriptor slots back when
/// it lands, as a submission that is waited for does
/// ([`crate::list::tables_done`]), and not when the lists are reset
/// [`bury_lag`] landings later (`VICTORY_TABLES_AT_LANDING=1`; off, then, as
/// before). Held
/// for 64 landings, the binocular zoom ran the slots out with 4,200 to
/// 4,600 asks a frame refused and a third of its draws gone (codec_d,
/// pareto-residency f4b48ad76). The card has read the slots by then; the
/// objects they name still wait for the bury.
pub(crate) fn tables_at_landing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TABLES_AT_LANDING").is_ok_and(|v| v.trim() == "1"))
}

/// Drops what waited for submissions up to `finished` ([`bury_lag`]). The
/// numbers come in order within each thread's shard.
fn bury(finished: u64) {
    let finished = finished.saturating_sub(bury_lag());
    let mut done: Vec<(u64, Retired)> = Vec::new();
    for shard in &RETIRED {
        let mut retired = shard.lock().unwrap_or_else(|e| e.into_inner());
        let n = retired.iter().take_while(|(at, _)| *at <= finished).count();
        if n > 0 {
            done.extend(retired.drain(..n));
        }
    }
    drop(done);
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !(trail_on() || dred_any()) || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
    if dred_any() {
        let data = host().map(|h| h.removal_data(&[])).unwrap_or_default();
        tracing::error!("device-removed data{}", if data.is_empty() { "; none" } else { &data });
    }
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    CARD_CLOCK_US[2].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time.
fn card_clock_line(frames: f64, ms: f64) {
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), the rest while the title \
         had not submitted; the submitting thread stood waiting for the card {stood:.1} ms",
        if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 }
    );
    #[cfg(feature = "diagnostics")]
    if let Some(sides) = crate::sidetime::line(frames) {
        // The driver's own paging can't be seen from here: what can is our
        // waits for new memory to be resident and the use against the budget.
        static LAST: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
        let [waits, us] = [0, 1].map(|i| {
            let now = RESIDENT[i].load(Ordering::Relaxed);
            now.saturating_sub(LAST[i].swap(now, Ordering::Relaxed)) as f64 / frames
        });
        let memory = host().and_then(|h| h.video_memory()).unwrap_or_default();
        tracing::info!(
            "the card's time per frame in side sends (VICTORY_SIDE_STAMPS=1): {sides}; new memory waited for {waits:.1} times, \
             {:.2} ms; the card's memory {} MB over its budget, {} MB of the PC's shared memory in use; mesh dispatches cut to \
             65,535 groups since the start: {}",
            us / 1000.0,
            memory[1].saturating_sub(memory[0]) >> 20,
            memory[3] >> 20,
            crate::draw::MESH_CUT.load(Ordering::Relaxed)
        );
    }
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory; 631
/// lists were alive in run 65, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The process's private bytes (0 if the query fails): around the making
/// of lists (diagnostics), and against the PC's RAM ([`near_commit_limit`]).
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU if it is made already.
pub(crate) fn made_host() -> Option<&'static Host> {
    HOST.get()?.as_ref()
}

/// Capture the budgets at the failing reset; an earlier periodic sample may
/// describe a different frame. The original error survives unavailable counters.
fn reset_error(part: &str, error: &windows::core::Error) -> String {
    use std::fmt::Write;
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut text = format!("a host command {part} did not reset ({error})");
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        // SAFETY: plain data the call fills in.
        let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: a sized structure.
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 {
            let _ = write!(
                text, "; available RAM {} MiB, available commit {} MiB",
                m.ullAvailPhys >> 20, m.ullAvailPageFile >> 20
            );
        }
        if let Some([local, shared]) = made_host().and_then(Host::video_memory_groups) {
            let _ = write!(
                text, "; DXGI usage/budget: local {}/{} MiB, non-local {}/{} MiB",
                local[1] >> 20, local[0] >> 20, shared[1] >> 20, shared[0] >> 20
            );
        }
    }
    text
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %format_args!("{e:#}"), "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

/// Makes the host GPU on its own thread while the title loads, so its first
/// device call does not wait for the driver (5.4 s of play-g's 8.1 s to the
/// first frame). `VICTORY_HOST_EARLY=0`: made on first use, as before.
pub fn start_host() {
    if std::env::var_os("VICTORY_HOST_EARLY").is_some_and(|v| v == "0") {
        return;
    }
    let spawned = std::thread::Builder::new().name("host device".into()).spawn(|| {
        let started = std::time::Instant::now();
        let made = host().is_some();
        // This thread's own processor time against the wall clock: near the
        // wall time means the driver was working, far under it waiting.
        let (user, kernel) = thread_times();
        tracing::info!(
            made,
            ms = started.elapsed().as_millis() as u64,
            user_ms = user,
            kernel_ms = kernel,
            "host device made ahead of the title"
        );
    });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "host device thread not started; made on first use");
    }
}

/// This thread's user and kernel time so far, in milliseconds.
fn thread_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own thread's pseudo handle and four out pointers.
    if unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
        return (0, 0);
    }
    let ms = |t: FILETIME| ((u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime)) / 10_000;
    (ms(user), ms(kernel))
}

/// The adapter to make the device on: the one `VICTORY_ADAPTER` names (a
/// case-insensitive substring of its description), else the one DXGI ranks
/// for high performance. On a hybrid laptop the default adapter is the
/// integrated GPU, and the title runs there at about a frame a second.
fn host_adapter() -> Option<IDXGIAdapter> {
    // SAFETY: plain DXGI enumeration with valid out parameters.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        if let Ok(wanted) = std::env::var("VICTORY_ADAPTER") {
            let mut index = 0;
            while let Ok(a) = factory.EnumAdapters1(index) {
                // `EnumAdapters1` lists the software adapter too; it stays
                // behind `VICTORY_D3D12_WARP`, whose path checks that DXGI
                // names it as software, so a name that matches it here is not
                // taken.
                if let Ok(desc) = a.GetDesc1()
                    && desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0
                    && names_adapter(&adapter_name(&desc), &wanted)
                {
                    let name = adapter_name(&desc);
                    tracing::info!(adapter = %name, "host adapter (VICTORY_ADAPTER)");
                    return a.cast().ok();
                }
                index += 1;
            }
            if !wanted.trim().is_empty() {
                tracing::warn!(
                    wanted = %wanted,
                    "VICTORY_ADAPTER names no adapter; taking the high-performance one"
                );
            }
        }
        // VICTORY_MONITOR_CARD=1: of the cards with memory of their own, the
        // first in Windows' order that a monitor is plugged into. A PC with
        // two cards then plays on the one its monitor is on instead of
        // sending every frame across to it.
        if crate::switch::on("VICTORY_MONITOR_CARD") {
            let mut index = 0;
            while let Ok(a) = factory.EnumAdapters1(index) {
                index += 1;
                let Ok(desc) = a.GetDesc1() else { continue };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
                    || (desc.DedicatedVideoMemory as u64) < CARD_MEMORY
                    || a.EnumOutputs(0).is_err()
                {
                    continue;
                }
                let name = adapter_name(&desc);
                tracing::info!(adapter = %name, "host adapter (the card the monitor is on, VICTORY_MONITOR_CARD)");
                return a.cast().ok();
            }
        }
        let high = factory
            .cast::<IDXGIFactory6>()
            .and_then(|f| {
                f.EnumAdapterByGpuPreference::<IDXGIAdapter1>(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            })
            .ok()?;
        let name = adapter_name(&high.GetDesc1().ok()?);
        tracing::info!(adapter = %name, "host adapter");
        high.cast().ok()
    }
}

/// An adapter's description without its NUL padding.
fn adapter_name(desc: &DXGI_ADAPTER_DESC1) -> String {
    let len = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..len])
}

/// Whether `wanted` names `adapter`: a case-insensitive substring of its
/// description, ignoring surrounding spaces. An empty `wanted` names none.
/// Own memory from which a card counts as one of its own, not an integrated GPU.
const CARD_MEMORY: u64 = 2 << 30;

fn names_adapter(adapter: &str, wanted: &str) -> bool {
    let wanted = wanted.trim();
    !wanted.is_empty() && adapter.to_lowercase().contains(&wanted.to_lowercase())
}

fn create() -> anyhow::Result<Host> {
    let started = std::time::Instant::now();
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_any() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                if dred_on() || dred_light() {
                    dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                tracing::info!(
                    breadcrumbs = dred_on() || dred_light(),
                    markers = dred_on(),
                    "crash tracing on (device-removed data)"
                );
            } else {
                tracing::warn!("crash tracing asked for and not there: no device-removed data settings");
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter().context("selecting the WARP adapter")?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                .context("D3D12CreateDevice(WARP, FL 12_0, ID3D12Device)")?;
            tracing::info!("host device on the software adapter (WARP)");
        } else if let Some(adapter) = chosen_adapter() {
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                .context("D3D12CreateDevice(chosen adapter, FL 12_0, ID3D12Device)")?;
            crate::tdr::log();
        } else {
            let adapter = host_adapter();
            let driver = std::time::Instant::now();
            match adapter {
                Some(adapter) => D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                    .context("D3D12CreateDevice(host adapter, FL 12_0, ID3D12Device)")?,
                None => {
                    // The one thing this patch exists to stop: a device made
                    // on whichever adapter DXGI defaults to, unannounced.
                    tracing::warn!("no adapter enumerated; the device is made on the default one");
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)
                        .context("D3D12CreateDevice(default adapter, FL 12_0, ID3D12Device)")?
                }
            }
            tracing::info!(ms = driver.elapsed().as_millis() as u64, "D3D12CreateDevice done");
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device
            .ok_or_else(windows::core::Error::empty)
            .context("D3D12CreateDevice returned an empty device")?;
        // Before anything is made with it (`VICTORY_RECORD`).
        #[cfg(feature = "diagnostics")]
        crate::replay::record::from_env(&device);
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // `VICTORY_DRIVER_CACHE=0`: the driver's own shader cache (and the
        // runtime's, for a driver without one) is not used by this run, so a
        // run with an empty `VICTORY_LIBRARY_DIR` makes its pipelines cold
        // without a VERSION bump. Needs Windows developer mode. Switched off
        // for this process only, never cleared.
        if std::env::var_os("VICTORY_DRIVER_CACHE").is_some_and(|v| v == "0") {
            let off = device.cast::<ID3D12Device9>().and_then(|device9| {
                device9.ShaderCacheControl(
                    D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED | D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER,
                    D3D12_SHADER_CACHE_CONTROL_FLAG_DISABLE,
                )
            });
            match off {
                Ok(()) => tracing::info!("the driver's shader cache is off for this run (VICTORY_DRIVER_CACHE=0)"),
                Err(e) => tracing::warn!(error = %e, "the driver's shader cache could not be switched off"),
            }
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
            // (RDNA has waves of 32 and 64, the vendor's older cards of 64 alone.)
            // `VICTORY_HOST_RDNA=1` takes any card for one, to time `VICTORY_RDNA_FLOAT`'s
            // host-neutral bits (2, the legacy products) elsewhere.
            let amd = crate::indirect::amd(&device);
            let forced = std::env::var_os("VICTORY_HOST_RDNA").is_some_and(|v| v == "1");
            victory_shader::emit::set_host_rdna(forced || amd && o1.WaveLaneCountMin == 32 && o1.WaveLaneCountMax == 64);
        }
        let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_7,
        };
        let sm67 = device
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut model).cast(),
                size_of_val(&model) as u32,
            )
            .is_ok()
            && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_7.0;
        victory_shader::emit::set_helper_lane_waves(sm67);
        tracing::info!(
            sm67,
            "host shader model 6.7 (wave operations over helper lanes)"
        );
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .context("CreateCommandQueue(DIRECT)")?;
        let fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .context("CreateFence")?;
        let event = CreateEventW(None, false, false, None).context("CreateEventW")?;
        let rtvs: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })
            .context("CreateDescriptorHeap(RTV)")?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })
            .context("CreateDescriptorHeap(DSV)")?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        // SAFETY: live descriptor heaps.
        let rtv_start = unsafe { rtvs.GetCPUDescriptorHandleForHeapStart().ptr };
        let dsv_start = unsafe { dsvs.GetCPUDescriptorHandleForHeapStart().ptr };
        tracing::info!(ms = started.elapsed().as_millis() as u64, "host D3D12 device ready");
        crate::upload::reserve_early();
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_start,
            rtv_size,
            dsvs,
            dsv_start,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
            adapter,
            fallback_view: OnceLock::new(),
            fallback_sampler: OnceLock::new(),
        })
    }
}

impl Host {
    /// Ticks per second of the queue's timestamps.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1)
    }

    /// The driver's version, as the system counts it (0: not known).
    pub(crate) fn driver(&self) -> i64 {
        let adapter = self.adapter.as_ref();
        // SAFETY: a plain query.
        adapter.map_or(0, |a| unsafe { a.CheckInterfaceSupport(&IDXGIDevice::IID) }.unwrap_or(0))
    }
}

/// The card `VICTORY_ADAPTER` names, for a PC with more than one: part of
/// its name, any case (`5060`, `RTX 5070`), or its index in DXGI's order
/// (`#1`). Unset: Windows' default card. A value that matches no hardware
/// card is logged with the list and ignored.
fn chosen_adapter() -> Option<IDXGIAdapter1> {
    let want = std::env::var("VICTORY_ADAPTER").ok()?;
    let want = want.trim();
    if want.is_empty() {
        return None;
    }
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let index = want.strip_prefix('#').and_then(|n| n.parse::<u32>().ok());
        let mut seen = Vec::new();
        for i in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(i) else { break };
            let Ok(desc) = adapter.GetDesc1() else { continue };
            let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..len]);
            let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
            let hit = match index {
                Some(n) => n == i,
                None => name.to_lowercase().contains(&want.to_lowercase()),
            };
            if hit && !software {
                tracing::warn!(
                    adapter = %name,
                    index = i,
                    memory_mb = desc.DedicatedVideoMemory / (1024 * 1024),
                    "host device on the card VICTORY_ADAPTER names"
                );
                return Some(adapter);
            }
            seen.push(format!("#{i} {name}"));
        }
        tracing::warn!(
            want,
            cards = %seen.join(", "),
            "VICTORY_ADAPTER names no hardware card; Windows' default card is used"
        );
        None
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
pub(crate) fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    /// Card stamps scale to the title's 100 MHz without overflow (a
    /// saturating multiply gave every one the same value).
    #[test]
    fn timestamps_scale_to_the_title_clock() {
        assert_eq!(super::title_ticks(26_509_312, 1_000_000_000), 2_650_931);
        assert_eq!(super::title_ticks(254_637, 10_000_000), 2_546_370);
        // A card clock three hours on at 1 GHz.
        assert_eq!(super::title_ticks(10_800_000_000_000, 1_000_000_000), 1_080_000_000_000);
        // A zero frequency, which a queue without timestamps reports.
        assert_eq!(super::title_ticks(1_000_000_000, 0), 100_000_000_000_000_000);
    }

    /// A fill takes the bytes of a copy at its place in front of it
    /// (`place::forwards`) only if they are all it reads and no step in
    /// between may write there.
    #[test]
    fn fills_take_a_placed_copys_bytes_by_step_order() {
        use super::place::{Order, forwards_of};
        type Got = Vec<((usize, usize), (usize, usize), usize)>;
        fn got(written: &[(usize, usize)], lists: &[&[Order]]) -> Got {
            forwards_of(written, lists.iter().map(|l| l.iter().copied())).iter().map(|f| (f.fill, f.copy, f.at)).collect()
        }
        let copy = Order::Placed(2000, 9000, 64, false);
        // In the same list, and in a later one.
        assert_eq!(got(&[], &[&[copy, Order::Fill(2016, 16)], &[Order::Other, Order::Fill(2000, 64)]]), vec![((0, 1), (0, 0), 16), ((1, 1), (0, 0), 0)]);
        // A later copy over the same bytes: those are the fill's.
        assert_eq!(got(&[], &[&[copy, Order::Placed(2000, 9500, 64, false), Order::Fill(2000, 8)]]), vec![((0, 2), (0, 1), 0)]);
        // A CPU copy over part of it, or a run, in between.
        assert_eq!(got(&[], &[&[copy, Order::Write(2060, 4), Order::Fill(2000, 8)]]), Got::new());
        assert_eq!(got(&[], &[&[copy], &[Order::Run, Order::Fill(2000, 8)]]), Got::new());
        // Work that writes host memory only (`Step::Host`) leaves it.
        assert_eq!(got(&[], &[&[copy, Order::Other, Order::Fill(2000, 8)]]).len(), 1);
        // A source only partly inside the destination, and a fill ahead of the copy.
        assert_eq!(got(&[], &[&[copy, Order::Fill(2056, 16), Order::Fill(1992, 16)]]), Got::new());
        assert_eq!(got(&[], &[&[Order::Fill(2000, 8), copy]]), Got::new());
        // No upload memory and a source flights may write: the bytes are not
        // known at submission. With upload memory they are.
        assert_eq!(got(&[(9000, 9010)], &[&[copy, Order::Fill(2000, 8)]]), Got::new());
        assert_eq!(got(&[(9000, 9010)], &[&[Order::Placed(2000, 9000, 64, true), Order::Fill(2000, 8)]]).len(), 1);
    }

    /// A list's preludes: one a use took comes back once the submission it
    /// was sent with is done, not before, and those of a later submission
    /// stay queued.
    #[test]
    fn a_prelude_is_free_again_when_its_submission_is_done() {
        let mut pool = super::Preludes::<u32>::default();
        pool.free.push(1);
        let first = pool.take(0).unwrap();
        pool.flight.push((u64::MAX, first));
        // A second use in the same submission: another one.
        assert_eq!(pool.take(0), None);
        pool.flight.push((u64::MAX, 2));
        pool.sent(5);
        pool.flight.push((u64::MAX, 3));
        pool.sent(6);
        assert_eq!(pool.take(4), None, "submission 5 is not done");
        assert_eq!(pool.take(5).map(|p| p < 3), Some(true));
        assert_eq!((pool.free.len(), pool.flight.len()), (1, 1));
        assert_eq!(pool.take(6), Some(3));
        assert_eq!(pool.take(6).map(|p| p < 3), Some(true));
        assert!(pool.flight.is_empty());
    }

    /// A list named twice by one submission records each use's moves on a
    /// prelude of its own: a reset of the first before it ran would lose
    /// its barriers.
    #[test]
    fn a_second_use_gets_its_own_prelude() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: plain creation; the out pointer is ours.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made)
        }
        .unwrap();
        // The list starts with it as a copy destination and leaves it
        // readable, its real state common: each use records a move.
        let (start, leave) = (D3D12_RESOURCE_STATE_COPY_DEST.0 as u32, D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32);
        let texture = std::sync::Arc::new(crate::resource::HostTexture::new(made.unwrap(), 0));
        let key = std::sync::Arc::as_ptr(&texture) as usize;
        list.states.lock().unwrap().insert(key, (start, leave, texture.clone()));
        let first = list.prelude(false, &[]).expect("a move is recorded");
        let second = list.prelude(false, &[]).expect("the second use records a move too");
        assert_ne!(
            windows::core::Interface::as_raw(&first),
            windows::core::Interface::as_raw(&second),
            "the second use must not reset the list the first recorded on"
        );
        assert_eq!(texture.state.load(std::sync::atomic::Ordering::Relaxed), leave);
        let preludes = list.preludes.lock().unwrap();
        assert_eq!((preludes.free.len(), preludes.flight.len()), (0, 2));
    }

    /// The first list of a part sent in one call behind others
    /// (`VICTORY_JOIN_ONE_CALL`) has a prelude with the barriers even with
    /// nothing to move.
    #[test]
    fn a_joined_parts_first_list_gets_a_prelude() {
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        list.close();
        assert!(list.prelude(false, &[]).is_none(), "nothing to move: no prelude");
        assert!(list.prelude(true, &[]).is_some(), "behind another part: a prelude with the barriers");
    }

    mod builder_memory {
        use super::super::Deferred;
        use crate::memory::BuilderMemoryUse::{Back, Write};

        const A: u64 = 0x1_0001_0000;
        const B: u64 = 0x3_0001_0000;

        #[test]
        fn admits_writes_eight_gib_apart() {
            let mut recording = Deferred::default();
            assert!(recording.first_builder_memory_use(Write(A)));
            assert!(
                recording.first_builder_memory_use(Write(B)),
                "write at {B:#x} after {A:#x}"
            );
        }

        #[test]
        fn admits_copy_backs_eight_gib_apart() {
            let mut recording = Deferred::default();
            assert!(recording.first_builder_memory_use(Back(A)));
            assert!(
                recording.first_builder_memory_use(Back(B)),
                "copy-back at {B:#x} after {A:#x}"
            );
        }

        #[test]
        fn deduplicates_exact_repeats_per_operation() {
            let mut recording = Deferred::default();
            assert!(recording.first_builder_memory_use(Write(A)));
            assert!(recording.first_builder_memory_use(Back(A)));
            assert!(!recording.first_builder_memory_use(Write(A)));
            assert!(!recording.first_builder_memory_use(Back(A)));
        }

        #[test]
        fn keeps_byte_offsets_within_a_dword() {
            let mut recording = Deferred::default();
            for at in A..A + 4 {
                let write = recording.first_builder_memory_use(Write(at));
                let back = recording.first_builder_memory_use(Back(at));
                assert_eq!((write, back), (true, true), "write and copy-back at {at:#x}");
            }
        }

        #[test]
        fn scopes_uses_to_each_recording_and_reset() {
            let mut first = Deferred::default();
            let mut second = Deferred::default();
            for key in [Write(A), Back(A)] {
                assert!(first.first_builder_memory_use(key));
                assert!(second.first_builder_memory_use(key));
            }
            // HostList::reset replaces its Deferred after the list finishes.
            first = Deferred::default();
            for key in [Write(A), Back(A)] {
                assert!(first.first_builder_memory_use(key));
                assert!(!second.first_builder_memory_use(key));
            }
        }
    }

    /// A failed call waits a second; a success clears the wait.
    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    /// A list of ours is found by its name, with the debug layer's note or not.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_is_found_by_the_number_in_its_name() {
        assert_eq!(super::list_number("victory host list 7"), Some(7));
        assert_eq!(super::list_number("victory host list 158 (debug layer indirect)"), Some(158));
        assert_eq!(super::list_number("victory host list 7x"), None);
        assert_eq!(super::list_number("victory host list 7 x"), None);
        assert_eq!(super::list_number("another list 7"), None);
        assert_eq!(super::list_number(""), None);
    }

    /// What is there of a range some of whose pages are not: those bytes,
    /// counted, and zeros for the rest, whatever the buffer held.
    #[test]
    fn what_is_there_is_read_and_the_rest_is_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const SIZE: usize = 0x8_0000;
        // SAFETY: a reservation of ours; only committed pages are written.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!at.is_null());
            // 0-0x1_1000 and 0x3_0000-0x4_0000 are there.
            for (from, len) in [(0, 0x1_1000), (0x3_0000, 0x1_0000)] {
                assert!(!VirtualAlloc(at.add(from).cast(), len, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.add(from).write_bytes(0x5a, len);
            }
            // From an address inside a page, as textures are packed.
            let mut out = vec![0x77u8; SIZE - 0x300];
            assert_eq!(super::read_there(at.add(0x300), &mut out), 0x1_1000 - 0x300 + 0x1_0000);
            for (i, &b) in out.iter().enumerate() {
                let from = i + 0x300;
                let there = from < 0x1_1000 || (0x3_0000..0x4_0000).contains(&from);
                assert_eq!(b, if there { 0x5a } else { 0 }, "byte {from:#x}");
            }
            // All there: one read.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x1000), &mut out), 0x1_0000);
            assert!(out.iter().all(|&b| b == 0x5a));
            // Nothing there.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x5_0000), &mut out), 0);
            assert!(out.iter().all(|&b| b == 0));
            assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// Only what lies between, before and after the runs is zeroed.
    #[test]
    fn only_bytes_outside_the_runs_are_zeroed() {
        let mut dst = [7u8; 16];
        super::zero_between(&mut dst, 100, &[(102, 3), (104, 2), (110, 4)]);
        assert_eq!(dst, [0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0]);
        let mut dst = [7u8; 4];
        super::zero_between(&mut dst, 100, &[(100, 4)]);
        assert_eq!(dst, [7; 4]);
        super::zero_between(&mut dst, 100, &[]);
        assert_eq!(dst, [0; 4]);
    }

    /// Zeros for a title destination that went meanwhile are no fault.
    #[test]
    fn zeros_where_nothing_is_are_lost() {
        use windows_sys::Win32::System::Memory::*;
        // SAFETY: a reservation of ours, its first page committed.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), 1 << 16, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE).is_null());
            at.write_bytes(7, 4096);
            super::zero(at, 4096);
            assert!(std::slice::from_raw_parts(at, 4096).iter().all(|&b| b == 0));
            super::zero(at.add(4096), 4096);
            VirtualFree(at.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    /// Read backs of one list share 512-byte-aligned chunks instead of
    /// taking a read-back buffer each, and the chunks are reused after a
    /// reset.
    #[test]
    fn readback_spaces_share_chunks_and_are_reused() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let (first, at, ptr) = list.readback_space(4 << 10).unwrap();
        let (second, at2, ptr2) = list.readback_space(4 << 10).unwrap();
        assert_eq!(first.as_raw(), second.as_raw(), "one chunk for the list's read backs");
        assert_eq!(at % 512, 0);
        assert_eq!(at2, at + (4 << 10));
        assert_eq!(ptr2 as usize - ptr as usize, 4 << 10);
        assert!(list.readback_space(SCRATCH_CHUNK + 1).is_none(), "a copy larger than a chunk");
        list.close();
        host.execute(&[&list]);
        list.reset();
        let (again, at3, _) = list.readback_space(4 << 10).unwrap();
        assert_eq!((again.as_raw(), at3), (first.as_raw(), at), "reused after reset");
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    /// A texture read back through the shared chunk: the texels put in the
    /// texture come back at the chunk's 512-byte-aligned offset, which
    /// D3D12 requires of a placed texture footprint.
    #[test]
    fn texture_readback_through_shared_chunks_round_trips_texels() {
        let Some(host) = host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation into our out pointer.
        unsafe {
            host.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut texture,
            )
        }
        .expect("texture");
        let texture = texture.expect("texture");
        let mut footprint = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 1];
        let mut rows = [0u32; 1];
        let mut total = 0u64;
        // SAFETY: arrays for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(&desc, 0, 1, 0, Some(footprint.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        let f = footprint[0];
        let pitch = f.Footprint.RowPitch as usize;
        let (upload, udata) = host.upload_buffer(total).expect("upload buffer");
        let mut pattern = [0u8; 64];
        for y in 0..4 {
            for x in 0..16 {
                let v = (1 + y * 16 + x) as u8;
                pattern[y * 16 + x] = v;
                // SAFETY: inside the upload buffer.
                unsafe { *udata.add(y * pitch + x) = v };
            }
        }
        let list = host.new_list().unwrap();
        let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
        };
        let placed = |r: &ID3D12Resource, mut p: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, offset: u64| {
            p.Offset = offset;
            D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: p },
            }
        };
        // SAFETY: an open list; the texture is COPY_DEST, the upload buffer
        // is upload memory.
        unsafe {
            list.list.CopyTextureRegion(&sub(&texture, 0), 0, 0, 0, &placed(&upload, f, 0), None);
            list.list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        }
        // Something ahead of it in the chunk, so the offset is not 0.
        let _ = list.readback_space(100).expect("read-back space");
        let (readback, at, rdata) = list.readback_space(total).expect("read-back space");
        assert_eq!(at, 512);
        // SAFETY: the texture is COPY_SOURCE; the chunk is COPY_DEST memory.
        unsafe {
            list.list.CopyTextureRegion(&placed(&readback, f, at), 0, 0, 0, &sub(&texture, 0), None);
        }
        list.close();
        host.execute(&[&list]);
        for y in 0..4 {
            for x in 0..16 {
                // SAFETY: inside the chunk, the card is done with it.
                let got = unsafe { *rdata.add(y * pitch + x) };
                assert_eq!(got, pattern[y * 16 + x], "texel row {y} byte {x}");
            }
        }
        list.reset();
    }

    /// `cargo test --release -p victory-gpu benchmark_readback_spaces -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: a read-back buffer per copy against the shared chunks"]
    fn benchmark_readback_spaces() {
        let Some(host) = host() else { return };
        const N: u32 = 2_000;
        const BYTES: u64 = 4 << 10;
        let started = std::time::Instant::now();
        for _ in 0..N {
            let buffer = host.new_read_back(BYTES).unwrap().unwrap();
            let mut data = std::ptr::null_mut();
            // SAFETY: a new read-back buffer.
            unsafe { buffer.Map(0, None, Some(&mut data)) }.unwrap();
        }
        let one_each = started.elapsed();
        let list = host.new_list().unwrap();
        let started = std::time::Instant::now();
        for _ in 0..N {
            let _ = list.readback_space(BYTES).unwrap();
        }
        let shared = started.elapsed();
        println!("{N} x {BYTES} B read-back spaces: one buffer each {one_each:?}, shared chunks {shared:?}");
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    #[test]
    fn reset_error_preserves_the_operation_and_hresult() {
        use windows::Win32::Foundation::{E_INVALIDARG, E_OUTOFMEMORY};
        use windows::core::Error;
        let oom = reset_error("allocator", &Error::from_hresult(E_OUTOFMEMORY));
        assert!(oom.starts_with("a host command allocator did not reset"));
        assert!(oom.contains("0x8007000E"));
        let invalid = Error::from_hresult(E_INVALIDARG);
        assert_eq!(reset_error("list", &invalid), format!("a host command list did not reset ({invalid})"));
    }

    #[test]
    #[ignore = "requires a D3D12 adapter with DXGI memory budgets"]
    fn reset_error_reports_live_memory_budgets() {
        let device = host().expect("D3D12 adapter");
        // SAFETY: the adapter belongs to the live device.
        let desc = unsafe { device.adapter.as_ref().expect("DXGI adapter").GetDesc1() }.unwrap();
        let n = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        eprintln!("D3D12 adapter: {}", String::from_utf16_lossy(&desc.Description[..n]));
        let error = windows::core::Error::from_hresult(windows::Win32::Foundation::E_OUTOFMEMORY);
        let report = reset_error("list", &error);
        eprintln!("{report}");
        for field in ["available RAM ", "available commit ", "DXGI usage/budget: local ", "non-local "] {
            assert!(report.contains(field), "{report}");
        }
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// The batches of clamped records in one scratch buffer
/// ([`HostList::batch_place`]).
const BATCHES: u32 = 8;

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Read-back memory for texture-to-buffer copies: chunks the list's
    /// read backs share, reused after a reset.
    readbacks: Mutex<Readbacks>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<ByCopy<TwinUse>>,
    /// Set once `twins` is non-empty so `twins_written` skips locking on
    /// lists with no depth twins.
    has_twins: std::sync::atomic::AtomicBool,
    /// The lists that run those transitions just before this one, one per
    /// use ([`Preludes`]); the first made when the list first has such
    /// transitions ([`lazy_prelude`]).
    preludes: Mutex<Preludes>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: std::sync::Arc<Mutex<std::collections::BTreeSet<String>>>,
    /// What its indirect executions ran, for the report of a removed device.
    #[cfg(feature = "diagnostics")]
    indirects: indirects::Shared,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted, and after
    /// it what the clamp notes about mesh records. In memory the CPU maps.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// The same words in the card's own memory ([`card_budget`]), which the
    /// clamp uses instead when there is one: zeroed in the list before its
    /// first clamp, its notes copied to `budget` when the list closes.
    card_budget: OnceLock<Option<ID3D12Resource>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// Scratch buffers of the batched clamps (`crate::indirect::batch`), and
    /// how many batches of this recording have a place in them.
    batch_args: Mutex<(Vec<ID3D12Resource>, u32)>,
    /// Refreshes of host copies that a thread of ours is still finding, and
    /// where among the list's steps they belong ([`aside`]).
    coming: Mutex<Option<(usize, std::sync::Arc<aside::Coming>)>>,
    /// Another number each time something changes that what a draw reads
    /// depends on ([`HostList::changed_something`]); no two lists have the
    /// same.
    epoch: std::sync::atomic::AtomicU64,
    /// The title calls that may record into the list ([`HostList::title_call`]).
    title_calls: std::sync::atomic::AtomicU64,
    /// Left open and empty, as made or reset, for the title's next reset
    /// ([`HostList::leave_open`]): that reset finds nothing to undo. The
    /// list's title calls and epoch then, each plus one (0: not left open).
    left_open: [std::sync::atomic::AtomicU64; 2],
    /// Whether this recording has any `Step::Placed` copy (`place::HostList::place_constant`).
    has_placed: std::sync::atomic::AtomicBool,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: std::sync::Arc<Mutex<Vec<Note>>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
    /// With `VICTORY_GPU_TIMES`: whether the last noted command has its
    /// stamp for when it was done ([`HostList::noted_done`]).
    #[cfg(feature = "diagnostics")]
    done: std::sync::atomic::AtomicBool,
    /// The batched clamps' passes on the card.
    #[cfg(feature = "diagnostics")]
    pub(crate) clamp_stamps: crate::indirect::batch::Stamps,
    /// With `VICTORY_GPU_STATS`: the card's counts from each note to the
    /// next.
    #[cfg(feature = "diagnostics")]
    pstats: pstats::Stats,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// A host list's preludes: those free to record on, and those queued, with
/// the fence value of the submission that runs them (`u64::MAX` until it is
/// sent). A submission may name a list twice, and a list sent ahead
/// ([`late`]) may be named again while it is in the air: each use records
/// on one of its own, and a reset of one still queued would lose its
/// barriers.
struct Preludes<T = Prelude> {
    free: Vec<T>,
    flight: Vec<(u64, T)>,
}

impl<T> Default for Preludes<T> {
    fn default() -> Self {
        Preludes { free: Vec::new(), flight: Vec::new() }
    }
}

impl<T> Preludes<T> {
    /// One free to record on, those whose submission the fence has passed
    /// (`done`) back in the pool first.
    fn take(&mut self, done: u64) -> Option<T> {
        for (number, prelude) in std::mem::take(&mut self.flight) {
            if number <= done {
                self.free.push(prelude);
            } else {
                self.flight.push((number, prelude));
            }
        }
        self.free.pop()
    }

    /// Those recorded since the last submission are run by submission
    /// `number`.
    fn sent(&mut self, number: u64) {
        for (n, _) in &mut self.flight {
            if *n == u64::MAX {
                *n = number;
            }
        }
    }
}

/// Whether the new host list each of the title's lists gets when it is
/// submitted is left open for the title's next reset ([`HostList::leave_open`];
/// `VICTORY_FRESH_OPEN=0`: closed at once and reset again, as before). What
/// runs on the card is the same: those lists are empty either way.
fn fresh_open() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FRESH_OPEN").is_none_or(|v| v != "0"))
}

/// Since the last frame line: host lists left open ([`fresh_open`]), the
/// title's resets that found one so and did nothing to it, and those closed
/// after all (submitted again without a reset, or recorded into before it).
static FRESH_OPEN: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// [`FRESH_OPEN`], for the frame-rate line.
fn fresh_open_line(frames: f64) {
    let [open, kept, closed] = [0, 1, 2].map(|i| FRESH_OPEN[i].swap(0, Ordering::Relaxed) as f64 / frames);
    if open + kept + closed > 0.0 {
        tracing::info!(
            "new host lists for the title's lists left open per frame (VICTORY_FRESH_OPEN=0: closed at once): {open:.1}, \
             {kept:.1} resets of the title found them so and had nothing to undo, {closed:.2} closed after all"
        );
    }
}

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// With `VICTORY_GPU_TIMES` a list's stamps are halves: the first for when
/// each command starts, the second for when it is done.
#[cfg(feature = "diagnostics")]
const HALF: u32 = STAMPS / 2;

/// The commands a list times.
#[cfg(feature = "diagnostics")]
fn timed() -> u32 {
    if gpu_times() && !timing() { HALF - 1 } else { STAMPS - 1 }
}

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = ByCopy<(u32, u32, std::sync::Arc<crate::resource::HostTexture>)>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// The title's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        crate::dxc::d3d_compile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            settle();
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED") && !dred_faults() && !dred_light())
}

/// Breadcrumbs and the faulting address, but no markers and no texts
/// (`VICTORY_D3D12_DRED=light`): which list a removed device stopped in, at
/// which of its commands, and the pipelines the list binds; formatting a text
/// for every command is what makes the whole of it slow.
pub(crate) fn dred_light() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some_and(|v| v == "light")
    })
}

/// Any of the three.
pub(crate) fn dred_any() -> bool {
    dred_on() || dred_light() || dred_faults()
}

/// Whether only the faulting address of a removed device is kept
/// (`VICTORY_D3D12_DRED=faults`): no breadcrumbs, no markers and no texts, so
/// it costs next to nothing and can stay on while the title is played.
pub(crate) fn dred_faults() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some_and(|v| v == "faults")
    })
}

/// The lists alive while device-removed data is on, by the number their name
/// ends in, with their marker texts and the pipelines they bind: a report made
/// on another thread than the one that submits has no lists of its own.
#[cfg(feature = "diagnostics")]
#[allow(clippy::type_complexity)]
static LIST_NOTES: Mutex<
    std::collections::BTreeMap<
        u32,
        (
            std::sync::Weak<Mutex<Vec<Note>>>,
            std::sync::Weak<Mutex<std::collections::BTreeSet<String>>>,
            std::sync::Weak<Mutex<indirects::Indirects>>,
        ),
    >,
> = Mutex::new(std::collections::BTreeMap::new());

/// The number in the name of a list of ours ("victory host list 7", which the
/// debug layer ends with a note of its own in brackets).
#[cfg(feature = "diagnostics")]
fn list_number(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("victory host list ")?;
    let digits = rest.split(' ').next()?;
    (rest.len() == digits.len() || rest[digits.len()..].starts_with(" (")).then(|| digits.parse().ok()).flatten()
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Indirect records that ran empty for want of their list's budget, the
/// largest cost of one (told from 2^22 on), and the records that asked for
/// no work at all (no vertices, instances or groups), likewise.
static BUDGET_NOTES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether a list's indirect-work budget lives in the card's own memory
/// (`VICTORY_INDIRECT_CARD_BUDGET=0` keeps it in memory the CPU maps, as
/// before). The clamp charges it with atomics, at least two in a row for
/// every clamp; in mapped memory on a card of its own each of them
/// crosses the bus.
fn card_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_CARD_BUDGET").is_none_or(|v| v != "0"))
}

/// What a list leaves out when it closes (`VICTORY_CLOSE_LEAN`): `notes` 1,
/// the copy of the clamp's notes out of the card's budget and the barriers
/// around it (the notes only feed the log: largest mesh group count,
/// records refused); any other value but `0`, 2, also the moves of the
/// list's indirect scratch buffers back to common. Each of these buffers
/// is the list's own and decays to common when the submission finishes,
/// as every buffer does, so the next recording finds it there anyway.
fn close_lean() -> u8 {
    static LEAN: OnceLock<u8> = OnceLock::new();
    *LEAN.get_or_init(|| match std::env::var("VICTORY_CLOSE_LEAN").as_deref() {
        Err(_) | Ok("0") => 0,
        Ok("notes") => 1,
        Ok(_) => 2,
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands and every command's time summed by
/// pipeline (`VICTORY_GPU_TIMING`, diagnostics builds).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Whether every noted command is timed on the card with the submissions
/// left as they are, summed by pipeline for the five-second report
/// (`VICTORY_GPU_TIMES`, diagnostics builds). `VICTORY_GPU_TIMING` reads
/// about ten times the real frame, as the card stands between its lists.
#[cfg(feature = "diagnostics")]
fn gpu_times() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMES"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on), or for
/// the frame recorder (`VICTORY_RECORD`).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::book::on()
        && let Ok(resource) = object.cast::<ID3D12Resource>()
    {
        let text = text();
        crate::book::made(&resource, || text.clone());
        return name_it(object, || text);
    }
    name_it(object, text);
}

fn name_it(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::replay::record::on() {
        let text = text();
        crate::replay::record::name(object, &text);
        if dred_any()
            && let Ok(object) = object.cast::<ID3D12Object>()
        {
            // SAFETY: a plain call on a live object.
            let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text)) };
        }
        return;
    }
    if dred_any()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping.
struct Chunk {
    buffer: ID3D12Resource,
    data: usize,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// Read-back chunks for texture-to-buffer copies, front to back: the
/// chunks, the one in use and the bytes used in it. A list's read backs
/// share them, and they are reused after a reset (upstream #2818: a
/// read-back buffer for every copy cost CreateCommittedResource and Map).
#[derive(Default)]
struct Readbacks {
    chunks: Vec<ReadbackChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of read-back buffers.
unsafe impl Send for Readbacks {}

struct ReadbackChunk {
    buffer: ID3D12Resource,
    mapped: usize,
    capacity: u64,
}

/// A list's first read-back chunk; the next ones double, up to
/// `SCRATCH_CHUNK`.
const READBACK_FIRST: u64 = 64 << 10;

impl Readbacks {
    /// `len` bytes (a multiple of 512, at most one chunk) in a read-back
    /// chunk, made on demand: the chunk and the offset in it.
    fn allocate(&mut self, host: &Host, len: u64) -> Option<(usize, u64)> {
        while let Some(chunk) = self.chunks.get(self.current) {
            if len <= chunk.capacity - self.used {
                break;
            }
            (self.current, self.used) = (self.current + 1, 0);
        }
        if self.current == self.chunks.len() {
            let next = self.chunks.last().map_or(READBACK_FIRST, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(READBACK_FIRST).max(next).min(SCRATCH_CHUNK);
            #[cfg(feature = "diagnostics")]
            let making = std::time::Instant::now();
            let buffer = host.new_read_back(capacity).ok()??;
            let mut data = std::ptr::null_mut();
            // SAFETY: a read-back buffer of ours, mapped once for good; the
            // CPU reads it only after the list that wrote it has run.
            unsafe { buffer.Map(0, None, Some(&mut data)) }.ok()?;
            if data.is_null() {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::ReadBacksMadeNs, making.elapsed().as_nanos() as u64);
                crate::copystat::add(crate::copystat::C::ReadBacksMade, 1);
                crate::copystat::add(crate::copystat::C::ReadBacksMadeBytes, capacity);
            }
            name(&buffer, || format!("list read-back chunk {capacity:#x}"));
            self.chunks.push(ReadbackChunk { buffer, mapped: data as usize, capacity });
            self.used = 0;
        }
        let at = self.used;
        self.used += len;
        Some((self.current, at))
    }
}
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them, unless
/// `VICTORY_SCRATCH_SPARE_MB` says otherwise ([`spare_most`]).
const SPARE_MOST: u64 = 128 << 20;
/// The same with [`keep_freed`].
const SPARE_KEPT: u64 = 1 << 30;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Whether a list gives its scratch chunks back when its submission lands
/// ([`HostList::release_scratch`]; on by default, `VICTORY_SCRATCH_AT_LANDING=0`: at its reset).
fn scratch_at_landing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_AT_LANDING").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes or more: one kept ([`SPARE`]) or a new
/// one; and its size.
///
/// A small first chunk ([`SCRATCH_FIRST`]) asked for while none of them is
/// kept is a kept 4 MiB one. Without that, on an RTX 5080 (Jason's balcony,
/// Series S) the spares sat at their bound the whole time, all of them
/// 4 MiB ones, while some 35 small first chunks a frame were made new and
/// as many came back past the bound and were freed (about 1 ms each, a
/// committed upload buffer).
fn take_chunk(host: &Host, size: u64) -> Option<(Chunk, u64)> {
    if scratch_pool() {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((chunk, size)) = take_kept(&mut spare, size, scratch_any_size()) {
            ALIVE[3].fetch_sub(1, Ordering::Relaxed);
            ALIVE[4].fetch_sub(size, Ordering::Relaxed);
            ALIVE[5].fetch_add(1, Ordering::Relaxed);
            return Some((chunk, size));
        }
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "list scratch".into());
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    let chunk = Chunk {
        buffer,
        data: data as usize,
    };
    Some((chunk, size))
}

/// Whether a list's small first chunk may be a kept 4 MiB one
/// (`VICTORY_SCRATCH_ANY_SIZE`, on; `=0`: only one of its own size).
fn scratch_any_size() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_ANY_SIZE").is_none_or(|v| v != "0"))
}

/// The most bytes of scratch chunks kept while no list holds them:
/// [`SPARE_MOST`], or `VICTORY_SCRATCH_SPARE_MB`.
///
/// On the balcony the lists hold some 140 MB of chunks at once, and with
/// 128 MB the spares stood at the bound the whole run: what came back past
/// it was freed and what was asked for next made new, some 12 chunks of
/// 4 MiB a frame, about 2 ms each to free.
fn spare_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        let mb = std::env::var("VICTORY_SCRATCH_SPARE_MB").ok();
        mb.and_then(|v| v.trim().parse::<u64>().ok()).map_or(SPARE_MOST, |mb| mb << 20)
    })
}

/// A kept chunk for a list that asks for `size` bytes, and its size: one of
/// that size, else, for a small first chunk with `any`, a kept 4 MiB one.
fn take_kept<T>(spare: &mut Spare<T>, size: u64, any: bool) -> Option<(T, u64)> {
    spare.take(size).map(|c| (c, size)).or_else(|| {
        (any && size == SCRATCH_FIRST)
            .then(|| spare.take(SCRATCH_CHUNK).map(|c| (c, SCRATCH_CHUNK)))
            .flatten()
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`spare_most`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                spare.give(chunk, size, if keep_freed() { SPARE_KEPT } else { spare_most() })
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// Copy sources of [`HostList::upload`] no list holds: upload buffers,
/// then shared buffers, by size class.
static UPLOAD_SPARE: [Mutex<Spare<Chunk>>; 2] = [const { Mutex::new(Spare::new()) }; 2];

/// The most bytes [`UPLOAD_SPARE`] keeps of each kind
/// (`VICTORY_UPLOAD_POOL_MB`, 1024; 0: no pool, a buffer per copy as
/// before). Every copy a list recorded made a buffer of its own and waited
/// for it to be resident: 19,324 shared buffers and 9.3 s of waits in a
/// 5-minute session (RTX 4090).
fn upload_pool_most() -> u64 {
    static MB: OnceLock<u64> = OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_UPLOAD_POOL_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024)
    }) << 20
}

fn take_upload(shared: bool, class: u64) -> Option<Chunk> {
    UPLOAD_SPARE[usize::from(shared)].lock().unwrap_or_else(|e| e.into_inner()).take(class)
}

/// The copy sources of a list whose submissions have run.
fn give_uploads(pooled: Vec<(Chunk, u64, bool)>) {
    if pooled.is_empty() {
        return;
    }
    let most = upload_pool_most();
    let mut freed = Vec::new();
    for (chunk, class, shared) in pooled {
        if let Some(chunk) = UPLOAD_SPARE[usize::from(shared)].lock().unwrap_or_else(|e| e.into_inner()).give(chunk, class, most) {
            freed.push(chunk);
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// Near the commit limit ([`near_commit_limit`]): the kept copy sources
/// down to [`SPARE_MOST`] of each kind, the ones given last kept.
fn trim_spare_uploads() {
    let over = UPLOAD_SPARE.iter().any(|spare| spare.lock().unwrap_or_else(|e| e.into_inner()).bytes > SPARE_MOST);
    if !over || !near_commit_limit() {
        return;
    }
    let mut freed = Vec::new();
    for spare in &UPLOAD_SPARE {
        let mut spare = spare.lock().unwrap_or_else(|e| e.into_inner());
        while spare.bytes > SPARE_MOST && !spare.kept.is_empty() {
            let (chunk, size) = spare.kept.remove(0);
            spare.bytes -= size;
            freed.push(chunk);
        }
    }
    drop(freed);
}

/// [`give_chunks`] for a list whose submission has landed but which is reset only after the
/// bury lag ([`HostList::release_scratch`]): the chunks the spare pool takes go there, and
/// the others are handed back, to be freed at that reset as before (nothing is freed earlier
/// than it was).
fn spare_what_fits(chunks: Vec<(Chunk, u64)>) -> Vec<(Chunk, u64)> {
    let mut kept = Vec::new();
    let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
    let limit = if keep_freed() { SPARE_KEPT } else { SPARE_MOST };
    for (chunk, size) in chunks {
        match spare.give(chunk, size, limit) {
            None => {
                ALIVE[3].fetch_add(1, Ordering::Relaxed);
                ALIVE[4].fetch_add(size, Ordering::Relaxed);
            }
            Some(chunk) => kept.push((chunk, size)),
        }
    }
    kept
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, odd where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// The fetch inside flights ([`inflight_fetch`]), if it is on and was
    /// made.
    inflight: Option<CardInflight>,
}

/// What the fetch inside flights needs ([`inflight_fetch`], [`CARD_FETCH`]).
struct CardInflight {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// PC memory, mapped (buffer, mapping, bytes): one dword per slot, the
    /// counter of the flight whose region starts at that slot; then, from
    /// [`CardInflight::entries`] on, one dword per dword of
    /// `CardWork::differs`, the entry of that piece.
    table: (ID3D12Resource, usize, u64),
    /// The card's own memory, [`CardInflight::slots`] slots of
    /// [`INFLIGHT_SLOT`] bytes: where the shader puts the pieces. A flight's
    /// region of it is written only by that flight and read only by its
    /// landing.
    staging: ID3D12Resource,
    slots: u32,
    /// The next region starts here ([`take_region`]).
    head: u32,
    /// The regions (first slot, slots) of flights in the air, oldest first.
    in_air: std::collections::VecDeque<(u32, u32)>,
    /// What fetches the old way brought (copy, offsets, and the highest
    /// send whose lists were surely on the queue ahead of the fetch: its
    /// order of the queue less one). A flight sent at or before that ran
    /// ahead of the fetch, so the CPU's side holds those bytes as the card
    /// had them after the flight, or newer: the flight's own slot is older
    /// there and is not merged there. Dropped once every flight sent up to
    /// it has landed.
    brought: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)>,
}

impl CardInflight {
    /// Where the entries start in `table`, in bytes.
    fn entries(&self) -> u32 {
        self.slots * 4
    }
}

/// What the thread that lands flights brings the staged pieces down with
/// ([`inflight_fetch`]): a copy queue of its own, so the copy runs beside
/// the flights the submitting thread keeps sending and not behind them, its
/// list, fence and event, and the PC's memory the pieces come down to.
struct InflightDown {
    queue: ID3D12CommandQueue,
    list: Prelude,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one group per 64 KiB piece (cut at multiples of
/// 64 KiB of title memory, as `Shadow::card_piece` cuts them): `differs`
/// gets a dword per piece, odd where a byte differs. The offsets are widened
/// to 16 bytes; buffers are multiples of 64 KiB, and a piece flagged for a
/// byte just outside the range costs a fetch that finds nothing.
///
/// A piece's 64 threads walk it side by side, each every 64th 16 bytes, and
/// one of them hashes their 64 hashes: 64 steps after one another. One thread
/// walked the piece before, 4,096 steps; the passes took 24.8 ms of a frame's
/// 82 on the card (capture replay1, 405 of them), most of it waiting for
/// those steps.
///
/// Built with `TABLE` ([`card_batch`]), a group takes its piece, its mode
/// and the dwords of `differs` it answers for from a table, so that one
/// dispatch holds every piece of a copy that a list hashes or compares.
///
/// Built with `PAGES` ([`card_pages`]), four threads take each 4 KiB page of
/// the piece and every thread keeps its own hash: the dword gets, above the
/// flag, which of the 16 pages differ, and only those are fetched.
pub(crate) const CARD_COMPARE: &str = r#"
#ifdef TABLE
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1), SRV(t1)"
#else
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
#endif
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
#ifdef TABLE
// 16 bytes per group from entry `first` on: the piece, the first of its
// dwords in the list at byte `out_at` (each the byte of one in `differs`),
// the mode (as `compare`), how many dwords. A piece is in one entry of a
// dispatch, so its hash is read before it is stored.
ByteAddressBuffer table : register(t1);
#endif
#ifdef PAGES
// 16 bytes per thread of each piece in `hashes`; bit p: page p differs.
groupshared uint pages_changed;
#define STEP 64u
#else
groupshared uint4 parts[64];
#define STEP 1024u
#endif
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_GroupID, uint thread : SV_GroupIndex) {
#ifdef PAGES
    if (thread == 0u) pages_changed = 0u;
    GroupMemoryBarrierWithGroupSync();
#endif
#ifdef TABLE
    uint4 entry = table.Load4((first + id.x) * 16u);
    uint piece = entry.x;
    uint mode = entry.z;
#else
    uint piece = first + id.x;
    uint mode = compare;
#endif
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) + thread;
#ifdef PAGES
    uint page = thread >> 2u;
    uint at = lo + page * 4096u + (thread & 3u) * 16u;
    hi = min(lo + (page + 1u) * 4096u, hi);
#else
    uint at = lo + thread * 16u;
#endif
    // Eight loads at once: one after the other, each waits for memory.
    [loop]
    for (; at + 7u * STEP < hi; at += 8u * STEP) {
        uint4 v0 = now.Load4(at + 0u);
        uint4 v1 = now.Load4(at + STEP);
        uint4 v2 = now.Load4(at + 2u * STEP);
        uint4 v3 = now.Load4(at + 3u * STEP);
        uint4 v4 = now.Load4(at + 4u * STEP);
        uint4 v5 = now.Load4(at + 5u * STEP);
        uint4 v6 = now.Load4(at + 6u * STEP);
        uint4 v7 = now.Load4(at + 7u * STEP);
        h = (h ^ v0) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v1) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v2) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v3) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v4) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v5) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v6) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v7) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    [loop]
    for (; at < hi; at += STEP) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
#ifdef PAGES
    uint mine = (piece * 64u + thread) * 16u;
    if (mode != 0u && any(h != hashes.Load4(mine))) InterlockedOr(pages_changed, 1u << page);
    if (mode != 1u) hashes.Store4(mine, h);
#else
    parts[thread] = h;
#endif
    GroupMemoryBarrierWithGroupSync();
    if (thread != 0u) return;
#ifdef PAGES
    if (mode == 0u) return;
    // The flag, and above it the pages.
    uint flag = (pages_changed << 1u) | (pages_changed != 0u ? 1u : 0u);
#else
    h = parts[0];
    [loop]
    for (uint i = 1u; i < 64u; ++i) {
        h = (h ^ parts[i]) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (mode == 0u) {
        hashes.Store4(piece * 16u, h);
        return;
    }
    // The flag, and above it 31 bits of the hash (`torn`).
    uint folded = h.x ^ (h.y * 3u) ^ (h.z * 5u) ^ (h.w * 7u);
    uint flag = (folded << 1u) | (any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
    // With the copy-back on its own thread: the next compare of the
    // piece finds what is written from here on.
    if (mode == 2u) hashes.Store4(piece * 16u, h);
#endif
#ifdef TABLE
    for (uint k = 0u; k < entry.w; ++k) differs.Store(table.Load(out_at + (entry.y + k) * 4u), flag);
#else
    differs.Store(out_at + id.x * 4u, flag);
#endif
}
"#;

/// `VICTORY_CARD_PAGES=1`: the compare tells which 4 KiB pages of a piece
/// differ and a fetch brings only those, where it brought the whole 64 KiB
/// for a byte. Its hashes take 64 times the room (1 KiB per piece). Not with
/// `VICTORY_FETCH_CHECK`, which needs the piece's hash.
pub(crate) fn card_pages() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_PAGES").is_some_and(|v| v == "1") && !torn::on())
}

/// [`CARD_COMPARE`] as it is built: with `PAGES` where [`card_pages`].
fn card_compare_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| if card_pages() { format!("#define PAGES 1\n{CARD_COMPARE}") } else { CARD_COMPARE.to_string() })
}

/// Of a result of the compare built with `PAGES`: the pages that differ.
pub(crate) fn changed_pages(result: u32) -> u16 {
    (result >> 1) as u16
}

/// The hash [`CARD_COMPARE`] gives of a piece with its flag.
#[cfg(test)]
pub(crate) fn piece_hash(bytes: &[u8]) -> u32 {
    torn::hash(bytes)
}

/// One pass of [`CARD_COMPARE`] over a span's pieces, into `differs` from
/// dword `at` on (`mode` as the shader's `compare`): a group per piece, in
/// parts a dispatch can hold.
///
/// # Safety
/// The list is open with the compare's root signature, pipeline and buffers
/// set, and the pieces lie in the buffer.
unsafe fn card_pass(list: &ID3D12GraphicsCommandList, span: &crate::shadow::CardSpan, size: u32, at: u32, mode: u32) {
    const MOST: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    let mut done = 0;
    while done < span.pieces {
        let n = (span.pieces - done).min(MOST);
        let constants = [span.first + done, span.skew, size, n, (at + done) * 4, mode];
        // SAFETY: the caller's.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.Dispatch(n, 1, 1);
        }
        done += n;
    }
}

/// Whether our hash and compare passes over copies in the card's memory go
/// in one dispatch per copy a list takes them for (`VICTORY_CARD_BATCH=1`;
/// off by default), each piece once, and not in a dispatch per range.
///
/// Profile-watch: 160-190 hash passes and 74-83 compare passes a frame, a
/// pass of 2.7 pieces on average at about 69 us, nearly all of it the
/// dispatch itself. A piece in several ranges of one compare (spans of a
/// copy that share a 64 KiB piece) is hashed once and its flag goes to each
/// range's dword: as passes, the second could find the hash the first one
/// stored.
fn card_batch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BATCH").is_some_and(|v| v == "1"))
}

/// [`CARD_COMPARE`] built with `TABLE`, and a table per host list.
struct CardBatch {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// By the list's pointer: the buffer (PC memory, mapped), its mapping,
    /// its bytes. Written while the list is recorded, which is after its
    /// last run finished.
    tables: Mutex<std::collections::HashMap<usize, (ID3D12Resource, usize, usize)>>,
}

// SAFETY: D3D12 objects (free-threaded), and mappings written under the lock.
unsafe impl Send for CardBatch {}
unsafe impl Sync for CardBatch {}

/// One copy's part of [`card_batched`]: its buffer and hashes (GPU
/// addresses), [`CardSpan`]'s size and skew, and its pieces with their
/// mode and the byte of their dword in `differs`, if any.
///
/// [`CardSpan`]: crate::shadow::CardSpan
struct CardJob {
    key: usize,
    now: u64,
    hashes: u64,
    size: u32,
    skew: u32,
    pieces: Vec<(u32, u32, Option<u32>)>,
}

/// The batch's pipeline, made once; `None` when off or it could not be made.
fn card_batch_work(device: &ID3D12Device) -> Option<&'static CardBatch> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::{D3D_SHADER_MACRO, ID3DBlob};
    static WORK: OnceLock<Option<CardBatch>> = OnceLock::new();
    if !card_batch() {
        return None;
    }
    WORK.get_or_init(|| {
        let defines = [
            D3D_SHADER_MACRO {
                Name: windows::core::s!("TABLE"),
                Definition: windows::core::s!("1"),
            },
            D3D_SHADER_MACRO::default(),
        ];
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source, names and macros live through the calls; the
        // root signature is inside the shader.
        let made = unsafe {
            (|| -> windows::core::Result<CardBatch> {
                D3DCompile(
                    card_compare_source().as_ptr().cast(),
                    card_compare_source().len(),
                    windows::core::s!("card compare table"),
                    Some(defines.as_ptr()),
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
                let code = code.ok_or_else(windows::core::Error::empty)?;
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
                let pso: ID3D12PipelineState = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
                Ok(CardBatch {
                    root,
                    pso,
                    tables: Mutex::new(std::collections::HashMap::new()),
                })
            })()
        };
        made.inspect_err(|e| tracing::warn!(error = %e, "the card's batched compare could not be made; passes per range")).ok()
    })
    .as_ref()
}

/// Records `jobs` on an open list: a dispatch per copy (in parts a dispatch
/// holds), with `differs` at GPU address `differs`. `kind` is the
/// [`CARD_PASSES`] counter the dispatches go to (0 hashes, 2 compares).
///
/// # Safety
/// The list is open, its last run finished, and the jobs' pieces lie in
/// their buffers. It leaves the batch's root signature set.
unsafe fn card_batched(host: &Host, batch: &CardBatch, list: &ID3D12GraphicsCommandList, differs: u64, jobs: Vec<CardJob>, kind: usize) -> windows::core::Result<()> {
    const MOST: usize = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION as usize;
    let mut entries: Vec<[u32; 4]> = Vec::new();
    let mut slots: Vec<u32> = Vec::new();
    let mut parts = Vec::with_capacity(jobs.len());
    for mut job in jobs {
        job.pieces.sort_unstable_by_key(|&(piece, ..)| piece);
        let first = entries.len();
        for &(piece, mode, slot) in &job.pieces {
            let fresh = entries.len() == first;
            match entries.last_mut() {
                Some(e) if !fresh && e[0] == piece => e[2] = e[2].max(mode),
                _ => entries.push([piece, slots.len() as u32, mode, 0]),
            }
            if let Some(slot) = slot {
                slots.push(slot);
                entries.last_mut().unwrap()[3] += 1;
            }
        }
        parts.push((job, first, entries.len() - first));
    }
    if entries.is_empty() {
        return Ok(());
    }
    let tail = entries.len() * 16;
    let bytes = tail + slots.len() * 4;
    let table = {
        let mut tables = batch.tables.lock().unwrap_or_else(|e| e.into_inner());
        let key = list.as_raw() as usize;
        if tables.get(&key).is_none_or(|t| t.2 < bytes) {
            // The one it replaces was last read by this list's last run,
            // which finished.
            let size = bytes.next_power_of_two().max(1 << 16);
            let (buffer, data) = host.shared_buffer(size as u64).ok_or_else(windows::core::Error::empty)?;
            name(&buffer, || "card compare table".into());
            tables.insert(key, (buffer, data as usize, size));
        }
        let table = &tables[&key];
        // SAFETY: the mapping holds `bytes`; the list that reads it is not
        // running (the caller's).
        unsafe {
            std::ptr::copy_nonoverlapping(entries.as_ptr().cast::<u8>(), table.1 as *mut u8, tail);
            std::ptr::copy_nonoverlapping(slots.as_ptr().cast::<u8>(), (table.1 + tail) as *mut u8, slots.len() * 4);
            table.0.GetGPUVirtualAddress()
        }
    };
    let mut dispatches = 0u64;
    // SAFETY: the caller's.
    unsafe {
        list.SetComputeRootSignature(&batch.root);
        list.SetPipelineState(&batch.pso);
        list.SetComputeRootUnorderedAccessView(3, differs);
        list.SetComputeRootShaderResourceView(4, table);
        for (job, first, count) in &parts {
            if *count == 0 {
                continue;
            }
            list.SetComputeRootShaderResourceView(1, job.now);
            list.SetComputeRootUnorderedAccessView(2, job.hashes);
            let mut done = 0;
            while done < *count {
                let n = (count - done).min(MOST);
                let constants = [(first + done) as u32, job.skew, job.size, n as u32, tail as u32, 0];
                list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                list.Dispatch(n as u32, 1, 1);
                dispatches += 1;
                done += n;
            }
        }
    }
    CARD_PASSES[kind].fetch_add(dispatches, Ordering::Relaxed);
    if kind == 0 {
        CARD_PASSES[1].fetch_add(entries.len() as u64, Ordering::Relaxed);
    }
    Ok(())
}

/// Whether the pieces a flight's compare flags come down to the PC inside
/// the flight itself (`VICTORY_CARD_FETCH_INFLIGHT=1`; off by default, and
/// only with [`relay`]).
///
/// With the relay the thread that lands flights fetched the flagged pieces
/// with a list of its own on the host's queue, the one the submitting
/// thread keeps filling: the fetch ran behind every flight sent meanwhile.
/// Measured on an RTX 3070 Laptop: 140.9 ms a frame of that thread waiting
/// for fetches, for 63 pieces (3.2 MB) in about five fetches, some 28 ms a
/// fetch spent queued behind newer flights; and the submitting thread
/// waited 81.7 ms a frame in "title memory read" settles for that thread.
///
/// With the switch the compare's list keeps the flagged pieces: right
/// behind the compare, [`CARD_FETCH`] copies each piece it flagged into a
/// slot of the flight's own region of a staging buffer in the card's memory
/// ([`CardInflight`]). That region is the card's bytes as the flight left
/// them, and nothing writes it again before the flight has landed; so the
/// landing brings down exactly the slots the flight used, one copy per
/// flight on a copy queue of its own (the flights the submitting thread
/// sent meanwhile are not ahead of it there), and merges them
/// (`Shadow::card_merge_relay`). The merge's "fetched after" is the
/// flight's own send number (`CardPlan::sent_as`): see
/// [`Host::relay_card_inflight`] for why that is exact and what still goes
/// the old way.
///
/// The first version had the shader store the pieces straight into the
/// PC's memory over the bus. Fine at 600 pieces (36 MB) a frame on an RTX
/// 3070 Laptop, it cost an RTX 5070 at 1,139 pieces (70 MiB) a frame its
/// frame rate (4.72 fps without the switch, 1.74 with it; the submitting
/// thread's waits for the card 83 to 354 ms a frame), and an RTX 4070 Ti
/// at about 1,280 pieces 40 to 160 ms a frame of card time: thousands of
/// 16-byte stores from a few groups into memory the CPU caches. In the
/// card's own memory they are cheap, and the bus sees one copy per flight.
///
/// shortcut: the copy is issued at landing, not in the compare's list: the
/// slots a flight uses are known only when it has run. A copy in the list
/// would have to be sized beforehand: every slot the flight might use
/// (each declared piece), or several sizes each predicated on the count,
/// and predicated copies are not predicated where predication rides on
/// Vulkan's conditional rendering (draws and dispatches only).
fn inflight_fetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_CARD_FETCH_INFLIGHT").is_some_and(|v| v == "1"))
}

/// The bytes of the card's memory the flights' flagged pieces are staged
/// in ([`inflight_fetch`]; `VICTORY_CARD_FETCH_INFLIGHT_MB`, 64): 1,024
/// slots ([`inflight_slots`]), shared by the flights in the air. A flight
/// that finds no room lands the old way, and pieces beyond its region do.
fn inflight_fetch_bytes() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_FETCH_INFLIGHT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64).max(1) << 20
    })
}

/// The most slots one flight's region may take
/// (`VICTORY_CARD_FETCH_INFLIGHT_MAX_KB`, in KiB of pieces; no limit by
/// default but the staging buffer's): a flight reserves one slot per piece
/// it compares, up to this. Pieces it flags beyond its region come down the
/// old way.
fn inflight_flight_most() -> u32 {
    static MOST: OnceLock<u32> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_FETCH_INFLIGHT_MAX_KB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(u32::MAX, |kb| (kb / 64).clamp(1, u64::from(u32::MAX)) as u32)
    })
}

/// The slots a flight comparing `compared` pieces reserves: one per piece
/// compared, never more than `most` (`VICTORY_CARD_FETCH_INFLIGHT_MAX_KB`),
/// or a quarter of the buffer when that is not set. Without the quarter one
/// flight comparing 1,024 pieces took the whole staging buffer though it
/// flagged a handful, and every other flight in the air came down the old
/// way, behind newer flights: the very wait the fetch inside flights is for.
/// Sizing by a guess of the share flagged instead left a tenth of the
/// flagged pieces beyond their regions in the world (the share swings from
/// flight to flight). Pieces flagged beyond the region come down the old way.
pub(crate) fn inflight_want(compared: u64, slots: u32, most: u32) -> u32 {
    let cap = if most == u32::MAX { (slots / 4).max(1) } else { most };
    compared.min(u64::from(cap)).min(u64::from(slots)) as u32
}

/// Where a region of `want` slots starts in a staging buffer of `slots`
/// slots whose next region would start at `head`, with the regions
/// `in_air` (first slot, slots) still held: at `head`, or at slot 0 where
/// it would run past the end; `None` where that overlaps a region held (or
/// `want` is 0 or more than the buffer). Regions are taken and given back
/// in the same order, so the free room is the stretch from the newest
/// region's end round to the oldest one's start.
pub(crate) fn take_region(in_air: &std::collections::VecDeque<(u32, u32)>, head: u32, slots: u32, want: u32) -> Option<u32> {
    if want == 0 || want > slots {
        return None;
    }
    let at = if head.saturating_add(want) > slots { 0 } else { head };
    let free = in_air.iter().all(|&(base, len)| at + want <= base || base + len <= at);
    free.then_some(at)
}

/// A slot of [`CARD_FETCH`]: a 64 KiB piece as the compare cuts it, widened
/// to 16 bytes at both ends (one more 16-byte step where the copy does not
/// start at a multiple of 16 in its piece).
pub(crate) const INFLIGHT_SLOT: u64 = (64 << 10) + 16;

/// The slots for `bytes` asked of [`inflight_fetch_bytes`]: one per 64 KiB,
/// at least 1 and at most 2^15 (the shader's byte offsets into the slots
/// are 32-bit). The buffer is `slots` times [`INFLIGHT_SLOT`], a little over
/// `bytes`.
pub(crate) fn inflight_slots(bytes: u64) -> u32 {
    (bytes >> 16).clamp(1, 1 << 15) as u32
}

/// [`CARD_FETCH`]'s table entry for a piece the compare did not flag.
pub(crate) const INFLIGHT_NONE: u32 = 0xffff_ffff;

/// Where piece `i` of `span` starts in its [`CARD_FETCH`] slot, as an
/// offset of the copy: the compare's lower bound of the piece, widened down
/// to 16 bytes (`CARD_COMPARE`, before clamping to the declared range).
/// What `Shadow::card_piece` gives for the piece lies at or above it, and
/// within [`INFLIGHT_SLOT`] of it.
pub(crate) fn inflight_piece_from(span: &crate::shadow::CardSpan, i: u32) -> usize {
    let lo = (span.first + i) as usize * 65536;
    let lo = lo.saturating_sub(span.skew as usize);
    lo & !15
}

/// The 1 KiB steps of a slot: what stage 1 of [`CARD_FETCH`] dispatches
/// across.
pub(crate) const INFLIGHT_STEPS: u32 = INFLIGHT_SLOT.div_ceil(1024) as u32;

/// The fetch inside a flight ([`inflight_fetch`]), run in the compare's list
/// behind the compares of the declared ranges, in two stages with a UAV
/// barrier between (and none of `groupshared` and group barriers, which
/// some compilers of HLSL refuse):
///
/// - `stage` 0, one thread per piece: where `differs` flags the piece, a
///   ticket of the flight's counter (`table`, dword `base`); the piece's
///   entry (from `entries` on, at its dword in `differs`) is the ticket or
///   [`INFLIGHT_NONE`].
/// - `stage` 1, groups of 64 threads, x the 1 KiB step of the slot
///   ([`INFLIGHT_STEPS`]), y the piece (looping on by `rows`): a piece whose
///   ticket is below `count` is copied, cut and widened as the compare hashes
///   it, into slot `base + ticket` of `staging`, in the card's memory.
///
/// A ticket of `count` or more got no slot: that piece comes down the old
/// way.
pub(crate) const CARD_FETCH: &str = r#"
#define RS "RootConstants(num32BitConstants=10, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
#define SLOT 65552u
#define NONE 0xffffffffu
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint base;     // the flight's first slot
    uint count;    // the flight's slots
    uint stage;    // 0: tickets; 1: copies
    uint entries;  // where the entries start in `table`, in bytes
    uint rows;     // pass 1: the rows of groups dispatched
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer differs : register(u0);
RWByteAddressBuffer table : register(u1);
RWByteAddressBuffer staging : register(u2);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 id : SV_DispatchThreadID, uint t : SV_GroupIndex) {
    if (stage == 0u) {
        if (id.x >= pieces) return;
        uint entry = NONE;
        if ((differs.Load(out_at + id.x * 4u) & 1u) != 0u) {
            table.InterlockedAdd(base * 4u, 1u, entry);
        }
        table.Store(entries + out_at + id.x * 4u, entry);
        return;
    }
    [loop]
    for (uint i = group.y; i < pieces; i += rows) {
        uint k = table.Load(entries + out_at + i * 4u);
        if (k >= count) continue;
        uint piece = first + i;
        uint lo = piece * 65536u;
        lo = lo > skew ? lo - skew : 0u;
        uint hi = min((piece + 1u) * 65536u - skew, size);
        lo = lo & ~15u;
        hi = (hi + 15u) & ~15u;
        uint at = lo + group.x * 1024u + t * 16u;
        if (at < hi) {
            staging.Store4((base + k) * SLOT + (at - lo), now.Load4(at));
        }
    }
}
"#;

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists: a pass takes the time of one piece however few its pieces are.
/// Run 102 sent up 514 ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_REHASH_RUNS").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by
/// default). The pieces the card's compare found differing come down in
/// one fetch and were then merged one after the other on the thread that
/// settles: about 630 pieces and 40 MB a frame (run 104), each compared
/// with the copy's CPU side byte by byte and put into title memory through
/// the kernel. A piece is its own bytes of its copy and of title memory,
/// so eight threads take an eighth of them each. (Not with [`relay`], whose
/// landings do not come here: `crate::land` has threads of its own.)
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_some_and(|v| v == "1"))
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
    /// Looked at by `crate::picked`, and not declared: what is found changed
    /// there was written through a view nobody declared. Not landed.
    picked: bool,
    /// Picked by [`lazy`]: compared and stored (mode 2), and what is found
    /// tells `lazy` which pieces were written lately.
    lazy: bool,
}

impl CardPlan {
    /// [`lazy`] is told which of its pieces the compare found written.
    fn lazy_found(&self, differs: &[u32]) {
        for r in self.ranges.iter().filter(|r| r.lazy && r.compared) {
            lazy::found(&r.shadow, &r.span, |p| differs.get((r.at + p) as usize).is_some_and(|&d| torn::flagged(d)));
        }
    }

    /// `crate::picked` is told where what it looked at changed, by the
    /// compare's results.
    fn picked_found(&self, differs: &[u32]) {
        for r in self.ranges.iter().filter(|r| r.picked) {
            let changed = (0..r.span.pieces).find(|p| differs.get((r.at + p) as usize).is_some_and(|&d| torn::flagged(d)));
            if let Some(p) = changed {
                crate::picked::found(&r.shadow, r.shadow.card_piece(&r.span, p).0);
            }
        }
    }
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With [`relay`]: the number of the send to copies in the card's memory
    /// that the submission ran behind ([`QUEUE_ORDER`]).
    sent_as: u64,
    /// With [`upcheck`]: the look at what it sent up.
    looks: Option<upcheck::Looks>,
    /// With [`send`]: the memory its sends come from.
    sends: Option<send::Memory>,
    /// Whether the compare's list stages the flagged pieces itself
    /// ([`inflight_fetch`]): every compared range has its [`CARD_FETCH`].
    inflight: bool,
    /// Its region of the staging buffer (first slot, slots), held until it
    /// has landed ([`inflight_regions_landed`]).
    region: Option<(u32, u32)>,
    /// Held through completion of the DIRECT commands consuming the copy.
    upload_slot: Option<upq::Slot>,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory, and how
    /// the card could bring the bytes instead.
    Fill(usize, usize, usize, Option<card_steps::FillTo>),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only, and the title memory it reads
    /// (from, to; empty for none).
    Host(Box<dyn Fn() + Send>, (usize, usize)),
    /// As `Host`, for work that reads what the texture checks before it
    /// found: they are run first.
    Checked(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
    /// A title buffer copy that the list does on the card ([`place`]): what
    /// is done on the CPU for it.
    Placed(place::Placed),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Copy sources from [`HostList::upload`]'s pool: the buffer, its size
    /// class, whether it is a shared buffer. Given back to the pool when the
    /// list is reset.
    pooled: Vec<(Chunk, u64, bool)>,
    /// Title textures whose upload the list's prelude records if the check at
    /// submission finds it due (`crate::texture::prelude_uploads`).
    prelude_uploads: Vec<crate::texture::PreludeUpload>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Upload buffers taken from the pool of shared buffers
    /// ([`upload_pool`]), with their mapped bytes and their size: given back
    /// to it when the list is reset (counted in `uploads`' place in
    /// [`ALIVE`]).
    view_pooled: Vec<(ID3D12Resource, usize, u64)>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// The pipelines [`HostList::bind_pipeline`] kept, by address.
    bound: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Ranges declared written by shaders since the list last wrote title
    /// memory (host copy, address, bytes): declaring them again would
    /// change nothing.
    gpu_written: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Ranges declared written by shaders since the open batched clamp
    /// ([`HostList::batch_opened`]).
    batch_writes: Option<crate::indirect::batch::WriteLog>,
    /// Scratch memory [`HostList::fill_scratch`] fills since the list last
    /// wrote title memory (source, bytes): its host address.
    filled: std::collections::HashMap<(usize, usize), u64, std::hash::BuildHasherDefault<AddressHash>>,
    /// Where the list's copies at their place write (from, to), for the
    /// fills behind them ([`HostList::place_constants`]).
    placed: Vec<(usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Last GPU source copied into each aliased host texture in this list.
    texture_aliases: std::collections::HashMap<usize, usize>,
    /// Builder writes and small-buffer copy-backs requested since the last reset.
    builder_memory: std::collections::HashSet<BuilderMemoryUse>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, where they
    /// are in a read-back chunk mapped for good (0: map the buffer, from its
    /// start), the title memory they go to, the bytes of a row, the rows,
    /// the title's row pitch.
    backs: Vec<(ID3D12Resource, usize, usize, [usize; 3])>,
    /// Title textures of the kinds copied back ([`crate::texback`]) that
    /// the list's shaders write, copied when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Those copies, whose texels go to title memory once the list has run.
    texture_backs: Vec<crate::texback::Back>,
    /// Resolved timestamps, which go to title memory in the title's clock
    /// once the list has run: the read-back buffer, where they are in it
    /// when it is a chunk mapped for good (0: map it, from its start), the
    /// title memory, how many.
    stamps: Vec<(ID3D12Resource, usize, usize, usize)>,
    /// Whether the indirect-work budget is zeroed for this recording (a
    /// step on the CPU, or in the list for the card's copy).
    budget_zeroed: bool,
    /// The indirect scratch buffer (`HostList::indirect_args`, by its
    /// interface pointer) that a clamp in this recording left readable as
    /// indirect arguments; 0 if none. It stays readable until the next
    /// clamp into it ([`HostList::indirect_state`]).
    indirect_readable: usize,
    /// Ranges `[dst, dst + len)` written on the GPU by `Step::Placed` in
    /// this recording, in order, with the destination host copy.
    placed_written: Vec<(usize, usize, std::sync::Arc<crate::shadow::Shadow>)>,
    /// Current per-recording GPU buffer `(buffer, capacity, used)` for
    /// inline placed constant snapshots ([`place`]).
    placed_cbv: Option<(ID3D12Resource, u64, u64)>,
}

impl Deferred {
    fn first_builder_memory_use(&mut self, key: BuilderMemoryUse) -> bool {
        self.builder_memory.insert(key)
    }

    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }

    /// The steps recorded last write title memory: what was brought up to
    /// date before them is not any more.
    fn title_written(&mut self) {
        self.refreshed.clear();
        self.gpu_written.clear();
        self.filled.clear();
    }
    /// Clears all recorded entries while preserving heap allocations for reuse.
    fn clear(&mut self) {
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
        self.copies.clear();
        self.uploads.clear();
        self.upload_bytes = 0;
        self.kept.clear();
        self.bound.clear();
        self.refreshed.clear();
        self.gpu_written.clear();
        self.filled.clear();
        self.textures.clear();
        self.texture_aliases.clear();
        self.builder_memory.clear();
        self.writes.clear();
        self.backs.clear();
        self.back_wanted.clear();
        self.texture_backs.clear();
        self.stamps.clear();
        self.budget_zeroed = false;
        self.indirect_readable = 0;
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        give_uploads(std::mem::take(&mut self.pooled));
        ALIVE[6].fetch_sub((self.uploads.len() + self.view_pooled.len()) as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
        // A list is reset only once its last submission has run
        // (`HostList::reset`): a buffer nothing else holds is read by no one.
        // The steps go first: a fill names the buffer it fills.
        if !self.view_pooled.is_empty() {
            self.copies.clear();
        }
        for (buffer, data, size) in self.view_pooled.drain(..) {
            if references(&buffer) == 1 {
                crate::shadow::view_pool_put(size, &buffer, data as *mut u8);
            }
        }
    }
}

/// Whether a list's upload buffers in shared memory (with
/// `VICTORY_CARD_STEPS` 2 or more, one a copy: `HostList::upload`) come from
/// the pool `shadow::view_pool_take` keeps by size and go back
/// to it when the list is reset (`VICTORY_UPLOAD_POOL=1`; off by default).
/// With the views' pool on, they were the most made buffers left: 23,000 of
/// 40,000, 1.1 GB, each waited for until it is mapped (run 4 of 2026-11-27).
/// Buffers up to 16 MiB.
fn upload_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_POOL").is_some_and(|v| v == "1"))
}

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1`; off by default, and off in the timing run, which
/// waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]).
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`; off by default; it turns
/// [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says another stage,
/// and is off with it, in the timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Whether the pieces the relay fetches from the card's memory come over the
/// card's copy queue (`VICTORY_RELAY_COPY=1`) and not over the queue that
/// draws.
///
/// On the drawing queue a fetch runs behind whatever was sent meanwhile,
/// and with the relay that is most of a frame: the thread that lands flights
/// stood 48 ms of a 120 ms frame waiting for seven fetches of 94 MB in all,
/// a few milliseconds of copying, and everything that waits for a landing
/// (the title's fences, reads of memory flights may write) waited with it.
///
/// What changes: the fetch runs when its flights are done and beside later
/// submissions, so of the sends of title bytes it is only known to have run
/// behind those its own flights ran behind (`CardPlan::sent_as`). Bytes sent
/// later are left out of the merge (`Shadow::card_merge_relay`): they are
/// the title's, newer than what the flights wrote. A later flight's bytes
/// may be in what is fetched, as before.
fn relay_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RELAY_COPY").is_some_and(|v| v == "1"))
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Since the last frame-rate line, with [`inflight_fetch`]: pieces merged
/// from the slots their flights staged them in, their bytes, pieces flagged
/// beyond their flight's region (or of a flight with none), landings that
/// fetched pieces the old way, microseconds the thread that lands flights
/// waited for the staged slots to come down, and their bytes.
static INFLIGHT_STATS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// After a fetch the old way taken with the queue's count at `after`, for
/// plans whose highest send is `landed`: up to which send the notes of
/// sends to card copies may go (`crate::shadow::relay_fetched`).
///
/// Without [`inflight_fetch`] every later merge is of a later fetch, whose
/// count is `after` or more: notes up to `after` are of no use, as before.
/// With it, a flight still in the air, sent before the fetch, merges its
/// own slot with "fetched after" its own send, which may be below `after`:
/// it needs the notes of the sends between, or it takes the title's bytes
/// that went up in them for its shaders' writes and puts its older card
/// bytes over them. (Flights 5, 6, 7 in the air; 5 lands the old way with
/// the count at 7; a title byte went up in send 7 into a piece 6's shaders
/// write; 6 lands from its slot, which holds that byte from before send 7.)
/// So the mark goes only to the landed plans' highest send: every flight
/// still to land was sent after it, and every later fetch the old way
/// takes a count at least that high. (`CardInflight::brought` is another
/// matter: the card's bytes the fetch put in the CPU's side, not the
/// title's.)
pub(crate) fn fallback_fetched_mark(inflight: bool, after: u64, landed: Option<u64>) -> u64 {
    match (inflight, landed) {
        (false, _) => after,
        (true, Some(landed)) => landed.min(after),
        (true, None) => 0,
    }
}

/// With [`inflight_fetch`], after a landing merged (either way): these
/// plans' regions of the staging buffer are given back, and may be taken
/// again by flights recorded from now on. Their slots were brought down and
/// merged before this; and a later flight's list runs on the host's queue
/// behind theirs, so its stores into a region given back come after their
/// stores into it.
fn inflight_regions_landed(plans: &[&CardPlan]) {
    let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(inflight) = work.as_mut().and_then(|w| w.inflight.as_mut()) else {
        return;
    };
    for region in plans.iter().filter_map(|p| p.region) {
        if let Some(i) = inflight.in_air.iter().position(|&r| r == region) {
            inflight.in_air.remove(i);
        }
    }
    // Every flight still to land was sent after these: what fetches the old
    // way brought up to them concerns none of them.
    if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
        inflight.brought.retain(|b| b.3 > last);
    }
}

/// Waits for landings given up with the line empty ([`relay_wait`]), since
/// the start.
static RELAY_IDLE_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range, in order: all
/// of them flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: &[(usize, u32, bool)]) -> Vec<(usize, u32)> {
    let mut pieces: Vec<(usize, u32)> = flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect();
    pieces.sort_unstable();
    pieces.dedup();
    pieces
}

#[cfg(test)]
mod relay_tests {
    use super::{flagged_pieces, landable};

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(&flags);
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }
}

/// With [`card_pages`]: for each piece a range of `plan` flagged, the pages
/// any of its ranges found changed. A second range over a piece may find
/// the hashes the first stored and flag nothing; its bytes come down if the
/// first range's pages hold them.
fn plan_pages(plan: &CardPlan, differs: &[u32]) -> std::collections::HashMap<(usize, u32), u16> {
    let mut pages = std::collections::HashMap::new();
    for r in plan.ranges.iter().filter(|r| !r.picked && r.compared) {
        let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
        for p in 0..r.span.pieces {
            if let Some(&d) = differs.get((r.at + p) as usize) {
                *pages.entry((copy, r.span.first + p)).or_insert(0) |= changed_pages(d);
            }
        }
    }
    pages.retain(|_, m| *m != 0);
    pages
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    let base = *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    });
    settle_room::list_limit(base, settle_room::on())
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// What was taken off it to be done on the card.
    own: Option<card_steps::Taken>,
    /// Title-memory ranges written when its shader textures are copied back.
    texture_back_ranges: Vec<(usize, usize)>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// The list of ours that ran ahead of them ([`card_steps`]).
    own: Option<card_steps::Taken>,
    texture_back_ranges: Vec<(usize, usize)>,
}

/// One thing in the line of what waits for the card, in order.
#[allow(clippy::large_enum_variant)] // the line holds one of them at a time, and only for a moment
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence (its address) signalled behind the flights before it.
    Signal(usize, u64),
}

#[cfg_attr(test, derive(Default))]
struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders and texture readbacks may write
    /// (address, end), in order and joined.
    written: Vec<(usize, usize)>,
    /// `written`, with big card copies' parts only where written lately
    /// ([`lazy::narrow`]; empty when that is off): what steps wait for.
    narrow: Vec<(usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
    /// With [`card_steps`]: what the flights being landed write.
    landing: Vec<card_steps::Writer>,
    /// With [`relay`]: things of the line landed so far, and how many the
    /// landing under way took ([`settle_unlocked`]).
    landed_items: u64,
    taking: u64,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    narrow: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
    landing: Vec::new(),
    landed_items: 0,
    taking: 0,
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

impl Line {
    /// What steps of a submission wait for (or are done on the card for):
    /// `written`, or with [`lazy::narrow`] its narrowed form.
    fn blocking(&self) -> &Vec<(usize, usize)> {
        if lazy::narrow() { &self.narrow } else { &self.written }
    }

    /// A flight's declared writes and texture read-backs join `written` (and
    /// `narrow`).
    fn joined(&mut self, back: &[Declared], textures: &[(usize, usize)]) {
        join_ranges(&mut self.written, back.iter().flat_map(|(_, r)| r.iter().copied()).chain(textures.iter().copied()));
        if lazy::narrow() {
            let mut more: Vec<(usize, usize)> = textures.to_vec();
            for (shadow, ranges) in back {
                if lazy::on(shadow) {
                    for &(from, to) in ranges {
                        lazy::hot_within(shadow, from, to, &mut more);
                    }
                } else {
                    more.extend(ranges.iter().copied());
                }
            }
            join_ranges(&mut self.narrow, more);
        }
    }
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture). Then: joined submissions
/// cut before a part that waits for flights ([`cut_waiting`]).
static LATE_STATS: [std::sync::atomic::AtomicU64; 19] = [const { std::sync::atomic::AtomicU64::new(0) }; 19];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// The kinds of read that make a submission wait for the card ([`READ_WAITED`]): the places
/// a `WHY_READ` settle can come from, in the order [`late_blocked`] tries them.
const READ_BUFFER_COPY: usize = 0;
const READ_COPY_PLACED: usize = 1;
const READ_FILL: usize = 2;
const READ_TEXTURE: usize = 3;
const READ_KINDS: [&str; 4] = [
    "a buffer copy's memory",
    "a buffer copy at its place",
    "a fill of upload memory",
    "a texture's memory",
];

/// Since the last frame-rate line: microseconds of `title memory read` waits, by what was read.
///
/// `LATE_STATS[13..]` counts how often each kind blocks a submission, about once a frame, while the wait they cause
/// is 226-249 ms a frame, so the count says nothing about which kind owns the time. This does.
static READ_WAITED: [std::sync::atomic::AtomicU64; READ_KINDS.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; READ_KINDS.len()];

thread_local! {
    /// Which kind of read the next `WHY_READ` settle is for; set by [`late_blocked`].
    static READ_KIND: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList]) -> Option<usize> {
    let (written, copies) = {
        let line = line();
        if line.blocking().is_empty() {
            return None;
        }
        (line.blocking().clone(), card_steps::copies(&line))
    };
    blocked_by(&written, runs, true).or_else(|| card_steps::copied_read(&copies, runs).then_some(WHY_READ))
}

/// Whether a small fill of upload memory, or copy of a title buffer, goes on without waiting for
/// flights when the title itself wrote the memory since they were given it
/// (`VICTORY_FILL_TITLE=1`; off by default).
///
/// The title's 174 MB heap is declared written as a whole (a root view has
/// no size), and the title keeps a ring of constants in it: one fill of 480
/// bytes a frame, at a new place each time, waited for every flight's
/// copy-back, some 30 ms of a 130 ms frame (run blame1: 38 of the 40 waits
/// named). The title writes those bytes just before the draw that reads
/// them, which takes the block's guard off ([`crate::watch::written_since_taken`]).
///
/// Or no shader's write went into that page in the last 600 frames
/// ([`crate::watch::landed_lately`]): the first alone left two thirds of
/// the wait, for constants at the end of a 64 KiB page that the title wrote
/// before the page was taken.
///
/// What can go wrong: a constant a flight's shader really writes, in a
/// block that the title (or an earlier flight's copy-back) also wrote, or
/// for the first time, is read as it was before that flight.
fn fill_is_titles(src: usize, len: usize) -> bool {
    const SMALL: usize = 64 << 10;
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_TITLE").is_some_and(|v| v == "1"));
    on && len <= SMALL && (crate::watch::written_since_taken(src, len) || !crate::watch::landed_lately(src, len, 600))
}

/// Diagnostics: the waits for flights, by what waited, why it was not done
/// on the card ([`card_steps`]), its bytes, and the declared write it lies
/// in (start, bytes).
#[cfg(feature = "diagnostics")]
type Blamed = (&'static str, &'static str, usize, usize, usize);
/// How often and how long, and the address the last one read.
#[cfg(feature = "diagnostics")]
static BLAME: Mutex<Option<std::collections::HashMap<Blamed, (u64, u64, usize)>>> = Mutex::new(None);
/// What the wait that comes next is for, and why not on the card.
#[cfg(feature = "diagnostics")]
static BLAMED: Mutex<(Option<(Blamed, usize)>, &'static str)> = Mutex::new((None, ""));

/// Diagnostics: shader writes declared over a megabyte or more: the range
/// (from, to) and what declares it ([`wide_write`]): how often, and its
/// shaders.
#[cfg(feature = "diagnostics")]
static WIDE_WRITES: Mutex<Option<std::collections::HashMap<(usize, usize, usize), (u64, String)>>> = Mutex::new(None);

/// Diagnostics: a draw or dispatch declares that it writes `len` bytes at
/// `address`, through a root view of `pipeline`, a view in a table (0) or a
/// raw address (1).
#[cfg(feature = "diagnostics")]
pub(crate) fn wide_write(address: u64, len: u64, pipeline: usize) {
    if len < 1 << 20 {
        return;
    }
    let mut all = WIDE_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let e = all
        .get_or_insert_with(Default::default)
        .entry((address as usize, (address + len) as usize, pipeline))
        .or_insert_with(|| {
            let shaders = match pipeline {
                0 => "a view in a table".into(),
                1 => "a raw address".into(),
                p => crate::pipeline::host_pipeline(p as crate::object::This).map_or_else(String::new, |p| {
                    p.stages.iter().map(|s| s.shader.hash.to_string()).collect::<Vec<_>>().join("+")
                }),
            };
            (0, shaders)
        });
    e.0 += 1;
}

#[cfg(feature = "diagnostics")]
fn blame(kind: &'static str, written: &[(usize, usize)], at: usize, len: usize) {
    let i = written.partition_point(|r| r.1 <= at);
    let (from, to) = written.get(i).copied().unwrap_or((0, 0));
    let mut blamed = BLAMED.lock().unwrap_or_else(|e| e.into_inner());
    let why = blamed.1;
    blamed.0 = Some(((kind, why, len, from, to - from), at));
}

/// Why [`card_steps`] left the submission to the wait that comes next.
#[cfg(feature = "diagnostics")]
fn refused(why: &'static str) {
    *BLAMED.lock().unwrap_or_else(|e| e.into_inner()) = (None, why);
}

/// The wait that comes next is for these ranges ([`stale_under_flights`]).
#[cfg(feature = "diagnostics")]
fn blame_stale(stale: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)]) {
    if let Some((shadow, from, _)) = stale.first() {
        let (start, len) = shadow.span();
        let key = ("title memory changed under shader writes", crate::shadow::stale_why(), 0, start as usize, len as usize);
        BLAMED.lock().unwrap_or_else(|e| e.into_inner()).0 = Some((key, *from));
    }
}

/// The wait for what [`blame`] last named, or a wait of another reason than
/// title memory read, took `us`.
#[cfg(feature = "diagnostics")]
fn blamed(why: usize, us: u64) {
    let (named, refused) = std::mem::replace(&mut *BLAMED.lock().unwrap_or_else(|e| e.into_inner()), (None, ""));
    let (key, at) = match (why, named) {
        (WHY_READ | WHY_STALE, Some(named)) => named,
        (WHY_READ, None) => (("a host copy of what a copy done on the card wrote", refused, 0, 0, 0), 0),
        _ => {
            let kind = match why {
                WHY_SYNC => "a synchronous submission, a present or the sweep",
                WHY_VALUE => "a memory value",
                WHY_STALE => "title memory changed under shader writes",
                WHY_RUN => "a step of ours",
                WHY_MOST => "lists in the air",
                WHY_FENCE => "a blocking fence wait",
                _ => "another reason",
            };
            ((kind, "", 0, 0, 0), 0)
        }
    };
    let mut all = BLAME.lock().unwrap_or_else(|e| e.into_inner());
    let e = all.get_or_insert_with(Default::default).entry(key).or_default();
    *e = (e.0 + 1, e.1 + us, at);
}

/// For the frame-rate line: the five that waited longest.
#[cfg(feature = "diagnostics")]
fn log_blame(frames: f64) {
    let all = BLAME.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let wide = WIDE_WRITES.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let mut all: Vec<_> = all.into_iter().collect();
    all.sort_unstable_by_key(|(_, (_, us, _))| std::cmp::Reverse(*us));
    for ((kind, why, len, from, size), (n, us, at)) in all.into_iter().take(5) {
        // The passes that declare it written, the ones that do so most.
        let mut passes: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
        for ((a, b, _), (n, shaders)) in &wide {
            if *a < at + len && at < *b {
                *passes.entry(shaders).or_default() += n;
            }
        }
        let mut passes: Vec<(&str, u64)> = passes.into_iter().collect();
        passes.sort_unstable_by_key(|p| std::cmp::Reverse(p.1));
        let passes: Vec<String> =
            passes.into_iter().take(5).map(|(shaders, n)| format!("{shaders} {:.1}x", n as f64 / frames)).collect();
        tracing::info!(
            "waited for flights, per frame: {:.1} ms in {:.2} waits for {kind} +{len:#x} (last at {at:#x}), in shader writes declared \
             over {from:#x}+{size:#x} (declared per frame over a megabyte or more there, by shader: {}){}{why}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            passes.join(", "),
            match (why.is_empty(), kind) {
                (true, _) => "",
                (_, "title memory changed under shader writes") => "; not put in under them: ",
                _ => "; not done on the card: ",
            },
        );
    }
}

/// Whether flights that write `written` stop a read of `len` bytes at `src`.
fn read_blocked(written: &[(usize, usize)], src: usize, len: usize) -> bool {
    touches(written, src, src + len) && !fill_is_titles(src, len)
}

/// Whether they stop a title buffer copy, and if so whether by its source.
fn copy_blocked(written: &[(usize, usize)], dst: usize, src: usize, len: usize) -> Option<bool> {
    let read = read_blocked(written, src, len);
    (read || touches(written, dst, dst + len)).then_some(read)
}

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    // Destinations of copies at their place, if a step reads them.
    let mut placed: Vec<(usize, usize)> = Vec::new();
    let forwards = place::forwards(written, runs);
    for (n, list) in runs.iter().enumerate() {
        for (i, step) in list.deferred().copies.iter().enumerate() {
            match step {
                &Step::Copy(dst, src, len) => {
                    if let Some(_read) = copy_blocked(written, dst, src, len) {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            if _read {
                                blame("a buffer copy's source", written, src, len);
                            } else {
                                blame("a buffer copy's destination", written, dst, len);
                            }
                        }
                        READ_KIND.with(|k| k.set(READ_BUFFER_COPY));
                        return Some(WHY_READ);
                    }
                }
                &Step::Fill(_, src, len, _) => {
                    if read_blocked(written, src, len) && !place::forwarded(&forwards, n, i) {
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a fill of upload memory", written, src, len);
                        }
                        READ_KIND.with(|k| k.set(READ_FILL));
                        return Some(WHY_READ);
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    if touches(written, at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a texture", written, at, len);
                        }
                        READ_KIND.with(|k| k.set(READ_TEXTURE));
                        return Some(WHY_READ);
                    }
                }
                Step::Run(_) => return Some(WHY_RUN),
                Step::Placed(p) => match p.waits(written) {
                    place::Waits::No => {}
                    place::Waits::ForReaders => placed.push((p.dst, p.dst + p.len)),
                    place::Waits::Yes => {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a buffer copy at its place", written, p.src, p.len);
                        }
                        READ_KIND.with(|k| k.set(READ_COPY_PLACED));
                        return Some(WHY_READ);
                    }
                },
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(..) | Step::Checked(_) => {}
            }
        }
    }
    (!placed.is_empty() && place::read_by(runs, placed, &forwards)).then_some(WHY_READ)
}

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    use std::sync::Arc;
    // Per copy, the span of what is refreshed (a flush brings that much
    // up to date at most).
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    // Where each copy is in it, if shaders may be writing it (none begins to
    // while this runs): a frame has 20,000 refreshes of some 4,500 copies.
    let mut places: ByCopy<Option<usize>> = ByCopy::default();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step {
                let place = places.entry(Arc::as_ptr(shadow) as usize).or_insert_with(|| {
                    shadow.writes_pending().then(|| {
                        spans.push((shadow.clone(), *address, *address + *len));
                        spans.len() - 1
                    })
                });
                if let Some(i) = *place {
                    let s = &mut spans[i];
                    (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len));
                }
            }
        }
    }
    stale_of(spans)
}

/// Of `spans` (a copy shaders may be writing, and the span of what a
/// submission refreshes of it), those [`stale_under_flights`] answers.
fn stale_of(
    mut spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there. With [`card_steps`] the pages of the
        // title's ring of constants come here too, which were waited for
        // before: hundreds of short runs each, at about 1.5 us a run.
        let mut runs_left = settle_room::merge_budget(card_steps::stage() > 0, settle_room::on());
        let busy: Vec<Vec<(usize, usize)>> = spans.iter().map(|(shadow, from, to)| shadow.busy(*from, *to - *from)).collect();
        let asked = crate::shadow::Writes::ask(
            spans.iter().zip(&busy).filter(|((shadow, ..), _)| !shadow.unwatched()).flat_map(|(_, b)| b.iter().copied()).collect(),
        );
        let mut busy = busy.into_iter();
        spans.retain(|(shadow, ..)| shadow.merge_under_writes(busy.next().unwrap_or_default(), &asked, &mut runs_left));
        return spans;
    }
    spans.retain(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from));
    spans
}

/// How many parts of submissions, and lists, the line's thread gives to the
/// host as one (`VICTORY_JOIN_PARTS=<n>`; off by default, 1; `VICTORY_JOIN_LISTS`,
/// 48).
///
/// The idea is fxceless/victory-fork's, commit 39731795 (GPL-3.0-or-later, as
/// this): the title's lists wait for memory values, which cuts a frame into
/// some 29 parts, each prepared on its own: the write watch asked, the heaps'
/// records, the title's bytes under the flights' writes, a list that sends
/// what changed and one that compares. (Run submit-1, the dorm, 12 of 28 parts
/// joined, eight at most: the thread works 51.4 ms a frame instead of 54.8, the
/// frame is 0.6 to 3 ms shorter.) What differs here: the parts are taken
/// off one queue's entries of the line (`crate::queue`), still go to the queue
/// in calls of their own, and everything they refresh is brought up to date in
/// front of the first ([`refresh_front`], without which a later part's title
/// bytes under an earlier part's declared writes would go missing).
pub(crate) fn joins() -> (usize, usize) {
    if ahead_of_writes() { joins_asked() } else { (1, 1) }
}

/// Whether the parts [`joins`] gives the host go to the queue in one call,
/// the first list of each part after the first behind a barrier that waits
/// for everything before it ([`join_barriers`]) (`VICTORY_JOIN_ONE_CALL=1`;
/// off by default, a call per part).
///
/// From Skyfrye's AnyXboxX#337 (`VICTORY_MERGE_SUBMISSIONS`), which holds the
/// title's submissions back and sends them as one host submission with these
/// barriers between them; here only the barriers and the one call, over the
/// parts the line's thread joins already.
fn join_one_call() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_JOIN_ONE_CALL"))
}

fn joins_asked() -> (usize, usize) {
    static MOST: OnceLock<(usize, usize)> = OnceLock::new();
    let of = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok());
    let (parts, lists) = *MOST.get_or_init(|| (of("VICTORY_JOIN_PARTS").unwrap_or(1).max(1), of("VICTORY_JOIN_LISTS").unwrap_or(48).max(1)));
    #[cfg(any(feature = "diagnostics", feature = "ab"))]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_PARTS") {
        return (if on { 8 } else { 1 }, lists);
    }
    (parts, lists)
}

/// With flights in the air: before which part of joined submissions
/// ([`joins`]) they are cut, so that the parts ahead of the first one that
/// waits for flights go to the card without waiting
/// (`VICTORY_CUT_WAITING=1`; off by default).
///
/// The parts are joined while they wait in the line, without looking at
/// the flights, and are prepared as one: a part that reads what a flight
/// writes held back those in front of it too. Leafy's run past the intro
/// (leafy-0314) waited three times a frame for fills of the title's ring
/// of constants, 113 to 222 ms, and the card stood 30 to 68 ms a frame
/// behind the thread that lands and prepares, with nothing of ours queued.
/// Cut, the parts in front run on the card while the rest waits.
///
/// Only for the relay's exact wait ([`relay_exact`]), which waits for the
/// flights that write what is read and not for the parts just sent; a
/// settle for everything would wait for those as well.
fn cut_waiting(runs: &[&HostList], parts: &[usize]) -> Option<usize> {
    if parts.is_empty() || !cut_waiting_on() || !relay_exact() {
        return None;
    }
    let (written, copies) = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        (line.written.clone(), card_steps::copies(&line))
    };
    let why = |runs: &[&HostList]| blocked_by(&written, runs, false).or_else(|| card_steps::copied_read(&copies, runs).then_some(WHY_READ));
    let mut cut = 0;
    for &end in parts.iter().chain(std::iter::once(&runs.len())) {
        match why(&runs[cut..end]) {
            None => cut = end,
            Some(WHY_READ) => break,
            Some(_) => return None,
        }
    }
    // (Copies at their place are looked at across parts.)
    if cut == 0 || cut == runs.len() || why(&runs[..cut]).is_some() {
        return None;
    }
    LATE_STATS[18].fetch_add(1, Ordering::Relaxed);
    Some(cut)
}

fn cut_waiting_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CUT_WAITING"))
}

/// Whether the part `next` can be prepared together with `ahead`, which goes
/// before it and has not been sent: nothing done on the CPU for it reads what
/// those give their shaders, write back or write when they are done, and it
/// changes no title memory that they read ([`Step::Copy`], [`Step::Run`]).
pub(crate) fn joinable<'a>(ahead: impl Iterator<Item = &'a HostList>, next: &[&HostList]) -> bool {
    if next.iter().any(|l| l.deferred().copies.iter().any(|s| matches!(s, Step::Copy(..) | Step::Run(_)))) {
        return false;
    }
    let ahead: Vec<&HostList> = ahead.collect();
    let mut written: Vec<(usize, usize)> = declared_writes(&ahead).into_iter().flat_map(|(_, ranges)| ranges).collect();
    for list in &ahead {
        let deferred = list.deferred();
        written.extend(deferred.writes.iter().map(|&(at, _, bytes)| (at, at + bytes)));
        written.extend(deferred.backs.iter().map(|&(_, _, at, [row, rows, pitch])| (at, at + rows.saturating_sub(1) * pitch + row)));
    }
    let mut joined = Vec::new();
    join_ranges(&mut joined, written);
    blocked_by(&joined, next, false).is_none()
}

/// Whether a bottom-of-pipe write of `list` meets `wait`.
pub(crate) fn will_meet(list: &HostList, wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    // The last write there says.
    let deferred = list.deferred();
    let last = deferred.writes.iter().rev().find(|(at, _, n)| *at < address + bytes && address < *at + *n);
    matches!(last, Some((at, crate::queue::Written::Value(v), n)) if *at == address && *n >= bytes && wait.holds_for(*v))
}

/// Whether a submission sent ahead is cut before its first list that waits
/// for flights, and the lists in front of it go to the card at once
/// (`VICTORY_SEND_FREE_FIRST=1`; off by default).
///
/// The card stood 37 ms a frame with nothing of ours queued while the
/// submitting thread worked, in Leafy's play run of 2026-11-27 (game
/// 90-180 s, 222 ms frames): a submission that reads what the flights write
/// waits for them to land, the card runs dry, and its lists are prepared
/// after that with the card standing. Its first lists may read nothing the
/// flights write: those now run behind the flights while the thread waits
/// and prepares the rest.
fn send_free_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_SEND_FREE_FIRST"))
}

/// Since the last frame-rate line ([`send_free_first`]): submissions cut,
/// the lists sent in front, and all the lists of those submissions.
static FREE_FIRST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// With [`send_free_first`]: before which of `runs` the submission is cut,
/// the first that flights in the air stop, if lists come before it that
/// they stop in no way (taken together, as a submission of their own).
fn free_cut(runs: &[&HostList]) -> Option<usize> {
    if !send_free_first() || runs.len() < 2 {
        return None;
    }
    let (written, copies) = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        (line.written.clone(), card_steps::copies(&line))
    };
    let cut = (0..runs.len()).find(|&n| blocked_by(&written, &runs[n..=n], false).is_some())?;
    let ahead = &runs[..cut];
    if cut == 0 || blocked_by(&written, ahead, false).is_some() || card_steps::copied_read(&copies, ahead) {
        return None;
    }
    FREE_FIRST[0].fetch_add(1, Ordering::Relaxed);
    FREE_FIRST[1].fetch_add(cut as u64, Ordering::Relaxed);
    FREE_FIRST[2].fetch_add(runs.len() as u64, Ordering::Relaxed);
    Some(cut)
}

/// Whether a submission's snapshots ([`crate::shadow::take_snapshots`]) are
/// taken on the kept threads while the submitting thread brings the copies
/// in the card's memory up to date, and waited for before anything writes
/// the copies again (`VICTORY_SNAP_BESIDE=1`; off by default). In Leafy's
/// play run of 2026-11-27 the two came one after the other on this thread
/// while the card stood idle: 5.5 and 11.5 ms a frame. They touch other
/// copies (no snapshots of copies in the card's memory), and only read the
/// copies' CPU sides.
fn snap_beside() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_SNAP_BESIDE"))
}

/// Lists and the order they run in.
type Run<T> = (Vec<T>, Vec<usize>);

/// `order`'s first `cut` places and the rest, each with the lists they name;
/// as it was if a list is named on both sides.
fn split<T>(lists: Vec<T>, order: Vec<usize>, cut: usize) -> Result<[Run<T>; 2], Run<T>> {
    let (first, second) = order.split_at(cut);
    if first.is_empty() || second.is_empty() || first.iter().any(|i| second.contains(i)) {
        return Err((lists, order));
    }
    let behind: Vec<bool> = (0..lists.len()).map(|i| second.contains(&i)).collect();
    let place = |i: usize| behind[..i].iter().filter(|&&b| b == behind[i]).count();
    let orders = [first, second].map(|half| half.iter().map(|&i| place(i)).collect::<Vec<_>>());
    let mut halves: [Vec<T>; 2] = [Vec::new(), Vec::new()];
    for (list, behind) in lists.into_iter().zip(&behind) {
        halves[usize::from(*behind)].push(list);
    }
    let [a, b] = halves;
    let [x, y] = orders;
    Ok([(a, x), (b, y)])
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: usize, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    if let Some(met) = late_met_known(wait) {
        return met;
    }
    settle_now(WHY_VALUE);
    wait.met()
}

/// [`late_met`] without a wait for the card; `None`: it has to be waited for.
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> Option<bool> {
    let (address, bytes) = wait.place();
    {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() {
            return Some(false);
        }
        let mut unsure = touches(&line.written, address, address + bytes);
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((at, what, n)),
            _ => None,
        }));
        for (at, what, n) in writes {
            if *at >= address + bytes || address >= *at + *n {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if *at == address && *n >= bytes => {
                    if wait.holds_for(*v) {
                        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
                        return Some(true);
                    }
                }
                _ => unsure = true,
            }
        }
        if !unsure {
            return Some(false);
        }
    }
    None
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    if relay() && settle_unlocked() {
        let mark = {
            let line = line();
            line.landed_items + line.taking + line.line.len() as u64
        };
        relay_wait(why, |line| line.landed_items >= mark);
        return;
    }
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "diagnostics")]
    let held = std::time::Instant::now();
    host.settle(&mut submitted, why);
    #[cfg(feature = "diagnostics")]
    crate::copystat::lap(crate::copystat::C::SubmitLockSettleNs, held);
}

/// `VICTORY_SETTLE_UNLOCKED=1` (off by default), with [`relay`]: a
/// [`settle_now`] waits until what was in the line when it was called has
/// landed, without the submission lock, so the submitting thread goes on
/// meanwhile. Under the lock it waited for the line to empty, and for the
/// submissions of other threads that held the lock first; the title cannot
/// count on those being done anyway, as nothing orders them before the
/// present or fence wait that settles.
fn settle_unlocked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SETTLE_UNLOCKED").is_some_and(|v| v == "1"))
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Whether a draw that finds no free dynamic descriptor slot first waits for
/// the next landing without the submission lock (`VICTORY_DYNAMIC_WAIT=1`;
/// off by default: the settle rules of `draw::dynamic_slots` alone, as
/// before).
///
/// The slots come back flight by flight, when the thread that lands flights
/// is done with one, and that thread never takes the submission lock. A
/// settle waits for all of them and keeps the lock meanwhile, so the
/// submitting thread stands at the lock with submissions ready and the card
/// runs dry behind it. With this the thread that translates the draw waits
/// until a landing gave slots back, and takes them; other submissions go on
/// meanwhile. With nothing in the air and nothing being landed no slot can
/// come back that way, and the settle rules decide as before.
fn dynamic_wait() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DYNAMIC_WAIT").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: asks that found no free dynamic
/// descriptor slot and waited for landings ([`dynamic_wait`]); the
/// microseconds they waited; those that had their slots after a landing;
/// those left to the settle rules all the same.
static DYNAMIC_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `take` found no free dynamic descriptor slots (it gives `None`): with
/// [`dynamic_wait`] and [`relay`], waits for flights to land, which gives
/// their lists' slots back, and asks again, without the submission lock and
/// only until `take` has its slots or nothing is left in the air. `None`
/// when it is off or nothing came back.
pub(crate) fn wait_for_slots(mut take: impl FnMut() -> Option<u32>) -> Option<u32> {
    if !(dynamic_wait() && relay()) {
        return None;
    }
    let began = std::time::Instant::now();
    DYNAMIC_WAITS[0].fetch_add(1, Ordering::Relaxed);
    let mut found;
    {
        let _waiting = crate::timeline::span("waiting for landings: dynamic descriptor slots");
        loop {
            // (Looked at before the asking: what lands after this look is
            // waited for below.)
            let nothing = {
                let line = line();
                line.line.is_empty() && !line.settling
            };
            found = take();
            if found.is_some() || nothing {
                break;
            }
            // A landing that ended between the asking and here is not
            // told again: looked at once a millisecond all the same.
            let line = line();
            drop(LINE_LANDED.wait_timeout(line, std::time::Duration::from_millis(1)).unwrap_or_else(|e| e.into_inner()));
        }
    }
    DYNAMIC_WAITS[if found.is_some() { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
    DYNAMIC_WAITS[1].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    found
}

/// For the frame-rate line: [`DYNAMIC_WAITS`] per frame, and the fewest
/// dynamic slots a view heap had free since the last line.
fn dynamic_wait_line(frames: f64) {
    let w: Vec<f64> = DYNAMIC_WAITS.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let low = crate::descriptor::host::DYNAMIC_LOW[0].swap(u32::MAX, Ordering::Relaxed);
    if low == u32::MAX && w[0] <= 0.0 {
        return;
    }
    tracing::info!(
        "dynamic descriptor slots: the fewest free since the last line {} of {}; per frame {:.2} asks found none and waited \
         {:.1} ms for landings, {:.2} had their slots after a landing, {:.2} were left to the settle rules \
         (VICTORY_DYNAMIC_WAIT=1 waits for the next landing without the submission lock: now {})",
        if low == u32::MAX { 0 } else { low },
        crate::descriptor::host::DYNAMIC_LOW[1].load(Ordering::Relaxed),
        w[0],
        w[1] / 1000.0,
        w[2],
        w[3],
        if dynamic_wait() && relay() { "on" } else { "off" }
    );
}
/// [`settle_now`] when the dynamic descriptor slots ran out: the flights'
/// lists give theirs back as they land. Counted with the settles for too
/// much in the air, which it is.
pub(crate) fn settle_for_tables() {
    #[cfg(feature = "diagnostics")]
    let began = std::time::Instant::now();
    settle_now(WHY_MOST);
    #[cfg(feature = "diagnostics")]
    crate::copystat::lap(crate::copystat::C::SettleTablesNs, began);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let landed: Vec<_> = {
        let mut line = line();
        let n = line.landed.len().saturating_sub(bury_lag() as usize);
        line.landed.drain(..n).collect()
    };
    for (lists, order) in landed {
        crate::list::recycle(lists, &order);
    }
}

#[cfg(test)]
#[test]
fn host_fence_completion_rejects_removed_device() {
    assert!(!fence_reached(u64::MAX, 1));
    assert!(!fence_reached(6, 7));
    assert!(fence_reached(7, 7));
    assert!(fence_reached(8, 7));
}

/// Whether a fence's completed value shows `target` run. `u64::MAX` is not
/// a value the card reaches: it is what `GetCompletedValue` answers when the
/// device is gone.
fn fence_reached(completed: u64, target: u64) -> bool {
    completed != u64::MAX && completed >= target
}

/// Waits for `target` on `fence` and reports what went wrong instead of
/// carrying on: a failed wait used to fall through to the landing step with
/// nothing run, settling flights off an unsent fence value, and a lost
/// device (a completed value of `u64::MAX`) made every wait pass at once.
///
/// # Safety
///
/// `event` is the calling thread's own, never closed while it waits.
unsafe fn wait_host_fence(fence: &ID3D12Fence, target: u64, event: HANDLE, context: &str) {
    // SAFETY: the caller's fence and event, alive for the whole wait.
    if let Err(e) = unsafe { fence.SetEventOnCompletion(target, event) } {
        fail(format_args!(
            "{context}: SetEventOnCompletion({target}) failed: {e}"
        ));
    }
    let status = unsafe { WaitForSingleObject(event, INFINITE) };
    if status != WAIT_OBJECT_0 {
        fail(format_args!("{context}: fence wait failed: {status:?}"));
    }
    let completed = unsafe { fence.GetCompletedValue() };
    if !fence_reached(completed, target) {
        fail(format_args!(
            "{context}: fence {completed} did not reach {target}"
        ));
    }
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    crate::cores::keep_on_fast_cores("victory-late-wait");
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            let mut helping = crate::timeline::span("helper: landed lists reset and given back");
            land();
            helping.next("helper: looking at what queues hold");
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        #[cfg(feature = "diagnostics")]
        let _card = crate::timeline::card_wait(Some(target));
        // SAFETY: our fence and event.
        unsafe { wait_host_fence(&host.fence, target, event, "late helper") };
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            let held = std::time::Instant::now();
            host.settle(&mut submitted, WHY_CARD_DONE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::lap(crate::copystat::C::SubmitLockDoneNs, held);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY=1)"
            );
        }
    });
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    let waiting = std::time::Instant::now();
    let _waiting = crate::timeline::span(match why {
        WHY_SYNC => "waiting for landings: everything (a present, the sweep)",
        WHY_VALUE => "waiting for landings: a memory value",
        WHY_STALE => "waiting for landings: title memory changed under shader writes",
        WHY_READ => "waiting for landings: title memory read",
        WHY_MOST => "waiting for landings: lists in the air",
        WHY_FENCE => "waiting for landings: a blocking fence wait",
        _ => "waiting for landings: another reason",
    });
    while !done(&line) {
        // Nothing left to land: no landing would wake this again (a range
        // still pending that no flight in the line holds). The caller
        // falls back to its own settle.
        if line.line.is_empty() && !line.settling {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            RELAY_IDLE_WAITS.fetch_add(1, Ordering::Relaxed);
            if !SAID.swap(true, Ordering::Relaxed) {
                tracing::warn!(why, "a wait for landings found the line empty and its wait not met; given up (counted in idle waits)");
            }
            break;
        }
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    if why == WHY_READ {
        let kind = READ_KIND.with(|k| k.replace(usize::MAX));
        if kind < READ_KINDS.len() {
            READ_WAITED[kind].fetch_add(us, Ordering::Relaxed);
        }
    } else {
        READ_KIND.with(|k| k.set(usize::MAX));
    }
    SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    blamed(why, us);
}

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    /// With [`relay_copy`]: the card's copy queue, which `list` is for.
    queue: Option<ID3D12CommandQueue>,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
    /// Pieces caught while a later flight was writing them.
    suspects: torn::Suspects,
    /// What earlier fetches brought of flights not landed yet.
    had: crate::land::Had,
    /// With [`inflight_fetch`]: what the staged slots come down with, made
    /// with the first of them.
    down: Option<InflightDown>,
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
fn relay_worker() {
    crate::cores::keep_on_fast_cores("victory-relay");
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.taking = n as u64;
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                if card_steps::stage() > 0 {
                    line.landing = card_steps::writers(&items);
                }
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                #[cfg(feature = "diagnostics")]
                let _card = crate::timeline::card_wait(Some(target));
                // SAFETY: the host's fence and this thread's event.
                unsafe { wait_host_fence(&host.fence, target, event, "relay") };
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards). In two steps:
/// what is to be looked through and what of it differs ([`RelayBack::look`],
/// which needs nothing that the fetch from the card's memory brings), and
/// the merge.
struct RelayBack {
    work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    found: Option<Vec<Vec<(usize, usize)>>>,
}

fn relay_hand_back(back: &[Declared]) -> RelayBack {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    RelayBack { work, found: None }
}

impl RelayBack {
    /// Looks the ranges through, once; its time counts as `stat`.
    fn look(&mut self, stat: usize) {
        if self.work.is_empty() || self.found.is_some() {
            return;
        }
        let scanning = std::time::Instant::now();
        // Looked through without a lock: a piece that changes meanwhile (a
        // later flight writing, the title's bytes going in) is found now or
        // by that flight's landing, and the merge looks again under the
        // lock.
        self.found = Some(crate::shadow::find_all(&self.work));
        crate::land::spent(stat, scanning);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    }

    /// What differs goes into title memory.
    fn merge(mut self) {
        use crate::land;
        self.look(land::stat::SCAN);
        let Some(found) = self.found else {
            return;
        };
        let merging = std::time::Instant::now();
        let mut written: Vec<u64> = Vec::new();
        if land::on() {
            // Pieces in the same block of title memory in one job, in order.
            let pieces: Vec<(usize, usize, usize)> =
                found.iter().enumerate().flat_map(|(i, found)| found.iter().map(move |&(at, n)| (i, at, n))).collect();
            let start = |i: usize| self.work[i].0.span().0 as usize;
            let mut order: Vec<usize> = (0..pieces.len()).collect();
            order.sort_by_key(|&p| (start(pieces[p].0) + pieces[p].1) / crate::land::BLOCK);
            let cut: Vec<(usize, usize)> = order.iter().map(|&p| (start(pieces[p].0) + pieces[p].1, pieces[p].2)).collect();
            let jobs = land::jobs(&cut);
            let put: Vec<std::sync::atomic::AtomicU64> = self.work.iter().map(|_| Default::default()).collect();
            land::each(jobs.len(), &|j| {
                let working = std::time::Instant::now();
                let mut notes = land::Notes::default();
                for &p in &order[jobs[j].clone()] {
                    let (i, at, n) = pieces[p];
                    put[i].fetch_add(self.work[i].0.land_back(at, n, &mut notes), Ordering::Relaxed);
                }
                notes.flush();
                land::spent(land::stat::BACK_ALL, working);
            });
            land::count(land::stat::JOBS, jobs.len() as u64);
            written.extend(put.iter().map(|p| p.load(Ordering::Relaxed)));
        } else {
            let (mut now, mut title) = (Vec::new(), Vec::new());
            for ((shadow, ..), found) in self.work.iter().zip(found) {
                let merge = |(at, n)| shadow.relay_merge(at, n, &mut now, &mut title) as u64;
                written.push(found.into_iter().map(merge).sum());
            }
        }
        land::spent(land::stat::BACK, merging);
        for ((_, from, to), written) in self.work.iter().zip(written) {
            COPY_BACK[1].fetch_add(written, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written as usize);
            let _ = (from, to);
        }
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    if send_free_first() {
        let f: Vec<f64> = FREE_FIRST.iter().map(|f| f.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "submissions cut in front of their first list that waits for flights, per frame (VICTORY_SEND_FREE_FIRST=1): {:.1},              {:.1} lists sent in front of {:.1}",
            f[0],
            f[1],
            f[2]
        );
    }
    if cut_waiting_on() {
        tracing::info!("joined submissions cut before a part that waits for flights, per frame (VICTORY_CUT_WAITING=1): {:.1}", s[18]);
    }
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}, \
             a reason with no place of its own: {:.1} and {:.1} (a wait that landed outside the WHY_ places above; \
             they were counted in the total but never named until now)",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
            w[0],
            w[1],
        );
        let reads: Vec<f64> = READ_WAITED.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        let named: f64 = reads.iter().sum();
        tracing::info!(
            "of the title memory read wait, per frame, by what was read: {} ({} in all, against {:.1} ms for why read)",
            READ_KINDS
                .iter()
                .zip(&reads)
                .map(|(name, ms)| format!("{name} {ms:.1} ms"))
                .collect::<Vec<_>>()
                .join(", "),
            format!("{named:.1} ms"),
            w[WHY_READ],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "the relay per frame (VICTORY_RELAY=1): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it)",
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if inflight_fetch() {
            let i: Vec<f64> = INFLIGHT_STATS.iter().map(|i| i.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "card copies' pieces staged inside their flights per frame (VICTORY_CARD_FETCH_INFLIGHT=1): {:.1} pieces \
                 ({:.2} MB) merged from their slots, which came down ({:.2} MB) in {:.1} ms of waiting on a copy queue; \
                 {:.1} pieces flagged beyond their flight's region; {:.1} landings fetched pieces the old way (the wait above)",
                i[0],
                i[1] / (1 << 20) as f64,
                i[5] / (1 << 20) as f64,
                i[4] / 1000.0,
                i[2],
                i[3],
            );
        }
        let idle = RELAY_IDLE_WAITS.load(Ordering::Relaxed);
        if idle > 0 {
            tracing::info!("waits for landings given up with nothing left to land: {idle} since the start");
        }
        if send::own() {
            let t: Vec<f64> = send::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "bytes of submissions' own sent up per frame (VICTORY_SEND_OWN): {:.1} MB in {:.0} runs, {:.0} words in scatters; {:.2} ms \
                 taking them out of the CPU sides, {:.2} of it copying the runs",
                t[0] / (1 << 20) as f64,
                t[1],
                t[2],
                t[3] / 1000.0,
                t[4] / 1000.0,
            );
        }
        let t: Vec<f64> = torn::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        if t[0] > 0.0 {
            tracing::info!(
                "fetched pieces checked per frame (VICTORY_FETCH_CHECK): {:.1}, of which {:.2} were not what their flight ended                  with (caught while a later flight was at them, or title bytes went up meanwhile); {:.2} were found whole when                  fetched again",
                t[0],
                t[1],
                t[2],
            );
        }
    }
    crate::land::log(frames);
    if place::on() {
        let t: Vec<f64> = place::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        let constants = place::CONSTANTS.swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "title buffer copies at their place in the lists, per frame (VICTORY_COPIES_IN_PLACE): {:.1} of {:.1} KB, {:.1} of them out              of what shaders wrote; {:.2} copies into textures out of what shaders wrote; {:.2} fills that took a copy's bytes              (VICTORY_FORWARD_PLACED={}); {:.2} copies out of what a shader of the same submission was              declared to write ahead of them, {:.2} of them with a fill that took their bytes; {:.1} constants read in place",
            t[0],
            t[1] / 1024.0,
            t[2],
            t[3],
            t[4],
            u8::from(place::forward_on()),
            t[5],
            t[6],
            constants,
        );
        if place::fill_in_place() {
            let f: Vec<f64> = place::FILLS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "constants behind a copy at its place in their list, per frame (VICTORY_FILL_IN_PLACE): {:.2} copied on the card at their place, {:.1} KB; {:.2} left to submission, no host copy of what shaders wrote holds them",
                f[0],
                f[1] / 1024.0,
                f[2],
            );
        }
    }
    card_steps::log(frames);
    fetch_waits::log(frames);
    dynamic_wait_line(frames);
    crate::port_journal::line(frames);
    crate::port::log(frames);
    crate::port_cbv::log(frames);
    crate::picked::log(frames);
    #[cfg(feature = "diagnostics")]
    log_blame(frames);
    if late_exact() {
        let [pages, runs, bytes, current, asked, alone] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB; {current:.0} pages not looked at, put in after the title's last write; \
             {asked:.0} ranges asked for all copies at once, {alone:.0} asked again on their own",
            bytes / (1 << 20) as f64
        );
    }
}

pub(crate) mod aside;
pub(crate) mod awake;
mod card_steps;
mod fetch_waits;
pub(crate) mod indirects;
#[cfg(feature = "diagnostics")]
pub(crate) mod panel;
#[cfg(feature = "diagnostics")]
mod frames;
mod font;
mod lazy;
mod menu;
mod place;
#[cfg(feature = "diagnostics")]
pub(crate) mod pstats;
mod send;
mod settle_room;
pub(crate) mod slot_wait;

/// Whether what goes up to copies in the card's memory goes from memory of
/// its own (`send::own`).
pub(crate) fn sends_own() -> bool {
    send::own()
}
pub(crate) use send::{goes_as_words, words_under};
#[cfg(feature = "diagnostics")]
pub(crate) mod spans;
mod torn;
#[cfg(feature = "diagnostics")]
mod traffic;
pub(crate) mod upcheck;
mod upq;
#[cfg(feature = "diagnostics")]
mod stamps;
#[cfg(feature = "diagnostics")]
pub(crate) use stamps::{timeline_ready, timeline_restart, timeline_stamps};

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// See [`HostList::defer_gpu_write`].
fn declare_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DECLARE_ONCE").map_or(true, |v| v != "0"))
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    // Where each copy is in it, and the ranges seen: dispatches declare the
    // same ranges thousands of times a frame.
    let mut places: ByCopy<usize> = ByCopy::default();
    let mut seen = std::collections::HashSet::<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>::default();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step {
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                if !seen.insert((key, *address, *len)) {
                    continue;
                }
                let i = *places.entry(key).or_insert_with(|| {
                    back.push((shadow.clone(), Vec::new()));
                    back.len() - 1
                });
                back[i].1.push((*address, *address + *len));
            }
        }
    }
    back
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; only with
/// `VICTORY_REFRESH_CHECK=1` as well: it costs about 100 ms a frame), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    use crate::land::{count, spent, stat};
    let started = std::time::Instant::now();
    // Textures shaders wrote that the title reads on the CPU
    // ([`crate::texback`]).
    for back in &list.deferred().texture_backs {
        crate::texback::land(back);
        count(stat::RB_TEXTURE_COUNT, 1);
    }
    let textures_done = spent(stat::RB_TEXTURES, started);
    for (buffer, mapped, at, [row, rows, pitch]) in &list.deferred().backs {
        count(stat::RB_BUFFER_COUNT, 1);
        count(stat::RB_BUFFER_BYTES, (*row * *rows) as u64);
        let mut data = *mapped as *mut std::ffi::c_void;
        // SAFETY: a read-back buffer of ours (or a chunk of one, mapped for
        // good), written by the list that has run; the title's destination
        // holds every row (checked when the copy was recorded).
        unsafe {
            if (*mapped != 0 || buffer.Map(0, None, Some(&mut data)).is_ok()) && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                    crate::watch::told(&[(*at + line * pitch, *row)]);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                if *mapped == 0 {
                    buffer.Unmap(0, None);
                }
            }
        }
    }
    if !list.deferred().stamps.is_empty() {
        land_stamps(&list.deferred().stamps);
    }
    spent(stat::RB_BUFFERS, textures_done);
}

/// Resolved host timestamps into title memory. The card's clock counts from
/// its own start, the title's ([`crate::queue::gpu_clock`]) from ours: each
/// stamp goes back by how long before the card's clock now it was taken.
fn land_stamps(stamps: &[(ID3D12Resource, usize, usize, usize)]) {
    let Some(host) = host() else {
        return;
    };
    let (mut card, mut pc) = (0, 0);
    // SAFETY: plain queries of our queue.
    let frequency = unsafe {
        if host.queue.GetClockCalibration(&mut card, &mut pc).is_err() {
            return;
        }
        host.queue.GetTimestampFrequency().unwrap_or(0)
    };
    let now = crate::queue::gpu_clock();
    for (buffer, mapped, at, count) in stamps {
        let mut data = *mapped as *mut std::ffi::c_void;
        // SAFETY: a read-back buffer of ours (or a chunk of one, mapped for
        // good) the list that has run resolved `count` stamps into; the
        // title's destination holds them (checked when it was recorded).
        unsafe {
            if (*mapped != 0 || buffer.Map(0, None, Some(&mut data)).is_ok()) && !data.is_null() {
                for i in 0..*count {
                    let taken = data.cast::<u64>().add(i).read_unaligned();
                    let ticks = now.saturating_sub(title_ticks(card.saturating_sub(taken), frequency)).max(1);
                    (*at as *mut u64).add(i).write_unaligned(ticks);
                }
                if *mapped == 0 {
                    buffer.Unmap(0, None);
                }
            }
        }
    }
}

/// `at` ticks of a clock of `host_hz` in the title's, [`crate::queue::TIMESTAMP_FREQUENCY`].
/// The product is taken in u128: a card's clock counts from its start, and
/// at 1 GHz its count times 100 MHz passes u64::MAX in about three minutes.
fn title_ticks(at: u64, host_hz: u64) -> u64 {
    let title_hz = u128::from(crate::queue::TIMESTAMP_FREQUENCY);
    (u128::from(at) * title_hz / u128::from(host_hz.max(1))) as u64
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() {
            // Done before (`Host::card_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its 174 MB heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (0.3-0.5 s) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let mut preludes = Preludes::default();
            if !lazy_prelude() {
                preludes.free.extend(self.make_prelude());
            }
            #[cfg(feature = "diagnostics")]
            let notes = std::sync::Arc::<Mutex<Vec<Note>>>::default();
            let pipelines = std::sync::Arc::<Mutex<std::collections::BTreeSet<String>>>::default();
            #[cfg(feature = "diagnostics")]
            let indirects = indirects::Shared::default();
            #[cfg(feature = "diagnostics")]
            if dred_on() || dred_light() {
                static NUMBER: AtomicU32 = AtomicU32::new(0);
                let number = NUMBER.fetch_add(1, Ordering::Relaxed);
                name(&list, || format!("victory host list {number}"));
                let mut all = LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner());
                if number.is_multiple_of(256) {
                    all.retain(|_, l| l.0.strong_count() > 0);
                }
                all.insert(
                    number,
                    (
                        std::sync::Arc::downgrade(&notes),
                        std::sync::Arc::downgrade(&pipelines),
                        std::sync::Arc::downgrade(&indirects),
                    ),
                );
            }
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                readbacks: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                has_twins: std::sync::atomic::AtomicBool::new(false),
                preludes: Mutex::new(preludes),
                pipelines,
                #[cfg(feature = "diagnostics")]
                indirects,
                budget: OnceLock::new(),
                card_budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                batch_args: Mutex::default(),
                coming: Mutex::default(),
                epoch: std::sync::atomic::AtomicU64::new(next_epoch()),
                title_calls: std::sync::atomic::AtomicU64::new(0),
                left_open: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
                has_placed: std::sync::atomic::AtomicBool::new(false),
                _alive: Counted::new(),
                #[cfg(feature = "diagnostics")]
                notes,
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                done: Default::default(),
                #[cfg(feature = "diagnostics")]
                clamp_stamps: Default::default(),
                #[cfg(feature = "diagnostics")]
                pstats: Default::default(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            CARD_CLOCK_US[0].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
            CARD_CLOCK_US[1].fetch_add(worked().saturating_sub(worked_then), Ordering::Relaxed);
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            crate::dxc::d3d_compile(
                card_compare_source().as_ptr().cast(),
                card_compare_source().len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            #[cfg(feature = "diagnostics")]
            name(&pso, || crate::replay::play::COMPARE.into());
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                inflight: if inflight_fetch() { self.make_card_inflight(RESULTS) } else { None },
            })
        }
    }

    /// [`CardInflight`] for `results` bytes of the compare's results; `None`
    /// (and the old fetch) if the shader or the memory cannot be had.
    ///
    /// Compiled with [`compile`] (the system's compiler), which logs what a
    /// compiler that refuses the shader says. shortcut: no DXC here; the
    /// shader keeps to what older compilers of HLSL take (no `groupshared`,
    /// no group barriers; `InterlockedAdd` on a raw buffer is what is left).
    fn make_card_inflight(&self, results: u64) -> Option<CardInflight> {
        let made = (|| -> windows::core::Result<CardInflight> {
            let bytes = compile(CARD_FETCH, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
            let slots = inflight_slots(inflight_fetch_bytes());
            let table_bytes = u64::from(slots) * 4 + results;
            let (table, table_data) = self.shared_buffer(table_bytes).ok_or_else(windows::core::Error::empty)?;
            name(&table, || "card fetch in flight: tickets".into());
            let staging = self.uav_buffer(u64::from(slots) * INFLIGHT_SLOT).ok_or_else(windows::core::Error::empty)?;
            name(&staging, || "card fetch in flight: pieces staged on the card".into());
            // SAFETY: the blob's own bytes; the root signature is inside
            // the shader.
            unsafe {
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, &bytes)?;
                let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
                Ok(CardInflight {
                    root,
                    pso,
                    table: (table, table_data as usize, table_bytes),
                    staging,
                    slots,
                    head: 0,
                    in_air: Default::default(),
                    brought: Vec::new(),
                })
            }
        })();
        match made {
            Ok(inflight) => {
                tracing::info!(
                    slots = inflight.slots,
                    "card copies' flagged pieces are staged inside their flights (VICTORY_CARD_FETCH_INFLIGHT=1)"
                );
                Some(inflight)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no fetch inside flights (VICTORY_CARD_FETCH_INFLIGHT=1): pieces are fetched the old way");
                None
            }
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    /// `picked`: ranges that may be written too, and ranges to look at
    /// (`crate::picked`).
    fn card_before(&self, declared: &[Declared], picked: &crate::picked::Ranges, flown: bool) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per copy.
        let declared: Vec<Declared> = declared.iter().filter(|(shadow, _)| shadow.carded()).cloned().collect();
        let declared = if crate::picked::lands() { crate::picked::with(declared, &picked.written) } else { declared };
        // (Of a copy with a declared range here, that says it.)
        let looked: Vec<&crate::picked::Range> =
            picked.looked.iter().filter(|l| !declared.iter().any(|(s, _)| Arc::ptr_eq(s, &l.0))).collect();
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let gathering = crate::timeline::span("copies in the card's memory: what changed");
        let mut dirty = crate::shadow::card_dirty(sent_as);
        // Pieces whose hash is stale and that hold what the CPU's side holds
        // (`VICTORY_SEND_CHANGED`): hashed again with the others, nothing sent.
        let rehash = crate::shadow::card_rehash();
        gathering.end();
        if declared.is_empty() && dirty.is_empty() && looked.is_empty() && rehash.is_empty() {
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for shadow in declared.iter().map(|d| &d.0).chain(looked.iter().map(|l| &l.0)) {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        let mut looks = None;
        let mut upload_slot = None;
        // The bytes as they are now, whatever later submissions put there.
        let taking = crate::timeline::span("copies in the card's memory: taken");
        let own = if send::own() { send::take(self, &dirty) } else { None };
        taking.end();
        if !dirty.is_empty() || !rehash.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let _recording = crate::timeline::span("copies in the card's memory: the list");
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    #[cfg(feature = "diagnostics")]
                    let side = crate::sidetime::begin(&self.device, &self.queue, list, crate::sidetime::SYNC);
                    if upcheck::on() {
                        looks = upcheck::record(self, list, &dirty);
                    }
                    // Every copy first, then one barrier call, then every
                    // pass: a wait between each copy and its pass had the
                    // card do them one after the other.
                    let mut sent = Vec::new();
                    // A buffer twice in `dirty` is moved out of COPY_DEST once.
                    let mut transitioned = Vec::new();
                    if let Some(own) = &own {
                        // Only the private snapshot is immutable for the entire
                        // asynchronous transfer. Keep shared staging on DIRECT.
                        if upq::on() && looks.is_none() {
                            let (source, used) = own.source();
                            if upq::wants(used) {
                                upload_slot = upq::send(self, &self.queue, source, used);
                                if upload_slot.is_none() && used != 0 {
                                    upq::fallback();
                                }
                            }
                        }
                        own.record(self, list, &dirty, |shadow| looks.as_ref().is_some_and(|l| l.hold(shadow)),
                            upload_slot.as_ref().and_then(upq::Slot::bounce));
                    }
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, _)) = shadow.card_sides() else {
                            continue;
                        };
                        if own.is_none() {
                            for &(a, b) in ranges {
                                list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                            }
                        }
                        let bytes: u64 = ranges.iter().map(|&(a, b)| (b - a) as u64).sum();
                        crate::shadow::CARD_STATS[0].fetch_add(bytes, Ordering::Relaxed);
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        #[cfg(feature = "diagnostics")]
                        traffic::moved(traffic::UP, shadow.span(), ranges.iter().copied());
                        if own.is_none() && before.is_some() && !ranges.is_empty() {
                            let raw = shadow.buffer.as_raw();
                            if !transitioned.contains(&raw) {
                                transitioned.push(raw);
                                sent.push(transition(
                                    &shadow.buffer,
                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                ));
                            }
                        }
                    }
                    if !sent.is_empty() {
                        list.ResourceBarrier(&sent);
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::sidetime::end(list, side);
                    #[cfg(feature = "diagnostics")]
                    let side = crate::sidetime::begin(&self.device, &self.queue, list, crate::sidetime::HASH);
                    let batch = card_batch_work(&self.device);
                    let mut jobs = Vec::new();
                    if batch.is_none() {
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&work.pso);
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    }
                    for (shadow, ranges) in dirty.iter().chain(&rehash) {
                        let (Some((_, Some(before), _)), size) = (shadow.card_sides(), shadow.card_hash_size()) else {
                            continue;
                        };
                        // No twin (it cost the card the copy's size again,
                        // 399 MB in run 75): `before` holds a 128-bit hash
                        // of each 64 KiB piece. The pieces just sent up are
                        // hashed again, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        let joins = rehash_runs() || own.is_some();
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            // (In order: the next of the same piece, of the
                            // tens of thousands there are, says nothing.)
                            let mut runs: Vec<crate::shadow::CardSpan> = Vec::new();
                            for run in ranges.iter().map(|&(a, b)| shadow.card_run(a, b)) {
                                match runs.last_mut() {
                                    Some(last) if joins && (last.first, last.pieces) == (run.first, run.pieces) => last.to = last.to.max(run.to),
                                    _ => runs.push(run),
                                }
                            }
                            runs
                        };
                        // Pending pieces keep their hashes (`lazy`).
                        let runs = if lazy::on(shadow) && !whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            lazy::hold_back(shadow, runs)
                        } else {
                            runs
                        };
                        if batch.is_some() {
                            jobs.push(CardJob {
                                key: 0,
                                now: shadow.buffer.GetGPUVirtualAddress(),
                                hashes: before.GetGPUVirtualAddress(),
                                size: size as u32,
                                skew: runs.first().map_or(0, |r| r.skew),
                                pieces: runs.iter().flat_map(|r| (r.first..r.first + r.pieces).map(|p| (p, 0, None))).collect(),
                            });
                            continue;
                        }
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if joins { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            card_pass(list, &run, size as u32, 0, 0);
                        }
                    }
                    if let Some(batch) = batch {
                        card_batched(self, batch, list, work.differs.0.GetGPUVirtualAddress(), jobs, 0)?;
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::sidetime::end(list, side);
                    list.Close()?;
                    if let Some(own) = &own {
                        own.copied();
                    }
                    before_send(&self.queue);
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty
                .into_iter()
                .chain(rehash)
                .map(|(shadow, _)| shadow)
                .collect(),
            lists: used,
            sent_as,
            looks,
            sends: own.map(send::Own::memory),
            inflight: false,
            region: None,
            upload_slot,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            let lazy_copy = twin && lazy::on(shadow);
            let start = shadow.span().0 as usize;
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                if lazy_copy {
                    // Only what `lazy` picks; the rest is left pending.
                    for (i0, i1) in lazy::select(shadow, &span) {
                        let (lo, hi) = (shadow.card_piece(&span, i0).0, shadow.card_piece(&span, i1 - 1).1);
                        if lo >= hi {
                            continue;
                        }
                        let span = shadow.card_run(lo, hi);
                        let compared = u64::from(at + span.pieces) * 4 <= work.differs.2;
                        plan.ranges.push(CardRange {
                            shadow: shadow.clone(),
                            declared: (start + lo, start + hi),
                            span,
                            at,
                            compared,
                            picked: false,
                            lazy: true,
                        });
                        if compared {
                            at += span.pieces;
                        }
                    }
                    continue;
                }
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                    picked: false,
                    lazy: false,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        // `lazy`: the pieces forced by a send and the sweep's next ones, of
        // the copies declared or sent up here.
        if lazy::min().is_some() {
            let mut copies: Vec<&Arc<crate::shadow::Shadow>> = Vec::new();
            for shadow in declared.iter().map(|d| &d.0).chain(plan._synced.iter()) {
                if lazy::on(shadow)
                    && shadow.card_sides().is_some_and(|(_, before, _)| before.is_some())
                    && !copies.iter().any(|s| Arc::ptr_eq(s, shadow))
                {
                    copies.push(shadow);
                }
            }
            for shadow in copies {
                let start = shadow.span().0 as usize;
                for span in lazy::sweep(shadow) {
                    let compared = u64::from(at + span.pieces) * 4 <= work.differs.2;
                    plan.ranges.push(CardRange {
                        shadow: shadow.clone(),
                        declared: (start + span.from as usize, start + span.to as usize),
                        span,
                        at,
                        compared,
                        picked: false,
                        lazy: true,
                    });
                    if compared {
                        at += span.pieces;
                    }
                }
            }
        }
        for (shadow, from, len) in looked {
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            if let Some(span) = shadow.card_span(*from, from + len)
                && twin
                && u64::from(at + span.pieces) * 4 <= work.differs.2
            {
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (*from, from + len),
                    span,
                    at,
                    compared: true,
                    picked: true,
                    lazy: false,
                });
                at += span.pieces;
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            // The fetch inside the flight (`inflight_fetch`): a region of the
            // staging buffer, a slot per piece compared (`inflight_want`),
            // its counter zeroed before the list
            // goes to the queue (its last user has landed).
            if flown && let Some(inflight) = work.inflight.as_mut() {
                let compared: u64 = plan.ranges.iter().filter(|r| r.compared).map(|r| u64::from(r.span.pieces)).sum();
                let want = inflight_want(compared, inflight.slots, inflight_flight_most());
                if let Some(base) = take_region(&inflight.in_air, inflight.head, inflight.slots, want) {
                    inflight.in_air.push_back((base, want));
                    inflight.head = base + want;
                    // SAFETY: the counters' mapping holds a dword per slot.
                    unsafe { (inflight.table.1 as *mut u32).add(base as usize).write_volatile(0) };
                    plan.region = Some((base, want));
                }
            }
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    #[cfg(feature = "diagnostics")]
                    let side = crate::sidetime::begin(&self.device, &self.queue, list, crate::sidetime::CHECK);
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    if let Some(batch) = card_batch_work(&self.device) {
                        // A job per copy, its ranges in one dispatch.
                        let mut jobs: Vec<CardJob> = Vec::new();
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let (Some((_, Some(before), _)), size) = (r.shadow.card_sides(), r.shadow.card_hash_size()) else {
                                continue;
                            };
                            let key = Arc::as_ptr(&r.shadow) as usize;
                            let at = match jobs.iter().position(|j| j.key == key) {
                                Some(at) => at,
                                None => {
                                    jobs.push(CardJob {
                                        key,
                                        now: r.shadow.buffer.GetGPUVirtualAddress(),
                                        hashes: before.GetGPUVirtualAddress(),
                                        size: size as u32,
                                        skew: r.span.skew,
                                        pieces: Vec::new(),
                                    });
                                    jobs.len() - 1
                                }
                            };
                            let mode = if r.lazy || flown && !r.picked { 2 } else { 1 };
                            jobs[at].pieces.extend((0..r.span.pieces).map(|i| (r.span.first + i, mode, Some((r.at + i) * 4))));
                        }
                        card_batched(self, batch, list, work.differs.0.GetGPUVirtualAddress(), jobs, 2)?;
                    } else {
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&work.pso);
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let (Some((_, Some(before), _)), size) = (r.shadow.card_sides(), r.shadow.card_hash_size()) else {
                                continue;
                            };
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                            // (What is only looked at keeps its hashes: what
                            // changed there is the next declared range's to land.)
                            // (`lazy`'s keep what they find: a piece left out
                            // keeps its old hashes, one compared takes new ones.)
                            let mode = if r.lazy || flown && !r.picked { 2 } else { 1 };
                            card_pass(list, &r.span, size as u32, r.at, mode);
                            CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // The fetch inside the flight (`inflight_fetch`): behind
                    // every compare of the list, tickets for the pieces they
                    // flagged, then the pieces into the flight's slots of
                    // the staging buffer, in the card's memory.
                    if let (Some(inflight), Some((base, count))) = (&work.inflight, plan.region) {
                        let uav = || D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                        };
                        list.ResourceBarrier(&[uav()]);
                        list.SetComputeRootSignature(&inflight.root);
                        list.SetPipelineState(&inflight.pso);
                        list.SetComputeRootUnorderedAccessView(2, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, inflight.table.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, inflight.staging.GetGPUVirtualAddress());
                        let mut every = true;
                        let mut staged: Vec<(u64, [u32; 10])> = Vec::new();
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                every = false;
                                continue;
                            };
                            let rows = r.span.pieces.clamp(1, 65535);
                            let constants =
                                [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, base, count, 0, inflight.entries(), rows];
                            list.SetComputeRoot32BitConstants(0, 10, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.Dispatch(r.span.pieces.div_ceil(64), 1, 1);
                            staged.push((r.shadow.buffer.GetGPUVirtualAddress(), constants));
                        }
                        // The tickets, then the copies.
                        list.ResourceBarrier(&[uav()]);
                        for (now, mut constants) in staged {
                            constants[7] = 1;
                            list.SetComputeRoot32BitConstants(0, 10, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, now);
                            list.Dispatch(INFLIGHT_STEPS, constants[9], 1);
                        }
                        plan.inflight = every;
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::sidetime::end(list, side);
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        if let Some(slot) = plan.upload_slot {
            upq::give_back(slot);
        }
        if let Some(looks) = plan.looks {
            looks.read();
        }
        if let Some(memory) = plan.sends {
            send::give_back(memory);
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        // SAFETY: plain object creation.
        let copying = relay_copy().then(|| unsafe {
            (|| -> windows::core::Result<(Prelude, ID3D12CommandQueue)> {
                let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                let list: ID3D12GraphicsCommandList =
                    self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
                list.Close()?;
                let queue: ID3D12CommandQueue = self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                    ..Default::default()
                })?;
                Ok(((allocator, list), queue))
            })()
        });
        let (list, queue) = match copying {
            Some(Ok((list, queue))) => (list, Some(queue)),
            Some(Err(e)) => {
                tracing::warn!(error = %e, "no copy queue: what shaders wrote to the card's memory is fetched on the drawing queue");
                (self.make_prelude()?, None)
            }
            None => (self.make_prelude()?, None),
        };
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                queue,
                fence,
                event,
                sent: 0,
                buffer: None,
                suspects: torn::Suspects::default(),
                had: crate::land::Had::default(),
                down: None,
            })
        }
    }

    /// [`InflightDown`]: a copy queue, its list, a fence and an event.
    fn make_inflight_down(&self) -> Option<InflightDown> {
        // SAFETY: plain objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue = self
                .device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })
                .ok()?;
            name(&queue, || "card fetch in flight: staged pieces come down".into());
            let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None).ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(InflightDown {
                queue,
                list: (allocator, list),
                fence,
                event,
                sent: 0,
                buffer: None,
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up. `plans` come with the value signalled behind
    /// their flight. `meanwhile` is called while the fetch is on its way, if
    /// there is one.
    ///
    /// With [`inflight_fetch`] the pieces are first taken from where the
    /// flights' own compare lists staged them ([`Host::relay_card_inflight`]);
    /// only those that are not there are fetched, as here, and merged after.
    fn relay_card_back(&self, plans: &[(&CardPlan, u64)], fetch: &mut Option<RelayFetch>, meanwhile: &mut dyn FnMut()) {
        use crate::land;
        let gathering = std::time::Instant::now();
        let differs = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        if let (Some(fetch), Some(first)) = (fetch.as_mut(), plans.iter().map(|p| p.1).min()) {
            fetch.had.landing(first);
        }
        let staged = if inflight_fetch() {
            let only: Vec<&CardPlan> = plans.iter().map(|p| p.0).collect();
            let staged = self.relay_card_inflight(&only, fetch);
            inflight_regions_landed(&only);
            staged
        } else {
            None
        };
        // SAFETY: the mapping of the results. The compares of these plans
        // have run, and their dwords are given to no other plan before these
        // are done with (`card_done`).
        let differs = differs.map(|(differs, dwords)| unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) });
        if let Some(left) = staged {
            // What the slots did not hold, fetched as below.
            for (shadow, lo, hi) in left {
                pieces.push((shadow, lo, hi, total));
                total += (hi - lo) as u64;
            }
        } else if let Some(differs) = differs {
            // Pieces caught while a later flight was at them (`torn`).
            let suspects = fetch.as_ref().map(|f| &f.suspects);
            let mut seen = std::collections::HashSet::new();
            for &(plan, behind) in plans {
                plan.picked_found(differs);
                plan.lazy_found(differs);
                let flags: Vec<(usize, u32, bool)> = plan
                    .ranges
                    .iter()
                    .filter(|r| !r.picked)
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| torn::flagged(d));
                            let wanted = suspects.map_or(flagged, |s| s.wanted((r.shadow.id(), r.span.first + p), flagged));
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, wanted)
                        })
                    })
                    .collect();
                let fetched = flagged_pieces(&flags);
                let pages = card_pages().then(|| plan_pages(plan, differs));
                for r in plan.ranges.iter().filter(|r| !r.picked) {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    if upcheck::on() {
                        upcheck::declared(r.shadow.id(), r.span.from as usize, r.span.to as usize);
                    }
                    for p in 0..r.span.pieces {
                        if fetched.binary_search(&(copy, r.span.first + p)).is_err() {
                            continue;
                        }
                        let mask = pages.as_ref().map_or(u16::MAX, |m| m.get(&(copy, r.span.first + p)).copied().unwrap_or(u16::MAX));
                        for (lo, hi) in r.shadow.card_pages(&r.span, p, mask) {
                        if lo < hi && !seen.contains(&(copy, lo, hi)) {
                            let key = (r.shadow.id(), r.shadow.span().0 as usize + lo, hi - lo);
                            if fetch.as_ref().is_some_and(|f| f.had.holds(behind, key)) {
                                land::count(land::stat::HAD, 1);
                                land::count(land::stat::HAD_BYTES, (hi - lo) as u64);
                                if land::skip_on() {
                                    continue;
                                }
                            }
                            if upcheck::on() {
                                upcheck::fetched(r.shadow.id(), lo);
                            }
                            seen.insert((copy, lo, hi));
                            pieces.push((r.shadow.clone(), lo, hi, total));
                            total += (hi - lo) as u64;
                        }
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                }
            }
        }
        if land::on() {
            // In the order they are landed, each byte of a copy once.
            let keys: Vec<land::Piece> =
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.id(), shadow.span().0 as usize + lo, hi - lo)).collect();
            total = 0;
            pieces = land::ordered(&keys)
                .into_iter()
                .map(|(i, title, len)| {
                    let shadow = &pieces[i].0;
                    let lo = title - shadow.span().0 as usize;
                    let at = total;
                    total += len as u64;
                    (shadow.clone(), lo, lo + len, at)
                })
                .collect();
        }
        land::spent(land::stat::GATHER, gathering);
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        // On the copy queue a later flight may be at a piece while it is
        // fetched: the pieces come down whole, to be hashed.
        let check = (torn::on() && fetch.as_ref().is_some_and(|f| f.queue.is_some()))
            .then(|| torn::Check::new(plans.iter().map(|p| p.0), differs.unwrap_or_default(), &mut pieces));
        if let Some(check) = &check {
            total = check.total;
        }
        #[cfg(feature = "diagnostics")]
        for (shadow, lo, hi, _) in &pieces {
            traffic::moved(traffic::FETCHED, shadow.span(), std::iter::once((*lo, *hi)));
        }
        // The sends these flights ran behind, all a fetch on the copy queue
        // is known to have run behind.
        let behind = plans.iter().map(|p| p.0.sent_as).max().unwrap_or(0);
        if pieces.is_empty() {
            // Nothing to fetch: every send so far is older than the next
            // fetch, if that goes to the queue the sends went to. With the
            // fetch inside flights the next merge may be of a flight in the
            // air now, with bytes from when it ran: only sends up to these
            // plans' are older than it.
            let after = match inflight_fetch() || fetch.as_ref().is_some_and(|f| f.queue.is_some()) {
                true => behind,
                false => *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()),
            };
            crate::shadow::relay_fetched(after);
            return;
        }
        if inflight_fetch() {
            INFLIGHT_STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let copy_queue = f.queue.clone();
        let waiting = std::time::Instant::now();
        // The flights the fetch runs behind: those in the line by now have
        // been given to the queue. On the copy queue, those landed here.
        let ahead = match copy_queue {
            Some(_) => 0,
            None => line().target,
        };
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in check.as_ref().map_or(&pieces, |c| &c.copies) {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                let _fetching = crate::timeline::span("landing: fetching pieces");
                if let Some(queue) = &copy_queue {
                    before_send(queue);
                    queue.ExecuteCommandLists(&lists);
                    queue.Signal(&fence, sent)?;
                    meanwhile();
                    fence.SetEventOnCompletion(sent, event)?;
                    #[cfg(feature = "diagnostics")]
                    let _card = crate::timeline::card_wait(None);
                    WaitForSingleObject(event, INFINITE);
                    return Ok(behind);
                }
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                before_send(&self.queue);
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                meanwhile();
                fence.SetEventOnCompletion(sent, event)?;
                #[cfg(feature = "diagnostics")]
                let _card = crate::timeline::card_wait(None);
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        // This fetch may bring bytes of flights in the air now: their own
        // slots are older than what it puts in the CPU's side there
        // (`relay_card_inflight`). Sends up to `after` are numbered, and the
        // submission numbered `after` may not have its lists on the queue
        // yet; every one before it has (the submission lock).
        if inflight_fetch()
            && after > 1
            && let Some(inflight) = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|w| w.inflight.as_mut())
        {
            inflight.brought.extend(pieces.iter().map(|(shadow, lo, hi, _)| (shadow.clone(), *lo, *hi, after - 1)));
        }
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        if fetch_waits::on() {
            let who = if copy_queue.is_some() { fetch_waits::RELAY_COPY } else { fetch_waits::RELAY_MAIN };
            let fetched = check.as_ref().map_or(&pieces, |c| &c.copies);
            fetch_waits::fetched(who, fetched.iter().map(|(shadow, lo, hi, _)| (shadow.span().0 as usize, *lo, *hi)), us, 0);
        }
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        let merging = std::time::Instant::now();
        let _merging = crate::timeline::span("landing: merging fetched pieces");
        let mut written = 0;
        if land::on() {
            let titles: Vec<(usize, usize)> =
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.span().0 as usize + lo, hi - lo)).collect();
            let jobs = land::jobs(&titles);
            let put = std::sync::atomic::AtomicU64::new(0);
            land::each(jobs.len(), &|j| {
                let working = std::time::Instant::now();
                if let Some(check) = &check {
                    for i in jobs[j].clone() {
                        // SAFETY: as below.
                        unsafe { check.look(i, data) };
                    }
                }
                let mut notes = land::Notes::default();
                for run in pieces[jobs[j].clone()].chunk_by(|a, b| std::sync::Arc::ptr_eq(&a.0, &b.0)) {
                    let addressed: Vec<(usize, usize, usize)> =
                        run.iter().map(|(_, lo, hi, at)| (*lo, *hi, data + *at as usize)).collect();
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run, and the next waits
                    // for this call.
                    put.fetch_add(unsafe { run[0].0.land_card(&addressed, after, &mut notes) }, Ordering::Relaxed);
                }
                notes.flush();
                land::spent(land::stat::CARD_ALL, working);
            });
            land::count(land::stat::JOBS, jobs.len() as u64);
            written = put.into_inner();
        } else {
            let mut title = Vec::new();
            for (i, (shadow, lo, hi, at)) in pieces.iter().enumerate() {
                if let Some(check) = &check {
                    // SAFETY: as below.
                    unsafe { check.look(i, data) };
                }
                // SAFETY: the mapping of the fetched bytes, `total` of them;
                // the copy into it has run.
                let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
                let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title);
                shadow.walls_merged(gpu.len(), merged);
                written += merged as u64;
            }
        }
        land::spent(land::stat::CARD, merging);
        COPY_BACK[1].fetch_add(written, Ordering::Relaxed);
        if let Some(f) = fetch.as_mut() {
            if let Some(check) = &check {
                check.done(&mut f.suspects);
            }
            f.had.fetched(
                ahead,
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.id(), shadow.span().0 as usize + lo, hi - lo)),
            );
        }
        // Sends up to `after` are of no use to any later merge; with the
        // fetch inside flights only those up to these plans' are
        // (`fallback_fetched_mark`).
        let landed = plans.iter().map(|p| p.0.sent_as).max();
        crate::shadow::relay_fetched(fallback_fetched_mark(inflight_fetch(), after, landed));
    }

    /// [`Host::relay_card_back`] with [`inflight_fetch`]: the pieces these
    /// plans' compares flagged that their compare lists staged
    /// ([`CARD_FETCH`]) come down, one copy per flight of the slots it used,
    /// all in one list on a copy queue of this thread's own and one wait,
    /// and are merged plan by plan in the line's order. Returns the pieces
    /// that were not staged (flagged beyond the flight's region, of a range
    /// not compared, of a flight with no region), each once, for the old
    /// fetch, which the caller merges after these; `None` if there is
    /// nothing to stage with (and the caller fetches every piece the old
    /// way).
    ///
    /// The merge's "fetched after" (`Shadow::card_merge_relay`) is the
    /// plan's own `sent_as`. `QUEUE_ORDER` is raised only in
    /// `Host::card_before`, under the submission lock, and that lock is held
    /// until the submission's lists and its compare's list are on the queue
    /// behind its sync list: so when the compare's list ran, the card had
    /// received exactly the sends 1..=`sent_as` of title bytes and no later
    /// one, which is what the merge's skip of later sends asks. A flight's
    /// region is written by its own list only and given back only after its
    /// landing (`inflight_regions_landed`), so what comes down is the card's
    /// bytes as the flight left them, however late the copy runs. (The old
    /// fetch, by contrast, reads the copies when it reaches the queue, and
    /// takes the count at its own ExecuteCommandLists.)
    ///
    /// Order of the merges: every slot of the landing first, then the old
    /// fetch of what was left. A fetch the old way brings the card's bytes
    /// as they are now, newer than the slots of every flight in the air then;
    /// merged last it is the newest thing merged into any byte. For the same
    /// reason the bytes such a fetch brought (`CardInflight::brought`) are
    /// left out of the slots of flights whose lists were on the queue ahead
    /// of it, in later landings: the CPU's side holds what the card held
    /// after them there. The rest of their slots is merged, newer than
    /// anything merged there before. Plans land in the line's order, each
    /// plan's slot as it was right behind its own lists, so every merge is
    /// of bytes at least as new as the CPU's side holds. The notes of sends
    /// the merges need are kept by `fallback_fetched_mark`.
    fn relay_card_inflight(
        &self,
        plans: &[&CardPlan],
        fetch: &mut Option<RelayFetch>,
    ) -> Option<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>> {
        use std::sync::Arc;
        // A stretch of a staged piece to merge: copy, offsets, the slot's
        // region, where in the region the stretch is, the plan's send.
        type Staged = (Arc<crate::shadow::Shadow>, usize, usize, u32, usize, u64);
        let mut staged: Vec<Staged> = Vec::new();
        let mut left: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        // Regions to bring down: first slot, slots used.
        let mut down: Vec<(u32, u32)> = Vec::new();
        let staging = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let work = work.as_ref()?;
            let inflight = work.inflight.as_ref()?;
            let dwords = (work.differs.2 / 4) as usize;
            // SAFETY: the mappings of the results, the counters and the
            // entries. The compares and fetches of these plans have run, and
            // their dwords and regions are given to no other plan before
            // these are done with (`card_done`, `inflight_regions_landed`).
            let (differs, counters, entries) = unsafe {
                (
                    std::slice::from_raw_parts(work.differs.1 as *const u32, dwords),
                    std::slice::from_raw_parts(inflight.table.1 as *const u32, inflight.slots as usize),
                    std::slice::from_raw_parts((inflight.table.1 as *const u32).add(inflight.slots as usize), dwords),
                )
            };
            let mut seen_left = std::collections::HashSet::new();
            for plan in plans {
                plan.picked_found(differs);
                let region = plan.region.filter(|_| plan.inflight);
                if let Some((base, count)) = region {
                    let used = counters.get(base as usize).copied().unwrap_or(0).min(count);
                    if used > 0 {
                        down.push((base, used));
                    }
                }
                // Pieces (copy, number) flagged, with their slot where staged.
                let mut slots: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
                let mut flags: Vec<(usize, u32, bool)> = Vec::new();
                // What is only looked at keeps its changes for the next
                // declared range (`CardPlan::picked_found`).
                for r in plan.ranges.iter().filter(|r| !r.picked) {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    if upcheck::on() {
                        upcheck::declared(r.shadow.id(), r.span.from as usize, r.span.to as usize);
                    }
                    for p in 0..r.span.pieces {
                        let at = (r.at + p) as usize;
                        let flagged = !r.compared || differs.get(at).is_none_or(|&d| torn::flagged(d));
                        if flagged
                            && r.compared
                            && let Some((_, count)) = region
                            && let Some(&k) = entries.get(at)
                            && k != INFLIGHT_NONE
                            && k < count
                        {
                            slots.insert((copy, r.span.first + p), k);
                        }
                        flags.push((copy, r.span.first + p, flagged));
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                }
                let fetched = flagged_pieces(&flags);
                let mut seen = std::collections::HashSet::new();
                for r in plan.ranges.iter().filter(|r| !r.picked) {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece = r.span.first + p;
                        if !fetched.contains(&(copy, piece)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo >= hi || !seen.insert((copy, lo, hi)) {
                            continue;
                        }
                        let from = inflight_piece_from(&r.span, p);
                        let slot = slots.get(&(copy, piece)).copied().filter(|_| lo >= from && (hi - from) as u64 <= INFLIGHT_SLOT);
                        let (Some(k), Some((base, _))) = (slot, region) else {
                            INFLIGHT_STATS[2].fetch_add(1, Ordering::Relaxed);
                            if seen_left.insert((copy, lo, hi)) {
                                left.push((r.shadow.clone(), lo, hi));
                            }
                            continue;
                        };
                        // Bytes a fetch the old way brought after this flight ran.
                        let newer: Vec<(usize, usize)> = inflight
                            .brought
                            .iter()
                            .filter(|b| b.3 >= plan.sent_as && Arc::ptr_eq(&b.0, &r.shadow) && b.1 < hi && b.2 > lo)
                            .map(|b| (b.1, b.2))
                            .collect();
                        let into = k as usize * INFLIGHT_SLOT as usize;
                        for (a, b) in crate::shadow::outside(lo, hi, newer) {
                            staged.push((r.shadow.clone(), a, b, base, into + (a - from), plan.sent_as));
                        }
                    }
                }
            }
            inflight.staging.clone()
        };
        if staged.is_empty() {
            return Some(left);
        }
        // The regions, one after the other, in the PC's memory.
        let mut into = std::collections::HashMap::new();
        let mut total = 0u64;
        for &(base, used) in &down {
            into.insert(base, total);
            total += u64::from(used) * INFLIGHT_SLOT;
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.down.is_none() {
            f.down = self.make_inflight_down();
        }
        // Said once: a landing without a copy queue or memory to bring the
        // staged pieces into comes down the old way (its regions are given
        // back all the same, `inflight_regions_landed`).
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let Some(d) = f.down.as_mut() else {
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!("no copy queue for the staged pieces (VICTORY_CARD_FETCH_INFLIGHT=1): fetched the old way");
            }
            return None;
        };
        if d.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            d.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = d.buffer.clone() else {
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(total, "no memory to bring the staged pieces down (VICTORY_CARD_FETCH_INFLIGHT=1): fetched the old way");
            }
            return None;
        };
        d.sent += 1;
        let ((allocator, list), queue, fence, event, sent) = (d.list.clone(), d.queue.clone(), d.fence.clone(), d.event, d.sent);
        let waiting = std::time::Instant::now();
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers; the queue, fence and event are this thread's. The regions
        // are these flights' own and the card has run them; buffers are in
        // the common state between submissions, and a copy queue takes them
        // from there.
        let ran = unsafe {
            (|| -> windows::core::Result<()> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for &(base, used) in &down {
                    let to = into[&base];
                    list.CopyBufferRegion(&buffer, to, &staging, u64::from(base) * INFLIGHT_SLOT, u64::from(used) * INFLIGHT_SLOT);
                }
                list.Close()?;
                queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);
                queue.Signal(&fence, sent)?;
                if fence.GetCompletedValue() < sent {
                    fence.SetEventOnCompletion(sent, event)?;
                    WaitForSingleObject(event, INFINITE);
                }
                Ok(())
            })()
        };
        if let Err(e) = ran {
            fail(format_args!("the staged pieces could not be brought down ({e}, VICTORY_CARD_FETCH_INFLIGHT=1)"));
        }
        INFLIGHT_STATS[4].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
        INFLIGHT_STATS[5].fetch_add(total, Ordering::Relaxed);
        let mut title = Vec::new();
        let (mut written, mut bytes) = (0, 0u64);
        for (shadow, lo, hi, base, at, after) in &staged {
            let Some(&region) = into.get(base) else {
                continue;
            };
            // SAFETY: inside what came down: the slot is below the region's
            // slots used, and the stretch within the slot (checked above).
            let gpu = unsafe { std::slice::from_raw_parts((data + region as usize + at) as *const u8, hi - lo) };
            let merged = shadow.card_merge_relay(*lo, gpu, *after, &mut title);
            shadow.walls_merged(gpu.len(), merged);
            written += merged;
            bytes += (hi - lo) as u64;
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        INFLIGHT_STATS[0].fetch_add(staged.len() as u64, Ordering::Relaxed);
        INFLIGHT_STATS[1].fetch_add(bytes, Ordering::Relaxed);
        Some(left)
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        let mut stage = crate::timeline::span("landing: what is to be looked through");
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let plans: Vec<(&CardPlan, u64)> = flights.iter().filter_map(|f| Some((f.card.as_ref()?, f.fence))).collect();
                // (Each byte once: the ranges are joined.)
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    for (shadow, ranges) in flight.back.iter().filter(|(shadow, _)| !shadow.carded()) {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => all.extend_from_slice(ranges),
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                let mut hand_back = relay_hand_back(&back);
                stage.next("landing: copies in the card's memory");
                self.relay_card_back(&plans, fetch, &mut || {
                    if crate::land::on() {
                        let _looking = crate::timeline::span("landing: looking through copies, beside the fetch");
                        hand_back.look(crate::land::stat::SCAN_BESIDE);
                    }
                });
                stage.next("landing: copies in the PC's memory");
                hand_back.merge();
                stage.next("landing: ranges done with");
                let finishing = std::time::Instant::now();
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                crate::land::spent(crate::land::stat::DONE, finishing);
                COPY_BACK[0].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        stage.next("landing: read-backs, telling the title");
        let mut last = 0;
        let mut landed = Vec::new();
        let mut telling = std::time::Instant::now();
        for item in items {
            telling = match item {
                Late::Run(mut flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    let done_from = std::time::Instant::now();
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    if let Some(own) = flight.own {
                        own.landed();
                    }
                    crate::land::spent(crate::land::stat::RB_DONE, done_from);
                    let mut lists = flight.lists;
                    for detached in &lists {
                        detached.release_scratch();
                    }
                    if tables_at_landing() {
                        for list in &mut lists {
                            list.release_tables();
                        }
                    }
                    crate::list::tables_early(&mut lists);
                    landed.push((lists, flight.order));
                    crate::land::spent(crate::land::stat::READ_BACKS, telling)
                }
                Late::Write(at, what, bytes) => {
                    what.write(at as *mut u8, bytes);
                    crate::land::spent(crate::land::stat::TOLD, telling)
                }
                Late::Signal(fence, value) => {
                    crate::fence::set(fence as crate::object::This, value);
                    crate::land::spent(crate::land::stat::TOLD, telling)
                }
            };
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        stage.next("landing: the rest");
        self.log_messages();
        crate::land::spent(crate::land::stat::REST, telling);
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                narrow,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                landing,
                landed_items,
                taking,
                ..
            } = &mut *line;
            kept.extend(landed);
            *landed_items += std::mem::take(taking);
            in_hand.clear();
            landing.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            narrow.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            let mut narrowed = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                    ranges.extend(flight.texture_back_ranges.iter().copied());
                    if lazy::narrow() {
                        narrowed.extend(flight.texture_back_ranges.iter().copied());
                        for (shadow, r) in &flight.back {
                            if lazy::on(shadow) {
                                for &(from, to) in r {
                                    lazy::hot_within(shadow, from, to, &mut narrowed);
                                }
                            } else {
                                narrowed.extend(r.iter().copied());
                            }
                        }
                    }
                }
            }
            join_ranges(written, ranges);
            join_ranges(narrow, narrowed);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        plan.picked_found(differs);
        plan.lazy_found(differs);
        let pages = card_pages().then(|| plan_pages(plan, differs));
        for (i, r) in plan.ranges.iter().enumerate().filter(|(_, r)| !r.picked) {
            let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
            for p in 0..r.span.pieces {
                if r.compared && !torn::flagged(differs[(r.at + p) as usize]) {
                    continue;
                }
                let mask = pages.as_ref().map_or(u16::MAX, |m| m.get(&(copy, r.span.first + p)).copied().unwrap_or(u16::MAX));
                for (lo, hi) in r.shadow.card_pages(&r.span, p, mask) {
                    if lo < hi && seen.insert((copy, lo, hi)) {
                        // Pages next to one another come down in one copy.
                        match pieces.last_mut() {
                            Some(last) if pages.is_some() && last.0 == i && last.2 == lo => last.2 = hi,
                            _ => pieces.push((i, lo, hi, total)),
                        }
                        total += (hi - lo) as u64;
                    }
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            // SAFETY: our fence.
            let ahead = fetch_waits::on().then(|| submitted.saturating_sub(unsafe { self.fence.GetCompletedValue() }));
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    #[cfg(feature = "diagnostics")]
                    let side = crate::sidetime::begin(&self.device, &self.queue, list, crate::sidetime::FETCH);
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::sidetime::end(list, side);
                    list.Close()?;
                    before_send(&self.queue);
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            if let Some(ahead) = ahead {
                fetch_waits::fetched(
                    fetch_waits::SUBMITTING,
                    pieces.iter().map(|&(i, lo, hi, _)| (plan.ranges[i].shadow.span().0 as usize, lo, hi)),
                    waiting.elapsed().as_micros() as u64,
                    ahead,
                );
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // Behind what the title's queues were given before it.
        crate::queue::before_sync();
        let _ = self.submit(runs, &[], false);
    }

    /// [`Host::execute`] of a closed upload-only list of our own, without
    /// waiting for the title's queues (`queue::before_sync`): safe from
    /// `VICTORY_THIN=1` recording threads.
    pub(crate) fn execute_sync_internal(&self, list: &HostList) {
        let _ = self.submit(&[list], &[], false);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        self.execute_parts(lists, order, &[]);
    }

    /// [`Host::execute_late`] of parts of submissions in one: another begins
    /// at each place of `order` in `parts` ([`joins`]).
    pub(crate) fn execute_parts(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>, parts: &[usize]) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        // Fills behind a copy at its place take its bytes
        // (`place::forwarding`, Grimkaaah f2586ef), before anything looks at
        // what steps read: the cut below, the submission's waits and what is
        // done on the card.
        if place::forwarding() {
            let written = line().written.clone();
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            place::forward_fills(&runs, &written);
        }
        if let Some(cut) = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            card_steps::cut_for(&runs).or_else(|| free_cut(&runs)).or_else(|| cut_waiting(&runs, parts))
        } {
            match split(lists, order, cut) {
                Ok([(lists, order), (more, behind)]) => {
                    let first: Vec<usize> = parts.iter().copied().filter(|&p| p < cut).collect();
                    let second: Vec<usize> = parts.iter().filter(|&&p| p > cut).map(|p| p - cut).collect();
                    self.execute_parts(lists, order, &first);
                    self.execute_parts(more, behind, &second);
                }
                Err((lists, order)) => self.send_late(lists, order, parts),
            }
            return;
        }
        self.send_late(lists, order, parts);
    }

    fn send_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>, parts: &[usize]) {
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, parts, true)
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let sizes = crate::memory::asked_sizes(order.iter().map(|&i| lists[i].host()));
        // Keep the same limit if an `alt` phase ends while this flight is waiting.
        let limit = late_most();
        let most = {
            let mut line = line();
            line.joined(&sent.back, &sent.texture_back_ranges);
            let Line { line: items, lists: flying, target, .. } = &mut *line;
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                own: sent.own,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= limit
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if sizes {
            self.settle(&mut submitted, WHY_READ);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            #[cfg(feature = "diagnostics")]
            let began = std::time::Instant::now();
            relay_wait(WHY_MOST, |line| line.lists < limit);
            #[cfg(feature = "diagnostics")]
            crate::copystat::lap(crate::copystat::C::SettleListsNs, began);
        } else if most {
            #[cfg(feature = "diagnostics")]
            let began = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            #[cfg(feature = "diagnostics")]
            crate::copystat::lap(crate::copystat::C::SettleListsNs, began);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        let _landing = crate::timeline::span("landed lists reset and given back");
        land();
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], &[], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            line.joined(&sent.back, &sent.texture_back_ranges);
            let Line { line: items, lists: flying, target, .. } = &mut *line;
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                own: sent.own,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let _settling = crate::timeline::span("settling flights");
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.narrow.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // SAFETY: the fence and event are ours, under the
                // submission lock; `submitted` is the last value signalled.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            working(false);
                            #[cfg(feature = "diagnostics")]
                            let _card = crate::timeline::card_wait(Some(*submitted));
                            WaitForSingleObject(self.event, INFINITE);
                            stood(waiting);
                            working(true);
                        }
                        Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
                    }
                }
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                hand_back(&back);
                COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(mut flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        let mut lists = flight.lists;
                        for detached in &lists {
                            detached.release_scratch();
                        }
                        if tables_at_landing() {
                            for list in &mut lists {
                                list.release_tables();
                            }
                        }
                        crate::list::tables_early(&mut lists);
                        landed.push((lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// For a submission of `runs` that refreshes the ranges `stale` of host
    /// copies, which flights may be writing and the title wrote too
    /// ([`stale_under_flights`]): the flights land first. `ranges` are
    /// those of the submission's one asking, which is made again.
    ///
    /// With [`card_steps`] only the flights that declared them: a range of
    /// a copy nobody declared any more is refreshed like any other. What
    /// they landed is title memory written, so everything is asked and
    /// looked at again; what is left after three times waits as before.
    fn settle_stale(
        &self,
        submitted: &mut u64,
        runs: &[&HostList],
        mut stale: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        ranges: &[(usize, usize)],
    ) {
        if relay_exact() && card_steps::stage() > 0 {
            for _ in 0..3 {
                #[cfg(feature = "diagnostics")]
                blame_stale(&stale);
                relay_wait(WHY_STALE, |_| stale.iter().all(|(shadow, from, to)| !shadow.writes_pending_in(*from, *to)));
                crate::watch::ask_once(ranges);
                stale = stale_under_flights(runs);
                if stale.is_empty() {
                    return;
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        blame_stale(&stale);
        self.settle(submitted, WHY_STALE);
        crate::watch::ask_once(ranges);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    ///
    /// `parts`: the lists of `runs` at which another part begins
    /// ([`joins`]): those go to the queue in calls of their own, one after the
    /// other, as they would have.
    fn submit<'a>(&'a self, runs: &[&HostList], parts: &[usize], late: bool) -> Option<(std::sync::MutexGuard<'a, u64>, Sent)> {
        // What the heap pass beside the recording finds is waited for here,
        // ahead of the submission lock, and no longer at the list's close:
        // the close waited 15-30 ms a frame for it on a recording thread in
        // the prof runs of 2026-11-25 (17 ms a frame on the Intel run), with
        // the title's Close call held that long. Nothing reads a list's
        // refreshes before this (`joinable` and `will_meet` pass them over),
        // and its descriptors are only read once it runs.
        for list in runs {
            list.take_coming();
        }
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut stage = crate::timeline::span("submit: the submission lock");
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let was_working = working(true);
        submitting(1, runs.len());
        // With [`heap_sync_first`]: the heaps' records begun now, beside the
        // planning of steps done on the card and the look at flights.
        let first_heaps = (late && heap_sync_first()).then(crate::descriptor::begin_sync_heaps);
        #[cfg(feature = "diagnostics")]
        let ahead = Split::begin();
        #[cfg(feature = "diagnostics")]
        SPLIT[11].fetch_add(called.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // `VICTORY_ASK_FIRST`: the submission's one asking begun now, on a
        // thread of its own, beside the planning and the look at flights
        // below; it is waited for where it was made before.
        let asked_first = (late && crate::watch::first() && !upcheck::on()).then(|| {
            let mut spans: ByCopy<(usize, usize)> = ByCopy::default();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && shadow.asked_ahead()
                    {
                        let span = spans
                            .entry(std::sync::Arc::as_ptr(shadow) as usize)
                            .or_insert((*address, *address + *len));
                        *span = (span.0.min(*address), span.1.max(*address + *len));
                    }
                }
            }
            crate::watch::ask_first(spans.into_values().collect())
        });
        // Whether flights were waited for, or settled for the sweep,
        // meanwhile.
        let mut waited_for_flights = false;
        let mut settled_for_sweep = false;
        stage.next("submit: steps given to the card");
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        // Or the reads are done on the card, behind the flights.
        let own = if late { self.card_steps(runs) } else { None };
        #[cfg(feature = "diagnostics")]
        ahead.part(12);
        stage.next("submit: waiting for flights it reads from");
        let mut waited_for_flights = false;
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if own.is_none()
            && let Some(why) = late_blocked(runs)
        {
            waited_for_flights = true;
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                relay_wait(why, |line| {
                    blocked_by(line.blocking(), runs, false).is_none() && !card_steps::copied_read(&card_steps::copies(line), runs)
                });
            } else {
                self.settle(&mut submitted, why);
            }
            card_steps::waited(true);
        } else if own.is_none() {
            card_steps::waited(false);
        }
        #[cfg(feature = "diagnostics")]
        ahead.part(13);
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted.
        // Renderer port #7 (from renderer patch #4): the round asked for when
        // the title submitted it, if that is known (`crate::port_journal`).
        // With [`heap_sync_first`] the round begun first is taken, or after a
        // wait for flights, whose landings wrote title memory, another is
        // begun, as the first saw them; the first is not waited for.
        let heaps_on = crate::port_journal::on();
        let heaps_synced = match (first_heaps, waited_for_flights) {
            (Some(first), false) => {
                HEAP_SYNC_FIRST[0].fetch_add(1, Ordering::Relaxed);
                first
            }
            (first, _) => {
                if let Some(first) = first {
                    HEAP_SYNC_FIRST[1].fetch_add(1, Ordering::Relaxed);
                    std::mem::forget(first);
                }
                match crate::port_journal::needed() {
                    0 => crate::descriptor::begin_sync_heaps(),
                    mark => crate::descriptor::begin_sync_heaps_after(mark),
                }
            }
        };
        // That list runs first, and is one of the submission's from here on.
        let with: Vec<&HostList>;
        let runs = match &own {
            Some(own) => {
                with = std::iter::once(own.list()).chain(runs.iter().copied()).collect();
                &with[..]
            }
            None => runs,
        };
        let parts: Vec<usize> = parts.iter().map(|&n| n + usize::from(own.is_some())).collect();
        stage.next("submit: the sweep");
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            if upcheck::on() {
                upcheck::at_rest(self);
            }
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            settled_for_sweep = true;
            if upcheck::on() {
                upcheck::at_rest(self);
            }
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        stage.next("submit: gathering what to ask about");
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        ahead.part(14);
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::reused();
        #[cfg(feature = "diagnostics")]
        let split = Split::begin();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            batch.copy_spans(runs);
            let Batch { spans, ranges, .. } = &mut batch;
            ranges.clear();
            ranges.extend(spans.iter().filter(|s| s.0.asked_ahead()).map(|s| (s.1, s.2)));
            #[cfg(feature = "diagnostics")]
            SPLIT[8].fetch_add(split.so_far(), Ordering::Relaxed);
            stage.next("submit: asking the write watch");
            match asked_first {
                // Begun beside the first steps: waited for here. Made again,
                // whole, if flights were waited for or settled since (what
                // they landed is title memory written); else only what it
                // does not hold is asked: the ranges of the list of steps
                // done on the card, which was made after it.
                Some(first) => {
                    first.wait();
                    ASKED_FIRST[0].fetch_add(1, Ordering::Relaxed);
                    if waited_for_flights || settled_for_sweep {
                        ASKED_FIRST[1].fetch_add(1, Ordering::Relaxed);
                        crate::watch::ask_once(&ranges);
                    } else if crate::watch::ask_more(&ranges) {
                        ASKED_FIRST[2].fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => crate::watch::ask_once(&ranges),
            }
            #[cfg(feature = "diagnostics")]
            split.part(0);
            stage.next("submit: title bytes under flights' writes");
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if upcheck::on() {
                upcheck::order(runs);
                if behind {
                    upcheck::reads(runs);
                }
            }
            let stale = if behind { batch.stale_under_flights() } else { Vec::new() };
            if !stale.is_empty() {
                let settling = std::time::Instant::now();
                self.settle_stale(&mut submitted, runs, stale, &batch.ranges);
                settled += settling.elapsed();
            }
        } else if behind && !stale_under_flights(runs).is_empty() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        #[cfg(feature = "diagnostics")]
        split.part(1);
        // What flights in the air may write.
        let air = line().written.clone();
        stage.next("submit: refreshes ahead of writes");
        if refresh_front() {
            batch.refresh_front(runs, &parts);
        } else if ahead_of_writes() {
            batch.refresh_ahead(runs, |list, i| own.as_ref().is_some_and(|own| own.takes(list, i)));
        }
        stage.next("submit: the lists' steps");
        let one_call = late && !parts.is_empty() && join_one_call();
        batch.forwards = place::forwards(&air, runs);
        place::count_written_ahead(runs, &batch.forwards);
        for (n, list) in runs.iter().enumerate() {
            list.run_deferred(&mut batch, &air, n, |i| own.as_ref().is_some_and(|own| own.takes(list, i)));
            firsts.push(lists.len());
            // (None unless `VICTORY_PRELUDE_UPLOADS=1`.)
            let uploads = list.prelude_uploads(&mut batch);
            if let Some(prelude) = list.prelude(one_call && parts.contains(&n), &uploads) {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.fill();
        stage.next("submit: the last refreshes");
        batch.flush();
        crate::watch::ask_once_end();
        #[cfg(feature = "diagnostics")]
        split.part(2);
        stage.next("submit: snapshots");
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        // (With `VICTORY_SNAP_BESIDE=1` on the kept threads, while this one
        // goes on to the copies in the card's memory.)
        #[cfg(feature = "diagnostics")]
        let snapshots = std::time::Instant::now();
        let mut snapping = None;
        if snap_beside() {
            snapping = Some(crate::shadow::snapshots_beside());
        } else {
            crate::shadow::take_snapshots();
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        // Waits for the snapshots; its time is counted with theirs.
        let snapped = |snapping: &mut Option<crate::shadow::Beside>| {
            if let Some(snapping) = snapping.take() {
                #[cfg(feature = "diagnostics")]
                let waiting = std::time::Instant::now();
                snapping.join();
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::SnapshotUs, waiting.elapsed().as_micros() as u64);
            }
        };
        #[cfg(feature = "diagnostics")]
        split.part(3);
        stage.next("submit: copies in the card's memory");
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if flown && self.card_results_low() {
            snapped(&mut snapping);
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
            #[cfg(feature = "diagnostics")]
            crate::copystat::lap(crate::copystat::C::SettleResultsNs, settling);
        }
        // What shaders may write through views they pick: last, so that no
        // refresh of this submission leaves it alone.
        let picked = crate::picked::declare(runs);
        for (shadow, at, len) in &picked.written {
            shadow.writing(*at, *len);
        }
        let declared = declared_writes(runs);
        let card = self.card_before(&declared, &picked, flown);
        // (Before anything writes the copies again.)
        snapped(&mut snapping);
        // The sends are taken: from here on the references hold what the
        // copies in the lists put.
        for (shadow, at, bytes) in batch.placed.drain(..) {
            shadow.placed_copy(at, &bytes);
        }
        // A title list can leave a buffer promoted to COPY_DEST. Buffers
        // decay to COMMON between ExecuteCommandLists calls, so submit
        // the compare separately before signalling this submission's fence.
        // Keep it in the list array so the timing end markers follow it.
        let compare_at = card.as_ref().and_then(|c| c.check.clone()).map(|list| {
            let at = lists.len();
            lists.push(Some(list));
            at
        });
        #[cfg(feature = "diagnostics")]
        split.part(4);
        stage.next("submit: waiting for the heaps' records");
        let heaps_began = std::time::Instant::now();
        drop(heaps_synced);
        crate::port_journal::heaps_waited(heaps_began.elapsed().as_nanos() as u64, heaps_on);
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run.
        crate::descriptor::make_waiting();
        #[cfg(feature = "diagnostics")]
        split.part(15);
        let prepared = std::time::Instant::now();
        stage.next("submit: sending");
        // When it runs on the card, for the timeline.
        #[cfg(feature = "diagnostics")]
        let stamp = stamps::take();
        #[cfg(feature = "diagnostics")]
        let compare_at = if let Some((ahead, behind)) = stamp.and_then(stamps::lists) {
            lists.insert(0, Some(ahead));
            lists.push(Some(behind));
            firsts.iter_mut().for_each(|first| *first += 1);
            compare_at.map(|at| at + 1)
        } else {
            compare_at
        };
        #[cfg(feature = "diagnostics")]
        let sending = crate::timeline::ticks();
        if late {
            *submitted += 1;
            for list in runs {
                list.preludes_sent(*submitted);
            }
            #[cfg(feature = "diagnostics")]
            let compare_at = if let Some([front, behind]) = self.span_lists(*submitted) {
                lists.insert(0, Some(front));
                lists.push(Some(behind));
                firsts.iter_mut().for_each(|first| *first += 1);
                compare_at.map(|at| at + 1)
            } else {
                compare_at
            };
            let (lists, compare) = lists.split_at(compare_at.unwrap_or(lists.len()));
            // With [`join_one_call`] the parts' barriers stand in for the cuts.
            let cuts: Vec<usize> = if one_call { Vec::new() } else { parts.iter().filter_map(|&n| firsts.get(n).copied()).collect() };
            // Fills on learned pages, done when the card gets here
            // ([`card_steps::put_off`]).
            if let Some(fills) = own.as_ref().map(|own| own.put_off()).filter(|fills| !fills.is_empty())
                && let Some((fence, value)) = card_steps::put_off::send(&self.device, &self.fence, *submitted - 1, fills)
            {
                // SAFETY: a live fence its thread always signals.
                let _ = unsafe { self.queue.Wait(&fence, value) };
            }
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                let mut from = 0;
                for to in cuts.into_iter().chain([lists.len()]) {
                    before_send(&self.queue);
                    self.queue.ExecuteCommandLists(&lists[from..to]);
                    from = to;
                }
                if !compare.is_empty() {
                    self.queue.ExecuteCommandLists(compare);
                }
                self.queue.Signal(&self.fence, *submitted)
            };
            awake::sent(&self.device);
            #[cfg(feature = "diagnostics")]
            crate::timeline::sent(sending, *submitted, stamp);
            self.card_sent();
            stage.next("submit: noting what is in the air");
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            #[cfg(feature = "diagnostics")]
            split.part(5);
            let back = crate::picked::with(declared, &picked.written);
            let mut texture_back_ranges = Vec::new();
            for list in runs {
                texture_back_ranges.extend(list.deferred().texture_backs.iter().map(crate::texback::Back::range));
            }
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                own,
                texture_back_ranges,
                writes,
            };
            batch.give_back();
            working(was_working);
            return Some((submitted, sent));
        }
        let (lists, compare) = lists.split_at(compare_at.unwrap_or(lists.len()));
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .enumerate()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                // Include any leading timeline marker in the first timed group.
                .map(|((i, &a), &b)| &lists[if i == 0 { 0 } else { a }..b])
                .collect()
        } else {
            vec![lists]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [lists];
        submitting(2, runs.len());
        for list in runs {
            list.preludes_sent(*submitted + groups.len() as u64);
        }
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // While the overlay shows, between two timestamps (its GPU
            // busy time).
            let timed = crate::overlay::time_lists(&self.device, &self.queue, group);
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                before_send(&self.queue);
                self.queue
                    .ExecuteCommandLists(timed.as_deref().unwrap_or(group));
                if i + 1 == groups.len() && !compare.is_empty() {
                    self.queue.ExecuteCommandLists(compare);
                }
                self.card_sent();
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                #[cfg(feature = "diagnostics")]
                crate::timeline::sent(sending, *submitted, stamp);
                match waited {
                    Ok(()) => {
                        working(false);
                        #[cfg(feature = "diagnostics")]
                        let _card = crate::timeline::card_wait(Some(*submitted));
                        WaitForSingleObject(self.event, INFINITE);
                        stood(started);
                        working(true);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            if timed.is_some() {
                crate::overlay::time_read();
            }
            // With timing on each group is one of `runs`: every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        stage.next("submit: copying back");
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        let back = crate::picked::with(declared, &picked.written);
        hand_back(&back);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        batch.give_back();
        working(was_working);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories read as far as they
        // are mapped (`readable_history`).
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
                #[cfg(feature = "diagnostics")]
                text.push_str(&crate::book::at_loss(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            if dred_faults() {
                return text;
            }
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history = readable_history(p.pCommandHistory, p.BreadcrumbCount);
                    text.push_str(&stopped_at(&history, p.BreadcrumbCount, done));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    let named = [
                        (!p.pCommandListDebugNameW.is_null()).then(|| p.pCommandListDebugNameW.to_string().unwrap_or_default()),
                        (!p.pCommandListDebugNameA.is_null()).then(|| {
                            std::ffi::CStr::from_ptr(p.pCommandListDebugNameA.cast()).to_string_lossy().into_owned()
                        }),
                    ]
                    .into_iter()
                    .flatten()
                    .next()
                    .unwrap_or_default();
                    text.push_str(&format!(" of \"{named}\""));
                    // Not one of the caller's: by the name.
                    #[cfg(feature = "diagnostics")]
                    let by_name = list_number(&named).and_then(|n| {
                        let all = LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner());
                        all.get(&n).map(|(notes, pipelines, _)| (notes.upgrade(), pipelines.upgrade()))
                    });
                    #[cfg(feature = "diagnostics")]
                    let indirects = run.map(|r| r.indirects.clone()).or_else(|| {
                        let n = list_number(&named)?;
                        LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner()).get(&n)?.2.upgrade()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    #[allow(clippy::type_complexity)]
                    let by_name: Option<(Option<()>, Option<std::sync::Arc<Mutex<std::collections::BTreeSet<String>>>>)> = None;
                    #[cfg(feature = "diagnostics")]
                    let noted = run.map(|r| r.notes.clone()).or_else(|| by_name.as_ref().and_then(|b| b.0.clone()));
                    let bound = run.map(|r| r.pipelines.clone()).or_else(|| by_name.as_ref().and_then(|b| b.1.clone()));
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = noted.map_or_else(Vec::new, |notes| {
                        let notes = notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 && (done as usize) < history.len() {
                        text.push_str(&in_flight(&history, &markers, done as usize));
                    }
                    #[cfg(feature = "diagnostics")]
                    if let Some(indirects) = indirects {
                        text.push_str(&indirects_around(&history, done as usize, &indirects));
                    }
                    // (Without markers a list that stopped midway says what
                    // it may have been at only by what it binds.)
                    if (done == 0 || markers.is_empty())
                        && let Some(bound) = bound
                    {
                        let pipelines = bound.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.rtv_start + (slot * self.rtv_size) as usize,
            };
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.dsv_start + (slot * self.dsv_size) as usize,
            };
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() && !gpu_times() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Every noted command's time on the card to the sums by pipeline, once
    /// the list's last submission ran (`VICTORY_GPU_TIMES`). Of a list sent
    /// twice since its reset only the last run counts, and a list closed
    /// and never sent counts nothing: its first and last stamps are zeroed
    /// once read.
    #[cfg(feature = "diagnostics")]
    fn times_taken(&self) {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let count = notes.len().min(HALF as usize - 1);
        let (Some((_, back)), Some(host), 1..) = (self.stamps(), host(), count) else {
            return;
        };
        let half = HALF as usize;
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (half + count) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into; the
        // card writes it again only after the next close.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return;
            }
            let mapped = std::slice::from_raw_parts_mut(data.cast::<u64>(), half + count);
            let stamps = mapped.to_vec();
            (mapped[0], mapped[count]) = (0, 0);
            back.Unmap(0, Some(&range));
            stamps
        };
        if stamps[0] == 0 || stamps[count] == 0 {
            return;
        }
        // SAFETY: a plain query.
        let frequency = unsafe { host.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
        let ms = |ticks: u64| ticks as f64 * 1e3 / frequency as f64;
        // What runs between a command's end and the next one's start (the
        // next one's barriers, clamp and copies) is summed as "before" it.
        for (i, (op, _)) in notes.iter().take(count).enumerate() {
            let done = stamps[half + i];
            crate::copystat::command_took(op, ms(done.saturating_sub(stamps[i])));
            let between = stamps[i + 1].saturating_sub(done);
            if between > 0 {
                match notes.get(i + 1).filter(|_| i + 1 < count) {
                    Some((next, _)) => crate::copystat::command_took(&format!("before {next}"), ms(between)),
                    None => crate::copystat::command_took("closing list", ms(between)),
                }
            }
        }
    }

    pub(crate) fn close(&self) {
        let _closing = crate::timeline::span("closing a host list");
        self.left_open[0].store(0, Ordering::Release);
        let mut lap = Lap::begin();
        // Each step of the close is timed as a command of its own
        // (`VICTORY_GPU_TIMES`), not summed into "closing list".
        #[cfg(feature = "diagnostics")]
        self.note_op(|| "closing: twins to depth".into());
        self.twins_to_depth();
        lap.to(0);
        #[cfg(feature = "diagnostics")]
        self.note_op(|| "closing: textures back".into());
        self.textures_back();
        lap.to(1);
        #[cfg(feature = "diagnostics")]
        self.note_op(|| "closing: indirect arguments common".into());
        let scratch = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((scratch, _)) = scratch {
            if close_lean() >= 2 {
                self.deferred().indirect_readable = 0;
            } else {
                self.indirect_common(&scratch);
            }
        }
        lap.to(2);
        #[cfg(feature = "diagnostics")]
        self.note_op(|| "closing: batches common".into());
        self.batches_common();
        lap.to(3);
        #[cfg(feature = "diagnostics")]
        self.note_op(|| "closing: card budget out".into());
        self.card_budget_out();
        lap.to(4);
        if coming_at_close() {
            self.take_coming();
        }
        lap.to(5);
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let noted = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            let count = noted.min(timed());
            let halves = gpu_times() && !timing() && count > 0;
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                if halves && noted == count && !self.done.swap(true, Ordering::Relaxed) {
                    self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, HALF + count - 1);
                }
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
                if halves {
                    self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, HALF, count, back, u64::from(HALF) * 8);
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        self.clamp_stamps.resolve(&self.list);
        #[cfg(feature = "diagnostics")]
        self.pstats.resolve(&self.list);
        lap.to(6);
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
        lap.to(7);
    }

    /// `work` finds, on a thread of ours, refreshes that belong at this
    /// place among the list's steps; they are there when the list is closed.
    pub(crate) fn refreshes_coming(&self, work: impl FnOnce() -> Vec<aside::Refresh> + Send + 'static) {
        self.take_coming();
        let at = self.deferred().copies.len();
        let coming = aside::begin(work);
        *self.coming.lock().unwrap_or_else(|e| e.into_inner()) = Some((at, coming));
    }

    /// The refreshes that were still to come are put among the steps: when
    /// the list is submitted, or at its close with `VICTORY_COMING_AT_CLOSE=1`.
    fn take_coming(&self) {
        let Some((at, coming)) = self.coming.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        let found = coming.wait();
        let mut deferred = self.deferred();
        let at = at.min(deferred.copies.len());
        deferred
            .copies
            .splice(at..at, found.into_iter().map(|(shadow, from, to)| Step::Refresh(shadow, from as usize, (to - from) as usize)));
    }

    /// A new host list for one of the title's lists (`crate::list::detach`),
    /// open and empty as it was made or reset: left so, for the title's next
    /// call on its list, a reset, which then has nothing to undo
    /// ([`fresh_open`]); or closed now, as before.
    ///
    /// Each of the title's lists got a new host list at every submission,
    /// closed at once on the submitting thread (an empty list's close, some
    /// 5 us in the driver, 200 a frame) and reset again by the title's reset
    /// (allocator and list) after the reset that made it ready for reuse.
    pub(crate) fn leave_open(&self) {
        if !fresh_open() {
            self.close();
            return;
        }
        // (The marks first: a reset that sees the list left open sees them.)
        self.left_open[1].store(self.epoch() + 1, Ordering::Relaxed);
        self.left_open[0].store(self.title_calls() + 1, Ordering::Release);
        FRESH_OPEN[0].fetch_add(1, Ordering::Relaxed);
    }

    /// A host list taken from one of the title's lists to run
    /// (`crate::list::detach`): closed first if it was left open
    /// ([`HostList::leave_open`]), as it is when the title submits its list
    /// again without a reset.
    pub(crate) fn close_if_left_open(&self) {
        if self.left_open[0].load(Ordering::Acquire) != 0 {
            FRESH_OPEN[2].fetch_add(1, Ordering::Relaxed);
            self.close();
        }
    }

    /// Whether the list is as [`HostList::leave_open`] left it: nothing was
    /// recorded into it since, as far as its title calls and its epoch say.
    /// It is no longer marked so; one left open that something was recorded
    /// into is closed here, for the reset that follows.
    fn still_left_open(&self) -> bool {
        let calls = self.left_open[0].swap(0, Ordering::AcqRel);
        let epoch = self.left_open[1].swap(0, Ordering::Relaxed);
        if calls == 0 {
            return false;
        }
        if calls == self.title_calls() + 1 && epoch == self.epoch() + 1 {
            return true;
        }
        FRESH_OPEN[2].fetch_add(1, Ordering::Relaxed);
        // SAFETY: our list, open; what went into it goes with the reset.
        let _ = unsafe { self.list.Close() };
        false
    }

    /// Its submission has landed: the card read its scratch chunks for the last time, so
    /// the next lists may take them now ([`spare_what_fits`]) rather than at its reset,
    /// after the bury lag (16 or 64 submissions, [`bury_lag`]; a list held its chunks all
    /// that time, 130 MB for 42 landed lists in the memory bench at only 8 lists a frame).
    /// Only with shared chunks ([`scratch_pool`]); `VICTORY_SCRATCH_AT_LANDING=0`: at the
    /// reset, as before.
    pub(crate) fn release_scratch(&self) {
        if !scratch_pool() || !scratch_at_landing() {
            return;
        }
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        (scratch.current, scratch.used) = (0, 0);
        let chunks = std::mem::take(&mut scratch.chunks);
        scratch.chunks = spare_what_fits(chunks);
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        // Before the epoch moves on below.
        let left_open = self.still_left_open();
        self.changed_something();
        self.has_placed.store(false, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let _resetting = crate::timeline::span("resetting a host list");
        self._alive.used();
        // A list closed and never submitted: the pass may still be writing
        // its descriptors.
        if let Some((_, coming)) = self.coming.lock().unwrap_or_else(|e| e.into_inner()).take() {
            coming.wait();
        }
        self.deferred().clear();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.has_twins.store(false, Ordering::Relaxed);
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.indirects.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        if gpu_times() && !timing() {
            self.times_taken();
        }
        #[cfg(feature = "diagnostics")]
        {
            let mut notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
            self.pstats.read(&notes);
            notes.clear();
        }
        #[cfg(feature = "diagnostics")]
        self.clamp_stamps.read();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        {
            let mut readbacks = self.readbacks.lock().unwrap_or_else(|e| e.into_inner());
            (readbacks.current, readbacks.used) = (0, 0);
        }
        // Open and empty since it was made or last reset: the allocator holds
        // nothing, the list nothing recorded (`HostList::leave_open`).
        if left_open {
            FRESH_OPEN[1].fetch_add(1, Ordering::Relaxed);
        } else {
            // SAFETY: the list is not executing.
            if let Err(e) = unsafe { self.allocator.Reset() } {
                fail(format_args!("{}", reset_error("allocator", &e)));
            }
            // SAFETY: the list is not executing and its allocator is ready.
            if let Err(e) = unsafe { self.list.Reset(&self.allocator, None) } {
                fail(format_args!("{}", reset_error("list", &e)));
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// Something changed in this list that bringing a draw's textures and
    /// buffers up to date depends on: a texture's state, a twin, what was
    /// refreshed.
    pub(crate) fn changed_something(&self) {
        self.epoch.store(next_epoch(), Ordering::Relaxed);
    }

    /// See [`HostList::changed_something`].
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// A title call that may record into the list (`crate::list::host_list`).
    pub(crate) fn title_call(&self) {
        self.title_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// The title calls recorded into the list so far.
    pub(crate) fn title_calls(&self) -> u64 {
        self.title_calls.load(Ordering::Relaxed)
    }

    /// Whether steps or texture state changes were recorded since the list
    /// was made or last reset.
    pub(crate) fn has_recorded_work(&self) -> bool {
        !self.deferred().copies.is_empty() || !self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture, the state it leaves and the (legal) state it
    /// takes, if that is a change. The list's first use of a texture records
    /// nothing: its prelude brings the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32, u32)> {
        let copy = crate::resource::host_copy(resource)?;
        // SAFETY: as in `crate::resource::host_resource`; the list's states
        // hold the copy from here on.
        let texture: &'static ID3D12Resource = unsafe { &*std::ptr::from_ref(&copy.texture) };
        let wanted = legal_for(copy.flags(), wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                self.changed_something();
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                if before != wanted {
                    self.changed_something();
                }
                (before != wanted).then_some((texture, before, wanted))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_for(copy.flags(), wanted);
        self.changed_something();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        self.has_twins.store(true, Ordering::Relaxed);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let was = (u.needs, u.behind, u.writing);
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        if was != (u.needs, u.behind, u.writing) || copy || colour {
            self.changed_something();
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Before a draw or dispatch binds anything: every depth texture with a
    /// stencil plane whose depth plane's twin an earlier draw or dispatch of
    /// this list wrote (the tiled lighting's compute writes each pixel's
    /// material class there) gets the twin copied and the depth plane's twin
    /// drawn into the plane now, not when the list closes. Without it the
    /// shaders that read the classes back in the same list see the classes
    /// of the frame before (`twin_use`'s shortcut). True if a pass was
    /// recorded: it set a heap, root signature, pipeline, depth target,
    /// viewport, scissor and topology of its own, and the caller sets the
    /// title's again (`State::after_our_pass`). VICTORY_PLANE_NOW=1, off by
    /// default.
    pub(crate) fn planes_now(&self) -> bool {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        static ON: OnceLock<bool> = OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_PLANE_NOW").is_some_and(|v| v == "1")) {
            return false;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        if twins.is_empty() {
            return false;
        }
        let due: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) && u.twin.plane.is_some())
            .map(|u| {
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone())
            })
            .collect();
        drop(twins);
        for (depth, twin) in &due {
            self.twin_copy(depth, twin, false);
            self.pass_plane(depth, twin);
            // Shaders may read it by a heap index with no view bound: left
            // readable as `twins_to_depth` leaves it.
            self.settle(depth, SHADER_RESOURCE);
        }
        if !due.is_empty() {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!("depth planes written by shaders are drawn into the plane before the next draw or dispatch (VICTORY_PLANE_NOW=1)");
            });
        }
        !due.is_empty()
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. A caller
    /// that goes on with the title's list sets the title's bindings again
    /// (`record_plane_pass`, `State::after_our_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of the title no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        // A target whose ticket went stale (the title reused or remapped the
        // memory it tiles into) is registered again, or its read-back is
        // dropped as one of memory the texture no longer holds.
        let existing = deferred
            .back_wanted
            .iter_mut()
            .find(|target| target.owner() == resource as usize);
        match existing {
            Some(entry) if !entry.fresh() => {
                if let Some(target) = crate::texback::Target::new(resource) {
                    *entry = target;
                }
            }
            None => deferred.back_wanted.extend(crate::texback::Target::new(resource)),
            Some(_) => {}
        }
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            self.settle(&target.copy, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, target) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        if !self.has_twins.load(Ordering::Relaxed) {
            return;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
            self.changed_something();
        }
    }

    /// For tests: whether the list's read-back wants a *fresh* target for
    /// `resource` — one whose ticket has not gone stale. A stale target is
    /// one the title's memory reuse invalidated, which the list must
    /// re-register on the next write of `resource`.
    #[cfg(test)]
    pub(crate) fn back_wanted_fresh(&self, resource: crate::object::This) -> bool {
        self.deferred()
            .back_wanted
            .iter()
            .any(|t| t.owner() == resource as usize && t.fresh())
    }

    /// For tests: how many texture read-backs the list recorded at its close.
    #[cfg(test)]
    pub(crate) fn texture_back_count(&self) -> usize {
        self.deferred().texture_backs.len()
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind. Title
    /// textures whose upload is decided at submission (`uploads`,
    /// `crate::texture::prelude_uploads`) are copied into their host copies
    /// here, right before the list runs (Belobug/VIctory-Belobug-AMD
    /// 5d6bd84).
    /// With `join` the list is the first of a part sent in one call behind
    /// others ([`join_one_call`]): the prelude is made in any case, and first
    /// waits for everything before it ([`join_barriers`]).
    fn prelude(&self, join: bool, uploads: &[(crate::texture::PreludeUpload, bool)]) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        // The uploads decided at submission: each host copy to the copy
        // destination state, and from there to the state the list starts
        // with (back to its real state if the list does not use it).
        let mut aliasing = Vec::new();
        for (upload, first_upload) in uploads {
            let copy = &upload.copy;
            let key = std::sync::Arc::as_ptr(copy) as usize;
            // A texture cut from a block may lie where another did: what the
            // card holds of that one is done with before the first upload
            // into this one.
            if crate::pool::on() && *first_upload {
                aliasing.push(D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                            pResourceBefore: std::mem::ManuallyDrop::new(None),
                            // SAFETY: a borrowed pointer the barrier never releases.
                            pResourceAfter: unsafe { std::mem::transmute_copy(&copy.texture) },
                        }),
                    },
                });
            }
            let real = copying.get(&key).copied().unwrap_or_else(|| copy.state.load(Ordering::Relaxed));
            if real != COPY_DEST {
                to_copy.push(transition(&copy.texture, host_state(real), host_state(COPY_DEST)));
                if !states.contains_key(&key) {
                    barriers.push(transition(&copy.texture, host_state(COPY_DEST), host_state(real)));
                }
            }
            copying.insert(key, COPY_DEST);
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        if !join && barriers.is_empty() && syncs.is_empty() && uploads.is_empty() {
            return None;
        }
        // SAFETY: a plain query.
        let done = host().map_or(0, |h| unsafe { h.fence.GetCompletedValue() });
        let free = self.preludes.lock().unwrap_or_else(|e| e.into_inner()).take(done);
        let Some((allocator, list)) = free.or_else(|| host().and_then(|h| h.make_prelude())) else {
            fail(format_args!("no host list for a state prelude"));
        };
        // SAFETY: the prelude is not executing (never queued, or the
        // submission that ran it is done); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(&allocator, None)?;
                if join {
                    list.ResourceBarrier(&join_barriers());
                }
                if !aliasing.is_empty() {
                    list.ResourceBarrier(&aliasing);
                }
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(&list, &depth.texture, twin, *colour);
                }
                for (upload, _) in uploads {
                    upload.record(&list);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        let queued = list.cast().ok();
        self.preludes.lock().unwrap_or_else(|e| e.into_inner()).flight.push((u64::MAX, (allocator, list)));
        queued
    }

    /// The preludes recorded for this submission are run by submission
    /// `number` ([`Preludes::sent`]).
    fn preludes_sent(&self, number: u64) {
        self.preludes.lock().unwrap_or_else(|e| e.into_inner()).sent(number);
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// The command noted last was just recorded (`VICTORY_GPU_TIMES`): its
    /// time on the card ends here, and what is recorded before the next
    /// note counts as that one's setup.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn noted_done(&self) {
        if !gpu_times() || timing() {
            return;
        }
        let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
        if let Some((heap, _)) = self.stamps()
            && (1..=timed()).contains(&at)
            && !self.done.swap(true, Ordering::Relaxed)
        {
            // SAFETY: an open list and our query heap.
            unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, HALF + at - 1) };
        }
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        // A timing run stamps and keeps the texts without the markers, so
        // the sums by pipeline need no device-removed data.
        let dred = dred_on();
        if !dred && !timing() && !gpu_times() && !pstats::on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() {
                let most = timed();
                // The last command's end, if it has none of its own.
                let done = self.done.swap(false, Ordering::Relaxed);
                // SAFETY: an open list and our query heap.
                unsafe {
                    if gpu_times() && !timing() && (1..=most).contains(&at) && !done {
                        self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, HALF + at - 1);
                    }
                    if at < most {
                        self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at);
                    }
                }
            }
            if pstats::on() {
                self.pstats.at_note(&self.list, at);
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            if dred {
                let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
                const UNICODE: u32 = 0;
                // SAFETY: an open list; the marker is copied when recorded.
                unsafe {
                    self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
                };
            }
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.textures.is_empty()
                && deferred.builder_memory.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty()
                && deferred.texture_backs.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes an indirect execution, while device-removed data is kept
    /// ([`indirects`]).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn ran_indirect(&self, pipeline: usize, name: &str, kind: u32, count: u32, user: &[u32]) {
        if dred_on() || dred_light() {
            self.indirects.lock().unwrap_or_else(|e| e.into_inner()).note(pipeline, name, kind, count, user);
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    /// The list binds a pipeline: it and its root signature are kept until
    /// the list is reset, and its name noted, the first time only (a kept
    /// pipeline's address names no other).
    pub(crate) fn bind_pipeline(&self, name: &str, pso: &ID3D12PipelineState, root: &ID3D12RootSignature) {
        {
            let mut deferred = self.deferred();
            if !deferred.bound.insert(pso.as_raw() as usize) {
                return;
            }
            deferred.kept.push(pso.clone().into());
            deferred.kept.push(root.clone().into());
        }
        self.used_pipeline(name);
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        let copy_dst = dst as usize;
        deferred.placed_written.retain(|(a, b, _)| *b <= copy_dst || *a >= copy_dst.saturating_add(len));
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.title_written();
        self.changed_something();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len, None));
    }

    /// The host address of `len` bytes of scratch memory that hold the title
    /// memory at `src` as of the list's submission: one fill for every ask
    /// until the list next writes title memory.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        let known = self.deferred().filled.get(&key).copied();
        if known.is_some() {
            return known;
        }
        let (gpu, to) = self.scratch(len)?;
        self.defer_fill(to, src, len as usize);
        self.deferred().filled.insert(key, gpu);
        Some(gpu)
    }

    /// As [`HostList::defer_fill`], for upload memory that only `slot` (a
    /// title heap's address, and the slot) shows.
    pub(crate) fn defer_constants(&self, dst: *mut u8, src: *const u8, len: usize, slot: (usize, u32)) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len, Some(card_steps::FillTo::Slot(slot.0, slot.1))));
    }

    /// As [`HostList::defer_fill`], for the beginning of a buffer
    /// [`HostList::upload`] made.
    pub(crate) fn defer_fill_buffer(&self, buffer: &ID3D12Resource, dst: *mut u8, src: *const u8, len: usize) {
        let to = (card_steps::stage() > 1).then(|| card_steps::FillTo::Buffer(buffer.clone()));
        self.deferred().copies.push(Step::Fill(dst as usize, src as usize, len, to));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    /// Declared once until the list next writes title memory (but with
    /// `VICTORY_DECLARE_ONCE=0`): the dispatches of a frame declared the
    /// same ranges thousands of times, each a step of its own.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        if let Some(log) = &mut deferred.batch_writes {
            log.note(address as u64, len as u64);
        }
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) || !declare_once() {
            deferred.copies.push(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// A batched clamp was recorded: what shaders are declared to write
    /// from here on is noted for [`HostList::batch_written`]
    /// (`VICTORY_INDIRECT_BATCH=3`).
    pub(crate) fn batch_opened(&self) {
        self.deferred().batch_writes = Some(crate::indirect::batch::WriteLog::default());
    }

    /// Shaders of the list write `len` bytes from title `address` on, in
    /// memory no step is deferred for (title memory itself).
    pub(crate) fn note_batch_write(&self, address: u64, len: u64) {
        if let Some(log) = &mut self.deferred().batch_writes {
            log.note(address, len);
        }
    }

    /// Whether shaders may have written any of `len` bytes from title
    /// `address` on since the last batched clamp (true if none was noted).
    pub(crate) fn batch_written(&self, address: u64, len: u64) -> bool {
        self.deferred().batch_writes.as_ref().is_none_or(|log| log.touches(address, len))
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. `mapped` is where they are
    /// when the buffer is a read-back chunk ([`HostList::readback_space`]),
    /// else null and they start the buffer. The buffer lives until the list
    /// is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, mapped: *const u8, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        self.deferred().backs.push((buffer, mapped as usize, dst as usize, [row, rows, pitch]));
    }

    /// `len` bytes of read-back memory for a texture-to-buffer copy, at an
    /// offset that is a multiple of 512 (a placed footprint's alignment):
    /// the buffer, the offset in it and the CPU pointer there. The list's
    /// read backs share chunks of up to 4 MiB, reused after a reset; `None`
    /// for a larger copy (it takes a buffer of its own) or with
    /// `VICTORY_READ_BACK_CHUNKS=0`.
    pub(crate) fn readback_space(&self, len: u64) -> Option<(ID3D12Resource, u64, *const u8)> {
        let len = len.max(1).checked_next_multiple_of(512)?;
        if len > SCRATCH_CHUNK || !crate::toggled!("VICTORY_READ_BACK_CHUNKS") {
            return None;
        }
        let host = host()?;
        let mut s = self.readbacks.lock().unwrap_or_else(|e| e.into_inner());
        let (chunk, at) = s.allocate(host, len)?;
        let chunk = &s.chunks[chunk];
        Some((chunk.buffer.clone(), at, (chunk.mapped + at as usize) as *const u8))
    }

    /// Timestamps resolved into `buffer` (at `mapped` when it is a chunk
    /// mapped for good, else null and from its start): once the list has
    /// run, `count` of them go to `dst` in the title's clock.
    pub(crate) fn defer_stamps(&self, buffer: ID3D12Resource, mapped: *const u8, dst: *mut u8, count: usize) {
        self.deferred().stamps.push((buffer, mapped as usize, dst as usize, count));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of at least `size` bytes, alive until the list
    /// is reset: from a pool by size class ([`UPLOAD_SPARE`]), so that it may
    /// hold what an earlier list wrote. Callers write or zero the bytes they
    /// use and copy by offset, never the whole buffer.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        // With [`card_steps`] the card may be the one that fills it.
        if card_steps::stage() > 1 && upload_pool() && size <= 16 << 20 {
            let class = crate::shadow::view_pool_size(size);
            let (buffer, data) = match crate::shadow::view_pool_take(class) {
                Some(taken) => taken,
                None => host.shared_buffer(class)?,
            };
            name(&buffer, || format!("list upload {size:#x}"));
            let mut deferred = self.deferred();
            ALIVE[6].fetch_add(1, Ordering::Relaxed);
            ALIVE[7].fetch_add(class, Ordering::Relaxed);
            deferred.upload_bytes += class;
            deferred.view_pooled.push((buffer.clone(), data as usize, class));
            return Some((buffer, data));
        }
        let shared = card_steps::stage() > 1;
        if upload_pool_most() > 0 {
            let class = size.max(64 << 10).next_power_of_two();
            let chunk = match take_upload(shared, class) {
                Some(chunk) => chunk,
                None => {
                    let (buffer, data) = if shared { host.shared_buffer(class)? } else { host.upload_buffer(class)? };
                    name(&buffer, || format!("list upload, pooled {class:#x}"));
                    Chunk { buffer, data: data as usize }
                }
            };
            let out = (chunk.buffer.clone(), chunk.data as *mut u8);
            self.deferred().pooled.push((chunk, class, shared));
            return Some(out);
        }
        let (buffer, data) = if shared { host.shared_buffer(size)? } else { host.upload_buffer(size)? };
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let (buffer, at, data) = self.scratch_at(len)?;
        // SAFETY: a live buffer.
        Some((unsafe { buffer.GetGPUVirtualAddress() } + at, data))
    }

    /// [`HostList::scratch`] and the buffer it lies in, for a descriptor
    /// slot to hold: the slot outlives the list's use of it.
    pub(crate) fn scratch_held(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let (buffer, at, data) = self.scratch_at(len)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() } + at;
        Some((buffer, gpu, data))
    }

    /// [`HostList::scratch`]: the buffer, the offset in it and the CPU
    /// pointer.
    fn scratch_at(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            self.deferred().hold_upload(buffer.clone(), len);
            return Some((buffer, 0, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let (chunk, size) = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, buffer) = (s.chunks[s.current].0.data, s.chunks[s.current].0.buffer.clone());
        let at = s.used;
        s.used += len;
        Some((buffer, at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let card = self.card_budget.get_or_init(|| self.make_card_budget()).as_ref();
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all; the records refused, the
            // largest cost; the records that asked for nothing; with the
            // card's copy, as
            // the list's end copied them here): taken, then zeroed.
            let notes = Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                let (refused, biggest) = (words.add(3).read_volatile(), words.add(4).read_volatile());
                BUDGET_NOTES[0].fetch_add(u64::from(refused), Ordering::Relaxed);
                BUDGET_NOTES[1].fetch_max(u64::from(biggest), Ordering::Relaxed);
                BUDGET_NOTES[2].fetch_add(u64::from(words.add(5).read_volatile()), Ordering::Relaxed);
                for k in 0..6 {
                    words.add(k).write_volatile(0);
                }
            });
            deferred.copies.push(Step::Host(notes, (0, 0)));
            drop(deferred);
            if let Some(card) = card {
                self.zero_card_budget(card);
            }
        }
        // SAFETY: a live buffer.
        Some(unsafe { card.unwrap_or(buffer).GetGPUVirtualAddress() })
    }

    /// The budget's words in the card's own memory ([`card_budget`]), if
    /// the list can zero them itself (`WriteBufferImmediate`).
    fn make_card_budget(&self) -> Option<ID3D12Resource> {
        if !card_budget() {
            return None;
        }
        self.list.cast::<ID3D12GraphicsCommandList2>().ok()?;
        let buffer = host()?.uav_buffer(256)?;
        name(&buffer, || "indirect budget".to_owned());
        Some(buffer)
    }

    /// Records, before the list's first clamp, the zeroing of the card's
    /// copy of the budget: common, written as a copy, then left for
    /// unordered access until the list closes ([`HostList::card_budget_out`]).
    fn zero_card_budget(&self, card: &ID3D12Resource) {
        let Ok(list2) = self.list.cast::<ID3D12GraphicsCommandList2>() else {
            return;
        };
        // SAFETY: a live buffer.
        let at = unsafe { card.GetGPUVirtualAddress() };
        let zeros = [0u64, 4, 8, 12, 16, 20].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at + offset,
            Value: 0,
        });
        // SAFETY: an open list; the buffer is the list's own and in the
        // common state at the start of every recording.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }

    /// Before the list closes, if this recording charged the card's copy
    /// of the budget: its notes on mesh records go to the mapped words
    /// (read at the next submission), and it goes back to common.
    fn card_budget_out(&self) {
        if !self.deferred().budget_zeroed || close_lean() >= 1 {
            return;
        }
        let (Some(Some(card)), Some(Some((mapped, _)))) =
            (self.card_budget.get(), self.budget.get())
        else {
            return;
        };
        // SAFETY: an open list; both buffers are the list's own. The
        // mapped one is promoted from common to the copy destination.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            self.list.CopyBufferRegion(mapped, 4, card, 4, 20);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            // It ends the list common, as every scratch buffer does.
            self.indirect_common(&old_buf);
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Where the next batch of clamped records goes: a device-local buffer
    /// the list keeps from recording to recording, the batch's place in it
    /// and the state the buffer is in. Every batch of a recording has a place
    /// of its own: the executions before it may still read theirs.
    pub(crate) fn batch_place(&self) -> Option<(ID3D12Resource, u32, D3D12_RESOURCE_STATES)> {
        use crate::indirect::batch::{JOBS, SLOT};
        let mut args = self.batch_args.lock().unwrap_or_else(|e| e.into_inner());
        let (buffer, i) = ((args.1 / BATCHES) as usize, args.1 % BATCHES);
        if buffer == args.0.len() {
            let size = u64::from(BATCHES * JOBS * SLOT);
            let made = host()?.uav_buffer(size)?;
            name(&made, || format!("indirect batches {size:#x}"));
            args.0.push(made);
        }
        args.1 += 1;
        // (Its first batch finds it as the recording before left it:
        // `batches_common`.)
        let state = if i == 0 { D3D12_RESOURCE_STATE_COMMON } else { D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT };
        Some((args.0[buffer].clone(), i * JOBS * SLOT, state))
    }

    /// Before the list closes: the scratch buffers of the batched clamps go
    /// back to the common state, where the next recording expects them.
    fn batches_common(&self) {
        let mut args = self.batch_args.lock().unwrap_or_else(|e| e.into_inner());
        let used = std::mem::take(&mut args.1).div_ceil(BATCHES) as usize;
        if close_lean() >= 2 {
            return;
        }
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = args.0[..used]
            .iter()
            .map(|b| transition(b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON))
            .collect();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers this recording left in
            // `INDIRECT_ARGUMENT`.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
    }

    /// Moves `scratch` back to the common state if a clamp left it readable
    /// as indirect arguments ([`HostList::indirect_readable`]): where the
    /// list's first clamp expects it. Done before the list closes and before
    /// a bigger scratch buffer replaces it: one barrier per list (where
    /// there was one per indirect execution), and the list ends as it
    /// begins however it is submitted.
    fn indirect_common(&self, scratch: &ID3D12Resource) {
        {
            let mut deferred = self.deferred();
            if deferred.indirect_readable != scratch.as_raw() as usize {
                return;
            }
            deferred.indirect_readable = 0;
        }
        let barrier = transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        // SAFETY: an open list and a live buffer this recording left in
        // `INDIRECT_ARGUMENT`.
        unsafe { self.list.ResourceBarrier(&[barrier]) };
    }

    /// The state the list's indirect scratch buffer `scratch` is in at this
    /// point of the recording: indirect arguments if a clamp in this
    /// recording left it so ([`HostList::indirect_readable`]), otherwise
    /// common: the state a buffer is made in, and the one a recording
    /// leaves it in when it closes or replaces it ([`HostList::indirect_common`]).
    pub(crate) fn indirect_state(&self, scratch: &ID3D12Resource) -> D3D12_RESOURCE_STATES {
        if self.deferred().indirect_readable == scratch.as_raw() as usize {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        }
    }

    /// A clamp recorded in this list left `scratch` readable as indirect
    /// arguments. It stays so for every execution that reads it, until the
    /// next clamp moves it back to unordered access.
    pub(crate) fn indirect_readable(&self, scratch: &ID3D12Resource) {
        self.deferred().indirect_readable = scratch.as_raw() as usize;
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Whether this recording first requests this builder operation and byte address.
    pub(crate) fn first_builder_memory_use(&self, key: BuilderMemoryUse) -> bool {
        self.deferred().first_builder_memory_use(key)
    }

    /// Whether `key` was asked about ([`HostList::first_use`]) since the
    /// reset.
    pub(crate) fn used(&self, key: usize) -> bool {
        self.deferred().textures.contains(&key)
    }

    /// Whether `source` is not what was last copied into `view` in this
    /// list, noting it as that now.
    pub(crate) fn alias_source_changed(&self, view: &ID3D12Resource, source: &ID3D12Resource) -> bool {
        let view = view.as_raw() as usize;
        let source = source.as_raw() as usize;
        self.deferred().texture_aliases.insert(view, source) != Some(source)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.title_written();
        drop(deferred);
        self.changed_something();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work), (0, 0)));
    }

    /// As [`HostList::defer_host`], for work that reads what an earlier
    /// [`HostList::defer_texture`] check found (with wide checks they wait
    /// for the batch's flush otherwise).
    pub(crate) fn defer_host_checked(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Checked(Box::new(work)));
    }

    /// The list's submission-time steps, as a submission of it alone runs them.
    #[cfg(test)]
    pub(crate) fn run_steps(&self) {
        let mut batch = Batch::default();
        self.run_deferred(&mut batch, &[], 0, |_| false);
        batch.fill();
        batch.flush();
    }

    /// As [`HostList::defer_host`], for work that reads `len` bytes of title
    /// memory at `from`.
    pub(crate) fn defer_host_read(&self, from: usize, len: usize, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work), (from, from + len)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// [`HostList::defer_texture`] of a texture whose upload the list's
    /// prelude records if the check finds it due
    /// (`crate::texture::prelude_uploads`; [`HostList::prelude_uploads`]).
    pub(crate) fn defer_prelude_upload(&self, upload: crate::texture::PreludeUpload) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Texture(upload.staging()));
        deferred.prelude_uploads.push(upload);
    }

    /// At submission, after the list's steps: the textures whose upload its
    /// prelude records, each with whether it is the first upload into its
    /// host copy. Their staging buffers are refreshed first (those the
    /// submission's steps left pending). Ported from
    /// Belobug/VIctory-Belobug-AMD 5d6bd84.
    fn prelude_uploads(&self, batch: &mut Batch) -> Vec<(crate::texture::PreludeUpload, bool)> {
        let waiting = self.deferred().prelude_uploads.clone();
        if waiting.is_empty() {
            return Vec::new();
        }
        batch.refresh_textures();
        waiting
            .into_iter()
            .filter_map(|upload| upload.finish_refresh().map(|first| (upload, first)))
            .collect()
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges). Copies and fills that
    /// `taken` names (by their place) are left out: they are done on the
    /// card ([`card_steps`]). `n`: the list's place in the submission.
    fn run_deferred(&self, batch: &mut Batch, air: &[(usize, usize)], n: usize, taken: impl Fn(usize) -> bool) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        // The time and the bytes of the copies and fills, some 13,000 a
        // frame between 20,000 refreshes: summed here and added once.
        let (mut us, mut bytes) = (std::time::Duration::ZERO, 0u64);
        #[cfg(feature = "diagnostics")]
        let (mut copies, mut refreshes, mut refreshed) = (0u64, 0u64, 0u64);
        for (i, step) in self.deferred().copies.iter().enumerate() {
            if matches!(step, Step::Copy(..) | Step::Fill(..)) && taken(i) {
                continue;
            }
            let (started, copied) = match step {
                &Step::Copy(dst, src, len) => {
                    let started = std::time::Instant::now();
                    #[cfg(feature = "diagnostics")]
                    {
                        copies += 1;
                    }
                    batch.fill();
                    batch.title_written();
                    // Renderer port #7: into a heap's records, the rounds
                    // asked earlier did not see it.
                    if crate::descriptor::in_shader_heap(dst, len) {
                        crate::port_journal::prep_wrote();
                    }
                    crate::port::copied(dst, src, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    // What the card holds there may be another value although
                    // ours did not change.
                    if send::own() {
                        crate::shadow::send_copied(dst, len);
                    }
                    (started, len)
                }
                &Step::Fill(dst, src, len, _) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        copies += 1;
                    }
                    crate::port::filled(src, len);
                    if let Some(f) = batch.forwards.iter().find(|f| f.fill == (n, i))
                        && let Some(from) = batch.kept.iter().find(|k| k.0 == f.copy).and_then(|k| k.1.get(f.at..f.at + len))
                    {
                        let started = std::time::Instant::now();
                        // (A kept copy too short for the fill, as an earlier
                        // submission's was in bp5-wg, is not taken: the fill
                        // reads title memory below.)
                        // SAFETY: the fill's upload memory, `len` bytes.
                        unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), dst as *mut u8, len) };
                        place::STATS[4].fetch_add(1, Ordering::Relaxed);
                        (started, len)
                    } else if wide_fills() {
                        batch.fills.push((dst, src, len));
                        bytes += len as u64;
                        continue;
                    } else {
                        let started = std::time::Instant::now();
                        copy_mapped(dst as *mut u8, src as *const u8, len, false);
                        (started, len)
                    }
                }
                Step::Placed(p) => {
                    let started = std::time::Instant::now();
                    if crate::descriptor::in_shader_heap(p.dst, p.len) {
                        crate::port_journal::prep_wrote();
                    }
                    let free = !touches(air, p.src, p.src + p.len) && !touches(air, p.dst, p.dst + p.len);
                    batch.fill();
                    if free {
                        batch.title_written();
                    }
                    let kept = batch.forwards.iter().any(|f| f.copy == (n, i));
                    match p.prepare(free, kept) {
                        Some((bytes, true)) => {
                            if kept {
                                batch.kept.push(((n, i), bytes.clone()));
                            }
                            batch.placed.push((p.shadow.clone(), p.dst, bytes));
                        }
                        Some((bytes, false)) if kept => batch.kept.push(((n, i), bytes)),
                        _ => {}
                    }
                    (started, p.len)
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        refreshes += 1;
                        refreshed += *len as u64;
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work, _) | Step::Checked(work) => {
                    batch.fill();
                    if matches!(step, Step::Checked(_)) && !batch.due.is_empty() {
                        batch.refresh_textures();
                    }
                    #[cfg(feature = "diagnostics")]
                    let started = std::time::Instant::now();
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                        // (It may write anywhere, a heap's records too.)
                        crate::port_journal::prep_wrote();
                    }
                    let _step = crate::timeline::span("a step of ours");
                    work();
                    // It wrote title memory, where is not known here: the
                    // submission's one asking ends, later refreshes ask anew.
                    if matches!(step, Step::Run(_)) {
                        crate::watch::ask_once_end();
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame).
                    batch.flush_one(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    shadow.writing(*address, *len);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        batch.due.push(staging.clone());
                        if !wide_textures() {
                            batch.refresh_textures();
                        }
                    }
                    continue;
                }
            };
            us += started.elapsed();
            bytes += copied as u64;
        }
        PREPARING[0].fetch_add(us.as_micros() as u64, Ordering::Relaxed);
        PREPARING[1].fetch_add(bytes, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            add(C::Copies, copies);
            add(C::CopyBytes, bytes);
            add(C::Refreshes, refreshes);
            add(C::RefreshBytes, refreshed);
            SPLIT[7].fetch_add(us.as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

/// The numbers of [`HostList::epoch`].
fn next_epoch() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A table by a copy's address.
pub(crate) type ByCopy<T> = std::collections::HashMap<usize, T, std::hash::BuildHasherDefault<AddressHash>>;

/// A hash for addresses of ours: the bits that tell them apart, spread. The
/// tables it serves are asked 40,000 times a frame, and hold no one's keys
/// but our own.
#[derive(Default)]
pub(crate) struct AddressHash(u64);

impl std::hash::Hasher for AddressHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_usize(usize::from(b));
        }
    }
    fn write_usize(&mut self, n: usize) {
        let h = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 32);
    }
    fn write_u64(&mut self, n: u64) {
        self.write_usize(n as usize);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_usize(n as usize);
    }
}

/// Since the last frame-rate line, nanoseconds of the submitting thread's
/// preparing: the write-watch asking for a submission; the title's bytes put
/// in under flights' writes; the lists' steps, the refreshes among them; the
/// snapshots; what goes to copies in the card's memory; the lists sent; and
/// of the steps: the refreshes, and the copies and fills (those with the
/// refreshes a copy of title memory flushes). Of the asking, which ranges it
/// is of. Of the refreshes, those before ranges are given to shaders, one
/// copy at a time: nanoseconds, and how many. Ahead of all that: the wait for
/// the submission lock, the steps done on the card planned ([`card_steps`]),
/// the flights waited for, the heaps' records and what is swept; and, at the
/// end, the wait for the heaps' records.
#[cfg(feature = "diagnostics")]
static SPLIT: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// Since the last frame line, with `VICTORY_ASK_FIRST` (`crate::watch::first`):
/// submissions whose asking was begun beside their first steps; those that
/// made it again after flights were waited for or settled; and those that
/// asked for more where it was made before.
static ASKED_FIRST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Since the last frame-rate line: flushes of refreshes, their refreshes,
/// those known to copy nothing, and those done on several threads.
#[cfg(feature = "diagnostics")]
static FLUSHES: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Since the last frame-rate line, of the refreshes in front of submissions
/// ([`refresh_front`]): nanoseconds sifting out those known to copy nothing
/// (with few, or [`front_sift`]), asking the write watch ahead, and the
/// threads' copies; and the ranges asked ahead.
#[cfg(feature = "diagnostics")]
static FRONT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Whether a submission sent ahead begins the round of the heaps' records
/// (`crate::descriptor::begin_sync_heaps`) as soon as it holds the
/// submission lock, before the steps done on the card are planned and
/// flights are looked at (`VICTORY_HEAP_SYNC_FIRST=1`; off by default: after
/// them, as before). With `VICTORY_PORT_JOURNAL=1` it does nothing: the
/// journal asks for the round at the title's submit, earlier still.
///
/// From trustno1-coder's fork: there the submitting thread waited for those
/// rounds at the end of its preparing, 2.7-5.8 ms a frame, while the
/// planning ahead of the round's start took 2-4 ms a frame. What changes:
/// the round reads the heaps' pages earlier, by that planning's time. A
/// record the title writes for this submission after submitting it (which
/// the console's card could miss as well) is left to the next round; a
/// submission that waited for flights to land begins another round after
/// the wait, as the round begun here saw what they landed.
fn heap_sync_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_HEAP_SYNC_FIRST") && !crate::port_journal::on())
}

/// Since the last frame line, with [`heap_sync_first`]: submissions whose
/// round of the heaps' records was begun first, and those that began
/// another after a wait for flights.
static HEAP_SYNC_FIRST: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Counts into [`SPLIT`].
#[cfg(feature = "diagnostics")]
struct Split(std::cell::Cell<std::time::Instant>);

#[cfg(feature = "diagnostics")]
impl Split {
    fn begin() -> Split {
        Split(std::cell::Cell::new(std::time::Instant::now()))
    }

    /// Nanoseconds since the last part.
    fn so_far(&self) -> u64 {
        self.0.get().elapsed().as_nanos() as u64
    }

    /// What came since the last part was part `i`.
    fn part(&self, i: usize) {
        let now = std::time::Instant::now();
        SPLIT[i].fetch_add((now - self.0.replace(now)).as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Whether a list's close waits for the heap pass beside its recording
/// (`VICTORY_COMING_AT_CLOSE=1`, as before) instead of its submission.
fn coming_at_close() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMING_AT_CLOSE").is_some_and(|v| v == "1"))
}

/// Since the last frame line: the nanoseconds of each step of the lists'
/// closes, in the order of `copystat::C::CloseTwinsNs` and on, and the
/// number of closes. dup-record measured 195 closes a frame of 295-538 us
/// each (2026-11-25): about 60-100 ms a frame on the recording threads.
#[cfg(feature = "diagnostics")]
static CLOSE_NS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// Times the steps of a list's close ([`HostList::close`]); nothing without
/// diagnostics.
struct Lap {
    #[cfg(feature = "diagnostics")]
    at: std::time::Instant,
}

impl Lap {
    fn begin() -> Self {
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::Closes, 1);
            CLOSE_NS[8].fetch_add(1, Ordering::Relaxed);
        }
        Lap {
            #[cfg(feature = "diagnostics")]
            at: std::time::Instant::now(),
        }
    }

    /// Step `i` of the close ended now.
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    fn to(&mut self, i: usize) {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::C;
            let now = std::time::Instant::now();
            let ns = (now - std::mem::replace(&mut self.at, now)).as_nanos() as u64;
            let c = match i {
                0 => C::CloseTwinsNs,
                1 => C::CloseTexBackNs,
                2 => C::CloseIndirectNs,
                3 => C::CloseBatchesNs,
                4 => C::CloseBudgetNs,
                5 => C::CloseComingNs,
                6 => C::CloseQueryNs,
                _ => C::CloseDriverNs,
            };
            crate::copystat::add(c, ns);
            CLOSE_NS[i.min(7)].fetch_add(ns, Ordering::Relaxed);
        }
    }
}

/// The split of the preparing, per frame, for the frame-rate line.
#[cfg(feature = "diagnostics")]
fn split_line(frames: f64) {
    let ms: Vec<f64> = SPLIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames).collect();
    let f: Vec<f64> = FLUSHES.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let e: Vec<f64> = crate::watch::early_asks().iter().map(|&n| n as f64 / frames).collect();
    let a: Vec<f64> = crate::watch::asks().iter().map(|&n| n as f64 / frames).collect();
    let h: Vec<f64> = crate::draw::HEAP_SPLIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames).collect();
    let s: Vec<f64> = crate::shadow::send_stats().iter().map(|&n| n as f64 / frames).collect();
    tracing::info!(
        "preparing per frame, the submitting thread: {:.1} ms asking the write watch ({:.1} of them finding what to ask), {:.1} putting the title's bytes under flights' \
         writes, {:.1} in the lists' steps (refreshes {:.1} in {:.0} flushes of {:.0}, {:.0} of them known to copy nothing, {:.0} on \
         several threads, {:.1} one at a time in {:.0} before shader writes; copies and fills {:.1}, the refreshes before copies among them), {:.1} snapshots, {:.1} for copies in the \
         card's memory, {:.1} sending the lists (VICTORY_LIGHT_REFRESH is {}); ahead of these {:.1} waiting for the submission lock, {:.1} \
         planning steps done on the card, {:.1} waiting for flights, {:.1} the heaps' records and what is swept; at the end {:.1} waiting \
         for the heaps' records; asked early by a thread of its own {:.0} pieces of 2 MiB, {:.1} ms (VICTORY_ASK_EARLY)",
        ms[0],
        ms[8],
        ms[1],
        ms[2],
        ms[6],
        f[0],
        f[1],
        f[2],
        f[3],
        ms[9],
        ms[10] * 1e6,
        ms[7],
        ms[3],
        ms[4],
        ms[5],
        if crate::watch::light() { "on" } else { "off" },
        ms[11],
        ms[12],
        ms[13],
        ms[14],
        ms[15],
        e[0],
        e[1] / 1e3,
    );
    let r: Vec<f64> = FRONT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "refreshes in front of submissions per frame: {:.2} ms sifting, {:.2} asking the write watch ahead over {:.0} ranges, {:.2} \
         copying on the threads (VICTORY_FRONT_SIFT is {})",
        r[0] / 1e6,
        r[1] / 1e6,
        r[3],
        r[2] / 1e6,
        if front_sift() { "on" } else { "off" },
    );
    tracing::info!(
        "asking about title memory per frame, all threads: {:.0} kernel askings over {:.0} MB ({:.1} ms, {:.0} pages of 4 KiB found \
         written, {:.0} askings refused); {:.0} questions about holes; view ranges: {:.0} answered by their \
         bits, {:.0} by the guards whole, {:.0} block by block; {:.0} blocks made read-only, {:.0} opened by ourselves; {:.1} ms \
         waiting for the guards, {:.1} ms changing protections; pages taken for the card: {:.1} MB compared, {:.1} MB of them \
         as they were, {:.1} MB to go up in {:.0} runs (VICTORY_SEND_CHANGED)",
        a[0],
        a[1] / 1024.0,
        a[11] / 1e6,
        a[12],
        a[10],
        a[9],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7] / 1e6,
        a[8] / 1e6,
        s[0] / 1048576.0,
        s[1] / 1048576.0,
        s[2] / 1048576.0,
        s[3],
    );
    let p = crate::watch::protects();
    let row = |i: usize| {
        let r = p[i];
        format!("{:.1} calls, {:.1} blocks, {:.2} ms, {:.2} slow", r[0] as f64 / frames, r[1] as f64 / frames, r[2] as f64 / 1e6 / frames, r[3] as f64 / frames)
    };
    tracing::info!(
        "title memory protections changed per frame, by who: under the guards' lock, other views as a copy takes the page {}, \
         parted blocks closed {}, runs opened {}, blocks opened {}, parts opened for faults {}, view blocks an asking guards {}; with the lock let go, runs an \
         asking guards {}, other views after the asking {}, blocks faults open with no lock {}; put back as they were {}",
        row(0),
        row(3),
        row(4),
        row(5),
        row(7),
        row(9),
        row(1),
        row(2),
        row(6),
        row(8),
    );
    tracing::info!(
        "the pass over the heap's views per frame: {:.1} ms looking at slots, {:.1} their copies' spans and refreshes, {:.1} textures; the \
         first two beside the recording (VICTORY_HEAP_PASS_ASIDE), {:.1} ms waited for at the lists' submission",
        h[0],
        h[1],
        h[2],
        aside::WAITED.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames,
    );
    let c: Vec<f64> = CLOSE_NS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64).collect();
    let each = c[..8].iter().sum::<f64>() / 1e3 / c[8].max(1.0);
    let c: Vec<f64> = c.iter().map(|n| n / frames).collect();
    tracing::info!(
        "host lists closed per frame: {:.0}, {:.0} us each; {:.1} ms copying depth twins back, {:.1} textures read back, {:.1} the          indirect clamps, {:.1} the batched clamps, {:.1} the card budget's notes, {:.1} waiting for the heap pass ({}), {:.1}          timestamps, {:.1} the driver's Close",
        c[8],
        each,
        c[0] / 1e6,
        c[1] / 1e6,
        c[2] / 1e6,
        c[3] / 1e6,
        c[4] / 1e6,
        c[5] / 1e6,
        if coming_at_close() { "VICTORY_COMING_AT_CLOSE=1" } else { "at the submission instead" },
        c[6] / 1e6,
        c[7] / 1e6,
    );
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH=0` disables it; `=1` forces it).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (four or more) are
/// spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_refresh_enabled(
            std::env::var("VICTORY_WIDE_REFRESH").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_refresh_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

#[cfg(test)]
mod wide_refresh_tests {
    use super::wide_refresh_enabled;

    #[test]
    fn parallel_refresh_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_refresh_enabled(None, Some(4)));
        assert!(wide_refresh_enabled(None, Some(16)));
        assert!(!wide_refresh_enabled(None, Some(2)));
        assert!(!wide_refresh_enabled(None, None));
        assert!(wide_refresh_enabled(Some("1"), Some(1)));
        assert!(!wide_refresh_enabled(Some("0"), Some(16)));
        assert!(wide_refresh_enabled(Some("auto"), Some(4)));
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them, or into ranges more than a page apart
/// ([`apart`]): done before a step that writes title memory or gives a
/// range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<Refresh>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: ByCopy<usize>,
    /// Staging buffers (by address) refreshed since the last title write,
    /// or due to be.
    textures: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Those due ([`Batch::refresh_textures`]).
    due: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// What copies at their place in the submission's lists put into title
    /// memory ([`place`]): the host copy, the title address, the bytes.
    placed: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, Vec<u8>)>,
    /// Fills not done yet (destination, source, bytes): [`wide_fills`].
    fills: Vec<(usize, usize, usize)>,
    /// Fills that take a copy's bytes ([`place::forwards`]), and the bytes
    /// of those copies, by their place.
    forwards: Vec<place::Forward>,
    kept: Vec<((usize, usize), Vec<u8>)>,
    /// Copies (by address) brought up to date ahead of the steps, and the
    /// range (from, to) ([`Batch::refresh_ahead`]).
    ahead: ByCopy<(usize, usize)>,
    /// `ahead` holds everything the submission refreshes
    /// ([`Batch::refresh_front`]).
    front: bool,
    /// Per copy the submission refreshes, the span of all its refreshes, in
    /// the order the copies first come ([`Batch::copy_spans`]); where each
    /// copy (by address) is in it.
    spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    spans_at: ByCopy<usize>,
    /// Room for the submission's one asking's ranges, and for
    /// [`Batch::refresh_front`]'s walk over the steps.
    ranges: Vec<(usize, usize)>,
    touches: Vec<Touch>,
}

thread_local! {
    /// The last submission's [`Batch`] on this thread, emptied, with the
    /// room its tables and lists grew to: some 28 submissions a frame
    /// each made them anew, grown from nothing (hash tables rehashed as
    /// they grew, about 2% of the submitting thread in profile-y-prof60).
    static SPARE_BATCH: std::cell::Cell<Option<Batch>> = const { std::cell::Cell::new(None) };
}

/// A copy's refresh: the range (from, to), and with [`apart`] the ranges of
/// it more than a page from that one and from each other, copied on their
/// own instead of with the span between them.
type Refresh = (std::sync::Arc<crate::shadow::Shadow>, usize, usize, Vec<(usize, usize)>);

/// Each range of `refresh`.
fn ranges(refresh: &Refresh) -> impl Iterator<Item = (usize, usize)> + '_ {
    std::iter::once((refresh.1, refresh.2)).chain(refresh.3.iter().copied())
}

/// Whether a copy's refreshes more than a page apart are copied apart
/// (`VICTORY_REFRESH_APART=0`: the span that covers them, as before). A
/// copy asked for at both ends copied all between them.
fn apart() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_APART").is_none_or(|v| v != "0"))
}

/// Ranges up to this far apart are copied as one: a copy of the page between
/// costs less than another call.
const COALESCE: usize = crate::watch::PAGE;

/// `[from, to)` joined to a copy's refresh ranges: the range (`span`) and
/// those apart from it and each other by more than [`COALESCE`] (`others`).
/// What it lies near merges with it; the merged range is the span.
fn join(span: &mut (usize, usize), others: &mut Vec<(usize, usize)>, from: usize, to: usize) {
    let near = |(a, e): (usize, usize), from: usize, to: usize| {
        a <= to.saturating_add(COALESCE) && from <= e.saturating_add(COALESCE)
    };
    if others.is_empty() {
        if near(*span, from, to) {
            *span = (span.0.min(from), span.1.max(to));
        } else {
            others.push((from, to));
        }
        return;
    }
    // Every two of them are further apart than that: what is near the
    // merged range was near one of its parts.
    let (mut from, mut to) = (from, to);
    others.push(*span);
    let mut i = 0;
    while i < others.len() {
        if near(others[i], from, to) {
            let (a, e) = others.swap_remove(i);
            (from, to) = (from.min(a), to.max(e));
        } else {
            i += 1;
        }
    }
    *span = (from, to);
}

/// Of `refresh`, the ranges that `nothing_new` says hold nothing new are let
/// go; whether any is left.
fn keep_new(refresh: &mut Refresh) -> bool {
    let (shadow, from, to, others) = refresh;
    others.retain(|&(a, e)| !shadow.nothing_new(a, e - a));
    if !shadow.nothing_new(*from, *to - *from) {
        return true;
    }
    match others.pop() {
        Some((a, e)) => {
            (*from, *to) = (a, e);
            true
        }
        None => false,
    }
}

/// [`keep_new`] with the answers for [`ranges`] of `refresh` taken from
/// `same`, in that order ([`crate::shadow::nothing_new_all`]).
fn keep_new_by(refresh: &mut Refresh, same: &mut impl Iterator<Item = bool>) -> bool {
    let main = same.next().unwrap_or(false);
    let (_, from, to, others) = refresh;
    others.retain(|_| !same.next().unwrap_or(false));
    if !main {
        return true;
    }
    match others.pop() {
        Some((a, e)) => {
            (*from, *to) = (a, e);
            true
        }
        None => false,
    }
}

/// Whether the copies that a submission's shaders write are brought up to
/// date together, on several threads, ahead of its steps
/// (`VICTORY_REFRESH_AHEAD=0`: each before the step that gives it to them).
///
/// 580 copies a frame, one after the other on the submitting thread, 3.4 to
/// 6.5 ms (session play-1125-0203).
fn ahead_of_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_AHEAD").is_none_or(|v| v != "0") && wide_refresh() && crate::watch::light())
}

/// Whether everything a submission refreshes is brought up to date in one go,
/// on several threads, ahead of its steps (`VICTORY_REFRESH_FRONT=1`, and with
/// `VICTORY_JOIN_PARTS`, which needs it; else in groups where the steps ask
/// for it). Measured on its own at nothing (run submit-1: refreshes 17.1 ms a
/// frame without, 18.3 with).
///
/// A submission's refreshes were done in up to five groups one after the other
/// (the copies its shaders write, those between its steps, what is left),
/// each ending with the threads waiting for the last copy; and the submitting
/// thread went over all the steps once for each question it has (what to ask
/// about, what flights may write too, what shaders write). A refresh may come
/// earlier than its step: the title's bytes are the same until a step writes
/// title memory, and behind such a step everything is asked for again, as it
/// was.
fn refresh_front() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    ahead_of_writes() && (joins_asked().0 > 1 || *ON.get_or_init(|| crate::switch::on("VICTORY_REFRESH_FRONT")))
}

/// With very many refreshes in front of a submission, those known to copy
/// nothing are sifted out on all threads before the write watch is asked
/// ahead, as they are with few: it is asked for those left only, and the
/// threads then share what copies. Without it the asking ahead takes all of
/// them, and each is sifted beside the others' copies.
fn front_sift() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_FRONT_SIFT"))
}

/// [`front_sift`]: the spans that may copy something, in their order.
fn sift_front(spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    use std::sync::atomic::AtomicBool;
    let spans = std::sync::Arc::new(spans);
    let keep: std::sync::Arc<Vec<AtomicBool>> = std::sync::Arc::new((0..spans.len()).map(|_| AtomicBool::new(false)).collect());
    let (s, k) = (spans.clone(), keep.clone());
    crate::shadow::wide_each(spans.len(), move |i| {
        let (shadow, from, to) = &s[i];
        if !shadow.nothing_new(*from, *to - *from) {
            k[i].store(true, Ordering::Relaxed);
        }
    });
    let spans = std::sync::Arc::try_unwrap(spans).unwrap_or_else(|a| (*a).clone());
    let left: Vec<_> = spans.into_iter().zip(keep.iter()).filter(|(_, k)| k.load(Ordering::Relaxed)).map(|(s, _)| s).collect();
    #[cfg(feature = "diagnostics")]
    FLUSHES[2].fetch_add((keep.len() - left.len()) as u64, Ordering::Relaxed);
    left
}

/// Per copy what is refreshed in front of a submission's steps, as
/// [`written_spans`]'s but of all copies, up to the copy's first write by
/// shaders in each part of the submission (`Touch::Part`: another begins).
/// What a part refreshes of a copy it has given to its shaders stays where it
/// stands: around what they write. What a later part refreshes is the
/// title's bytes of before either.
fn front_spans(steps: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize)> {
    let mut at: ByCopy<usize> = ByCopy::default();
    // (Copy, from, to), in the order they come, and the part that gave it to
    // shaders last.
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    let mut given: Vec<u32> = Vec::new();
    let (mut part, mut last) = (1, usize::MAX);
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                // (Most come several times on end.)
                let i = if spans.get(last).is_some_and(|s| s.0 == copy) {
                    last
                } else {
                    *at.entry(copy).or_insert_with(|| {
                        spans.push((copy, usize::MAX, 0));
                        given.push(0);
                        spans.len() - 1
                    })
                };
                last = i;
                if given[i] != part {
                    let span = &mut spans[i];
                    (span.1, span.2) = (span.1.min(from), span.2.max(to));
                }
            }
            Touch::Write(copy) => {
                let i = *at.entry(copy).or_insert_with(|| {
                    spans.push((copy, usize::MAX, 0));
                    given.push(0);
                    spans.len() - 1
                });
                given[i] = part;
            }
            Touch::Part => part += 1,
            Touch::Title | Touch::Nothing => {}
        }
    }
    spans.retain(|s| s.1 < s.2);
    spans
}

/// What a step is to the copies' refreshes, for [`written_spans`].
enum Touch {
    /// A refresh of a copy (by address), from, to.
    Refresh(usize, usize, usize),
    /// The copy is given to shaders.
    Write(usize),
    /// Title memory is written.
    #[cfg_attr(not(test), allow(dead_code))] // no step makes it; the tests do
    Title,
    Nothing,
    /// Another part of the submission begins.
    Part,
}

/// Of `steps`, up to the first that writes title memory: the copies given to
/// shaders, each with the range its refreshes before that cover.
fn written_spans(steps: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize)> {
    // (From, to, given to shaders.)
    let mut spans: ByCopy<(usize, usize, bool)> = ByCopy::default();
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                let span = spans.entry(copy).or_insert((from, to, false));
                if !span.2 {
                    (span.0, span.1) = (span.0.min(from), span.1.max(to));
                }
            }
            // (Not refreshed before: nothing of it is due.)
            Touch::Write(copy) => {
                if let Some(span) = spans.get_mut(&copy) {
                    span.2 = true;
                }
            }
            Touch::Title => break,
            Touch::Nothing | Touch::Part => {}
        }
    }
    spans.into_iter().filter(|(_, s)| s.2).map(|(copy, (from, to, _))| (copy, from, to)).collect()
}

#[cfg(test)]
mod written_spans_tests {
    use super::{Touch, front_spans, split, written_spans};

    #[test]
    fn everything_is_refreshed_in_front_but_what_a_part_gave_its_shaders() {
        use Touch::{Nothing, Part, Refresh, Title, Write};
        let found = front_spans(
            [
                Refresh(1, 10, 20),
                Refresh(1, 12, 25),
                Refresh(2, 0, 5),
                Nothing,
                Write(1),
                // (Around what the part's own shaders write.)
                Refresh(1, 0, 100),
                Write(3),
                Refresh(3, 1, 2),
                Title,
                Refresh(4, 7, 9),
                Part,
                // The next part's bytes under the first one's writes.
                Refresh(1, 30, 40),
                Write(4),
                Refresh(4, 0, 9),
            ]
            .into_iter(),
        );
        assert_eq!(found, [(1, 10, 40), (2, 0, 5), (4, 7, 9)]);
    }

    #[test]
    fn a_submission_is_cut_between_its_lists() {
        // Lists c and e are in neither half's order: they go with the first.
        let Ok([first, second]) = split(vec!['a', 'b', 'c', 'd', 'e'], vec![1, 3, 0], 1) else {
            panic!("no list is named twice");
        };
        let run = |(lists, order): &(Vec<char>, Vec<usize>)| order.iter().map(|&i| lists[i]).collect::<String>();
        assert_eq!((run(&first), run(&second)), ("b".into(), "da".into()));
        assert_eq!(first.0.len() + second.0.len(), 5);
        assert!(split(vec!['a', 'b'], vec![0, 1, 0], 2).is_err());
        assert!(split(vec!['a', 'b'], vec![0, 1], 0).is_err());
    }

    #[test]
    fn what_is_written_is_refreshed_ahead_up_to_its_write_and_to_title_writes() {
        use Touch::{Nothing, Refresh, Title, Write};
        let mut found = written_spans(
            [
                Refresh(1, 10, 20),
                Refresh(2, 0, 5),
                Nothing,
                Refresh(1, 30, 40),
                Write(1),
                Refresh(1, 0, 100),
                Write(3),
                Refresh(3, 1, 2),
                Refresh(4, 7, 9),
                Write(4),
                Title,
                Write(2),
            ]
            .into_iter(),
        );
        found.sort_unstable();
        assert_eq!(found, [(1, 10, 40), (4, 7, 9)]);
    }
}

/// Whether the fills of a submission's upload memory are done together on
/// several threads (`VICTORY_WIDE_FILLS=0`: one after the other, where the
/// lists name them).
///
/// They are 7,000 to 12,000 a frame, some 500 bytes each, 5 ms of the
/// submitting thread. Each reads title memory and writes memory of its own
/// that the card alone reads, when the lists run; they are done before the
/// next step that writes title memory or runs code of ours, and before the
/// submission goes on from its steps.
fn wide_fills() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_FILLS").is_none_or(|v| v != "0") && wide_refresh())
}

/// Whether a submission's textures are brought up to date on several
/// threads (`VICTORY_WIDE_TEXTURES=0`: one after the other, where the lists
/// name them, as before).
///
/// They are read, hashed and, if they changed, detiled into their staging
/// buffers: 5 ms a frame of the submitting thread, for some 16 MB. Each has
/// its own buffer, notes and lock, and is refreshed by the title's own
/// threads as it is. They are done before the next step that writes title
/// memory, as the copies' refreshes are, and before the lists go to the
/// card. What is new: the note of a list that its upload takes what the
/// buffer then holds (`texture.rs`, `dirty`) may come ahead of the refresh
/// an earlier list of the submission asked for, not behind it. The texture
/// is then uploaded by a later list too, once more than needed.
fn wide_textures() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_TEXTURES").is_none_or(|v| v != "0") && wide_refresh())
}

impl Batch {
    /// This thread's last one, emptied ([`SPARE_BATCH`]), or a new one.
    fn reused() -> Batch {
        SPARE_BATCH.take().unwrap_or_default()
    }

    /// Emptied (the room kept) for the thread's next submission.
    fn give_back(mut self) {
        self.refreshes.clear();
        self.index.clear();
        self.textures.clear();
        self.due.clear();
        self.placed.clear();
        self.fills.clear();
        self.forwards.clear();
        self.kept.clear();
        self.ahead.clear();
        self.front = false;
        self.spans.clear();
        self.spans_at.clear();
        self.ranges.clear();
        self.touches.clear();
        SPARE_BATCH.set(Some(self));
    }

    /// Per copy that `runs` refresh, the span of all its refreshes, into
    /// [`Batch::spans`]: the submission's one asking asks about those
    /// ([`crate::watch::ask_once`]), and behind flights those of copies
    /// shaders may be writing are looked at ([`Batch::stale_under_flights`]).
    /// One walk over the steps for both, where there were two, each with a
    /// table of its own.
    fn copy_spans(&mut self, runs: &[&HostList]) {
        self.spans.clear();
        self.spans_at.clear();
        // (Most come several times on end.)
        let mut last = (0, 0);
        for list in runs {
            for step in &list.deferred().copies {
                let Step::Refresh(shadow, address, len) = step else {
                    continue;
                };
                let (from, to) = (*address, *address + *len);
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                let at = if last.0 == key && key != 0 {
                    last.1
                } else {
                    match self.spans_at.entry(key) {
                        std::collections::hash_map::Entry::Occupied(at) => *at.get(),
                        std::collections::hash_map::Entry::Vacant(place) => {
                            place.insert(self.spans.len());
                            self.spans.push((shadow.clone(), from, to));
                            self.spans.len() - 1
                        }
                    }
                };
                last = (key, at);
                let span = &mut self.spans[at];
                (span.1, span.2) = (span.1.min(from), span.2.max(to));
            }
        }
    }

    /// [`stale_under_flights`] of the submission's refreshes, from
    /// [`Batch::copy_spans`]: the copies shaders may be writing now, in the
    /// order they first come, with the span of all their refreshes.
    fn stale_under_flights(&self) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
        stale_of(self.spans.iter().filter(|s| s.0.writes_pending()).cloned().collect())
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if self.ahead.get(&key).is_some_and(|&(from, to)| from <= address && end <= to) {
            return;
        }
        match self.index.get(&key) {
            Some(&at) => {
                let r = &mut self.refreshes[at];
                if !apart() {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(
                        crate::copystat::C::RefreshGapBytes,
                        address.saturating_sub(r.2).max(r.1.saturating_sub(end)) as u64,
                    );
                    (r.1, r.2) = (r.1.min(address), r.2.max(end));
                    return;
                }
                let mut span = (r.1, r.2);
                #[cfg(feature = "diagnostics")]
                let gap = ranges(r)
                    .map(|(a, e)| address.saturating_sub(e).max(a.saturating_sub(end)))
                    .min()
                    .filter(|&gap| gap <= COALESCE)
                    .unwrap_or(0);
                join(&mut span, &mut r.3, address, end);
                (r.1, r.2) = span;
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::RefreshGapBytes, gap as u64);
            }
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end, Vec::new()));
            }
        }
    }

    /// Before shaders are given a range of `shadow`'s copy: its pending
    /// refresh is done, the others stay pending.
    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        // From here on it is refreshed around what the shaders write. (What
        // is in front of all the steps was, up to here.)
        if !self.front {
            self.ahead.remove(&(std::sync::Arc::as_ptr(shadow) as usize));
        }
        let Some(at) = self.index.remove(&(std::sync::Arc::as_ptr(shadow) as usize)) else {
            return;
        };
        let refresh = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let shadow = &refresh.0;
        for (from, to) in ranges(&refresh) {
            if !(crate::watch::light() && shadow.nothing_new(from, to - from)) {
                Self::update(shadow, from, to);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            let ns = started.elapsed().as_nanos() as u64;
            SPLIT[6].fetch_add(ns, Ordering::Relaxed);
            SPLIT[9].fetch_add(ns, Ordering::Relaxed);
            SPLIT[10].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let _one = crate::timeline::span("a copy's refresh");
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add, again};
            crate::copystat::range(from, to - from, copied, us, 0);
            add(C::RefreshUpdates, 1);
            if copied == 0 {
                add(C::RefreshEmpty, 1);
                add(C::RefreshEmptyUs, us);
            }
            if again(0, (shadow as *const crate::shadow::Shadow as usize, from, to)) {
                add(C::RefreshAgain, 1);
                add(C::RefreshAgainBytes, (to - from) as u64);
            }
        }
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.ahead.clear();
        self.textures.clear();
    }

    /// Before the steps of `runs` ([`refresh_front`]), a new part of the
    /// submission beginning at each of `parts`.
    fn refresh_front(&mut self, runs: &[&HostList], parts: &[usize]) {
        // The copies refreshed, by address: as the submission's one asking
        // found them, or found here.
        if self.spans.is_empty() {
            self.copy_spans(runs);
        }
        let mut due = Vec::new();
        self.touches.clear();
        for (n, list) in runs.iter().enumerate() {
            if parts.contains(&n) {
                self.touches.push(Touch::Part);
            }
            for step in &list.deferred().copies {
                match step {
                    Step::Refresh(shadow, at, len) => {
                        self.touches.push(Touch::Refresh(std::sync::Arc::as_ptr(shadow) as usize, *at, *at + *len));
                    }
                    Step::GpuWrite(shadow, ..) => self.touches.push(Touch::Write(std::sync::Arc::as_ptr(shadow) as usize)),
                    Step::Texture(staging) if wide_textures() => {
                        if self.texture_due(staging) {
                            due.push(staging.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        let found = front_spans(self.touches.drain(..));
        let spans: Vec<_> = found
            .into_iter()
            .filter_map(|(key, from, to)| Some((self.spans.get(*self.spans_at.get(&key)?)?.0.clone(), from, to)))
            .collect();
        self.front = true;
        for (shadow, from, to) in &spans {
            self.ahead.insert(std::sync::Arc::as_ptr(shadow) as usize, (*from, *to));
        }
        #[cfg(feature = "diagnostics")]
        {
            FLUSHES[0].fetch_add(1, Ordering::Relaxed);
            FLUSHES[1].fetch_add(spans.len() as u64, Ordering::Relaxed);
        }
        let _flush = crate::timeline::span("refreshes of copies");
        let started = std::time::Instant::now();
        let before = [0, 2].map(|i| PREPARING[i].load(Ordering::Relaxed));
        // Those known to copy nothing are said so here, unless they are very
        // many: then beside the others.
        const FEW: usize = 256;
        let mut spans = spans;
        let batch = crate::shadow::new_batch();
        let mut many = spans.len() > FEW && !batch;
        #[cfg(feature = "diagnostics")]
        let mut lap = std::time::Instant::now();
        if !many {
            #[cfg(feature = "diagnostics")]
            let all = spans.len();
            if batch {
                let mut same = crate::shadow::nothing_new_all(spans.iter().map(|(shadow, from, to)| (&**shadow, *from, *to))).into_iter();
                spans.retain(|_| !same.next().unwrap_or(false));
            } else {
                spans.retain(|(shadow, from, to)| !shadow.nothing_new(*from, *to - *from));
            }
            #[cfg(feature = "diagnostics")]
            FLUSHES[2].fetch_add((all - spans.len()) as u64, Ordering::Relaxed);
        } else if front_sift() {
            spans = sift_front(spans);
            many = false;
        }
        #[cfg(feature = "diagnostics")]
        {
            FRONT[0].fetch_add(lap.elapsed().as_nanos() as u64, Ordering::Relaxed);
            lap = std::time::Instant::now();
        }
        let ahead = spans.len() > 1;
        if ahead {
            let ranges: Vec<(usize, usize)> = spans.iter().filter(|r| r.0.asked_ahead()).map(|r| (r.1, r.2)).collect();
            #[cfg(feature = "diagnostics")]
            FRONT[3].fetch_add(ranges.len() as u64, Ordering::Relaxed);
            crate::watch::ask_ahead(&ranges);
        }
        #[cfg(feature = "diagnostics")]
        {
            FRONT[1].fetch_add(lap.elapsed().as_nanos() as u64, Ordering::Relaxed);
            lap = std::time::Instant::now();
        }
        #[cfg(feature = "diagnostics")]
        FLUSHES[3].fetch_add(spans.len() as u64, Ordering::Relaxed);
        // (The textures one to a thread; the copies as `VICTORY_WIDE_CHUNK` says.)
        crate::shadow::wide_each_from(spans.len() + due.len(), due.len(), crate::shadow::wide_chunk(), move |i| {
            // The textures first: they are the longer pieces.
            let Some(i) = i.checked_sub(due.len()) else {
                due[i].refresh();
                return;
            };
            let (shadow, from, to) = &spans[i];
            if many && shadow.nothing_new(*from, *to - *from) {
                #[cfg(feature = "diagnostics")]
                FLUSHES[2].fetch_add(1, Ordering::Relaxed);
                return;
            }
            Self::update(shadow, *from, *to);
        });
        #[cfg(feature = "diagnostics")]
        FRONT[2].fetch_add(lap.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if ahead {
            crate::watch::ask_ahead_done();
        }
        // The time counted is the time it took, not the threads' sum.
        let took = started.elapsed().as_micros() as u64;
        let summed = [
            PREPARING[0].load(Ordering::Relaxed).saturating_sub(before[0]),
            PREPARING[2].load(Ordering::Relaxed).saturating_sub(before[1]),
        ];
        PREPARING[2].fetch_sub(summed[1], Ordering::Relaxed);
        PREPARING[0].fetch_sub(summed[0], Ordering::Relaxed);
        PREPARING[0].fetch_add(took, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        SPLIT[6].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Before the steps of `runs` ([`ahead_of_writes`]); copies and fills
    /// that `taken` names are not done on the CPU.
    fn refresh_ahead(&mut self, runs: &[&HostList], taken: impl Fn(&HostList, usize) -> bool) {
        let mut copies: ByCopy<std::sync::Arc<crate::shadow::Shadow>> = ByCopy::default();
        let mut touches = Vec::new();
        'steps: for list in runs {
            for (i, step) in list.deferred().copies.iter().enumerate() {
                touches.push(match step {
                    Step::Refresh(shadow, at, len) => {
                        let key = std::sync::Arc::as_ptr(shadow) as usize;
                        copies.entry(key).or_insert_with(|| shadow.clone());
                        Touch::Refresh(key, *at, *at + *len)
                    }
                    Step::GpuWrite(shadow, ..) => Touch::Write(std::sync::Arc::as_ptr(shadow) as usize),
                    Step::Copy(..) if taken(list, i) => Touch::Nothing,
                    Step::Copy(..) | Step::Run(_) | Step::Placed(_) => break 'steps,
                    _ => Touch::Nothing,
                });
            }
        }
        let due: Vec<_> = written_spans(touches.into_iter())
            .into_iter()
            .filter_map(|(key, from, to)| Some((copies.get(&key)?.clone(), from, to)))
            .collect();
        for (shadow, from, to) in &due {
            self.ahead.insert(std::sync::Arc::as_ptr(shadow) as usize, (*from, *to));
        }
        #[cfg(feature = "diagnostics")]
        let began = std::time::Instant::now();
        let started = std::time::Instant::now();
        let before = PREPARING[0].load(Ordering::Relaxed);
        crate::shadow::wide_each(due.len(), move |i| {
            let (shadow, from, to) = &due[i];
            if !shadow.nothing_new(*from, *to - *from) {
                Self::update(shadow, *from, *to);
            }
        });
        // The time counted is the time it took, not the threads' sum.
        let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
        PREPARING[0].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            let ns = began.elapsed().as_nanos() as u64;
            SPLIT[6].fetch_add(ns, Ordering::Relaxed);
            SPLIT[9].fetch_add(ns, Ordering::Relaxed);
        }
    }

    /// The fills noted are done.
    fn fill(&mut self) {
        // Below this many the threads cost more than they save.
        const FEW: usize = 96;
        const PART: usize = 64;
        if self.fills.is_empty() {
            return;
        }
        let started = std::time::Instant::now();
        if self.fills.len() < FEW {
            for (dst, src, len) in self.fills.drain(..) {
                copy_mapped(dst as *mut u8, src as *const u8, len, false);
            }
        } else {
            let fills = std::mem::take(&mut self.fills);
            crate::shadow::wide_each(fills.len().div_ceil(PART), move |i| {
                for &(dst, src, len) in fills.iter().skip(i * PART).take(PART) {
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                }
            });
        }
        let took = started.elapsed();
        PREPARING[0].fetch_add(took.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        SPLIT[7].fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }

    /// The textures due are brought up to date.
    fn refresh_textures(&mut self) {
        if self.due.len() < 2 || !wide_textures() {
            self.due.drain(..).for_each(|staging| staging.refresh());
            return;
        }
        let _textures = crate::timeline::span("textures brought up to date");
        // The time counted is the time it took, not the threads' sum.
        let started = std::time::Instant::now();
        let before = PREPARING[2].load(Ordering::Relaxed);
        let due = std::mem::take(&mut self.due);
        crate::shadow::wide_each(due.len(), move |i| due[i].refresh());
        let summed = PREPARING[2].load(Ordering::Relaxed).saturating_sub(before);
        PREPARING[2].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
    }

    fn flush(&mut self) {
        self.refresh_textures();
        self.index.clear();
        if self.refreshes.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let (began, all) = (std::time::Instant::now(), self.refreshes.len());
        let _flush = crate::timeline::span("refreshes of copies");
        self.flush_all();
        #[cfg(feature = "diagnostics")]
        {
            SPLIT[6].fetch_add(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
            FLUSHES[0].fetch_add(1, Ordering::Relaxed);
            FLUSHES[1].fetch_add(all as u64, Ordering::Relaxed);
        }
    }

    fn flush_all(&mut self) {
        // Most of a flush's copies are as they were: they are told so here,
        // from what the submission's asking found, before anything is asked
        // and any thread woken. (Thousands of them, the heap's buffer views
        // once a frame: on the threads, below.)
        const FEW: usize = 256;
        let light = crate::watch::light();
        let batch = crate::shadow::new_batch();
        if light && (self.refreshes.len() <= FEW || batch) {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let all = self.refreshes.len();
            if batch {
                let mut same = crate::shadow::nothing_new_all(
                    self.refreshes.iter().flat_map(|r| ranges(r).map(move |(from, to)| (&*r.0, from, to))),
                )
                .into_iter();
                self.refreshes.retain_mut(|r| keep_new_by(r, &mut same));
            } else {
                self.refreshes.retain_mut(keep_new);
            }
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            FLUSHES[2].fetch_add((all - self.refreshes.len()) as u64, Ordering::Relaxed);
        }
        // One write-watch asking for the copies that lie close together,
        // not one per copy.
        let ahead = self.refreshes.len() > 1;
        if ahead {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = self
                .refreshes
                .iter()
                .filter(|r| r.0.asked_ahead())
                .flat_map(ranges)
                .collect();
            crate::watch::ask_ahead(&ranges);
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if wide_refresh() && self.refreshes.len() >= 4 {
            // Each copy on its own, several at a time: a copy's refresh
            // writes that copy and reads title memory, nothing of another's.
            // The time counted is the time it took, not the threads' sum.
            let started = std::time::Instant::now();
            let before = PREPARING[0].load(Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            FLUSHES[3].fetch_add(self.refreshes.len() as u64, Ordering::Relaxed);
            if light {
                // A number to a free thread, not a job to each copy.
                let refreshes = std::mem::take(&mut self.refreshes);
                crate::shadow::wide_each_from(refreshes.len(), 0, crate::shadow::wide_chunk(), move |i| {
                    let shadow = &refreshes[i].0;
                    for (from, to) in ranges(&refreshes[i]) {
                        if refreshes.len() > FEW && !batch && shadow.nothing_new(from, to - from) {
                            #[cfg(feature = "diagnostics")]
                            FLUSHES[2].fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        Self::update(shadow, from, to);
                    }
                });
            } else {
                let jobs: Vec<Box<dyn FnOnce() + Send>> = self
                    .refreshes
                    .drain(..)
                    .map(|r| {
                        Box::new(move || ranges(&r).for_each(|(from, to)| Self::update(&r.0, from, to)))
                            as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
            }
            let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
            PREPARING[0].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
        } else {
            for r in self.refreshes.drain(..) {
                ranges(&r).for_each(|(from, to)| Self::update(&r.0, from, to));
            }
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }
}

/// Copies of at least this many bytes are cut into pieces for the kept
/// threads of [`crate::shadow::wide_run`] (`VICTORY_WIDE_COPY=1`; off by
/// default, an A/B switch). Pieces are [`WIDE_COPY_PIECE`] bytes each.
const WIDE_COPY_MIN: usize = 8 << 20;
const WIDE_COPY_PIECE: usize = 2 << 20;

fn wide_copy_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_COPY").is_some_and(|v| v == "1"))
}

/// The `[from, to)` pieces that cover `len` bytes, `piece` bytes each (the
/// last one may be shorter).
fn piece_bounds(len: usize, piece: usize) -> Vec<(usize, usize)> {
    (0..len.div_ceil(piece)).map(|i| (i * piece, ((i + 1) * piece).min(len))).collect()
}

/// [`copy_mapped`] of a big copy, piece by piece on several threads. Each
/// piece is shorter than [`WIDE_COPY_MIN`], so it does not come back here.
fn copy_pieces(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    let (d, s) = (dst as usize, src as usize);
    let jobs: Vec<Box<dyn FnOnce() + Send>> = piece_bounds(len, WIDE_COPY_PIECE)
        .into_iter()
        .map(|(a, b)| {
            Box::new(move || copy_mapped((d + a) as *mut u8, (s + a) as *const u8, b - a, title_dst))
                as Box<dyn FnOnce() + Send>
        })
        .collect();
    crate::shadow::wide_run(jobs);
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    if len >= WIDE_COPY_MIN && wide_copy_on() {
        copy_pieces(dst, src, len, title_dst);
        return;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let since = crate::watch::changes();
    // (A title destination that is away only while the runtime changes a
    // mapping beside it is written when that is done.)
    let whole = read(src, dst, len) || title_dst && crate::watch::again_after_change(since, || read(src, dst, len));
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        crate::watch::lost("a copy of the title's", dst as usize, len);
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    zero(to, n);
                }
            });
        }
    });
}

/// Zeros at `to`, in one 4 KiB page or two. It may be title memory (the
/// destination of a title's copy), which can go between the question whether
/// it is there and this.
fn zero(to: *mut u8, n: usize) {
    static ZEROS: [u8; 4096] = [0; 4096];
    if n <= ZEROS.len() && crate::safecopy::on() {
        // SAFETY: a destination that is not there fails the copy.
        if !unsafe { crate::safecopy::copy(to, ZEROS.as_ptr(), n) } {
            crate::watch::lost("zeros", to as usize, n);
        }
        return;
    }
    // SAFETY: inside the destination, which is mapped.
    unsafe { to.write_bytes(0, n) };
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Reads what is there of the `dst.len()` bytes of title memory at `src`,
/// zeros for the rest; returns how many bytes were there. Safe while the
/// runtime remaps title memory (the kernel, or a copy that takes the fault,
/// reads). What the runtime's record says is mapped is read in one go each,
/// and by pieces where that fails; memory it has no record of by pieces.
pub(crate) fn read_there(src: *const u8, dst: &mut [u8]) -> usize {
    let (start, len) = (src as usize, dst.len());
    let mut got = 0;
    let mut runs = crate::inplace::shown_runs(start, len);
    if runs.is_empty() {
        runs.push((start, len));
    }
    // Every byte of a run is read or zeroed below: only the rest is zeroed
    // here, not the whole buffer first (one write over all of it less).
    zero_between(dst, start, &runs);
    for (at, n) in runs {
        let piece = &mut dst[at - start..at - start + n];
        if read_into(at as *const u8, piece) {
            got += n;
            continue;
        }
        // (What failed may have written its first pages.)
        piece.fill(0);
        for_pieces(at, n, 64 << 10, |from, n| {
            let piece = &mut dst[from - start..from - start + n];
            if read_into(from as *const u8, piece) {
                got += n;
                return;
            }
            piece.fill(0);
            for_pieces(from, n, 4 << 10, |from, n| {
                let piece = &mut dst[from - start..from - start + n];
                if read_into(from as *const u8, piece) {
                    got += n;
                } else {
                    piece.fill(0);
                }
            });
        });
    }
    got
}

/// Zeros what of `dst` (title memory from `start`) `runs` (in order,
/// within `dst`: address, bytes) leave out.
fn zero_between(dst: &mut [u8], start: usize, runs: &[(usize, usize)]) {
    let mut done = 0;
    for &(at, n) in runs {
        let from = at - start;
        if from > done {
            dst[done..from].fill(0);
        }
        done = done.max(from + n);
    }
    dst[done..].fill(0);
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if crate::safecopy::on() {
        // SAFETY: the destination is mapped writable memory of `len` bytes;
        // a source page that is not there fails the copy.
        return unsafe { crate::safecopy::copy(dst, src, len) };
    }
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let since = crate::watch::changes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        if crate::safecopy::on() {
            // SAFETY: the source is ours; a destination page that is not
            // there, or not writable, fails the copy.
            return unsafe { crate::safecopy::copy(at as *mut u8, from, n) };
        }
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                if write(at, n) {
                    return;
                }
                // The kernel's write (`VICTORY_SAFE_COPY=0`) fails where a
                // refresh guarded the block again since the caller opened
                // it; and any where the runtime was changing a mapping.
                crate::watch::writing(at, n);
                if !write(at, n) && !crate::watch::again_after_change(since, || write(at, n)) {
                    crate::watch::lost("a host copy's bytes", at, n);
                }
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A host query heap of `count` queries for a console heap of `kind`
    /// (the heap types match: 0 occlusion, 1 timestamp, 2 pipeline
    /// statistics); none for other kinds.
    pub(crate) fn query_heap(&self, kind: u32, count: u32) -> Option<ID3D12QueryHeap> {
        let kind = match kind {
            0 => D3D12_QUERY_HEAP_TYPE_OCCLUSION,
            1 => D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            2 => D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
            _ => return None,
        };
        let mut heap: Option<ID3D12QueryHeap> = None;
        // SAFETY: plain object creation.
        unsafe { self.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: kind, Count: count.max(1), NodeMask: 0 }, &mut heap) }.ok()?;
        let heap = heap?;
        name(&heap, || format!("query heap {} x{count}", kind.0));
        Some(heap)
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need. One made before is given again when
    /// nobody else holds it: every holder keeps its buffer until the list
    /// that copies into it has run (`defer_read_back`, `texback::Back`), so
    /// a buffer only the pool holds is not in use. dup-record saw about two
    /// CreateCommittedResource calls a frame here (2026-11-25).
    /// `VICTORY_READ_BACK_POOL=0`: a new buffer each time, as before.
    ///
    /// Buffers up to 16 MiB were kept, 32 at most, never let go: in run
    /// play-f's open world 12 a frame were still made against 6 given
    /// again, and the textures-read-back step of the lists' close took
    /// 74 ms a frame (all threads), about 6 ms a buffer. Now buffers up to
    /// `LARGEST` are kept, up to `KEPT_BYTES` in all, and one nobody holds
    /// makes room for a new size when the pool is full.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        /// Buffers up to this size are kept: up to 16 MiB rounded up to a
        /// power of two from 64 KiB, larger ones to a multiple of 8 MiB.
        const LARGEST: u64 = 256 << 20;
        const KEPT_BYTES: u64 = 1 << 30;
        static POOL: Mutex<Vec<(u64, ID3D12Resource)>> = Mutex::new(Vec::new());
        static ON: OnceLock<bool> = OnceLock::new();
        // How many are kept; with `VICTORY_READ_BACK_KEEP=<n>` more (play-g:
        // with 48, each frame made about 11 while lists in the air held the
        // kept ones, slower as the run went on), and the pool's lock is not
        // held while one is made.
        static KEEP: OnceLock<Option<usize>> = OnceLock::new();
        let keep = *KEEP.get_or_init(|| {
            std::env::var("VICTORY_READ_BACK_KEEP").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n > 0).map(|n| n.min(4096))
        });
        let kept = keep.unwrap_or(48);
        // SAFETY: a live buffer of the pool's; the count before the
        // reference taken here tells whether anyone else holds it, and only
        // the pool hands it out, under its lock.
        let idle = |buffer: &ID3D12Resource| unsafe {
            let vtable = Interface::vtable(buffer);
            let count = (vtable.base__.base__.base__.base__.AddRef)(buffer.as_raw());
            (vtable.base__.base__.base__.base__.Release)(buffer.as_raw());
            count == 2
        };
        let pooled = size <= LARGEST && *ON.get_or_init(|| std::env::var_os("VICTORY_READ_BACK_POOL").is_none_or(|v| v != "0"));
        let size = match size {
            _ if !pooled => size,
            0..=0x100_0000 => size.max(64 << 10).next_power_of_two(),
            _ => size.next_multiple_of(8 << 20),
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ReadBackPoolNs, started.elapsed().as_nanos() as u64);
        if pooled
            && let Some((_, buffer)) = pool.iter().find(|(kept, buffer)| *kept == size && idle(buffer))
        {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::ReadBacksReused, 1);
            return Some(buffer.clone());
        }
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::ReadBacksMade, 1);
            crate::copystat::add(crate::copystat::C::ReadBacksMadeBytes, size);
        }
        let mut pool = if keep.is_some() {
            drop(pool);
            None
        } else {
            Some(pool)
        };
        #[cfg(feature = "diagnostics")]
        let making = std::time::Instant::now();
        let made = self.new_read_back(size);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ReadBacksMadeNs, making.elapsed().as_nanos() as u64);
        match made {
            Ok(buffer) => {
                if pooled && let Some(buffer) = &buffer {
                    let mut pool = pool.take().unwrap_or_else(|| POOL.lock().unwrap_or_else(|e| e.into_inner()));
                    let full = |pool: &Vec<(u64, ID3D12Resource)>| {
                        pool.len() >= kept || pool.iter().map(|(kept, _)| kept).sum::<u64>() + size > KEPT_BYTES
                    };
                    while full(&pool) {
                        let Some(i) = pool.iter().position(|(_, b)| idle(b)) else {
                            break;
                        };
                        pool.swap_remove(i);
                    }
                    if !full(&pool) {
                        pool.push((size, buffer.clone()));
                    }
                }
                buffer
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    /// A new buffer of `size` bytes in the read-back heap, in the state
    /// copies into it need.
    fn new_read_back(&self, size: u64) -> windows::core::Result<Option<ID3D12Resource>> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        }?;
        Ok(buffer)
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    buffer_not_resident(),
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    buffer_resident(&self.device, b)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        #[cfg(feature = "diagnostics")]
        if made.is_ok()
            && let Some(b) = &buffer
        {
            crate::book::made(b, || "an upload buffer".into());
        }
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

/// Swap chain back buffers, VICTORY_SWAP_BUFFERS (2 to 16, default 2). With
/// 3, a frame can render while the display still holds the other two.
fn buffers() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_SWAP_BUFFERS").ok().and_then(|v| v.trim().parse().ok());
        n.map_or(2, |n: u32| n.clamp(2, 16))
    })
}

/// The swap chain and whether making or resizing it may be tried now: a
/// desktop can become presentable again, and DXGI calls during an outage
/// are throttled to one a second.
#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    /// `Some(result)` when the call may be made now (a failed call waits a
    /// second before the next), `None` while it may not.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if self.next_attempt.is_some_and(|next| now < next) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
    /// Puts a present's two planes together ([`crate::compose`]); made at
    /// the first present of two.
    composer: Option<crate::compose::Composer>,
    #[cfg(feature = "diagnostics")]
    panel: panel::Panel,
    menu: menu::Menu,
    #[cfg(feature = "diagnostics")]
    frames: frames::Frames,
    /// The debug overlay (Ctrl+F10), drawn into each frame it shows.
    overlay: crate::overlay::Overlay,
}

/// The times between presents over the last 5 s: their median, 95th and
/// 99th percentiles, the longest, and how many were over 50 and 100 ms (a
/// frame rate's steadiness, which the averages of the frame-rate line hide).
#[cfg(feature = "diagnostics")]
fn frame_times_line(mut ms: Vec<f32>) {
    if ms.is_empty() {
        return;
    }
    ms.sort_by(f32::total_cmp);
    let at = |q: f32| ms[((ms.len() - 1) as f32 * q).round() as usize];
    let over = |limit: f32| ms.iter().filter(|&&m| m > limit).count();
    tracing::info!(
        "frame times over the last 5 s: {} frames, median {:.1} ms, p95 {:.1}, p99 {:.1}, longest {:.1}; {} over 50 ms, {} over 100 ms",
        ms.len(),
        at(0.5),
        at(0.95),
        at(0.99),
        ms[ms.len() - 1],
        over(50.0),
        over(100.0)
    );
}

/// The last second's frame rate in tenths, for the F1 menu's line.
pub(crate) static METER_FPS_TENTHS: AtomicU32 = AtomicU32::new(0);

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
    /// The last present, and the times between presents since the last
    /// line of every 5 s (ms), for the frame-time line ([`frame_times_line`]).
    #[cfg(feature = "diagnostics")]
    last_frame: std::time::Instant,
    #[cfg(feature = "diagnostics")]
    intervals: Vec<f32>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
            #[cfg(feature = "diagnostics")]
            last_frame: now,
            #[cfg(feature = "diagnostics")]
            intervals: Vec::with_capacity(512),
        }
    }

    /// Whether the next frame counted makes the frame-rate line.
    #[cfg(feature = "diagnostics")]
    fn line_due(&self) -> bool {
        self.since.elapsed() >= std::time::Duration::from_secs(1)
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        crate::pipeline::presented();
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        // Launch time as the player sees it: from the process's creation,
        // which is before anything of ours runs.
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            use windows_sys::Win32::Foundation::FILETIME;
            use windows_sys::Win32::System::SystemInformation::GetSystemTimeAsFileTime;
            use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
            let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
            let (mut created, mut exited, mut kernel, mut user, mut now) = (zero, zero, zero, zero, zero);
            // SAFETY: our own process and valid out pointers.
            let ok = unsafe {
                GetSystemTimeAsFileTime(&mut now);
                GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user)
            } != 0;
            let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
            if ok {
                let ms = ticks(now).saturating_sub(ticks(created)) / 10_000;
                tracing::info!(ms, "first frame presented");
            }
        });
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
            let now = std::time::Instant::now();
            self.intervals.push((now - self.last_frame).as_secs_f32() * 1000.0);
            self.last_frame = now;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        METER_FPS_TENTHS.store((fps * 10.0).round() as u32, Ordering::Relaxed);
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        crate::overlay::second(crate::overlay::Second {
            draws,
            prepare_ms: cpu,
            gpu_wait_ms: gpu,
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        let refused = BUDGET_NOTES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        let biggest = BUDGET_NOTES[1].swap(0, Ordering::Relaxed);
        let empty = BUDGET_NOTES[2].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame; {refused:.2} indirect records a frame ran empty for want of their list's budget (2^30), the largest cost of one {biggest} (0: under 2^22); {empty:.0} indirect records a frame ran empty as asked (no work, or a slot the title had not filled)"
        );
        {
            let texture_us = crate::draw::HEAP_TEXTURE_MICROS.swap(0, Ordering::Relaxed);
            // (Neptune) Where that time went (`draw::HEAP_TEXTURE_PARTS`).
            let p = crate::draw::HEAP_TEXTURE_PARTS.each_ref().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames);
            let ms = |us: f64| us / 1e3;
            if texture_us > 0 {
                tracing::info!(
                    "the pass over textures picked by index per frame: {:.1} ms; {:.0} slots looked at, {:.0} uploads recorded; \
                     {:.1} ms before the slots; {:.1} ms looking at slots (views, mapping, room, handing out), \
                     {:.1} ms making host slots and copies ({:.1} host copies made in {:.1} ms), \
                     {:.1} ms staging and recording uploads ({:.1} staging buffers made in {:.1} ms, \
                     {:.1} of their upload buffers made new in {:.1} ms), {:.1} ms the rest (mapped whole, fingerprint, queue); \
                     {:.1} slots handed to the prepare threads, {:.1} passed over waiting for them",
                    texture_us as f64 / 1e3 / frames,
                    p[0], p[1], ms(p[14]), ms(p[2]), ms(p[3]), p[6], ms(p[7]), ms(p[4]), p[8], ms(p[9]), p[10], ms(p[11]),
                    ms(p[5]), p[12], p[13]
                );
            }
            if crate::texture::prepare::threads() > 0 {
                let s = crate::texture::prepare::STATS.each_ref().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames);
                let [pending, ready, out] = crate::texture::prepare::census();
                tracing::info!(
                    "texture prepare threads per frame (VICTORY_TEXTURE_PREPARE_THREADS={}): {:.1} textures made ready ({:.1} MB), \
                     {:.1} host copies made; {:.1} ms of the threads' time ({:.1} ms making host copies, {:.1} ms making staging buffers, \
                     {:.1} ms reading and detiling); {:.1} found the card without room, {:.1} made ready and let go untaken; \
                     now {pending} handed out, {ready} ready, {} MB out",
                    crate::texture::prepare::threads(),
                    s[0], s[2] / 1e6, s[1], ms(s[3]), ms(s[4]), ms(s[5]), ms(s[6]), s[7], s[8], out >> 20
                );
            }
        }
        upq::line(frames);
        crate::shadow::exact_line(frames);
        crate::descriptor::sync_line(frames);
        if heap_sync_first() {
            let [first, again] = [0, 1].map(|i| HEAP_SYNC_FIRST[i].swap(0, Ordering::Relaxed) as f64 / frames);
            tracing::info!(
                "the heaps' records begun as a submission took the lock, per frame (VICTORY_HEAP_SYNC_FIRST=1): {first:.1}, and \
                 {again:.1} begun again after a wait for flights"
            );
        }
        late_line(frames);
        slot_wait::log(frames);
        crate::queue::behind_line(frames);
        card_clock_line(frames, ms);
        if place::forwarding() {
            tracing::info!(
                "fills of upload memory that took their bytes from a copy at its place, per frame (VICTORY_FILL_FORWARD=1): {:.1}",
                place::FORWARDED.swap(0, Ordering::Relaxed) as f64 / frames
            );
        }
        #[cfg(feature = "diagnostics")]
        spans::line(frames);
        crate::shadow::back_line(frames);
        fresh_open_line(frames);
        {
            let [begun, made_again, more] = [0, 1, 2].map(|i| ASKED_FIRST[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if begun > 0.0 {
                tracing::info!(
                    "the submissions' asking begun beside their first steps, per frame (VICTORY_ASK_FIRST=1): {begun:.1}, {made_again:.1}                      made again after flights were waited for, {more:.1} asked for more where it was made before"
                );
            }
        }
        #[cfg(feature = "diagnostics")]
        split_line(frames);
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            let wc_left = crate::shadow::WC_LEFT.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames;
            if passes + compares + wc_left > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes; \
                     {wc_left:.1} MB declared written in write-combined memory left on the card (VICTORY_WC_NO_BACK)",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        {
            let [own, asking, still, all] =
                [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it); the rest is reading, hashing and detiling"
                );
            }
            let [through, left, filled, short] =
                [0, 1, 2, 3].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left + filled > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there), \
                     {filled:.1} first fillings from the pages there (VICTORY_TEXTURE_FILL=0: left empty), {short:.1} of them \
                     with less than was mapped (looked at again)"
                );
            }
            let [read, waited] =
                [0, 1].map(|i| crate::texture::EXACT_UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if read + waited > 0.0 {
                tracing::info!(
                    "exact texture refreshes (uploads before a bake) that found title memory being remapped, per frame: \
                     {read:.1} read whole through the kernel without waiting, {waited:.1} waited (a page was not there) \
                     (VICTORY_EXACT_NO_WAIT=0: all wait)"
                );
            }
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        // Passes over the heap's buffer views for shaders that pick them by index.
        let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_ms =
            crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        // Copies between depth textures and their colour twins, each way.
        let [twin_to_depth, twin_to_colour] =
            [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Depth planes drawn from their twins.
        let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        // Copies in the card's memory (`crate::shadow::cards`).
        let card = crate::shadow::card_stats();
        #[cfg(feature = "diagnostics")]
        let slow = crate::watch::slow_asks();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            frame_times_line(std::mem::take(&mut self.intervals));
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed)
            );
            let short = &crate::draw::DYNAMIC_SHORT;
            let (found, refused) = (short[0].load(Ordering::Relaxed), short[1].load(Ordering::Relaxed));
            if found + refused > 0 {
                tracing::info!(
                    "dynamic descriptor slots run out, since the start: {found} found after settling the flights in the air, {refused} refused (their draws dropped; VICTORY_TABLE_SETTLE=0: not settled)"
                );
            }
            crate::tiles::status();
            {
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} dropped), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
                if !crate::descriptor::host::queue_host_textures() {
                    tracing::info!(
                        "records of textures the host writes left out of the queue (VICTORY_QUEUE_HOST_TEXTURES=0), since the start: {}",
                        crate::descriptor::host::HOST_TEXTURES_LEFT_OUT.load(Ordering::Relaxed)
                    );
                }
            }
            if crate::resource::held_only() {
                let [copies, bytes, grown, whole] = [0, 1, 2, 3].map(|i| crate::resource::HELD[i].load(Ordering::Relaxed));
                let [flipped, missed] = [0, 1].map(|i| crate::descriptor::host::FLIPPED[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures from the first level viewed (VICTORY_HELD_LEVELS=0: whole), since the start: {copies} host copies of the \
                     levels viewed alone ({} MB on the card, {} MB of title memory whole), {grown} replaced by one with more levels, \
                     {flipped} slots made from their new view after its upload had run ({missed} written again first); {} uploads left \
                     out as the host copy held those bytes (VICTORY_TEXTURE_SAME=0: made); {} MB of texture blocks free; \
                     {} looks for textures to let go, {:.2} ms each, the longest {:.1}",
                    bytes >> 20,
                    whole >> 20,
                    crate::draw::HEAP_TEXTURES_SAME.load(Ordering::Relaxed),
                    crate::pool::texture_room() >> 20,
                    crate::resource::SWEPT_US[0].load(Ordering::Relaxed),
                    crate::resource::SWEPT_US[1].load(Ordering::Relaxed) as f64
                        / 1e3
                        / crate::resource::SWEPT_US[0].load(Ordering::Relaxed).max(1) as f64,
                    crate::resource::SWEPT_US[2].load(Ordering::Relaxed) as f64 / 1e3
                );
            }
            {
                let mode = crate::watch::page_writes();
                let [writes, bytes, guarded, plain] = [0, 1, 2, 3].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                let [kept, dropped, slots, uploaded] =
                    [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                tracing::info!(
                    "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the                      title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to                      plain addresses finished ({}); {kept} ranges under texture views kept for the texture pass ({dropped} dropped);                      {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({})                      (VICTORY_PAGE_WRITES: unset or guards opens the guarded blocks, 0 counts only, 1, slots, all)",
                    bytes >> 20,
                    if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                    if mode & 4 != 0 { "told as well" } else { "counted only" },
                    if mode & 2 != 0 { "made due again" } else { "left as they were" }
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                     ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
            }
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms;                  texture slots written so far: {placed} placed, {objects} of objects; \
                 textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                 {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                refused = format!(
                    "{} ({} copies made of views off their buffer's elements); depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
                    crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                    crate::shadow::OWN_MADE.load(Ordering::Relaxed),
                    crate::resource::TWINS[0].load(Ordering::Relaxed),
                    crate::resource::TWINS[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[2].load(Ordering::Relaxed)
                ),
                making = crate::pipeline::making(),
                placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
            );
            #[cfg(feature = "diagnostics")]
            {
                let put = [0, 1, 2, 3].map(|i| crate::shadow::PUT[i].swap(0, Ordering::Relaxed) as f64 / frames);
                tracing::info!(
                    "what shaders wrote put into title memory per frame (VICTORY_EXACT_BACK): {:.0} pieces in {:.0} writes, {:.1} MB; {:.0} pieces joined over gaps",
                    put[0],
                    put[1],
                    put[2] / 1e6,
                    put[3]
                );
                traffic::line(self.copy_stats_frames as f64);
                let report_frames = std::mem::take(&mut self.copy_stats_frames);
                crate::copystat::log(report_frames as f64);
                // The caption has a shorter interval. Keep the reasons until
                // this report so the counts and frame total cover the same window.
                for (why, count) in crate::pipeline::take_drops().into_iter().take(40) {
                    let per_frame = count as f64 / report_frames as f64;
                    tracing::info!(count, frames = report_frames, "dropped {per_frame:.1}/frame: {why}");
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on), with `overlay`, in its
    /// own host state (which it stays in), put over it ([`crate::compose`]).
    /// Planes the composer cannot read, or no composer: `texture` alone.
    /// `None` if nothing was shown; else, in diagnostics builds, whether the
    /// frame was black.
    pub(crate) fn present_planes(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: Option<(&ID3D12Resource, u32)>,
        frame: u64,
    ) -> Option<bool> {
        static NO_COMPOSER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        // (Handed over only to be saved, with `VICTORY_TWO_PLANES=0`: not shown.)
        let planes = overlay.filter(|_| crate::compose::on() && !NO_COMPOSER.load(Ordering::Relaxed)).and_then(|(over, over_state)| {
            // SAFETY: a live resource.
            let over_desc = unsafe { over.GetDesc() };
            Some((
                over,
                D3D12_RESOURCE_STATES(over_state as i32),
                crate::compose::view(&desc)?,
                crate::compose::view(&over_desc)?,
            ))
        });
        let format = if planes.is_some() {
            DXGI_FORMAT_R8G8B8A8_UNORM
        } else {
            let Some(format) = swap_format(desc.Format) else {
                tracing::warn!(format = desc.Format.0, "cannot present this format");
                return None;
            };
            format
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        if presenter.is_none() {
            *presenter = match retry.attempt(Instant::now(), || self.presenter(width, height, format))? {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if planes.is_some() && p.composer.is_none() {
            match crate::compose::Composer::new(self, format) {
                Ok(composer) => {
                    tracing::info!(frame, width, height, "presents of two planes are put together");
                    p.composer = Some(composer);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no composer for two planes; the first is shown alone");
                    NO_COMPOSER.store(true, Ordering::Relaxed);
                    drop(presentation);
                    return self.present_planes(texture, state, None, frame);
                }
            }
        }
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission was waited for.
                unsafe {
                    p.swap_chain
                        .ResizeBuffers(buffers(), width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
                }
            })?;
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            #[cfg(feature = "diagnostics")]
            let read_now = (crate::capture::wanted(frame) && !crate::capture::ahead()) || crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            // The overlay rewrites its upload buffer between presents, so
            // while it shows each present is waited for.
            let overlay_shown = crate::overlay::shown();
            let read_now = read_now || overlay_shown;
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            let list = &copy.list;
            let composed = match (planes, p.composer.as_mut()) {
                (Some((over, over_state, base_view, over_view)), Some(composer)) => {
                    const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                    copy.keep(texture);
                    copy.keep(over);
                    let moves = |to_read: bool, back_from: D3D12_RESOURCE_STATES, back_to: D3D12_RESOURCE_STATES| {
                        let mut moves = Vec::with_capacity(3);
                        for (plane, at) in [(texture, state), (over, over_state)] {
                            if at != READ {
                                moves.push(if to_read { transition(plane, at, READ) } else { transition(plane, READ, at) });
                            }
                        }
                        moves.push(transition(&back, back_from, back_to));
                        moves
                    };
                    list.ResourceBarrier(&moves(true, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
                    let views = slot.map_or(0, |i| i + 1);
                    match composer.record(self, views, list, [(texture, base_view), (over, over_view)], &back, (width, height)) {
                        Ok(()) => {
                            list.ResourceBarrier(&moves(false, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST));
                            true
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "two planes not put together; the first is shown alone");
                            list.ResourceBarrier(&moves(false, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));
                            false
                        }
                    }
                }
                _ => false,
            };
            // A texture already in the copy source state takes no barrier
            // to it and back (the same state on both sides is refused).
            let moved = state != D3D12_RESOURCE_STATE_COPY_SOURCE;
            if !composed {
                let mut pre = Vec::with_capacity(2);
                if moved {
                    pre.push(transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                pre.push(transition(&back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST));
                list.ResourceBarrier(&pre);
                list.CopyResource(&back, texture);
            }
            // (A frame put together is read from the back buffer.)
            let shape = (width, height, format);
            p.overlay.record(self, list, &back, shape, p.hwnd, D3D12_RESOURCE_STATE_COPY_DEST, overlay_shown);
            // The picture shown is read from the back buffer, put together
            // or copied; `VICTORY_CAPTURE_PLANE` reads one plane instead.
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| {
                    let at = if composed { state } else { D3D12_RESOURCE_STATE_COPY_SOURCE };
                    let (read, at): (&ID3D12Resource, _) = match crate::capture::plane() {
                        Some(crate::capture::Plane::Scene) => (texture, at),
                        Some(crate::capture::Plane::Interface) => {
                            let (over, over_state) = overlay?;
                            (over, D3D12_RESOURCE_STATES(over_state as i32))
                        }
                        None => (&back, D3D12_RESOURCE_STATE_COPY_DEST),
                    };
                    let moved = at != D3D12_RESOURCE_STATE_COPY_SOURCE;
                    if moved {
                        list.ResourceBarrier(&[transition(read, at, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    }
                    let taken = crate::capture::read_back(&self.device, list, read);
                    if moved {
                        list.ResourceBarrier(&[transition(read, D3D12_RESOURCE_STATE_COPY_SOURCE, at)]);
                    }
                    taken
                })
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() || composed { None } else { crate::capture::probe(&self.device, list, texture) };
            #[cfg(feature = "diagnostics")]
            let watched = if composed { None } else { crate::popwatch::take(&self.device, list, texture) };
            #[cfg(feature = "diagnostics")]
            p.panel.draw(self, copy, &back, (width, height), format);
            p.menu.draw(self, copy, &back, (width, height), format);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            if composed {
                list.ResourceBarrier(&[transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                )]);
            } else {
                let mut post = Vec::with_capacity(2);
                if moved {
                    post.push(transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state));
                }
                post.push(transition(&back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT));
                list.ResourceBarrier(&post);
            }
            copy.close();
            let flown = match slot {
                Some(i) => {
                    let flown = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                    flown
                }
                None => {
                    self.execute(&[copy]);
                    0
                }
            };
            #[cfg(feature = "diagnostics")]
            if let Some(taken) = watched {
                crate::popwatch::sent(taken, flown, frame);
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = flown;
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                if slot.is_some() {
                    crate::capture::save_after(frame, buffer, footprint, flown);
                } else {
                    crate::capture::save_read_back(frame, &buffer, &footprint);
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::replay::record::present(texture, state);
            let mut showing = crate::timeline::span("the host's Present");
            #[cfg(feature = "diagnostics")]
            let presenting = std::time::Instant::now();
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            showing.next("the frame meter");
            #[cfg(feature = "diagnostics")]
            let line = p.meter.line_due();
            #[cfg(feature = "diagnostics")]
            p.frames.frame(self, presenting.elapsed());
            #[cfg(feature = "diagnostics")]
            p.panel.frame(line);
            p.menu.frame();
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            if line {
                p.frames.after();
            }
            showing.end();
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: buffers(),
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            // What the title draws against what the screen shows. A title that
            // renders 3840x2160 for a 2560x1600 panel pays 2.25 times the
            // pixels, and 9 times for a 1280x720 one; a run reported as slow
            // that draws the console's size while the mode it was given was
            // the screen's own needs this line to say so, since nothing else
            // in the log compares the two.
            //
            // Of the switches on this path none lowers it, and a reader who
            // takes the ratio for a switch to turn is misled:
            // `VICTORY_DISPLAY_MODE` names the mode the title is given, and a title that ignores the mode renders
            // its own targets anyway (the first 5070 Laptop run reported
            // 3840x2160 either way); `VICTORY_WINDOW_MAX` fits the window to
            // the screen without changing the title's render targets
            // (`victory_hle::modules::lifecycle::fit`).
            //
            // The switch that does lower it is not here: `VICTORY_OUTPUT`
            // (`victory_loader::resolution`) changes the title's own settings
            // at load, so the size it renders into is the size asked for, and
            // this line then reports the smaller one. Without it a ratio well
            // above 1 is a diagnosis of a slow run, not a lever.
            let screen = crate::dxgi::host_display_size();
            let drawn_of_the_screen = screen.map(|(w, h)| {
                (f64::from(width) * f64::from(height)) / (f64::from(w) * f64::from(h))
            });
            tracing::info!(
                width,
                height,
                screen = ?screen,
                drawn_of_the_screen,
                format = format.0,
                "presenting to a window"
            );
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
                composer: None,
                #[cfg(feature = "diagnostics")]
                panel: panel::Panel::new(),
                menu: menu::Menu::new(),
                #[cfg(feature = "diagnostics")]
                frames: frames::Frames::new(self),
                overlay: crate::overlay::Overlay::default(),
            })
        }
    }
}

/// The property our hook of the title's window procedure puts on the
/// windows the title makes (`TITLE_PROC` in `victory_hle::modules::lifecycle`).
const TITLE_HOOK: PCWSTR = w!("victory.wndproc");

/// A visible top-level window of this process, as [`title_window`] finds it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    hwnd: usize,
    /// Carries [`TITLE_HOOK`]: a window the title made.
    hooked: bool,
    /// Its client area is not empty.
    sized: bool,
}

/// Which of `candidates` (in z-order, topmost first) is the title's: the
/// one our hook marked, else the first with a client area. Pure, for testing.
fn pick_title_window(candidates: &[Candidate]) -> Option<usize> {
    candidates
        .iter()
        .find(|c| c.hooked)
        .or_else(|| candidates.iter().find(|c| c.sized))
        .map(|c| c.hwnd)
}

/// A visible top-level window of this process: the one the title made, if
/// it made one. Not simply the first visible one: started under a
/// pseudoconsole (from Windows Terminal, or by a process that has one), this
/// process also owns a visible `PseudoConsoleWindow` of 0x0 pixels. When
/// that one was above the title's window in z-order, the swap chain was made
/// for it, and the title's window stayed white with no frame ever shown.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                let mut area = RECT::default();
                let sized = GetClientRect(hwnd, &mut area).is_ok() && area.right > area.left && area.bottom > area.top;
                let hooked = !GetPropW(hwnd, TITLE_HOOK).0.is_null();
                (*(found.0 as *mut Vec<Candidate>)).push(Candidate { hwnd: hwnd.0 as usize, hooked, sized });
            }
        }
        true.into()
    }
    let mut found: Vec<Candidate> = Vec::new();
    // SAFETY: the callback only adds to `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    let picked = pick_title_window(&found);
    if !found.iter().any(|c| c.hooked) {
        tracing::info!(
            windows = found.len(),
            "no window of the title's among ours: presenting to the first with a client area, if any"
        );
    }
    picked.map(|hwnd| HWND(hwnd as *mut _))
}

#[cfg(test)]
mod title_window_tests {
    use super::{Candidate, pick_title_window};

    fn window(hwnd: usize, hooked: bool, sized: bool) -> Candidate {
        Candidate { hwnd, hooked, sized }
    }

    /// A pseudoconsole's 0x0 window above the title's in z-order.
    #[test]
    fn the_titles_window_wins_over_a_pseudo_console_above_it() {
        let found = [window(1, false, false), window(2, true, true)];
        assert_eq!(pick_title_window(&found), Some(2));
    }

    #[test]
    fn the_titles_window_counts_even_while_its_client_area_is_empty() {
        let found = [window(1, false, true), window(2, true, false)];
        assert_eq!(pick_title_window(&found), Some(2));
    }

    #[test]
    fn without_one_of_the_titles_the_first_with_a_client_area() {
        let found = [window(1, false, false), window(2, false, true), window(3, false, true)];
        assert_eq!(pick_title_window(&found), Some(2));
    }

    #[test]
    fn a_pseudo_console_alone_is_no_window_to_present_to() {
        assert_eq!(pick_title_window(&[window(1, false, false)]), None);
        assert_eq!(pick_title_window(&[]), None);
    }
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A shader-visible view heap of one null buffer view, for lists that
    /// set no view heap (codec_d, pareto-residency e05d4633a).
    pub(crate) fn fallback_view_heap(&self) -> Option<&ID3D12DescriptorHeap> {
        self.fallback_view
            .get_or_init(|| {
                // SAFETY: a heap of one descriptor and a null view in it.
                unsafe {
                    let heap: ID3D12DescriptorHeap = self
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .ok()?;
                    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_UINT,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV::default() },
                    };
                    self.device.CreateShaderResourceView(None, Some(&desc), heap.GetCPUDescriptorHandleForHeapStart());
                    Some(heap)
                }
            })
            .as_ref()
    }

    /// A shader-visible sampler heap of one point sampler, for lists that
    /// set no sampler heap.
    pub(crate) fn fallback_sampler_heap(&self) -> Option<&ID3D12DescriptorHeap> {
        self.fallback_sampler
            .get_or_init(|| {
                // SAFETY: a heap of one descriptor and a sampler in it.
                unsafe {
                    let heap: ID3D12DescriptorHeap = self
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .ok()?;
                    let desc = D3D12_SAMPLER_DESC {
                        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        MaxLOD: f32::MAX,
                        ..Default::default()
                    };
                    self.device.CreateSampler(&desc, heap.GetCPUDescriptorHandleForHeapStart());
                    Some(heap)
                }
            })
            .as_ref()
    }

    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        if let Some(made) = crate::pool::shared(&self.device, size) {
            return Some(made);
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    buffer_not_resident(),
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    buffer_resident(&self.device, b)?;
                    buffer_mapped(&self.device)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        #[cfg(feature = "diagnostics")]
        if made.is_ok()
            && let Some(b) = &buffer
        {
            crate::book::made(b, || "a shared buffer".into());
        }
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// Whether buffers can lie in the card's memory and be mapped by the CPU
    /// (`D3D12_HEAP_TYPE_GPU_UPLOAD`: a resizable BAR, and a system and driver
    /// that offer it); logged once.
    pub(crate) fn mapped_card_memory(&self) -> bool {
        static THERE: OnceLock<bool> = OnceLock::new();
        *THERE.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS16,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.GPUUploadHeapSupported.as_bool();
            tracing::info!(supported, "card memory the CPU maps");
            supported
        })
    }

    /// A buffer in the card's own memory that the CPU maps, shaders read and
    /// (with `written`) may write. The CPU should only write it: a read of the
    /// mapping crosses the bus a few bytes at a time.
    pub(crate) fn mapped_card_buffer(&self, size: u64, written: bool) -> Option<(ID3D12Resource, *mut u8)> {
        if !self.mapped_card_memory() {
            return None;
        }
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if written {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    buffer_not_resident(),
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    buffer_resident(&self.device, b)?;
                    hot(&self.device, b);
                    buffer_mapped(&self.device)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no buffer in card memory the CPU maps");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                buffer_not_resident(),
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        #[cfg(feature = "diagnostics")]
        if let Some(b) = &buffer {
            crate::book::made(b, || "an unordered access buffer".into());
        }
        let made = made.and_then(|()| buffer.as_ref().map_or_else(|| buffer_mapped(&self.device), |b| if wait_resident_buffers() { resident_soon(&self.device, b, true) } else { Ok(()) }));
        if made.is_ok()
            && let Some(b) = &buffer
        {
            hot(&self.device, b);
        }
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
/// For a list that stopped at command `done`: what the indirect executions
/// there ran, from the one before it to the next barrier (four at most), and
/// what the heap slots their user data names hold now.
#[cfg(feature = "diagnostics")]
fn indirects_around(history: &[D3D12_AUTO_BREADCRUMB_OP], done: usize, all: &indirects::Shared) -> String {
    let indirect = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT;
    let all = all.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    let mut shown = 0;
    let mesh = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH;
    for (at, op) in history.iter().enumerate().skip(done.saturating_sub(1)) {
        if shown == 4 || (at > done && *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER) {
            break;
        }
        if !indirect(op) && !mesh(op) {
            continue;
        }
        shown += 1;
        let direct = mesh(op);
        let nth = if direct { indirects::before(history, at, mesh) } else { indirects::before(history, at, indirect) };
        let Some((what, words)) = all.nth(nth, direct) else {
            continue;
        };
        text.push_str(&format!("; command {at} ran {what}"));
        for (word, slot, what) in crate::descriptor::host::describe_slots(words) {
            text.push_str(&format!(", word {word} slot {slot}: {what}"));
        }
    }
    text
}

/// The first `count` commands of a breadcrumb node's history, up to the
/// first page that cannot be read: the runtime's count can run past the
/// history it committed (a report on an RTX 3070 Ti faulted reading
/// command 15,360 of a long list).
fn readable_history(history: *const D3D12_AUTO_BREADCRUMB_OP, count: u32) -> Vec<D3D12_AUTO_BREADCRUMB_OP> {
    let mut ops = Vec::new();
    let mut readable = true;
    // Page by page: the kernel copies a range whole or fails it.
    for_pieces(history as usize, 4 * count as usize, 4 << 10, |at, n| {
        let mut piece = vec![D3D12_AUTO_BREADCRUMB_OP(0); n / 4];
        readable = readable && read(at as *const u8, piece.as_mut_ptr().cast(), n);
        if readable {
            ops.extend(piece);
        }
    });
    ops
}

/// Where a list stopped, from the part of its history that could be read
/// and the runtime's count of its commands.
fn stopped_at(history: &[D3D12_AUTO_BREADCRUMB_OP], count: u32, done: u32) -> String {
    let op = history.get(done as usize).map_or_else(|| "unknown".to_owned(), |op| op.0.to_string());
    let mut text = format!("; a list stopped at command {done} of {count} (op {op})");
    if history.len() < count as usize {
        text.push_str(&format!(", but only its first {} commands can be read", history.len()));
    }
    text
}

fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// What the first list of a part sent in one call behind others waits for
/// ([`join_one_call`]): every shader write before it (an unordered-access
/// barrier on no resource) and every use of memory before it (an aliasing
/// barrier between no resources), as a call of its own made the card wait
/// before. Lists of one `ExecuteCommandLists` may overlap on the card; lists
/// of two calls do not. (Skyfrye's, AnyXboxX#337.)
fn join_barriers() -> [D3D12_RESOURCE_BARRIER; 2] {
    [
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        },
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    pResourceAfter: std::mem::ManuallyDrop::new(None),
                }),
            },
        },
    ]
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let display = borderless().then(primary_display).flatten();
        let place = window_place(width, height, display)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            place.style | WS_VISIBLE,
            place.x,
            place.y,
            place.width,
            place.height,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

/// `VICTORY_BORDERLESS=1`: the window covers the primary display with no border or title bar
/// (windowed borderless, not exclusive full screen: the swap chain stretches the picture to it,
/// the title's render size is unchanged, alt-tab and other windows work as before).
fn borderless() -> bool {
    std::env::var("VICTORY_BORDERLESS").is_ok_and(|v| v.trim() == "1")
}

/// The primary display's rectangle on the desktop; `None` when it cannot be asked.
fn primary_display() -> Option<RECT> {
    use windows::Win32::Foundation::POINT;
    use windows::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MONITOR_DEFAULTTOPRIMARY, MONITORINFO, MonitorFromPoint,
    };
    let mut info = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: the primary display's handle, and a MONITORINFO of the size it declares.
    let asked = unsafe {
        let monitor = MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY);
        GetMonitorInfoW(monitor, &mut info).as_bool()
    };
    asked.then_some(info.rcMonitor)
}

/// Where the window goes and how it looks.
#[derive(Debug, PartialEq, Eq)]
struct WindowPlace {
    style: WINDOW_STYLE,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

/// A bordered window whose client area is `width` x `height` where the system puts it, or with a
/// display given, a borderless window that covers exactly that display.
fn window_place(
    width: u32,
    height: u32,
    display: Option<RECT>,
) -> windows::core::Result<WindowPlace> {
    if let Some(d) = display {
        return Ok(WindowPlace {
            style: WS_POPUP,
            x: d.left,
            y: d.top,
            width: d.right - d.left,
            height: d.bottom - d.top,
        });
    }
    let mut rect = RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    };
    // SAFETY: adjusts a rectangle we own.
    unsafe { AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)? };
    Ok(WindowPlace {
        style: WS_OVERLAPPEDWINDOW,
        x: CW_USEDEFAULT,
        y: CW_USEDEFAULT,
        width: rect.right - rect.left,
        height: rect.bottom - rect.top,
    })
}

#[cfg(test)]
mod window_place_tests {
    use super::*;

    #[test]
    fn a_display_given_makes_a_borderless_window_that_covers_it() {
        // A secondary display left of the primary one: negative desktop coordinates.
        let display = RECT {
            left: -2560,
            top: 0,
            right: 0,
            bottom: 1440,
        };
        let place = window_place(3840, 2160, Some(display)).unwrap();
        assert_eq!(
            place,
            WindowPlace {
                style: WS_POPUP,
                x: -2560,
                y: 0,
                width: 2560,
                height: 1440,
            }
        );
    }

    #[test]
    fn without_a_display_the_window_is_bordered_around_the_client_size() {
        let place = window_place(1920, 1080, None).unwrap();
        assert_eq!(place.style, WS_OVERLAPPEDWINDOW);
        assert_eq!((place.x, place.y), (CW_USEDEFAULT, CW_USEDEFAULT));
        // The border and title bar are added around the client area.
        assert!(place.width > 1920 && place.height > 1080);
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }

    /// A list's small first chunk takes a kept 4 MiB one when none of its own
    /// size is kept (`VICTORY_SCRATCH_ANY_SIZE`), and says so by its size; a
    /// kept small one comes first, and other sizes take only their own.
    #[test]
    fn a_small_first_chunk_takes_a_kept_4_mib_one() {
        use super::{SCRATCH_CHUNK, SCRATCH_FIRST, take_kept};
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.give(1, SCRATCH_CHUNK, u64::MAX), None);
        assert_eq!(take_kept(&mut spare, SCRATCH_FIRST, false), None, "switched off: its own size only");
        assert_eq!(take_kept(&mut spare, 8 << 20, true), None, "only the small first chunk");
        assert_eq!(take_kept(&mut spare, SCRATCH_FIRST, true), Some((1, SCRATCH_CHUNK)));
        assert_eq!(spare.bytes, 0);
        // A kept small one is taken before a 4 MiB one.
        assert_eq!(spare.give(2, SCRATCH_CHUNK, u64::MAX), None);
        assert_eq!(spare.give(3, SCRATCH_FIRST, u64::MAX), None);
        assert_eq!(take_kept(&mut spare, SCRATCH_FIRST, true), Some((3, SCRATCH_FIRST)));
        assert_eq!(take_kept(&mut spare, SCRATCH_CHUNK, true), Some((2, SCRATCH_CHUNK)));
        assert_eq!(take_kept(&mut spare, SCRATCH_FIRST, true), None);
    }
}

#[cfg(test)]
mod refresh_tests {
    use super::Batch;
    use crate::shadow::Shadow;
    use std::sync::Arc;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RESERVE, MapViewOfFile, PAGE_READWRITE,
        VirtualAlloc,
    };

    const PAGE: usize = 64 << 10;
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Title memory for the tests (never given back: copies of it stay in
    /// the guards): write-watched, or a view of a section.
    fn title(len: usize, view: bool) -> usize {
        // SAFETY: plain allocations.
        let at = unsafe {
            if view {
                let section =
                    CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
                // As the runtime tells of it.
                crate::title_memory_section(section as usize, len as u64, true);
                crate::title_memory_remapped(at, len);
                crate::title_memory_view(at, len, section as usize, 0);
                at
            } else {
                VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize
            }
        };
        assert_ne!(at, 0);
        at
    }

    /// Title memory of `blocks` pages in a placeholder, one view to a page,
    /// and every fourth page not there: as a title that streams has it.
    fn with_holes(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS,
            VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a placeholder and a section of our own.
        unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(at, 0);
            let len = blocks * PAGE;
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, (len >> 32) as u32, len as u32, std::ptr::null());
            for block in (0..blocks).filter(|b| b % 4 != 3) {
                let to = at + block * PAGE;
                VirtualFree(to as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    to as *const _,
                    (block * PAGE) as u64,
                    PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, to);
                crate::title_memory_remapped(to, PAGE);
            }
            at
        }
    }

    fn memory(len: usize) -> *mut u8 {
        // SAFETY: a plain allocation.
        unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast() }
    }

    struct Random(u64);

    impl Random {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as usize % n
        }
    }

    /// Prepares as a submission does: one asking for all the ranges, the
    /// refreshes in several flushes, the asking's end.
    fn submission(ranges: &[(Arc<Shadow>, usize, usize)], flushes: usize) {
        // One at a time, as submissions are.
        static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let asked: Vec<(usize, usize)> = ranges.iter().filter(|r| !r.0.unwatched()).map(|r| (r.1, r.1 + r.2)).collect();
        crate::watch::ask_once(&asked);
        let mut batch = Batch::default();
        for part in ranges.chunks(ranges.len().div_ceil(flushes).max(1)) {
            for (shadow, at, len) in part {
                batch.add(shadow, *at, *len);
            }
            batch.flush();
        }
        crate::watch::ask_once_end();
    }

    /// Whatever is written where, by the title or by us, and whichever
    /// copies (of both kinds of memory, over one another, none on a page
    /// line) are refreshed in which ranges: a refreshed range holds what
    /// title memory holds.
    #[test]
    fn refreshed_ranges_hold_title_memory() {
        // SAFETY (all blocks): title memory and copies of the test's own,
        // written by this thread and, in `Batch::flush`, by the threads that
        // refresh, which are done when it returns.
        let mut random = Random(0x1234_5678_9ABC_DEF1);
        const LEN: usize = 24 * PAGE;
        let regions = [title(LEN, false), title(LEN, true)];
        let mut copies = Vec::new();
        for &region in &regions {
            for (from, to) in [(0, LEN), (100, LEN - 5000), (3 * PAGE + 777, 9 * PAGE + 10), (8 * PAGE, 8 * PAGE + 300), (PAGE, 20 * PAGE)] {
                let data = memory(to - from);
                copies.push((Shadow::unbacked((region + from) as u64, data, (to - from) as u64, false), data));
            }
        }
        let byte = |at: usize| unsafe { (at as *const u8).read_volatile() };
        for round in 0..600 {
            for _ in 0..random.below(4) {
                let at = regions[random.below(2)] + random.below(LEN - 64);
                let n = 1 + random.below(60);
                if random.below(3) == 0 {
                    // What shaders wrote.
                    let was: Vec<u8> = (0..n).map(|i| byte(at + i)).collect();
                    let now: Vec<u8> = was.iter().map(|b| if random.below(2) == 0 { *b } else { random.below(256) as u8 }).collect();
                    Shadow::put(at, &was, &now);
                    assert!(now.iter().enumerate().all(|(i, &b)| byte(at + i) == b), "not put");
                    continue;
                }
                for i in 0..n {
                    unsafe { ((at + i) as *mut u8).write_volatile(random.below(256) as u8) };
                }
            }
            let mut ranges = Vec::new();
            for (shadow, _) in &copies {
                if random.below(3) == 0 {
                    continue;
                }
                let (start, len) = shadow.span();
                let (start, len) = (start as usize, len as usize);
                // Often the range of the time before, as a list's is.
                let from = if random.below(2) == 0 { 0 } else { random.below(len) };
                let n = if random.below(2) == 0 { len - from } else { 1 + random.below(len - from) };
                ranges.push((shadow.clone(), start + from, n));
            }
            if ranges.is_empty() {
                continue;
            }
            let flushes = 1 + random.below(3);
            submission(&ranges, flushes);
            for (shadow, at, n) in &ranges {
                let data = copies.iter().find(|c| Arc::ptr_eq(&c.0, shadow)).unwrap().1;
                let start = shadow.span().0 as usize;
                let same = unsafe {
                    std::slice::from_raw_parts(*at as *const u8, *n) == std::slice::from_raw_parts(data.add(at - start), *n)
                };
                assert!(same, "round {round}: {at:#x}+{n:#x} of a copy at {start:#x}; first byte {}", byte(*at));
            }
        }
    }

    /// A copy in the card's memory, with the card's side and the sends to it
    /// played by the test.
    struct Carded {
        shadow: Arc<Shadow>,
        data: usize,
        start: usize,
        len: usize,
        /// The last send, the last send every fetch to come is behind, and
        /// the card's bytes: the queue's order is the order it is locked in.
        card: std::sync::Mutex<(u64, u64, Vec<u8>)>,
        /// Whether a flight that writes it has not landed.
        flying: std::sync::atomic::AtomicBool,
        /// The dword of each eight its shaders write.
        class: usize,
    }

    impl Carded {
        fn new(start: usize, len: usize, class: usize) -> Carded {
            let data = memory(len);
            Carded {
                shadow: Shadow::unbacked(start as u64, data, len as u64, true),
                data: data as usize,
                start,
                len,
                card: std::sync::Mutex::new((0, 0, vec![0; len])),
                flying: std::sync::atomic::AtomicBool::new(false),
                class,
            }
        }

        /// What changed on the CPU's side goes up.
        fn send(&self) {
            let mut card = self.card.lock().unwrap();
            card.0 += 1;
            let (number, merged) = (card.0, card.1);
            for (a, b) in self.shadow.send_up(number, merged) {
                // SAFETY: the copy's bytes; nothing writes those that are to
                // go up.
                unsafe { std::ptr::copy_nonoverlapping((self.data + a) as *const u8, card.2[a..b].as_mut_ptr(), b - a) };
            }
        }
    }

    #[test]
    fn settle_room_merges_without_waiting_and_preserves_flight_bytes() {
        // Five 64 KiB blocks, each with 128 separate changes too long for
        // a word scatter. The flight also writes in the gaps between them.
        const LEN: usize = 5 * PAGE;
        const STRIDE: usize = 512;
        let mut expected = vec![0u8; LEN];
        for chunk in expected.chunks_exact_mut(STRIDE) {
            chunk[..320].fill(0x35);
            chunk[STRIDE - 4..].fill(0x72);
        }
        let run = |extra| {
            let at = title(LEN, false);
            let copy = Carded::new(at, LEN, 0);
            copy.shadow.update(at, LEN);
            copy.send();
            copy.shadow.writing(at, LEN);
            let (fence, fetched) = {
                let mut card = copy.card.lock().unwrap();
                for (i, chunk) in card.2.chunks_exact_mut(STRIDE).enumerate() {
                    chunk[STRIDE - 4..].fill(0x72);
                    // SAFETY: inside the test's private title allocation.
                    unsafe { std::ptr::write_bytes((at + i * STRIDE) as *mut u8, 0x35, 320) };
                }
                (card.0, card.2.clone())
            };
            let mut budget = super::settle_room::merge_budget(false, extra);
            let wait = copy.shadow.merge_alone(at, LEN, &mut budget);
            assert_eq!(wait, !extra, "640 changed runs");
            if !wait {
                copy.send();
            }
            copy.shadow.card_merge_relay(0, &fetched, fence, &mut Vec::new());
            copy.shadow.copy_back_done(&[(at, at + LEN)]);
            if wait {
                copy.shadow.update(at, LEN);
                copy.send();
            }
            // SAFETY: the CPU is finished writing both private allocations.
            for address in [at, copy.data] {
                let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, LEN) };
                assert!(bytes == expected, "title or upload bytes, extra room {extra}");
            }
            copy.card.into_inner().unwrap().2
        };
        let normal = run(false);
        assert!(normal == expected);
        assert!(run(true) == normal);
    }

    /// Everything at once: the title writes its memory on two threads, the
    /// submitting thread puts its bytes in under shader writes
    /// (`Shadow::merge_under_writes`), refreshes copies that overlap (two in
    /// the card's memory, one not), sends up what changed and declares what
    /// the next flight's shaders write, and what flights wrote is landed
    /// meanwhile (`crate::land`, beside the guards). When all of it stands
    /// still and the copies are refreshed once more, each of them, and the
    /// card's side of those that have one, hold what title memory holds, and
    /// title memory what was written last.
    ///
    /// Into the first copy shaders write besides through a view nobody
    /// declared, words the title does not write: nothing that goes up ever
    /// puts an older value over those.
    ///
    /// (It fails in the first rounds if `landed_beside` notes nothing, or
    /// only takes the block out of the guards' clean ones and leaves its bit,
    /// or if a fault leaves the bit; in some thousand rounds,
    /// `VICTORY_TEST_ROUNDS`, if what is landed is noted before it is in.)
    #[test]
    fn landings_title_writes_and_light_refreshes_at_once() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        assert!(crate::safecopy::on());
        const LEN: usize = 24 * PAGE;
        let rounds: usize = std::env::var("VICTORY_TEST_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let region = title(LEN, true);
        let dword = |at: usize| unsafe { &*((region + at) as *const AtomicU32) };
        let carded = [Carded::new(region, LEN, 3), Carded::new(region + 4 * PAGE + 1000, LEN - 6 * PAGE - 4, 5)];
        let plain = {
            let (start, len) = (region + 700, LEN - 9000);
            let data = memory(len);
            (Shadow::unbacked(start as u64, data, len as u64, false), start, len, data as usize)
        };
        // What was last written so, by offset.
        let mut undeclared = std::collections::HashMap::new();
        let exact = std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0");
        let stop = AtomicBool::new(false);
        // Also when an assertion fails on one of the threads.
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        // Held shared by who writes title memory, alone while it is looked at.
        let still = std::sync::RwLock::new(());
        let (fly, flights) = std::sync::mpsc::channel::<(usize, Vec<(usize, usize)>, u32)>();
        std::thread::scope(|s| {
            for t in 0..2usize {
                let (stop, still) = (&stop, &still);
                s.spawn(move || {
                    let mut random = Random(0x7777_1234_0000_0001 + t as u64 * 0x1111);
                    let mut value = 0u32;
                    while !stop.load(Ordering::Relaxed) {
                        {
                            let _moving = still.read().unwrap();
                            for _ in 0..3 {
                                value += 1;
                                // Of each eight dwords, one a thread.
                                let at = random.below(LEN / 32) * 32 + 4 * (1 + t);
                                dword(at).store(value, Ordering::Relaxed);
                            }
                        }
                        std::thread::sleep(std::time::Duration::from_micros(20));
                    }
                });
            }
            // Lands flights.
            {
                let (carded, stop) = (&carded, &stop);
                s.spawn(move || {
                    let _stop = Stop(stop);
                    while let Ok((which, ranges, value)) = flights.recv() {
                        let c = &carded[which];
                        // Fetched: the card's bytes, behind every send so far.
                        let (after, now) = {
                            let card = c.card.lock().unwrap();
                            (card.0, card.2.clone())
                        };
                        // By piece: 64 KiB of title memory.
                        let mut pieces = Vec::new();
                        for &(from, to) in &ranges {
                            let mut at = from;
                            while at < to {
                                let next = ((c.start + at) / PAGE * PAGE + PAGE - c.start).min(to);
                                pieces.push((at, next, now.as_ptr() as usize + at));
                                at = next;
                            }
                        }
                        let parts: Vec<_> = pieces.chunks(3).collect();
                        crate::land::each(parts.len(), &|i| {
                            let mut notes = crate::land::Notes::default();
                            // SAFETY: the bytes of `now`; each piece once.
                            unsafe { c.shadow.land_card(parts[i], after, &mut notes) };
                            notes.flush();
                        });
                        for &(from, to) in &ranges {
                            for at in (from..to).filter(|at| (c.start - region + at) % 32 == 4 * c.class) {
                                // (The next fetch may have brought more.)
                                assert!(dword(c.start - region + at).load(Ordering::Relaxed) >= value, "not landed");
                            }
                        }
                        let done: Vec<_> = ranges.iter().map(|&(a, b)| (c.start + a, c.start + b)).collect();
                        c.shadow.copy_back_done(&done);
                        c.card.lock().unwrap().1 = after;
                        c.flying.store(false, Ordering::Release);
                    }
                });
            }
            let _stop = Stop(&stop);
            let mut random = Random(0xABCD_EF01_2345_6789);
            let check = |round: usize, random: &mut Random, quiet: bool| {
                for c in &carded {
                    if c.shadow.writes_pending() {
                        let mut runs = usize::MAX;
                        assert!(!c.shadow.merge_alone(c.start, c.len, &mut runs), "round {round}");
                    }
                }
                let mut ranges = Vec::new();
                for (shadow, start, len) in
                    carded.iter().map(|c| (&c.shadow, c.start, c.len)).chain([(&plain.0, plain.1, plain.2)])
                {
                    // All of it, or a part; often the same part.
                    let (from, n) = match random.below(3) {
                        _ if quiet => (0, len),
                        0 => (0, len),
                        1 => (len / 3 / 4 * 4, len / 2 / 4 * 4),
                        _ => {
                            let from = random.below(len / 4) * 4;
                            (from, (4 + random.below((len - from) / 4) * 4).min(len - from))
                        }
                    };
                    ranges.push((shadow.clone(), start + from, n));
                }
                submission(&ranges, 1 + random.below(3));
                carded.iter().for_each(Carded::send);
            };
            let mut value = 1u32;
            for round in 0..rounds {
                check(round, &mut random, false);
                if exact {
                    let c = &carded[0];
                    let mut card = c.card.lock().unwrap();
                    for _ in 0..40 {
                        value += 1;
                        let at = random.below(LEN / 32) * 32 + 4 * 6;
                        card.2[at..at + 4].copy_from_slice(&value.to_le_bytes());
                        undeclared.insert(at, value);
                    }
                }
                // The flight of this submission: its shaders' writes.
                for (which, c) in carded.iter().enumerate() {
                    if c.flying.swap(true, Ordering::Acquire) {
                        continue;
                    }
                    value += 1;
                    let mut ranges: Vec<(usize, usize)> = Vec::new();
                    let first = random.below(6) * PAGE / 2;
                    for k in 0..2 {
                        let from = first + k * 9 * PAGE + 4 * random.below(9000);
                        let to = (from + 4 * (1 + random.below(7 * PAGE / 4))).min(c.len);
                        if from < to {
                            ranges.push((from, to));
                        }
                    }
                    for &(from, to) in &ranges {
                        c.shadow.writing(c.start + from, to - from);
                    }
                    {
                        let mut card = c.card.lock().unwrap();
                        for &(from, to) in &ranges {
                            for at in (from..to).filter(|at| (c.start - region + at) % 32 == 4 * c.class) {
                                card.2[at..at + 4].copy_from_slice(&value.to_le_bytes());
                            }
                        }
                    }
                    fly.send((which, ranges, value)).unwrap();
                }
                if round % 20 != 19 {
                    continue;
                }
                // Still: no flight in the air, no title write.
                let _still = still.write().unwrap();
                while carded.iter().any(|c| c.flying.load(Ordering::Acquire)) {
                    assert!(!stop.load(Ordering::Relaxed), "the landing thread failed");
                    std::thread::yield_now();
                }
                check(round, &mut random, true);
                // SAFETY: the test's own memory, written by nobody now.
                let bytes = |at: usize, len: usize| unsafe { std::slice::from_raw_parts(at as *const u8, len) };
                let first = |a: &[u8], b: &[u8]| a.iter().zip(b).position(|(x, y)| x != y);
                for c in &carded {
                    let (held, there) = (bytes(c.data, c.len), bytes(c.start, c.len));
                    assert_eq!(first(held, there), None, "round {round}: a card copy's CPU side (class {})", c.class);
                    let card = c.card.lock().unwrap();
                    let mut side = card.2.clone();
                    if c.class == carded[0].class {
                        // (They may have been landed meanwhile, by a flight
                        // that declared their range, and may not.)
                        for (&at, &value) in &undeclared {
                            let on_card = u32::from_le_bytes(side[at..at + 4].try_into().unwrap());
                            assert_eq!(on_card, value, "round {round}: an older value at {at:#x}");
                            side[at..at + 4].copy_from_slice(&there[at..at + 4]);
                        }
                    }
                    assert_eq!(first(&side, there), None, "round {round}: a card copy's card side (class {})", c.class);
                }
                assert_eq!(first(bytes(plain.3, plain.2), bytes(plain.1, plain.2)), None, "round {round}: the other copy");
            }
            drop(fly);
        });
    }

    /// The title's bytes put in under shader writes, for a copy of 174 MB
    /// whose shader writes were declared as 40 ranges that overlap, with 12
    /// pages written since. Run with `--ignored --nocapture`, and with
    /// `VICTORY_LIGHT_REFRESH=0` for the way it was.
    #[test]
    #[ignore]
    fn bytes_under_overlapping_ranges() {
        const BIG: usize = 174 << 20;
        let big = title(BIG, true);
        // SAFETY (both): the test's own memory.
        unsafe { std::ptr::write_bytes(big as *mut u8, 1, BIG) };
        let copy = Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true);
        copy.update(big, BIG);
        copy.take_dirty();
        for i in 0..40 {
            copy.writing(big + i * 5920, BIG - i * 5920);
        }
        let mut best = f64::MAX;
        for round in 0..30 {
            for i in 0..12 {
                unsafe { ((big + (i * 211 + round) % 2700 * PAGE + 64) as *mut u8).write_volatile(round as u8) };
            }
            let started = std::time::Instant::now();
            let mut runs = usize::MAX;
            assert!(!copy.merge_alone(big, BIG, &mut runs));
            best = best.min(started.elapsed().as_secs_f64() * 1e3);
            copy.take_dirty();
        }
        println!("VICTORY_LIGHT_REFRESH {}: best {best:.3} ms", crate::watch::light());
    }

    /// A range nothing wrote is known to be as it was, at its ends too, and
    /// is so no more once something wrote it.
    #[test]
    fn unchanged_ranges_are_known() {
        for view in [false, true] {
            let region = title(8 * PAGE, view);
            // (A page read for the first time counts as written.)
            // SAFETY: the test's own memory.
            unsafe { std::ptr::write_bytes(region as *mut u8, 1, 8 * PAGE) };
            let data = memory(8 * PAGE);
            let copy = Shadow::unbacked(region as u64, data, (8 * PAGE) as u64, false);
            // Ends in the middle of pages.
            let (at, len) = (region + 500, 5 * PAGE);
            let look = |expected: bool, what: &str| {
                // The submission's one asking, which takes two ranges. (Again
                // if a submission of another test ended it.)
                let mut known = false;
                for _ in 0..100 {
                    crate::watch::ask_once(&[(at, at + len), (region + 7 * PAGE, region + 8 * PAGE)]);
                    known = copy.nothing_new(at, len);
                    if known || !expected {
                        break;
                    }
                }
                assert_eq!(known, expected, "view {view}: {what}");
                let copied = copy.update(at, len);
                crate::watch::ask_once_end();
                copied
            };
            // With VICTORY_WHOLE_PAGES (on by default) a watched copy's refresh
            // takes the pages the range touches, so its ends too.
            let whole = std::env::var_os("VICTORY_WHOLE_PAGES").is_none_or(|v| v != "0");
            let part = if view { PAGE } else { 4096 };
            assert_eq!(look(false, "never taken"), if whole { 6 * PAGE } else { len });
            assert_eq!(look(true, "taken"), 0);
            assert_eq!(look(true, "taken"), 0);
            // SAFETY: the test's own memory.
            unsafe { ((region + 600) as *mut u8).write_volatile(1) };
            // (The kernel names the 4 KiB, the guards only the block.)
            assert_eq!(look(false, "written at its end"), if whole { part } else { part - 500 });
            assert_eq!(look(true, "taken again"), 0);
            // Outside the range, on its last page.
            // SAFETY: as above.
            unsafe { ((region + 5 * PAGE + 600) as *mut u8).write_volatile(1) };
            assert_eq!(look(false, "written on its last page"), if whole { part } else { 500 });
            // A longer range was never taken whole.
            assert!(!copy.nothing_new(region, if whole { 7 * PAGE } else { 6 * PAGE }));
        }
    }

    /// Of a copy in the card's memory only the dwords title memory changed go
    /// up, and none beside them, not even between two in 64 bytes; a page
    /// never taken goes up whole.
    #[test]
    fn only_what_differs_goes_up() {
        let region = title(4 * PAGE, true);
        let data = memory(4 * PAGE);
        let copy = Shadow::unbacked(region as u64, data, (4 * PAGE) as u64, true);
        copy.update(region, 4 * PAGE);
        assert_eq!(copy.take_dirty(), [(0, 4 * PAGE)], "all of it, though zeros are no change");
        // What the relay does: shader writes into title memory, and into the
        // CPU's side.
        crate::watch::landing(region + PAGE + 640, 64);
        // SAFETY: the test's own memory.
        unsafe {
            std::ptr::write_bytes((region + PAGE + 640) as *mut u8, 7, 64);
            std::ptr::write_bytes(data.add(PAGE + 640), 7, 64);
        }
        copy.update(region, 4 * PAGE);
        let again = copy.take_dirty();
        // SAFETY: as above.
        unsafe {
            ((region + 2 * PAGE + 129) as *mut u8).write_volatile(9);
            ((region + 2 * PAGE + 136) as *mut u32).write_volatile(9);
            ((region + 2 * PAGE + 140) as *mut u32).write_volatile(9);
            // The value that was there.
            ((region + 2 * PAGE + 132) as *mut u32).write_volatile(0);
        }
        copy.update(region, 4 * PAGE);
        let written = copy.take_dirty();
        if std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0") {
            assert_eq!(again, []);
            assert_eq!(written, [(2 * PAGE + 128, 2 * PAGE + 132), (2 * PAGE + 136, 2 * PAGE + 144)]);
        } else {
            assert_eq!(again, [(PAGE, 2 * PAGE)]);
            assert_eq!(written, [(2 * PAGE, 3 * PAGE)]);
        }
        // SAFETY: as above.
        assert_eq!(unsafe { data.add(2 * PAGE + 129).read() }, 9);
    }

    /// The refreshes of a frame as run record-1 had them: 30 submissions, the
    /// first with a copy for each of 4,400 ranges, the others with 60 and a
    /// big copy that 55 landed pieces and 9 title writes opened, while a
    /// thread lands what shaders wrote. Run with `--ignored --nocapture`,
    /// and with `VICTORY_LIGHT_REFRESH=0` for the way it was.
    #[test]
    #[ignore]
    fn frame_of_refreshes() {
        let mut random = Random(0x9876_5432_1234_5671);
        const EACH: usize = 224 << 10;
        const BIG: usize = 174 << 20;
        let watched = title(3000 * EACH, false);
        // `VICTORY_BENCH_HOLES=0`: views without holes.
        let holes = std::env::var_os("VICTORY_BENCH_HOLES").is_none_or(|v| v != "0");
        let views = if holes { with_holes(1400 * EACH / PAGE) } else { title(1400 * EACH, true) };
        let there = |at: usize| !holes || (at - views) / PAGE % 4 != 3;
        let big = title(BIG, true);
        // SAFETY: the test's own memory.
        unsafe {
            std::ptr::write_bytes(watched as *mut u8, 1, 3000 * EACH);
            for at in (views..views + 1400 * EACH).step_by(PAGE).filter(|&at| there(at)) {
                std::ptr::write_bytes(at as *mut u8, 1, PAGE);
            }
            std::ptr::write_bytes(big as *mut u8, 1, BIG);
        }
        let mut small = Vec::new();
        for i in 0..4400 {
            let (start, at) = if i < 3000 { (watched, i * EACH) } else { (views, (i - 3000) * EACH) };
            let data = memory(EACH);
            // The range views read, not all of the copy.
            small.push((Shadow::unbacked((start + at) as u64, data, EACH as u64, false), start + at + 4000, EACH - 9000));
        }
        let heap = (Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true), big, BIG);
        // The same memory, as another copy: what is landed is landed through
        // it, and the refreshes are of the first.
        let lands = Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true);
        let (mut best, mut asking) = ([f64::MAX; 2], [0f64; 2]);
        let (mut sum, mut landing) = (0.0, 0.0);
        const FRAMES: usize = 12;
        for frame in 0..FRAMES {
            let mut took = [0f64; 2];
            for i in 0..30 {
                // The title's writes, and the relay's beside the submission.
                for _ in 0..9 {
                    let at = views + random.below(1400 * EACH / 8) * 8;
                    if there(at) {
                        // SAFETY: the test's own memory.
                        unsafe { (at as *mut u64).write_volatile(frame as u64) };
                    }
                }
                for _ in 0..60 {
                    let at = watched + random.below(3000 * EACH / 8) * 8;
                    // SAFETY: the test's own memory.
                    unsafe { (at as *mut u64).write_volatile(frame as u64) };
                }
                let first = random.below(BIG / PAGE - 60);
                let lands = lands.clone();
                let land = std::thread::spawn(move || {
                    let now = vec![frame as u8 ^ i as u8 ^ 0x55; PAGE];
                    let started = std::time::Instant::now();
                    for pieces in (first..first + 55).collect::<Vec<_>>().chunks(4) {
                        let pieces: Vec<_> =
                            pieces.iter().map(|p| (p * PAGE + 8, (p + 1) * PAGE - 8, now.as_ptr() as usize)).collect();
                        let mut notes = crate::land::Notes::default();
                        // SAFETY: the bytes of `now`; nothing else lands here.
                        unsafe { lands.land_card(&pieces, 0, &mut notes) };
                        notes.flush();
                    }
                    started.elapsed().as_secs_f64() * 1e3
                });
                let ranges: Vec<(Arc<Shadow>, usize, usize)> = if i == 0 {
                    small.iter().cloned().chain([heap.clone()]).collect()
                } else {
                    (0..60).map(|_| small[random.below(4400)].clone()).chain([heap.clone()]).collect()
                };
                let started = std::time::Instant::now();
                submission(&ranges, 3);
                took[usize::from(i != 0)] += started.elapsed().as_secs_f64() * 1e3;
                let put = land.join().unwrap();
                if frame >= 2 {
                    landing += put;
                }
                heap.0.take_dirty();
            }
            if frame >= 2 {
                sum += took[0] + took[1];
                for k in 0..2 {
                    best[k] = best[k].min(took[k]);
                    asking[k] += took[k];
                }
            }
        }
        println!(
            "VICTORY_LIGHT_REFRESH {}: a frame's refreshes {:.2} ms in the mean; the submission of all copies {:.2} (best {:.2}), the 29 \
             others {:.2} (best {:.2}); the landing thread's puts {:.2}",
            crate::watch::light(),
            sum / (FRAMES - 2) as f64,
            asking[0] / (FRAMES - 2) as f64,
            best[0],
            asking[1] / (FRAMES - 2) as f64,
            best[1],
            landing / (FRAMES - 2) as f64,
        );
        #[cfg(feature = "diagnostics")]
        println!("per frame, as `crate::watch::asks`: {:?}", crate::watch::asks().map(|n| n / FRAMES as u64));
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::{Batch, COALESCE, join};
    use std::sync::Arc;

    /// A copy's refreshes: what lies within a page of the others merges
    /// with them, what is further stays apart, and a range between two
    /// joins them.
    #[test]
    fn refreshes_further_than_a_page_apart_stay_apart() {
        let (mut span, mut others) = ((0x1_0000, 0x1_1000), Vec::new());
        join(&mut span, &mut others, 0x40_0000, 0x40_1000);
        assert_eq!((span, &others[..]), ((0x1_0000, 0x1_1000), &[(0x40_0000, 0x40_1000)][..]));
        assert!(others.capacity() > 0);
        // A page's gap merges ...
        join(&mut span, &mut others, 0x1_1000 + COALESCE, 0x1_2000 + COALESCE);
        assert_eq!(span, (0x1_0000, 0x1_2000 + COALESCE));
        assert_eq!(others, [(0x40_0000, 0x40_1000)]);
        // ... a byte more does not.
        join(&mut span, &mut others, 0x1_2001 + 2 * COALESCE, 0x1_3000 + 2 * COALESCE);
        assert_eq!(others.len(), 2);
        // Near the one apart: merged with it, it is now the span.
        join(&mut span, &mut others, 0x40_1000, 0x40_2000);
        assert_eq!(span, (0x40_0000, 0x40_2000));
        assert_eq!(others.len(), 2);
        // Over all of them: one range.
        join(&mut span, &mut others, 0x1_1000, 0x40_0000);
        assert_eq!((span, others.len()), ((0x1_0000, 0x40_2000), 0));
        // Ranges of one copy near each other never take memory for others.
        let (mut span, mut others) = ((0, 0x1000), Vec::new());
        join(&mut span, &mut others, 0x1000, 0x3000);
        join(&mut span, &mut others, 0x100, 0x200);
        assert_eq!((span, others.capacity()), ((0, 0x3000), 0));
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        let vram_idle_ms = |memory| super::vram_idle_for(memory, false);
        let keeps = |memory| super::vram_idle_for(memory, true);
        let gib = 1024 * 1024 * 1024;
        assert_eq!(keeps(Some([100, 67, 0, 0])), 600_000);
        assert_eq!(keeps(Some([100, 68, 0, 0])), 30_000);
        assert_eq!(keeps(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(keeps(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(keeps(None), 5_000);
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn textures_are_kept_while_the_card_has_room() {
        use super::vram_keep;
        let gib = 1 << 30;
        assert_eq!(vram_keep(100 * gib, 87 * gib).0, 600_000);
        // Past it: the longest idle ones, enough to come down to 85 %.
        let (ms, bytes) = vram_keep(100 * gib, 90 * gib);
        assert_eq!((ms, bytes), (5_000, 5 * gib));
        assert_eq!(vram_keep(100 * gib, 120 * gib), (1_500, 35 * gib));
        assert_eq!(vram_keep(0, 7).0, 5_000);
        // Around a line of 93 % instead: kept under 91 %, down to 88 %.
        use super::vram_keep_below;
        assert_eq!(vram_keep_below(100 * gib, 90 * gib, 93 * gib).0, 600_000);
        assert_eq!(vram_keep_below(100 * gib, 92 * gib, 93 * gib), (5_000, 4 * gib));
        assert_eq!(vram_keep_below(100 * gib, 99 * gib, 93 * gib), (1_500, 11 * gib));
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        let ram_idle_ms = |avail| super::ram_idle_for(avail, false);
        assert_eq!(super::ram_idle_for(Some(12 << 30), true), 300_000);
        assert_eq!(super::ram_idle_for(Some((12 << 30) - 1), true), 10_000);
        assert_eq!(super::ram_idle_for(None, true), 10_000);
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }

    #[test]
    fn freed_copies_not_kept_near_the_commit_limit() {
        use super::private_near_limit;
        let gib = 1u64 << 30;
        // 70 % exactly of the smaller one is near, a byte under it is not.
        assert!(private_near_limit(7 * gib, 10 * gib, 20 * gib));
        assert!(!private_near_limit(7 * gib - 1, 10 * gib, 20 * gib));
        assert!(private_near_limit(7 * gib, 20 * gib, 10 * gib));
        // A 32 GB PC with a 48 GB page file (an 80 GB commit limit): RAM
        // sets the mark, about 22.4 GB, not 56 GB.
        assert!(!private_near_limit(20 * gib, 32 * gib, 80 * gib));
        assert!(private_near_limit(23 * gib, 32 * gib, 80 * gib));
        assert!(private_near_limit(40 * gib, 32 * gib, 80 * gib));
        // A 64 GB PC at the README's 30 GB keeps its copies as before.
        assert!(!private_near_limit(30 * gib, 64 * gib, 80 * gib));
        // One unknown number: the other one; both unknown: never near.
        assert!(private_near_limit(23 * gib, 0, 32 * gib));
        assert!(private_near_limit(23 * gib, 32 * gib, 0));
        assert!(!private_near_limit(u64::MAX, 0, 0));
        // No overflow on huge values.
        assert!(private_near_limit(u64::MAX, u64::MAX, u64::MAX));
    }
}

#[cfg(test)]
mod breadcrumbs {
    use super::{readable_history, stopped_at};
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_AUTO_BREADCRUMB_OP, D3D12_AUTO_BREADCRUMB_OP_DISPATCH, D3D12_AUTO_BREADCRUMB_OP_SETMARKER,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4 << 10;

    /// The device-removed report of a list whose breadcrumb count runs a
    /// page past its committed history (as the runtime's did on an RTX
    /// 3070 Ti): the report takes the commands that are there and says the
    /// rest cannot be read, instead of faulting.
    #[test]
    fn a_history_shorter_than_its_count_is_read_as_far_as_it_is_there() {
        let op = |i: usize| -> D3D12_AUTO_BREADCRUMB_OP {
            if i.is_multiple_of(2) { D3D12_AUTO_BREADCRUMB_OP_SETMARKER } else { D3D12_AUTO_BREADCRUMB_OP_DISPATCH }
        };
        let committed = PAGE / 4;
        // SAFETY: a fresh reservation of two pages; the first is committed
        // and ours to fill.
        let base = unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS).cast::<D3D12_AUTO_BREADCRUMB_OP>();
            assert!(!base.is_null());
            assert!(!VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            for i in 0..committed {
                base.add(i).write(op(i));
            }
            base
        };
        let count = 2 * committed as u32;
        let history = readable_history(base, count);
        assert_eq!(history.len(), committed, "the committed page");
        assert!(history.iter().enumerate().all(|(i, o)| *o == op(i)), "its commands, in order");
        assert_eq!(
            stopped_at(&history, count, 7),
            "; a list stopped at command 7 of 2048 (op 6), but only its first 1024 commands can be read"
        );
        assert_eq!(
            stopped_at(&history, count, 1500),
            "; a list stopped at command 1500 of 2048 (op unknown), but only its first 1024 commands can be read"
        );
        // A history that is all there reads whole, without the note.
        let whole = readable_history(base, committed as u32);
        assert_eq!(whole.len(), committed);
        assert_eq!(stopped_at(&whole, committed as u32, 7), "; a list stopped at command 7 of 1024 (op 6)");
        // SAFETY: the reservation made above.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }
}

#[cfg(test)]
mod piece_copy_tests {
    use super::{copy_pieces, piece_bounds};

    /// The pieces cover the copy exactly once, the last one shorter.
    #[test]
    fn pieces_cover_the_copy_once() {
        assert_eq!(piece_bounds(0, 4), vec![]);
        assert_eq!(piece_bounds(4, 4), vec![(0, 4)]);
        assert_eq!(piece_bounds(10, 4), vec![(0, 4), (4, 8), (8, 10)]);
        let len = 9 << 20;
        let cut = piece_bounds(len, 2 << 20);
        assert_eq!(cut.len(), 5);
        assert_eq!(cut.iter().map(|(a, b)| b - a).sum::<usize>(), len);
        assert!(cut.windows(2).all(|w| w[0].1 == w[1].0));
    }

    /// A big copy made on several threads copies every byte, the last
    /// partial piece included.
    #[test]
    fn a_big_copy_in_pieces_copies_every_byte() {
        let len = (9 << 20) + 4097;
        let src: Vec<u8> = (0..len).map(|i| (i * 131 + i / 4099) as u8).collect();
        let mut dst = vec![0u8; len];
        copy_pieces(dst.as_mut_ptr(), src.as_ptr(), len, false);
        assert!(dst == src, "the pieces did not copy the same bytes");
    }
}
