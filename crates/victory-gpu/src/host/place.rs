//! The title's buffer copies, done on the card at their place in the list
//! (`VICTORY_COPIES_IN_PLACE`, on by default; `0`: all at submission, on the
//! CPU, as before).
//!
//! A buffer copy was done in title memory when its list was submitted, before
//! any draw of the submission ran, and the host copies refreshed after it.
//! That is another order than the list's: draws ahead of the copy read what
//! it put; of two copies to the same bytes with a draw behind each, both
//! draws read the second; a source that a dispatch of the same submission
//! writes first was read as it was before it; and a destination the
//! submissions in the air still read changed under them (where its host copy
//! has one side only, `crate::shadow::card_all`).
//!
//! Now the list copies, in the destination's host copy: out of the source's
//! host copy where shaders write that one, else out of upload memory of the
//! list that takes the source's bytes when the list is submitted (a source
//! shaders never wrote is the title's CPU's, or a copy's put there earlier in
//! the submission). What is written there is a shader's write from then on:
//! the bytes are declared, and what the card made of them is put into title
//! memory before the title is told of the submission.
//!
//! Title memory takes the copy at submission as well, as it did: fills of
//! upload memory, textures and host copies that lie over the same bytes read
//! it there. The reference of the destination's host copy takes the same
//! bytes ([`Shadow::placed_copy`]), so what the landing finds is what the card
//! made of them afterwards (a counter copied to zero and counted up; a source
//! that a shader wrote in between), and nothing is sent up over it. Where
//! flights in the air may write the source or the destination this is left to
//! the landing: what they wrote is not in title memory yet, and what they
//! will put there must not stand over this copy's bytes. If a step of the
//! submission reads the destination on the CPU, it waits for them instead
//! ([`Placed::waits`]).
//!
//! Copies to memory that has no host copy yet are done as before: nothing on
//! the card has read or written it. Large copies out of memory shaders do not
//! write also go at their place when scratch upload memory is available: a
//! deferred CPU copy would otherwise let a later write to the same bytes reach
//! an earlier GPU read in the same list or in a submission still in the air.

use super::*;
use crate::shadow::Shadow;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPIES_IN_PLACE").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: copies at their place, their bytes, those
/// out of a host copy, texture copies out of one, and fills that took a
/// copy's bytes ([`forwards`]); copies out of a host copy whose source a
/// shader of the same submission was declared to write ahead of them, and
/// those of them a fill took the bytes of ([`count_written_ahead`]).
pub(super) static STATS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// Constants read on the card directly from a placed copy's host copy.
pub(super) static CONSTANTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether constant buffers reading bytes written earlier in the same list by
/// [`HostList::place_copy`] are snapshotted on the GPU: on with
/// `VICTORY_CARD_STEPS=2` unless `VICTORY_PLACED_CONSTANTS=0`, and with
/// `VICTORY_PLACED_CONSTANTS=1` at any stage.
fn placed_constants() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        on() && match std::env::var("VICTORY_PLACED_CONSTANTS").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => super::card_steps::stage() == 2,
        }
    })
}

/// What the CPU does at submission for a copy at its place in a list.
pub(super) struct Placed {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) len: usize,
    /// The destination's host copy.
    pub(super) shadow: std::sync::Arc<Shadow>,
    /// Upload memory the list copies from, to be given the source's bytes.
    from: Option<usize>,
}

impl Placed {
    /// Whether flights that write `written` are waited for.
    pub(super) fn waits(&self, written: &[(usize, usize)]) -> Waits {
        let source = touches(written, self.src, self.src + self.len);
        if source && self.from.is_some() {
            return Waits::Yes;
        }
        match source || touches(written, self.dst, self.dst + self.len) {
            true => Waits::ForReaders,
            false => Waits::No,
        }
    }

    /// At submission, in the order of the steps: the list's upload memory
    /// takes the source's bytes and, if `free` (no flight in the air may write
    /// source or destination), title memory at the destination does. Gives
    /// the bytes if it took them (always if `keep`), and whether title memory
    /// holds them.
    pub(super) fn prepare(&self, free: bool, keep: bool) -> Option<(Vec<u8>, bool)> {
        if !free && self.from.is_none() && !keep {
            return None;
        }
        let mut bytes = vec![0u8; self.len];
        // (What is not there reads as zeros, as any copy's source.)
        copy_mapped(bytes.as_mut_ptr(), self.src as *const u8, self.len, false);
        if let Some(to) = self.from {
            // SAFETY: the list's own upload memory of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, self.len) };
        }
        if !free {
            return Some((bytes, false));
        }
        copy_mapped(self.dst as *mut u8, bytes.as_ptr(), self.len, true);
        crate::watch::wrote(self.dst, self.len);
        // Title memory that is not all there was left alone.
        let mut there = vec![0u8; self.len];
        let held = read_into(self.dst as *const u8, &mut there) && there == bytes;
        Some((bytes, held))
    }
}

/// Whether a copy waits for flights before its submission is prepared: not;
/// yes, as it takes a source on the CPU that they may write; or only if
/// another step of the submission reads its destination on the CPU.
pub(super) enum Waits {
    No,
    ForReaders,
    Yes,
}

/// Whether a step of `runs` reads one of `ranges` on the CPU (fills of
/// `forwards` read a copy's bytes instead).
pub(super) fn read_by(runs: &[&HostList], mut ranges: Vec<(usize, usize)>, forwards: &[Forward]) -> bool {
    join_ranges(&mut ranges, []);
    runs.iter().enumerate().any(|(n, list)| {
        list.deferred().copies.iter().enumerate().any(|(i, step)| match step {
            &Step::Copy(_, src, len) => touches(&ranges, src, src + len),
            &Step::Fill(_, src, len, _) => !forwarded(forwards, n, i) && touches(&ranges, src, src + len),
            Step::Texture(staging) => {
                let (at, len) = staging.memory();
                touches(&ranges, at, at + len)
            }
            Step::Host(_, (from, to)) => touches(&ranges, *from, *to),
            Step::Placed(p) => p.from.is_some() && touches(&ranges, p.src, p.src + p.len),
            _ => false,
        })
    })
}

/// Fills of upload memory that read only bytes an earlier copy at its place
/// in the submission put take the bytes the copy took at submission
/// (`VICTORY_FORWARD_PLACED`, on by default; `0`: from title memory, as
/// before). From title memory such a fill waited for flights that may write
/// the copy's destination, although the copy stands over what they write
/// (two of the three longest waits in the ft2 run of 2026-11-26). A copy out
/// of a host copy counts where no flight may write its source: title memory
/// there is what the wait would have found.
pub(super) fn forward_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FORWARD_PLACED").is_none_or(|v| v != "0"))
}

/// Constants behind a copy at its place in the same list that writes them are
/// copied on the card at the fill's place, out of the host copy, and the slot
/// shows that copy (`VICTORY_FILL_IN_PLACE=1`, off until it wins an A/B). As a
/// fill at submission it was done in front of the lists, ahead of the copy,
/// so card_steps refused it and the submission waited for every flight: 111
/// to 142 ms a frame in game 95-112 of capss-ahead and capss-old, over half
/// the frame.
pub(super) fn fill_in_place() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_FILL_IN_PLACE") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_IN_PLACE").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line ([`fill_in_place`]): fills done at their
/// place, their bytes, and fills behind such a copy left to submission (no
/// host copy that shaders write holds all they read).
pub(super) static FILLS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// A fill that takes its bytes from a copy at its place: each by its list's
/// place in the submission and its own in the list; `at`, where in the copy
/// the fill's bytes begin.
pub(super) struct Forward {
    pub(super) fill: (usize, usize),
    pub(super) copy: (usize, usize),
    pub(super) at: usize,
}

/// The fills of `runs` that read only what a copy at its place put in front
/// of them, with no step between that writes there. `written`: what flights
/// may write.
pub(super) fn forwards(written: &[(usize, usize)], runs: &[&HostList]) -> Vec<Forward> {
    if !on() || !forward_on() {
        return Vec::new();
    }
    let lists: Vec<Vec<Order>> = runs.iter().map(|list| list.deferred().copies.iter().map(Order::of).collect()).collect();
    forwards_of(written, lists)
}

/// What [`forwards`] looks at in a step.
#[derive(Clone, Copy)]
pub(super) enum Order {
    /// A fill's source and length.
    Fill(usize, usize),
    /// A CPU write of title memory: destination, length.
    Write(usize, usize),
    /// A copy at its place: destination, source, length, and whether it has
    /// upload memory to give the source's bytes.
    Placed(usize, usize, usize, bool),
    /// Work that may write title memory anywhere.
    Run,
    Other,
}

impl Order {
    fn of(step: &Step) -> Order {
        match step {
            &Step::Fill(_, src, len, _) => Order::Fill(src, len),
            &Step::Copy(dst, _, len) => Order::Write(dst, len),
            Step::Placed(p) => Order::Placed(p.dst, p.src, p.len, p.from.is_some()),
            Step::Run(_) => Order::Run,
            _ => Order::Other,
        }
    }
}

/// [`forwards`] over the steps of each list, in submission order.
pub(super) fn forwards_of<L: IntoIterator<Item = Order>>(written: &[(usize, usize)], lists: impl IntoIterator<Item = L>) -> Vec<Forward> {
    let mut found = Vec::new();
    let mut standing = Standing::default();
    for (n, list) in lists.into_iter().enumerate() {
        for (i, step) in list.into_iter().enumerate() {
            match step {
                Order::Fill(src, len) => found.extend(standing.fill((n, i), src, len)),
                Order::Write(dst, len) => standing.written(dst, len),
                Order::Placed(dst, src, len, from) => standing.copy((n, i), dst, len, from || !touches(written, src, src + len)),
                Order::Run => standing.0.clear(),
                Order::Other => {}
            }
        }
    }
    found
}

/// Copies at their place whose bytes stand, in the order of the steps:
/// destination, its end, the copy.
#[derive(Default)]
struct Standing(Vec<(usize, usize, (usize, usize))>);

impl Standing {
    fn written(&mut self, dst: usize, len: usize) {
        self.0.retain(|s| !(s.0 < dst + len && dst < s.1));
    }

    /// `known`: its bytes are known at submission.
    fn copy(&mut self, copy: (usize, usize), dst: usize, len: usize, known: bool) {
        self.written(dst, len);
        if known && len > 0 {
            self.0.push((dst, dst + len, copy));
        }
    }

    fn fill(&self, fill: (usize, usize), src: usize, len: usize) -> Option<Forward> {
        let &(dst, _, copy) = self.0.iter().find(|s| len > 0 && s.0 <= src && src + len <= s.1)?;
        Some(Forward { fill, copy, at: src - dst })
    }
}

/// Diagnostics: the copy on the card reads what a shader ahead of it in the
/// submission put in its source, a fill that took its bytes at submission
/// does not.
pub(super) fn count_written_ahead(runs: &[&HostList], forwards: &[Forward]) {
    if !on() {
        return;
    }
    let mut ahead: Vec<(usize, usize)> = Vec::new();
    let mut seen = std::collections::HashSet::<(usize, usize), std::hash::BuildHasherDefault<AddressHash>>::default();
    for (n, list) in runs.iter().enumerate() {
        for (i, step) in list.deferred().copies.iter().enumerate() {
            match step {
                &Step::GpuWrite(_, address, len) => {
                    if seen.insert((address, len)) {
                        ahead.push((address, address + len));
                    }
                }
                Step::Placed(p) if p.from.is_none() && ahead.iter().any(|r| r.0 < p.src + p.len && p.src < r.1) => {
                    STATS[5].fetch_add(1, Ordering::Relaxed);
                    if forwards.iter().any(|f| f.copy == (n, i)) {
                        STATS[6].fetch_add(1, Ordering::Relaxed);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Whether step `i` of the `n`th list is a fill of `forwards`.
pub(super) fn forwarded(forwards: &[Forward], n: usize, i: usize) -> bool {
    forwards.iter().any(|f| f.fill == (n, i))
}

/// Everything recorded so far is done, and seen, before what follows.
///
/// # Safety
/// An open list.
unsafe fn wait_for_all(host: &Host, list: &ID3D12GraphicsCommandList) {
    static ENHANCED: OnceLock<bool> = OnceLock::new();
    let enhanced = *ENHANCED.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a feature structure of its size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut o).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        known.is_ok() && o.EnhancedBarriersSupported.as_bool()
    });
    // SAFETY: the caller's.
    unsafe {
        if enhanced && let Ok(list) = list.cast::<ID3D12GraphicsCommandList7>() {
            let all = [D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_ALL,
                SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_COMMON,
                AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
            }];
            list.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_GLOBAL,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pGlobalBarriers: all.as_ptr(),
                },
            }]);
            return;
        }
        let alias = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    pResourceAfter: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        let uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        list.ResourceBarrier(&[alias, uav]);
    }
}

/// The host copy of `[src, src + len)` if shaders write it: what it holds at
/// a place in a list is not what title memory holds at submission.
fn shaders_own(src: usize, len: usize) -> Option<std::sync::Arc<Shadow>> {
    crate::shadow::peek(src as u64, len as u64).filter(|s| s.written() && !s.in_place())
}

impl HostList {
    /// Copies `len` bytes of title memory at `src` to `dst`, at this place of
    /// the list (see the file). `false` if it is to be done at submission.
    pub(crate) fn place_copy(&self, dst: usize, src: usize, len: usize) -> bool {
        if !on() {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let Some(to) = crate::shadow::peek(dst as u64, len as u64).filter(|s| !s.in_place()) else {
            return false;
        };
        let own = shaders_own(src, len);
        // Prefer a placed GPU copy whenever scratch (or a shader-owned source)
        // is available: deferring a large CPU `Step::Copy` until submission
        // collapses mid-list and cross-submission order (issue #844). The old
        // `MOST` gate sent streaming copies above 8 MiB down that path.
        let scratch = match &own {
            Some(_) => None,
            None => match self.scratch_at(len as u64) {
                Some(scratch) => Some(scratch),
                None => return false,
            },
        };
        let at = to.offset(dst as u64);
        // What the title's CPU wrote around and under it so far goes first.
        to.refresh(self, dst as u64, len as u64);
        // SAFETY (both): an open list; live buffers (the list holds the
        // copies and its scratch memory), ranges inside them.
        unsafe { wait_for_all(host, &self.list) };
        match (&own, &scratch) {
            (Some(from), _) => {
                from.refresh(self, src as u64, len as u64);
                let offset = from.offset(src as u64);
                if from.buffer.as_raw() == to.buffer.as_raw() {
                    let Some(between) = self.between(host, len as u64) else {
                        return false;
                    };
                    unsafe {
                        self.list.CopyBufferRegion(&between, 0, &from.buffer, offset, len as u64);
                        // The host copy was promoted to a copy's source: back
                        // to common, to be promoted to its destination.
                        self.list.ResourceBarrier(&[
                            transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
                            transition(&from.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                        ]);
                        wait_for_all(host, &self.list);
                        self.list.CopyBufferRegion(&to.buffer, at, &between, 0, len as u64);
                    }
                } else {
                    unsafe { self.list.CopyBufferRegion(&to.buffer, at, &from.buffer, offset, len as u64) };
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (None, Some((from, offset, _))) => unsafe {
                self.list.CopyBufferRegion(&to.buffer, at, from, *offset, len as u64);
            },
            (None, None) => return false,
        }
        // The destination was promoted to a copy's destination, a write
        // state: no use after it in the list (the draws' root views read it,
        // issue #350) could promote it again. Back to common, as the
        // submission's lists expect the host copies.
        // SAFETY: an open list; a live buffer.
        unsafe {
            self.list.ResourceBarrier(&[transition(&to.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            wait_for_all(host, &self.list);
        }
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(len as u64, Ordering::Relaxed);
        to.gpu_writes(self, dst as u64, len as u64);
        let mut deferred = self.deferred();
        deferred.placed.push((dst, dst + len));
        deferred.placed_written.push((dst, dst + len, to.clone()));
        deferred.copies.push(Step::Placed(Placed {
            dst,
            src,
            len,
            shadow: to,
            from: scratch.map(|s| s.2 as usize),
        }));
        deferred.title_written();
        self.has_placed.store(true, Ordering::Relaxed);
        self.changed_something();
        true
    }

    /// Snapshots `[src, src + len)` on the card when the most recent write to
    /// those bytes in this recording is a [`HostList::place_copy`] that
    /// covers the whole range, returning `(buffer, gpu_address)` in
    /// `D3D12_RESOURCE_STATE_GENERIC_READ`.
    pub(crate) fn place_constant(&self, src: usize, len: usize) -> Option<(ID3D12Resource, u64)> {
        if !placed_constants() || len == 0 || !self.has_placed.load(Ordering::Relaxed) {
            return None;
        }
        let end = src.checked_add(len)?;
        let host = host()?;
        let alloc = (len as u64).next_multiple_of(256);
        let (from, mut cbv_slot) = {
            let mut deferred = self.deferred();
            let mut matched = None;
            for (p_dst, p_end, shadow) in deferred.placed_written.iter().rev() {
                if *p_end <= src || *p_dst >= end {
                    continue;
                }
                if *p_dst <= src && end <= *p_end {
                    let (start, shadow_len) = shadow.span();
                    if !shadow.in_place() && (src as u64) >= start && (end as u64) <= start + shadow_len {
                        matched = Some(shadow.clone());
                    }
                }
                break;
            }
            let from = matched?;
            let slot = match deferred.placed_cbv.as_mut() {
                Some((buf, cap, used)) if *used + alloc <= *cap => {
                    let at = *used;
                    *used += alloc;
                    Some((buf.clone(), at, true))
                }
                _ => None,
            };
            (from, slot)
        };
        if cbv_slot.is_none() {
            let cap = alloc.max(64 << 10);
            let buf = host.uav_buffer(cap)?;
            name(&buf, || "placed constants on the card".into());
            let old = {
                let mut deferred = self.deferred();
                match deferred.placed_cbv.as_mut() {
                    Some((existing, existing_cap, used)) if *used + alloc <= *existing_cap => {
                        let at = *used;
                        *used += alloc;
                        cbv_slot = Some((existing.clone(), at, true));
                        None
                    }
                    _ => {
                        cbv_slot = Some((buf.clone(), 0, false));
                        deferred.placed_cbv.replace((buf, cap, alloc)).map(|(b, _, _)| b)
                    }
                }
            };
            if let Some(old) = old {
                self.keep(&old);
            }
        }
        let (cbv_buf, at, was_used) = cbv_slot?;
        // SAFETY: `self.list` is open; `cbv_buf` is kept alive in `Deferred`
        // until reset, `from` is a live host copy covering `[src, end)`, and
        // `at..at + len` fits in `cbv_buf`.
        unsafe {
            if was_used {
                self.list.ResourceBarrier(&[transition(
                    &cbv_buf,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
            wait_for_all(host, &self.list);
            self.list.CopyBufferRegion(&cbv_buf, at, &from.buffer, from.offset(src as u64), len as u64);
            self.list.ResourceBarrier(&[transition(
                &cbv_buf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )]);
            let gpu = cbv_buf.GetGPUVirtualAddress() + at;
            CONSTANTS.fetch_add(1, Ordering::Relaxed);
            Some((cbv_buf, gpu))
        }
    }

    /// For a copy of `[src, src + len)` into a texture: a buffer (readable by
    /// copies) that holds what shaders wrote there, at this place of the list.
    /// `None` where shaders do not write it.
    pub(crate) fn place_source(&self, src: usize, len: usize) -> Option<ID3D12Resource> {
        if !on() {
            return None;
        }
        let between = self.shaders_wrote(src, len, D3D12_RESOURCE_STATE_COPY_SOURCE)?;
        STATS[3].fetch_add(1, Ordering::Relaxed);
        Some(between)
    }

    /// For constants of `[src, src + len)` that a copy at its place earlier
    /// in this list writes: a buffer that holds them at this place of the
    /// list, and its card address. `None`: the fill is done as before.
    pub(crate) fn place_constants(&self, src: usize, len: usize) -> Option<(ID3D12Resource, u64)> {
        if !on() || !fill_in_place() {
            return None;
        }
        if !self.deferred().placed.iter().any(|&(from, to)| from < src + len && src < to) {
            return None;
        }
        let read = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        let Some(between) = self.shaders_wrote(src, len, read) else {
            FILLS[2].fetch_add(1, Ordering::Relaxed);
            return None;
        };
        FILLS[0].fetch_add(1, Ordering::Relaxed);
        FILLS[1].fetch_add(len as u64, Ordering::Relaxed);
        // SAFETY: a live buffer.
        let gpu = unsafe { between.GetGPUVirtualAddress() };
        Some((between, gpu))
    }

    /// A buffer that holds what shaders wrote at `[src, src + len)` at this
    /// place of the list, left in `state`. `None` where shaders do not write
    /// it.
    fn shaders_wrote(&self, src: usize, len: usize, state: D3D12_RESOURCE_STATES) -> Option<ID3D12Resource> {
        let host = host()?;
        let from = shaders_own(src, len)?;
        let between = self.between(host, len as u64)?;
        from.refresh(self, src as u64, len as u64);
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe {
            wait_for_all(host, &self.list);
            self.list.CopyBufferRegion(&between, 0, &from.buffer, from.offset(src as u64), len as u64);
            self.list.ResourceBarrier(&[transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
            wait_for_all(host, &self.list);
        }
        Some(between)
    }

    /// A buffer on the card for this recording alone.
    fn between(&self, host: &Host, len: u64) -> Option<ID3D12Resource> {
        let buffer = host.uav_buffer(len.next_multiple_of(1 << 16))?;
        name(&buffer, || "bytes between copies".into());
        self.keep(&buffer);
        Some(buffer)
    }
}

// From Grimkaaah/Victory-Grimkaaah f2586ef (audited clean by revisao-refs 27/11).
/// Whether a fill of upload memory that flights in the air may write, and
/// that stands behind a copy at its place of the same submission which puts
/// all of its bytes, takes them from that copy instead (`VICTORY_FILL_FORWARD=1`;
/// off by default; `VICTORY_AB` can switch it while the title runs).
///
/// Why: the title keeps small blocks of constants (0xb0 to 0xd60 bytes) in
/// buffers it puts new values in each frame with a buffer copy out of its
/// upload memory, and draws read them as constants, some ahead of the copy
/// (the old values), some behind it (the new ones). The copy of the frame
/// before is a flight's declared write there, so the fills wait for flights;
/// done on the card in front of all lists they would read the old values
/// behind the copy too, so the whole submission was refused
/// (`card_steps`, "a copy at its place in a list writes what it reads") and
/// stood until the card had run everything: 13 to 28 ms a frame with the
/// ray-tracing structure builder on (runs trace and trace2: lists 11 and
/// 13 behind the copy at list 11 step 313, lists 0, 7 and 11 ahead of the
/// one at step 367).
///
/// What the fill reads then is what the card puts there: a copy whose source
/// is taken on the CPU at submission ([`Placed::prepare`]) puts in its list's
/// own upload memory (`from`) the bytes the card copies, before any step
/// behind it is done. Only when nothing between the copy and the fill writes
/// any of the fill's bytes (another copy, a buffer copy, a shader's declared
/// write, a step of ours), and only for a fill that would wait otherwise.
/// The fills ahead of the copy still read the old values, on the card with
/// `VICTORY_PLACED_ORDER=1`, which takes what stands ahead of a copy at its
/// place.
pub(super) fn forwarding() -> bool {
    #[cfg(any(feature = "diagnostics", feature = "ab"))]
    if let Some(on) = crate::ab::pick("VICTORY_FILL_FORWARD") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_FILL_FORWARD"))
}

/// Since the last frame line: fills that took their bytes from a copy at its
/// place ([`forward_fills`]).
pub(super) static FORWARDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What [`fill_forwards`] looks at of a step.
#[derive(Clone, Copy, Debug)]
enum View {
    /// A copy at its place: where it puts its bytes, how many, and where its
    /// list holds them if it took them on the CPU at submission.
    Put { dst: usize, len: usize, from: Option<usize> },
    /// A write of `[at, at + len)`: a buffer copy, a shader's declared write.
    Writes { at: usize, len: usize },
    /// A step of ours: it may write anything.
    Run,
    /// A fill of upload memory out of `[src, src + len)`.
    Fill { src: usize, len: usize },
    /// Reads alone.
    Other,
}

impl View {
    fn of(step: &Step) -> View {
        match step {
            Step::Placed(p) => View::Put { dst: p.dst, len: p.len, from: p.from },
            &Step::Copy(dst, _, len) => View::Writes { at: dst, len },
            &Step::GpuWrite(_, at, len) => View::Writes { at, len },
            Step::Run(_) => View::Run,
            &Step::Fill(_, src, len, _) => View::Fill { src, len },
            _ => View::Other,
        }
    }
}

/// Which fills of `runs` (each: whether its list is named twice in the
/// submission, its steps in order) take their bytes from a copy at its
/// place ([`forwarding`]); `blocked` says which fills would wait for
/// flights. Each: (the run's place, the step's place, where its bytes are).
fn fill_forwards(runs: &[(bool, Vec<View>)], blocked: impl Fn(usize, usize) -> bool) -> Vec<(usize, usize, usize)> {
    // The copies whose bytes still stand: (where, how many, where held).
    let mut puts: Vec<(usize, usize, usize)> = Vec::new();
    let mut moves = Vec::new();
    // What writes `[at, at + n)` leaves no copy's bytes standing there.
    let wrote = |puts: &mut Vec<(usize, usize, usize)>, at: usize, n: usize| puts.retain(|p| !(at < p.0 + p.1 && p.0 < at + n));
    for (k, (twice, steps)) in runs.iter().enumerate() {
        for (i, step) in steps.iter().enumerate() {
            match *step {
                View::Put { dst, len, from } => {
                    wrote(&mut puts, dst, len);
                    if let Some(from) = from {
                        puts.push((dst, len, from));
                    }
                }
                View::Writes { at, len } => wrote(&mut puts, at, len),
                View::Run => puts.clear(),
                // (A list named twice in a submission does its steps twice:
                // left alone.)
                View::Fill { src, len } if !twice && len > 0 && blocked(src, len) => {
                    if let Some(&(dst, _, from)) = puts.iter().rev().find(|p| p.0 <= src && src + len <= p.0 + p.1) {
                        moves.push((k, i, from + (src - dst)));
                    }
                }
                View::Fill { .. } | View::Other => {}
            }
        }
    }
    moves
}

/// [`forwarding`]: before anything looks at what the steps of `runs` read,
/// with flights that write `written` in the air, the fills that can take
/// their bytes from a copy at its place ahead of them read those instead.
pub(super) fn forward_fills(runs: &[&HostList], written: &[(usize, usize)]) {
    if written.is_empty() {
        return;
    }
    let lists: Vec<usize> = runs.iter().map(|l| std::ptr::from_ref::<HostList>(l) as usize).collect();
    // One list's lock at a time: a submission may name a list twice.
    let views: Vec<(bool, Vec<View>)> = runs
        .iter()
        .enumerate()
        .map(|(k, list)| {
            let twice = lists.iter().filter(|&&l| l == lists[k]).count() > 1;
            (twice, list.deferred().copies.iter().map(View::of).collect())
        })
        .collect();
    let moves = fill_forwards(&views, |src, len| read_blocked(written, src, len));
    for &(k, i, from) in &moves {
        if let Some(Step::Fill(_, src, ..)) = runs[k].deferred().copies.get_mut(i) {
            *src = from;
        }
    }
    FORWARDED.fetch_add(moves.len() as u64, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Title memory of `len` bytes with a host copy (not in place).
    fn title_of(resource: usize, len: usize) -> (&'static mut [u8], usize, std::sync::Arc<Shadow>) {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: memory of our own, never freed.
        let base = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | WRITE_WATCH, PAGE_READWRITE) }.cast::<u8>();
        assert!(!base.is_null());
        crate::shadow::register(resource, base as u64, len as u64);
        let shadow = crate::shadow::find(base as u64, 64).expect("a copy");
        assert!(!shadow.in_place());
        // SAFETY: `len` bytes just committed.
        (unsafe { std::slice::from_raw_parts_mut(base, len) }, base as usize, shadow)
    }

    /// What the debug layer said (id, text), as it says it: the host's own
    /// reading of its queue at each submission empties the queue.
    static SAID: Mutex<Vec<(i32, String)>> = Mutex::new(Vec::new());

    unsafe extern "system" fn heard(
        _: D3D12_MESSAGE_CATEGORY,
        _: D3D12_MESSAGE_SEVERITY,
        id: D3D12_MESSAGE_ID,
        text: windows::core::PCSTR,
        _: *mut core::ffi::c_void,
    ) {
        // SAFETY: the layer's text, alive through the call.
        let text = unsafe { text.to_string() }.unwrap_or_default();
        SAID.lock().unwrap_or_else(|e| e.into_inner()).push((id.0, text));
    }

    /// Listens to the debug layer from now on; `false` without it.
    fn listen(host: &Host) -> bool {
        let Some(queue) = host.messages.as_ref().and_then(|q| q.cast::<ID3D12InfoQueue1>().ok()) else {
            return false;
        };
        let mut cookie = 0;
        // SAFETY: a function that lives as long as the process.
        unsafe { queue.RegisterMessageCallback(Some(heard), D3D12_MESSAGE_CALLBACK_FLAG_NONE, std::ptr::null_mut(), &mut cookie) }.is_ok()
    }

    /// What the debug layer said since the last call.
    fn messages() -> Vec<(i32, String)> {
        std::mem::take(&mut *SAID.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Reads at `read` as a root shader resource and writes at `write` as a
    /// root unordered access view, as draws of the list do.
    const ROOT_VIEWS: &str = r#"
#define RS "SRV(t0), UAV(u0)"
ByteAddressBuffer from : register(t0);
RWByteAddressBuffer to : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { to.Store(0, from.Load(0) + 1u); }
"#;

    /// Issue #350: a copy at its place leaves its buffers where later root
    /// views of the same list can take them (the debug layer said
    /// "Resource state (0x400: COPY_DEST) (promoted from COMMON state) ... is
    /// invalid for use as a root shader resource", id 538). Out of upload
    /// memory, out of another host copy, within one; in the card's memory
    /// and in the PC's. A diagnostics build with the debug layer:
    /// `VICTORY_D3D12_DEBUG=1 cargo test -p victory-gpu --features diagnostics
    /// --lib copies_in_place_leave -- --ignored`.
    #[test]
    #[ignore = "wants a diagnostics build, VICTORY_D3D12_DEBUG=1 and a process of its own"]
    fn copies_in_place_leave_their_buffers_to_root_views() {
        let Some(host) = super::host() else {
            return; // no host GPU
        };
        assert!(listen(host), "a diagnostics build with VICTORY_D3D12_DEBUG=1");
        assert!(on(), "copies at their place");
        let code = compile(ROOT_VIEWS, windows::core::s!("main"), windows::core::s!("cs_5_1")).unwrap();
        // SAFETY: the code lives through the calls; the root signature is in it.
        let (root, pso) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let out = host.uav_buffer(1 << 16).unwrap();
        let dispatch = |list: &HostList, read: u64, write: u64| {
            // SAFETY: an open list; live buffers.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootShaderResourceView(0, read);
                list.list.SetComputeRootUnorderedAccessView(1, write);
                list.list.Dispatch(1, 1, 1);
            }
        };
        // (In the card's memory from 4 MiB on, on a 16 GB card: the compare
        // that follows the lists in the same call reads what shaders and
        // these copies wrote as a root shader resource.)
        let mut resource = 0x3500;
        let mut wrong = Vec::new();
        for len in [256usize << 10, 8 << 20] {
            for case in 0..4 {
                let _ = messages();
                let compares = CARD_PASSES[2].load(Ordering::Relaxed);
                let (title, start, to) = title_of(resource, len);
                let (other, other_start, from) = title_of(resource + 1, len);
                resource += 2;
                title[..64].fill(1);
                other[..64].fill(2);
                let list = host.new_list().unwrap();
                match case {
                    // Out of upload memory: the title's CPU wrote the source.
                    0 => assert!(list.place_copy(start + 4096, start, 64)),
                    // Out of another host copy, which a shader writes.
                    1 => {
                        from.refresh(&list, other_start as u64, 64);
                        from.gpu_writes(&list, other_start as u64, 64);
                        assert!(list.place_copy(start + 4096, other_start, 64));
                    }
                    // Within one host copy that a shader writes.
                    2 => {
                        to.refresh(&list, start as u64, 64);
                        to.gpu_writes(&list, start as u64, 64);
                        assert!(list.place_copy(start + 4096, start, 64));
                    }
                    // As 0, and nothing of the list reads it after: only the
                    // compare does, for a copy in the card's memory.
                    _ => assert!(list.place_copy(start + 4096, start, 64)),
                }
                // As a draw's root view does (`VICTORY_ROOT_SRV_DIRECT=1`).
                if case < 3 {
                    dispatch(&list, to.gpu((start + 4096) as u64), out_gpu(&out));
                }
                list.close();
                host.execute(&[&list]);
                let said = messages();
                let carded = to.carded();
                let compared = CARD_PASSES[2].load(Ordering::Relaxed) - compares;
                eprintln!("{len:#x} case {case}: carded {carded}, compares {compared}, {} messages", said.len());
                for (id, text) in &said {
                    eprintln!("{len:#x} case {case} (carded {carded}, compares {compared}): D3D12 {id}: {text}");
                }
                if !said.is_empty() {
                    wrong.push((len, case, said.len()));
                }
                assert_eq!(title[4096], if case == 1 { 2 } else { 1 }, "the copy's bytes");
            }
        }
        assert_eq!(wrong, Vec::new(), "(bytes, case, messages) the debug layer spoke of");
    }

    fn out_gpu(out: &ID3D12Resource) -> u64 {
        // SAFETY: a live buffer.
        unsafe { out.GetGPUVirtualAddress() }
    }
    /// A fill takes a copy's bytes only if all it reads is that copy's, and
    /// nothing wrote over them in between.
    #[test]
    fn fills_take_the_bytes_of_the_copy_in_front() {
        let found = |s: &Standing, src, len| s.fill((9, 9), src, len).map(|f: Forward| (f.copy, f.at));
        let mut s = Standing::default();
        assert_eq!(found(&s, 2000, 8), None);
        s.copy((0, 1), 2000, 64, true);
        assert_eq!(found(&s, 2000, 64), Some(((0, 1), 0)));
        assert_eq!(found(&s, 2032, 16), Some(((0, 1), 32)));
        assert_eq!(found(&s, 2032, 64), None);
        assert_eq!(found(&s, 1990, 16), None);
        // A later copy over part of it: the bytes there are its.
        s.copy((1, 0), 2040, 8, true);
        assert_eq!(found(&s, 2000, 8), None);
        assert_eq!(found(&s, 2040, 8), Some(((1, 0), 0)));
        // One whose source flights may write.
        s.copy((1, 1), 2040, 8, false);
        assert_eq!(found(&s, 2040, 8), None);
        s.copy((1, 2), 3000, 16, true);
        s.written(3008, 1);
        assert_eq!(found(&s, 3000, 4), None);
    }

    /// What copies with two sides cost to make (ignored: a measurement for
    /// `VICTORY_CARD_ALL`): buffers of 64 KiB on the card and in the PC's
    /// memory.
    #[test]
    #[ignore]
    fn small_buffers_cost() {
        let Some(host) = super::host() else {
            return;
        };
        const N: usize = 1000;
        let used = || host.video_memory().map_or(0, |m| m[1] >> 20);
        let before = used();
        let t = std::time::Instant::now();
        let card: Vec<_> = (0..N).map(|_| host.uav_buffer(1 << 16)).collect();
        let card_us = t.elapsed().as_micros() as f64 / N as f64;
        let after = used();
        let t = std::time::Instant::now();
        let pc: Vec<_> = (0..N).map(|_| host.shared_buffer(1 << 16)).collect();
        let pc_us = t.elapsed().as_micros() as f64 / N as f64;
        println!(
            "{N} buffers of 64 KiB: on the card {card_us:.0} us each, {} MB of its memory; in the PC's memory {pc_us:.0} us each",
            after.saturating_sub(before)
        );
        drop((card, pc));
    }

    #[test]
    fn place_constant_reads_what_place_copy_wrote() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // (Off unless `VICTORY_CARD_STEPS=2` or `VICTORY_PLACED_CONSTANTS=1`.)
        let Some(host) = super::host().filter(|_| placed_constants()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        // SAFETY: fresh 64 KiB page-aligned allocation for the test.
        let base = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert!(base != 0);
        // Initialize source bytes outside the destination range.
        let src = base + 4096;
        let dst = base + 8192;
        // SAFETY: `base..base + LEN` is committed read-write memory.
        unsafe {
            for i in 0..512 {
                ((src + i) as *mut u8).write((i.wrapping_mul(37) ^ 0xa5) as u8);
            }
        }
        crate::shadow::register(0x501a_ce01, base as u64, LEN as u64);
        let _shadow = crate::shadow::find(dst as u64, 512).expect("shadow for registered range");
        let list = host.new_list().unwrap();
        assert!(list.place_copy(dst, src, 512));
        // Partial overlap or disjoint ranges must bail out cleanly.
        assert!(list.place_constant(dst - 16, 64).is_none());
        assert!(list.place_constant(dst + 480, 64).is_none());
        assert!(list.place_constant(base + 16384, 64).is_none());
        let (buf1, gpu1) = list.place_constant(dst, 256).expect("first slice inside placed copy");
        let (buf2, gpu2) = list.place_constant(dst + 256, 256).expect("second slice inside placed copy");
        assert_eq!(gpu1 % 256, 0);
        assert_eq!(gpu2 % 256, 0);
        assert!(!list.deferred().copies.iter().any(|s| matches!(s, super::Step::Fill(..))));

        let (out, out_ptr) = host.shared_buffer(512).unwrap();
        // SAFETY: `list` is open; offsets match `gpu1`/`gpu2` inside `buf1`/`buf2`.
        unsafe {
            let off1 = gpu1 - buf1.GetGPUVirtualAddress();
            let off2 = gpu2 - buf2.GetGPUVirtualAddress();
            list.list.CopyBufferRegion(&out, 0, &buf1, off1, 256);
            list.list.CopyBufferRegion(&out, 256, &buf2, off2, 256);
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.log_messages(), None);
        // SAFETY: `out_ptr` is a 512-byte shared buffer mapping, and `src` is inside `base..base + LEN`.
        let got = unsafe { std::slice::from_raw_parts(out_ptr, 512) };
        let expected = unsafe { std::slice::from_raw_parts(src as *const u8, 512) };
        assert_eq!(got, expected);
        crate::shadow::unregister(0x501a_ce01, base as u64);
        // SAFETY: `base` was allocated with `VirtualAlloc` and `list` has finished.
        unsafe {
            VirtualFree(base as *mut _, 0, MEM_RELEASE);
        }
    }
}

/// [`fill_forwards`]: which fills take the bytes of a copy at its place ahead of
/// them, and which keep reading title memory.
#[cfg(test)]
mod forward_tests {
    use super::{View, fill_forwards};

    fn put(dst: usize, len: usize, from: usize) -> View {
        View::Put { dst, len, from: Some(from) }
    }

    fn fill(src: usize, len: usize) -> View {
        View::Fill { src, len }
    }

    #[test]
    fn a_fill_behind_a_copy_takes_its_bytes() {
        let runs = vec![(false, vec![put(0x1000, 0xd0, 0x9000), View::Other, fill(0x1020, 0xb0)])];
        assert_eq!(fill_forwards(&runs, |_, _| true), vec![(0, 2, 0x9020)]);
        // In a later list too.
        let runs = vec![(false, vec![put(0x1000, 0xd0, 0x9000)]), (false, vec![View::Other, fill(0x1000, 0xd0)])];
        assert_eq!(fill_forwards(&runs, |_, _| true), vec![(1, 1, 0x9000)]);
    }

    #[test]
    fn fills_ahead_of_the_copy_keep_the_old_bytes() {
        let runs = vec![(false, vec![fill(0x1000, 0x20)]), (false, vec![fill(0x1040, 0x20), put(0x1000, 0xd0, 0x9000)])];
        assert!(fill_forwards(&runs, |_, _| true).is_empty());
    }

    #[test]
    fn the_last_copy_ahead_of_the_fill_is_the_one_taken() {
        let runs = vec![(false, vec![put(0x1000, 0xd0, 0x9000), put(0x1000, 0xd0, 0xa000), fill(0x1010, 0x10)])];
        assert_eq!(fill_forwards(&runs, |_, _| true), vec![(0, 2, 0xa010)]);
        // One that puts only part of the first leaves neither standing there.
        let runs = vec![(false, vec![put(0x1000, 0xd0, 0x9000), put(0x1080, 0x10, 0xa000), fill(0x1000, 0x10)])];
        assert!(fill_forwards(&runs, |_, _| true).is_empty());
    }

    #[test]
    fn anything_that_may_change_the_bytes_leaves_the_fill_alone() {
        let alone = |steps: Vec<View>, twice: bool, blocked: bool| fill_forwards(&[(twice, steps)], |_, _| blocked).is_empty();
        // A shader's declared write or a buffer copy in between.
        assert!(alone(vec![put(0x1000, 0xd0, 0x9000), View::Writes { at: 0x10cc, len: 4 }, fill(0x1000, 0xd0)], false, true));
        // A step of ours in between.
        assert!(alone(vec![put(0x1000, 0xd0, 0x9000), View::Run, fill(0x1000, 0xd0)], false, true));
        // A fill that reaches past the copy's bytes.
        assert!(alone(vec![put(0x1000, 0xd0, 0x9000), fill(0x1000, 0xe0)], false, true));
        // A copy that did not take its bytes on the CPU (out of a host copy).
        assert!(alone(vec![View::Put { dst: 0x1000, len: 0xd0, from: None }, fill(0x1000, 0xd0)], false, true));
        // A list named twice in the submission.
        assert!(alone(vec![put(0x1000, 0xd0, 0x9000), fill(0x1000, 0xd0)], true, true));
        // A fill that would not wait for flights: as before.
        assert!(alone(vec![put(0x1000, 0xd0, 0x9000), fill(0x1000, 0xd0)], false, false));
        // A write elsewhere leaves it standing.
        assert!(!alone(vec![put(0x1000, 0xd0, 0x9000), View::Writes { at: 0x10d0, len: 4 }, fill(0x1000, 0xd0)], false, true));
    }
}
