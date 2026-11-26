//! The compute pipelines of the console's ray-tracing structure builder
//! (`XboxBvhBuilder.dll`) run **all or none**. It builds and refits the
//! structures in title memory every frame through some fifty compute
//! dispatches, and its stages feed each other through memory: a stage that
//! runs on what a dropped stage never wrote computes its store offsets from
//! garbage. The structures lie in plain title memory, between the title's
//! own objects, and a root view has no size, so those stores land on the
//! neighbours (run 98, with 14 of 35 stages translated: the title died
//! within a second of the first builder submission, on pointers that had
//! become bounding-box bit patterns; upstream #2118 saw the same on
//! fxceless/victory-fork).
//!
//! So the builder's dispatches run only in frames where every pipeline it
//! has created has its host pipeline; otherwise none of them run, and the
//! structures keep what they held. The decision is made once per frame, so
//! a frame never runs part of the builder. Its pipelines are started at
//! once when the title creates them, so no frame's wait budget holds them
//! back. A stage that could not be bound stops the stages after it for the
//! rest of that frame and the next ([`rests`]).
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image
//! (upstream #2118). Before its image is known they are guessed by their
//! programs (`of_builder`: a spilled user-data table).
//!
//! shortcut: all of it only with `VICTORY_BVH_BUILDER=1`, until a run has
//! shown the whole builder and the host side of title memory by address
//! (`memory.rs`) working on a card. Without it the builder never runs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// Whether the builder may run at all (`VICTORY_BVH_BUILDER=1`).
pub(crate) fn allowed() -> bool {
    if cfg!(test) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1"))
}

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, base + size));
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the builder's image is known (its pipelines are then known by
/// origin alone).
pub(crate) fn mapped() -> bool {
    image().is_some()
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    let mut frames = [std::ptr::null_mut(); 24];
    // SAFETY: room for that many frames.
    let n = unsafe { RtlCaptureStackBackTrace(1, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) };
    frames[..usize::from(n)].iter().map(|&f| f as usize).collect()
}

/// The builder's pipelines (alive while the title holds them).
static PIPELINES: Mutex<Vec<Weak<Pending>>> = Mutex::new(Vec::new());

/// The builder created pipeline `p`: it counts towards [`runs`], and is
/// made at once if the builder may run.
pub(crate) fn note(p: &Arc<Pending>) {
    PIPELINES.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(p));
    if allowed() {
        make::begin(p);
    }
}

/// No stage runs before this frame.
static RESTS: AtomicU64 = AtomicU64::new(0);

/// A stage of the builder was dropped after all: none runs for the rest of
/// this frame and the next (what it fed is not there).
pub(crate) fn rests(why: &'static str) {
    RESTS.store(crate::frame::current() + 2, Ordering::Relaxed);
    static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if SAID.fetch_add(1, Ordering::Relaxed) < 20 {
        tracing::warn!(why, "the ray-tracing structure builder rests for this frame and the next: its stages run all or none");
    }
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, false));

/// Whether the builder's dispatches run in the current frame: decided once
/// per frame, so that a frame never runs part of the builder.
pub(crate) fn runs() -> bool {
    let frame = crate::frame::current();
    if !allowed() || frame < RESTS.load(Ordering::Relaxed) {
        return false;
    }
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        return decided.1;
    }
    let pipelines: Vec<Arc<Pending>> = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|p| p.strong_count() > 0);
        pipelines.iter().filter_map(Weak::upgrade).collect()
    };
    let states: Vec<Option<bool>> = pipelines.iter().map(|p| make::state(p)).collect();
    let open = !states.is_empty() && all_made(&states);
    if open != decided.1 && !states.is_empty() {
        let (made, refused) = (
            states.iter().filter(|s| **s == Some(true)).count(),
            states.iter().filter(|s| **s == Some(false)).count(),
        );
        let which: Vec<String> = pipelines
            .iter()
            .zip(&states)
            .filter(|(_, s)| **s == Some(false))
            .take(8)
            .map(|(p, _)| make::hashes(p))
            .collect();
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            ?which,
            "the ray-tracing structure builder {}",
            if open { "runs: all of its programs have host pipelines" } else { "is held back: not all of its programs have a host pipeline" }
        );
    }
    *decided = (frame, open);
    open
}

/// Whether every pipeline is made (`Some(true)`); refused ones are
/// `Some(false)`, those still being made `None`.
fn all_made(states: &[Option<bool>]) -> bool {
    states.iter().all(|s| *s == Some(true))
}

/// Tests that set the builder's image, one at a time; each starts with
/// no decision made.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, false);
    guard
}

/// No builder mapped (a title without it).
#[cfg(test)]
pub(crate) fn clear_image_for_tests() {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The builder runs only when every one of its pipelines is made: one
    /// refused or still being made holds all of them back.
    #[test]
    fn the_builder_runs_only_with_every_pipeline_made() {
        assert!(all_made(&[Some(true), Some(true)]));
        assert!(!all_made(&[Some(true), Some(false)]), "one refused");
        assert!(!all_made(&[Some(true), None]), "one still being made");
    }

    /// Pipelines are the builder's by their bytecode in its image, or by
    /// its code among the callers; not otherwise.
    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        clear_image_for_tests();
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        // Bytecode inside the image.
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        // A caller inside the image: this function's own code.
        *IMAGE.lock().unwrap() = Some((here, here + 0x4000));
        assert!(created_by_builder(&[]));
        clear_image_for_tests();
    }
}
