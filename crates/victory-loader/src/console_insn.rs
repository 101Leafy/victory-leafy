//! Instructions Zen 2 has and the Intel machines lack, handled where they fault with
//! STATUS_ILLEGAL_INSTRUCTION: `clzero`, `monitorx`, `mwaitx`, `rdpru`, and SSE4a's `extrq`,
//! `insertq`, `movntss` and `movntsd`. The hot ones are patched in place on their first fault.
//!
//! - `VICTORY_CLZERO_ASK=1` asks Windows about memory on every fault instead of keeping a
//!   short per-thread list of regions.
//! - `VICTORY_PATCH_WAITS=0` leaves `monitorx` and `mwaitx` faulting instead of rewriting them.
//! - `VICTORY_PATCH_CLZERO=0` runs `clzero` loops in the handler instead of sending them to a
//!   stub with plain stores.

#[path = "console_insn_decode.rs"]
mod decode;

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, FlushInstructionCache, M128A};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE, PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE, PAGE_WRITECOPY, VirtualAlloc, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::crash::{CMP_FLAGS, cmp_flags};
use decode::{Insn, decode};

/// The size of the area `near` hands out.
pub(crate) const STUBS_LEN: usize = 0x10000;

const PAGE: usize = 0x1000;
/// Code read ahead of the faulting instruction, for the loops that start before their `clzero`.
const BEFORE: usize = 8;
const WINDOW: usize = 32;
const SLOTS: usize = 8;
/// How many answers a remembered region gives before Windows is asked about it again.
const ANSWERS: u32 = 1024;

static EMULATED: AtomicU64 = AtomicU64::new(0);
static QUERIES: AtomicU64 = AtomicU64::new(0);
static ZEROED: AtomicU64 = AtomicU64::new(0);
static PATCHED: AtomicU64 = AtomicU64::new(0);

/// Our stub areas, as base and bytes used.
static AREAS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static PATCHING: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
struct Options {
    cache: bool,
    waits: bool,
    loops: bool,
}

impl Options {
    fn get() -> Self {
        static OPTIONS: OnceLock<Options> = OnceLock::new();
        let set = |name: &str| std::env::var_os(name).map(|v| !v.is_empty() && v != "0");
        *OPTIONS.get_or_init(|| Options {
            cache: set("VICTORY_CLZERO_ASK") != Some(true),
            waits: set("VICTORY_PATCH_WAITS") != Some(false),
            loops: set("VICTORY_PATCH_CLZERO") != Some(false),
        })
    }
}

#[derive(Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
    read: bool,
    write: bool,
    left: u32,
}

impl Region {
    const NONE: Region = Region { start: 0, end: 0, read: false, write: false, left: 0 };
}

thread_local! {
    static REGIONS: Cell<[Region; SLOTS]> = const { Cell::new([Region::NONE; SLOTS]) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loop {
    /// `clzero; add rax, 64; cmp rax, b; jb` back to the `clzero`, 12 bytes.
    A { b: u8 },
    /// `mov rax, a; add a, 64; clzero; cmp a, b; jb` back to the `mov`, 15 bytes with 7
    /// ahead of the `clzero`.
    B { a: u8, b: u8 },
}

/// Handles an illegal-instruction fault at `ctx.Rip` when it is one of the instructions above:
/// -1 (continue execution) when it did, 0 to leave it to the rest of the handler.
///
/// # Safety
/// `ctx` must be the context of the thread that faulted, handed to a vectored handler running
/// on that thread.
pub unsafe fn emulate(ctx: &mut CONTEXT) -> i32 {
    // SAFETY: as the caller promises
    if unsafe { handle(ctx, Options::get()) } { -1 } else { 0 }
}

unsafe fn handle(ctx: &mut CONTEXT, o: Options) -> bool {
    let rip = ctx.Rip as usize;
    let Some(code) = fetch(rip, o) else {
        return false;
    };
    let here = &code[BEFORE..];
    if ours(rip, here) {
        return true;
    }
    let Some(i) = decode(here) else {
        return false;
    };
    if i.vex || i.map != 1 {
        return false;
    }
    // SAFETY: each checks the memory it writes, and the patches only rewrite the faulting code
    let done = unsafe {
        match (i.op, i.modrm, i.rep) {
            (0x01, Some(0xfc), _) => clzero(ctx, &code, &i, o),
            // f3 0f 01 fa is mcommit
            (0x01, Some(0xfa | 0xfb), 0) => wait(ctx, here, &i, o),
            (0x01, Some(0xfd), _) => rdpru(ctx, &i),
            (0x78 | 0x79, Some(_), 0) if i.opsize => extrq(ctx, here, &i),
            (0x78 | 0x79, Some(_), 0xf2) => insertq(ctx, here, &i),
            (0x2b, Some(_), 0xf2 | 0xf3) => movnt(ctx, &i, o),
            _ => false,
        }
    };
    if done {
        let n = EMULATED.fetch_add(1, Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(
                emulated = n,
                queries = QUERIES.load(Relaxed),
                zeroed = ZEROED.load(Relaxed),
                patched = PATCHED.load(Relaxed),
                "AMD-only instructions handled"
            );
        }
    }
    done
}

/// The code from `BEFORE` bytes ahead of `rip`; bytes on pages it may not read stay 0.
fn fetch(rip: usize, o: Options) -> Option<[u8; WINDOW]> {
    let start = rip.checked_sub(BEFORE)?;
    let end = start + WINDOW;
    let page = rip & !(PAGE - 1);
    let next = page + PAGE;
    let lo = if start >= page || readable(start, page - start, o) { start } else { page };
    let hi = if end <= next || readable(next, end - next, o) { end } else { next };
    let mut code = [0; WINDOW];
    // SAFETY: rip's page holds the instruction that faulted, and the pages either side were
    // checked before being read
    unsafe { std::ptr::copy_nonoverlapping(lo as *const u8, code.as_mut_ptr().add(lo - start), hi - lo) };
    Some(code)
}

/// Bytes one of our patches left at `rip`. A thread that faulted on the old bytes can land here
/// after the patch went in; it runs them again.
fn ours(rip: usize, here: &[u8]) -> bool {
    match *here {
        [0x0f, 0x1f, 0xfa, ..] | [0xf3, 0x90, 0x90, ..] => true,
        [0xe9, a, b, c, d, ..] => {
            let to = (rip + 5).wrapping_add_signed(i32::from_le_bytes([a, b, c, d]) as isize);
            areas().iter().any(|&(base, _)| (base..base + STUBS_LEN).contains(&to))
        }
        _ => false,
    }
}

fn areas() -> MutexGuard<'static, Vec<(usize, usize)>> {
    AREAS.lock().unwrap_or_else(PoisonError::into_inner)
}

fn info(at: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    // SAFETY: an all-zero MEMORY_BASIC_INFORMATION is valid
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: m is as large as the length passed
    let n = unsafe { VirtualQuery(at as *const c_void, &mut m, size_of::<MEMORY_BASIC_INFORMATION>()) };
    (n != 0).then_some(m)
}

fn query(at: usize) -> Option<Region> {
    QUERIES.fetch_add(1, Relaxed);
    let m = info(at)?;
    if m.State != MEM_COMMIT {
        return None;
    }
    let p = if m.Protect & (PAGE_GUARD | PAGE_NOACCESS) != 0 { 0 } else { m.Protect };
    let write = p & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0;
    let read = write || p & (PAGE_READONLY | PAGE_EXECUTE | PAGE_EXECUTE_READ) != 0;
    let start = m.BaseAddress as usize;
    Some(Region { start, end: start + m.RegionSize, read, write, left: ANSWERS })
}

fn region(at: usize, o: Options) -> Option<Region> {
    if o.cache {
        let mut all = REGIONS.get();
        if let Some(r) = all.iter_mut().find(|r| r.left > 0 && (r.start..r.end).contains(&at)) {
            r.left -= 1;
            let r = *r;
            REGIONS.set(all);
            return Some(r);
        }
    }
    let r = query(at)?;
    if o.cache && r.read {
        let mut all = REGIONS.get();
        if let Some(slot) = all.iter_mut().min_by_key(|r| r.left) {
            *slot = r;
        }
        REGIONS.set(all);
    }
    Some(r)
}

fn allowed(mut at: usize, len: usize, write: bool, o: Options) -> bool {
    let Some(end) = at.checked_add(len) else {
        return false;
    };
    while at < end {
        match region(at, o) {
            Some(r) if r.read && (r.write || !write) => at = r.end,
            _ => return false,
        }
    }
    true
}

fn readable(at: usize, len: usize, o: Options) -> bool {
    allowed(at, len, false, o)
}

fn writable(at: usize, len: usize, o: Options) -> bool {
    allowed(at, len, true, o)
}

fn gpr(ctx: &mut CONTEXT, n: u8) -> &mut u64 {
    match n & 15 {
        0 => &mut ctx.Rax,
        1 => &mut ctx.Rcx,
        2 => &mut ctx.Rdx,
        3 => &mut ctx.Rbx,
        4 => &mut ctx.Rsp,
        5 => &mut ctx.Rbp,
        6 => &mut ctx.Rsi,
        7 => &mut ctx.Rdi,
        8 => &mut ctx.R8,
        9 => &mut ctx.R9,
        10 => &mut ctx.R10,
        11 => &mut ctx.R11,
        12 => &mut ctx.R12,
        13 => &mut ctx.R13,
        14 => &mut ctx.R14,
        _ => &mut ctx.R15,
    }
}

fn xmm(ctx: &mut CONTEXT, n: u8) -> &mut M128A {
    // SAFETY: on x64 the floating-point part of CONTEXT is always the XSAVE layout
    unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters[usize::from(n & 15)] }
}

fn set_flags(ctx: &mut CONTEXT, flags: u32) {
    ctx.EFlags = (ctx.EFlags & !CMP_FLAGS) | flags;
}

/// Zeroes `len` bytes at `at` once it knows they are writable.
unsafe fn zero(at: usize, len: usize, o: Options) -> bool {
    if !writable(at, len, o) {
        return false;
    }
    // SAFETY: checked just above
    unsafe { std::ptr::write_bytes(at as *mut u8, 0, len) };
    ZEROED.fetch_add(len as u64, Relaxed);
    true
}

unsafe fn clzero(ctx: &mut CONTEXT, code: &[u8; WINDOW], i: &Insn, o: Options) -> bool {
    if i.len == 3
        && let Some(l) = find_loop(code)
    {
        // SAFETY: the loop was matched byte for byte at rip
        if o.loops && unsafe { patch_loop(ctx.Rip as usize, code, l) } {
            return true;
        }
        // SAFETY: as for clzero
        return unsafe { run_loop(ctx, l, o) };
    }
    let rax = if i.addr32 { ctx.Rax & 0xffff_ffff } else { ctx.Rax };
    // SAFETY: zero checks the line first
    if !unsafe { zero(rax as usize & !63, 64, o) } {
        return false;
    }
    ctx.Rip += i.len as u64;
    true
}

/// A known `clzero` loop around the `clzero` at `code[BEFORE]`.
fn find_loop(code: &[u8; WINDOW]) -> Option<Loop> {
    let c = BEFORE;
    if code[c..c + 3] != [0x0f, 0x01, 0xfc] {
        return None;
    }
    if let [0x48, 0x83, 0xc0, 0x40, rex @ (0x48 | 0x4c), 0x39, m, 0x72, 0xf4, ..] = code[c + 3..] {
        let b = ((m >> 3) & 7) | ((rex & 4) << 1);
        if m & 0xc7 == 0xc0 && b != 0 && b != 4 {
            return Some(Loop::A { b });
        }
    }
    let [r1 @ (0x48 | 0x4c), 0x89, m1, r2 @ (0x48 | 0x49), 0x83, m2, 0x40] = code[c - 7..c] else {
        return None;
    };
    let [r3, 0x39, m3, 0x72, 0xf1, ..] = code[c + 3..] else {
        return None;
    };
    let a = ((m1 >> 3) & 7) | ((r1 & 4) << 1);
    let b = ((m3 >> 3) & 7) | ((r3 & 4) << 1);
    let ok = m1 & 0xc7 == 0xc0
        && m2 & 0xf8 == 0xc0
        && (m2 & 7) | ((r2 & 1) << 3) == a
        && r3 & 0xfa == 0x48
        && m3 & 0xc0 == 0xc0
        && (m3 & 7) | ((r3 & 1) << 3) == a
        && a != 0
        && b != 0
        && a != 4
        && b != 4
        && a != b;
    ok.then_some(Loop::B { a, b })
}

/// The whole loop in one go, leaving registers and flags as its last `cmp` would.
unsafe fn run_loop(ctx: &mut CONTEXT, l: Loop, o: Options) -> bool {
    match l {
        Loop::A { b } => {
            let (rax, b) = (ctx.Rax, *gpr(ctx, b));
            let n = if b > rax { (b - rax).div_ceil(64) } else { 1 };
            // SAFETY: zero checks the range first
            if !unsafe { zero(rax as usize & !63, (n * 64) as usize, o) } {
                return false;
            }
            ctx.Rax = rax.wrapping_add(n * 64);
            set_flags(ctx, cmp_flags(ctx.Rax, b));
            ctx.Rip += 12;
        }
        Loop::B { a, b } => {
            let (rax, a0, bv) = (ctx.Rax, *gpr(ctx, a), *gpr(ctx, b));
            let n = if bv > a0 { (bv - a0).div_ceil(64) } else { 0 };
            let (first, rest) = (rax as usize & !63, a0 as usize & !63);
            if !writable(first, 64, o) || !writable(rest, (n * 64) as usize, o) {
                return false;
            }
            // SAFETY: both checked just above
            unsafe {
                zero(first, 64, o);
                zero(rest, (n * 64) as usize, o);
            }
            let last = a0.wrapping_add(n * 64);
            if n > 0 {
                ctx.Rax = last - 64;
            }
            *gpr(ctx, a) = last;
            set_flags(ctx, cmp_flags(last, bv));
            ctx.Rip += 8;
        }
    }
    true
}

/// `push rcx; mov rcx, rax; and rcx, -64`, eight `mov qword [rcx+8n], 0`, `pop rcx`.
fn zero_line(out: &mut Vec<u8>) {
    out.extend([0x51, 0x48, 0x89, 0xc1, 0x48, 0x83, 0xe1, 0xc0]);
    out.extend([0x48, 0xc7, 0x01, 0, 0, 0, 0]);
    for d in (8..64).step_by(8) {
        out.extend([0x48, 0xc7, 0x41, d, 0, 0, 0, 0]);
    }
    out.push(0x59);
}

/// Sends the loop to a stub doing the same with plain stores, through a `jmp` over the
/// `clzero` and the two bytes after it. Threads only ever stand at the `clzero` or past the
/// loop, so none is left inside the bytes the `jmp` covers.
unsafe fn patch_loop(at: usize, code: &[u8; WINDOW], l: Loop) -> bool {
    let c = BEFORE;
    let mut body = Vec::with_capacity(96);
    zero_line(&mut body);
    let back = match l {
        Loop::A { .. } => {
            body.extend(&code[c + 3..c + 10]);
            body.extend([0x72, (-(body.len() as isize + 2)) as u8]);
            at + 12
        }
        Loop::B { .. } => {
            body.extend(&code[c + 3..c + 6]);
            body.extend([0x73, 9]);
            body.extend(&code[c - 7..c]);
            body.extend([0xeb, (-(body.len() as isize + 2)) as u8]);
            at + 8
        }
    };
    let Some(stub) = room(at, body.len() + 5) else {
        return false;
    };
    let (Some(out), Some(into)) = (jump(stub + body.len(), back), jump(at, stub)) else {
        return false;
    };
    body.extend(out);
    // SAFETY: room handed out these bytes to us alone, in memory we allocated writable
    unsafe {
        std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
        FlushInstructionCache(GetCurrentProcess(), stub as *const c_void, body.len());
    }
    // SAFETY: at is the clzero that faulted
    if !unsafe { poke(at, &into) } {
        return false;
    }
    tracing::info!(
        at = format_args!("{at:#x}"),
        stub = format_args!("{stub:#x}"),
        "clzero loop sent to plain stores (VICTORY_PATCH_CLZERO)"
    );
    true
}

/// `len` bytes of our stub space within a jump of `at`.
fn room(at: usize, len: usize) -> Option<usize> {
    let len = len.next_multiple_of(16);
    let mut areas = areas();
    let reaches = |base: usize| jump(at, base).is_some() && jump(base + STUBS_LEN, at).is_some();
    let i = match areas.iter().position(|&(base, used)| used + len <= STUBS_LEN && reaches(base)) {
        Some(i) => i,
        None => {
            areas.push((near(at)?, 0));
            areas.len() - 1
        }
    };
    let (base, used) = &mut areas[i];
    *used += len;
    Some(*base + *used - len)
}

/// Writes `bytes` over live code with one 8-byte store that stays inside a cache line, so a
/// thread running past sees all the old bytes or all the new ones.
unsafe fn poke(at: usize, bytes: &[u8]) -> bool {
    let w = at.min((at & !63) + 56);
    if at + bytes.len() > w + 8 {
        return false;
    }
    let _held = PATCHING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut old = 0;
    // SAFETY: w..w + 8 is mapped code inside one page
    if unsafe { VirtualProtect(w as *const c_void, 8, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        return false;
    }
    // SAFETY: made writable just above; the read is of the same 8 bytes
    unsafe {
        let mut v = (w as *const u64).read_unaligned().to_le_bytes();
        v[at - w..][..bytes.len()].copy_from_slice(bytes);
        let v = u64::from_le_bytes(v);
        std::arch::asm!("mov qword ptr [{w}], {v}", w = in(reg) w, v = in(reg) v, options(nostack, preserves_flags));
        VirtualProtect(w as *const c_void, 8, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), w as *const c_void, 8);
    }
    PATCHED.fetch_add(1, Relaxed);
    true
}

/// `monitorx` and `mwaitx` as `pause`. With VICTORY_PATCH_WAITS on, the first fault rewrites
/// them instead: `monitorx` into the nop `0f 1f fa`, `mwaitx` into `pause; nop`.
unsafe fn wait(ctx: &mut CONTEXT, here: &[u8], i: &Insn, o: Options) -> bool {
    let rip = ctx.Rip as usize;
    if o.waits && i.len == 3 {
        let monitor = here[2] == 0xfa;
        // SAFETY: rip is the instruction that faulted, matched by decode
        let done = unsafe {
            if monitor { poke(rip + 1, &[0x1f]) } else { poke(rip, &[0xf3, 0x90, 0x90]) }
        };
        if done {
            let what = if monitor { "monitorx" } else { "mwaitx" };
            tracing::info!(at = format_args!("{rip:#x}"), what, "AMD wait patched (VICTORY_PATCH_WAITS)");
            return true;
        }
    }
    std::hint::spin_loop();
    ctx.Rip += i.len as u64;
    true
}

/// ecx 0 reads MPERF and 1 APERF; the TSC stands in for both, which keeps their ratio at 1.
/// Other registers read 0. Flags stay as they are.
fn rdpru(ctx: &mut CONTEXT, i: &Insn) -> bool {
    // SAFETY: rdtsc has no requirements
    let v = if ctx.Rcx as u32 <= 1 { unsafe { std::arch::x86_64::_rdtsc() } } else { 0 };
    ctx.Rax = v & 0xffff_ffff;
    ctx.Rdx = v >> 32;
    ctx.Rip += i.len as u64;
    true
}

/// The register numbers in ModRM's reg and rm fields, for register-only forms.
fn regs(i: &Insn) -> Option<(u8, u8)> {
    let m = i.modrm?;
    (m >> 6 == 3).then_some((((m >> 3) & 7) | ((i.rex & 4) << 1), (m & 7) | ((i.rex & 1) << 3)))
}

/// SSE4a length and index from their 6-bit fields; a length of 0 means 64.
fn field(len: u64, index: u64) -> (u32, u32) {
    let len = (len & 63) as u32;
    (if len == 0 { 64 } else { len }, (index & 63) as u32)
}

fn mask(len: u32) -> u64 {
    if len >= 64 { u64::MAX } else { (1 << len) - 1 }
}

/// `extrq`: the field at the index in the destination's low quadword, moved to its bottom.
fn extrq(ctx: &mut CONTEXT, here: &[u8], i: &Insn) -> bool {
    let Some((reg, rm)) = regs(i) else {
        return false;
    };
    let (dst, (len, index)) = if i.op == 0x78 {
        if reg != 0 {
            return false;
        }
        (rm, field(u64::from(here[i.imm_at]), u64::from(here[i.imm_at + 1])))
    } else {
        let src = xmm(ctx, rm).Low;
        (reg, field(src, src >> 8))
    };
    let d = xmm(ctx, dst);
    d.Low = (d.Low >> index) & mask(len);
    ctx.Rip += i.len as u64;
    true
}

/// `insertq`: the source's low bits written into the destination's low quadword at the index.
fn insertq(ctx: &mut CONTEXT, here: &[u8], i: &Insn) -> bool {
    let Some((dst, src)) = regs(i) else {
        return false;
    };
    let s = *xmm(ctx, src);
    let (len, index) = if i.op == 0x78 {
        field(u64::from(here[i.imm_at]), u64::from(here[i.imm_at + 1]))
    } else {
        let high = s.High as u64;
        field(high, high >> 8)
    };
    let m = mask(len) << index;
    let d = xmm(ctx, dst);
    d.Low = (d.Low & !m) | ((s.Low << index) & m);
    ctx.Rip += i.len as u64;
    true
}

fn gs_base() -> u64 {
    let teb: u64;
    // SAFETY: gs:[0x30] is the TEB's pointer to itself on x64 Windows
    unsafe {
        std::arch::asm!("mov {}, qword ptr gs:[0x30]", out(reg) teb, options(nostack, readonly, preserves_flags))
    };
    teb
}

/// `movntss` and `movntsd`: the low 32 or 64 bits of a register stored to memory.
unsafe fn movnt(ctx: &mut CONTEXT, i: &Insn, o: Options) -> bool {
    let Some(m) = i.modrm else {
        return false;
    };
    let next = ctx.Rip + i.len as u64;
    let mut r = [0; 16];
    for (n, v) in r.iter_mut().enumerate() {
        *v = *gpr(ctx, n as u8);
    }
    let Some(at) = i.address(|n| r[usize::from(n & 15)], next) else {
        return false;
    };
    let base = match i.segment {
        0x65 => gs_base(),
        0x64 => return false,
        _ => 0,
    };
    let at = at.wrapping_add(base) as usize;
    let v = xmm(ctx, ((m >> 3) & 7) | ((i.rex & 4) << 1)).Low;
    let size = if i.rep == 0xf3 { 4 } else { 8 };
    if !writable(at, size, o) {
        return false;
    }
    // SAFETY: checked just above
    unsafe {
        if size == 4 {
            (at as *mut u32).write_unaligned(v as u32);
        } else {
            (at as *mut u64).write_unaligned(v);
        }
    }
    ctx.Rip = next;
    true
}

/// `jmp rel32` from `from` to `to`, or `None` when `to` is out of reach.
pub(crate) fn jump(from: usize, to: usize) -> Option<[u8; 5]> {
    let rel = i32::try_from((to as i64).wrapping_sub(from as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// A fresh `STUBS_LEN` of read-write-execute memory that `jump` reaches from `at` and back.
pub(crate) fn near(at: usize) -> Option<usize> {
    const GRAIN: usize = 0x10000;
    const REACH: usize = 0x7ff0_0000;
    let (lo, hi) = (at.saturating_sub(REACH), at.saturating_add(REACH));
    let alloc = |c: usize| {
        // SAFETY: asks for fresh pages at a free address; nothing mapped is touched
        let p = unsafe { VirtualAlloc(c as *const c_void, STUBS_LEN, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) };
        (!p.is_null()).then_some(p as usize)
    };
    let mut p = at;
    while p > lo {
        let Some(m) = info(p) else { break };
        let (base, end) = (m.BaseAddress as usize, m.BaseAddress as usize + m.RegionSize);
        if m.State == MEM_FREE && end - base >= STUBS_LEN {
            let c = (end - STUBS_LEN) & !(GRAIN - 1);
            if c >= base
                && c >= lo
                && let Some(got) = alloc(c)
            {
                return Some(got);
            }
        }
        let Some(below) = base.checked_sub(1) else { break };
        p = below;
    }
    let mut p = at;
    while p < hi {
        let Some(m) = info(p) else { break };
        let (base, end) = (m.BaseAddress as usize, m.BaseAddress as usize + m.RegionSize);
        let c = base.next_multiple_of(GRAIN);
        if m.State == MEM_FREE
            && c + STUBS_LEN <= end.min(hi)
            && let Some(got) = alloc(c)
        {
            return Some(got);
        }
        p = end;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const ON: Options = Options { cache: true, waits: true, loops: true };
    const OFF: Options = Options { cache: false, waits: false, loops: false };

    fn rwx(len: usize) -> *mut u8 {
        // SAFETY: fresh pages, kept for the rest of the test run
        let p = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) };
        assert!(!p.is_null());
        p.cast()
    }

    fn put(at: *mut u8, bytes: &[u8]) {
        // SAFETY: at points into a buffer from rwx
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, bytes.len()) };
    }

    fn get(at: *const u8, len: usize) -> Vec<u8> {
        // SAFETY: at points into a buffer from rwx
        unsafe { std::slice::from_raw_parts(at, len).to_vec() }
    }

    fn context(rip: *const u8) -> Box<CONTEXT> {
        // SAFETY: an all-zero CONTEXT is valid
        let mut ctx: Box<CONTEXT> = Box::new(unsafe { std::mem::zeroed() });
        ctx.Rip = rip as u64;
        ctx
    }

    fn run(ctx: &mut CONTEXT, o: Options) -> bool {
        // SAFETY: the code and data these tests point at are their own rwx buffers
        unsafe { handle(ctx, o) }
    }

    #[test]
    fn clzero_one_line() {
        let (code, data) = (rwx(PAGE), rwx(PAGE));
        put(code.wrapping_add(16), &[0x0f, 0x01, 0xfc, 0xc3]);
        put(data, &[0xaa; 256]);
        for o in [ON, OFF] {
            let mut ctx = context(code.wrapping_add(16));
            ctx.Rax = data as u64 + 70;
            assert!(run(&mut ctx, o));
            assert_eq!(ctx.Rip, code as u64 + 19);
            assert_eq!(get(data, 64), [0xaa; 64]);
            assert_eq!(get(data.wrapping_add(64), 64), [0; 64]);
            assert_eq!(get(data.wrapping_add(128), 64), [0xaa; 64]);
            assert_eq!(get(code.wrapping_add(16), 4), [0x0f, 0x01, 0xfc, 0xc3]);
        }
    }

    #[test]
    fn clzero_refuses_read_only() {
        let code = rwx(PAGE);
        put(code.wrapping_add(16), &[0x0f, 0x01, 0xfc]);
        let mut ctx = context(code.wrapping_add(16));
        ctx.Rax = 0;
        assert!(!run(&mut ctx, ON));
        assert_eq!(ctx.Rip, code as u64 + 16);
    }

    #[test]
    fn waits_step_over() {
        let code = rwx(PAGE);
        for op in [0xfa, 0xfb] {
            put(code.wrapping_add(16), &[0x0f, 0x01, op]);
            let mut ctx = context(code.wrapping_add(16));
            assert!(run(&mut ctx, OFF));
            assert_eq!(ctx.Rip, code as u64 + 19);
            assert_eq!(get(code.wrapping_add(16), 3), [0x0f, 0x01, op]);
        }
    }

    #[test]
    fn waits_patched() {
        let code = rwx(PAGE);
        for (op, after) in [(0xfa, [0x0f, 0x1f, 0xfa]), (0xfb, [0xf3, 0x90, 0x90])] {
            put(code.wrapping_add(60), &[0x0f, 0x01, op]);
            let mut ctx = context(code.wrapping_add(60));
            assert!(run(&mut ctx, ON));
            assert_eq!(ctx.Rip, code as u64 + 60);
            assert_eq!(get(code.wrapping_add(60), 3), after);
            assert!(run(&mut ctx, ON));
            assert_eq!(ctx.Rip, code as u64 + 60);
        }
    }

    #[test]
    fn mcommit_left_alone() {
        let code = rwx(PAGE);
        put(code.wrapping_add(16), &[0xf3, 0x0f, 0x01, 0xfa]);
        let mut ctx = context(code.wrapping_add(16));
        assert!(!run(&mut ctx, ON));
    }

    #[test]
    fn rdpru_reads() {
        let code = rwx(PAGE);
        put(code.wrapping_add(16), &[0x0f, 0x01, 0xfd]);
        let mut ctx = context(code.wrapping_add(16));
        ctx.Rcx = 7;
        ctx.Rax = 1;
        ctx.Rdx = 1;
        ctx.EFlags = 0x202;
        assert!(run(&mut ctx, ON));
        assert_eq!((ctx.Rax, ctx.Rdx, ctx.EFlags, ctx.Rip), (0, 0, 0x202, code as u64 + 19));
        ctx.Rip = code as u64 + 16;
        ctx.Rcx = 1;
        assert!(run(&mut ctx, ON));
        assert_ne!(ctx.Rax | ctx.Rdx, 0);
    }

    #[test]
    fn sse4a() {
        let code = rwx(PAGE);
        let mut ctx = context(code.wrapping_add(16));
        // extrq xmm0, 8, 16
        put(code.wrapping_add(16), &[0x66, 0x0f, 0x78, 0xc0, 0x08, 0x10]);
        xmm(&mut ctx, 0).Low = 0x1234_5678_9abc_def0;
        assert!(run(&mut ctx, ON));
        assert_eq!((xmm(&mut ctx, 0).Low, ctx.Rip), (0xbc, code as u64 + 22));
        // insertq xmm2, xmm1, 8, 16
        ctx.Rip = code as u64 + 16;
        put(code.wrapping_add(16), &[0xf2, 0x0f, 0x78, 0xd1, 0x08, 0x10]);
        xmm(&mut ctx, 1).Low = 0x55;
        xmm(&mut ctx, 2).Low = u64::MAX;
        assert!(run(&mut ctx, ON));
        assert_eq!(xmm(&mut ctx, 2).Low, 0xffff_ffff_ff55_ffff);
        // extrq xmm3, xmm4 with the length 4 and index 4 in xmm4
        ctx.Rip = code as u64 + 16;
        put(code.wrapping_add(16), &[0x66, 0x0f, 0x79, 0xdc]);
        xmm(&mut ctx, 3).Low = 0xabcd;
        xmm(&mut ctx, 4).Low = 0x0404;
        assert!(run(&mut ctx, ON));
        assert_eq!((xmm(&mut ctx, 3).Low, ctx.Rip), (0xc, code as u64 + 20));
    }

    #[test]
    fn movnt_stores() {
        let (code, data) = (rwx(PAGE), rwx(PAGE));
        put(data, &[0xaa; 16]);
        let mut ctx = context(code.wrapping_add(16));
        ctx.Rax = data as u64;
        xmm(&mut ctx, 1).Low = 0x1122_3344_5566_7788;
        // movntsd [rax], xmm1
        put(code.wrapping_add(16), &[0xf2, 0x0f, 0x2b, 0x08]);
        assert!(run(&mut ctx, ON));
        assert_eq!(get(data, 8), 0x1122_3344_5566_7788u64.to_le_bytes());
        // movntss [rax+8], xmm1
        ctx.Rip = code as u64 + 16;
        put(code.wrapping_add(16), &[0xf3, 0x0f, 0x2b, 0x48, 0x08]);
        assert!(run(&mut ctx, ON));
        assert_eq!(get(data.wrapping_add(8), 8), [0x88, 0x77, 0x66, 0x55, 0xaa, 0xaa, 0xaa, 0xaa]);
        assert_eq!(ctx.Rip, code as u64 + 21);
    }

    #[test]
    fn jump_round_trip() {
        let code = rwx(PAGE);
        let area = near(code as usize).unwrap();
        put(code, &jump(code as usize, area + 64).unwrap());
        put((area + 64) as *mut u8, &[0xb8, 0x2a, 0, 0, 0, 0xc3]);
        // SAFETY: a jmp to `mov eax, 42; ret`
        let f: extern "C" fn() -> u32 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(), 42);
        assert_eq!(jump(0, 1 << 40), None);
        assert_eq!(jump(0x1000, 0x1000), Some([0xe9, 0xfb, 0xff, 0xff, 0xff]));
    }

    // f(start, end): mov rax, rcx, then a form A loop up to rdx, ret with rax
    const FORM_A: [u8; 16] =
        [0x48, 0x89, 0xc8, 0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x72, 0xf4, 0xc3];
    // f(start, end): a form B loop on rcx up to rdx, ret with rcx
    const FORM_B: [u8; 19] = [
        0x48, 0x89, 0xc8, 0x48, 0x83, 0xc1, 0x40, 0x0f, 0x01, 0xfc, 0x48, 0x39, 0xd1, 0x72, 0xf1, 0x48, 0x89,
        0xc8, 0xc3,
    ];

    fn window(code: *const u8, at: usize) -> [u8; WINDOW] {
        get(code.wrapping_add(at - BEFORE), WINDOW).try_into().unwrap()
    }

    #[test]
    fn loops_found() {
        let code = rwx(PAGE);
        put(code.wrapping_add(32), &FORM_A);
        assert_eq!(find_loop(&window(code, 35)), Some(Loop::A { b: 2 }));
        put(code.wrapping_add(128), &FORM_B);
        assert_eq!(find_loop(&window(code, 135)), Some(Loop::B { a: 1, b: 2 }));
        put(code.wrapping_add(200), &[0x0f, 0x01, 0xfc, 0xc3]);
        assert_eq!(find_loop(&window(code, 200)), None);
    }

    #[test]
    fn loops_run_in_handler() {
        let (code, data) = (rwx(PAGE), rwx(PAGE));
        let (start, end) = (data as u64 + 64, data as u64 + 320);

        put(code.wrapping_add(32), &FORM_A);
        put(data, &[0xaa; 512]);
        let mut ctx = context(code.wrapping_add(35));
        (ctx.Rax, ctx.Rdx) = (start, end);
        assert!(run(&mut ctx, OFF));
        assert_eq!((ctx.Rip, ctx.Rax), (code as u64 + 47, end));
        assert_eq!(ctx.EFlags & CMP_FLAGS, cmp_flags(end, end));
        assert_eq!(get(data, 512), [[0xaa; 64].as_slice(), &[0; 256], &[0xaa; 192]].concat());

        put(code.wrapping_add(128), &FORM_B);
        put(data, &[0xaa; 512]);
        let mut ctx = context(code.wrapping_add(135));
        (ctx.Rax, ctx.Rcx, ctx.Rdx) = (start, start + 64, end);
        assert!(run(&mut ctx, OFF));
        assert_eq!((ctx.Rip, ctx.Rax, ctx.Rcx), (code as u64 + 143, end - 64, end));
        assert_eq!(get(data, 512), [[0xaa; 64].as_slice(), &[0; 256], &[0xaa; 192]].concat());
    }

    #[test]
    fn loops_patched_to_stubs() {
        let (code, data) = (rwx(PAGE), rwx(PAGE));
        let (start, end) = (data as usize + 64, data as usize + 320);
        for (at, bytes) in [(32, FORM_A.as_slice()), (128, FORM_B.as_slice())] {
            put(code.wrapping_add(at), bytes);
            put(data, &[0xaa; 512]);
            let c = if at == 32 { 35 } else { 135 };
            let mut ctx = context(code.wrapping_add(c));
            assert!(run(&mut ctx, ON));
            assert_eq!(ctx.Rip, code as u64 + c as u64);
            assert_eq!(get(code.wrapping_add(c), 1), [0xe9]);
            assert!(run(&mut ctx, ON));
            // SAFETY: the loop now runs through its stub; both take (start, end) in rcx and rdx
            let f: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(code.wrapping_add(at)) };
            assert_eq!(f(start, end), end);
            assert_eq!(get(data, 512), [[0xaa; 64].as_slice(), &[0; 256], &[0xaa; 192]].concat());
        }
    }
}
