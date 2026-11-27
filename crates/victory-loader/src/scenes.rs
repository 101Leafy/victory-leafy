//! `VICTORY_SKIP_TO_IIS=1` (off without it): the title's own `-skipToIIS`
//! switch, on without a command line.
//!
//! What the title does (game_scarlett_master.exe 1.0.2.1, image offsets):
//! +0x2216220 runs at start with argc and argv, and sets the byte at
//! +0x74fc238 when argv[1] starts with `-skipToIIS`. Its one reader, the
//! scene playback update at +0x2348ed0, plays each scene marked for it
//! (bit 0 of its byte +0x7c) at rate 100 (the float at +0x6159c58), so the
//! opening scenes run through in a moment. VIctory hands the title no
//! command line, so here the argc test becomes the store: the flag is set
//! every launch and the function returns.

use crate::frame_rate::Patched;
use crate::resolution::Site;

/// +0x2216220 from +0x221625c: `cmp ecx, 2 ; jl end ; mov rcx, [rdx+8]`.
pub const ARGV_TEST: Site = Site {
    at: 0x221625c,
    what: "-skipToIIS switch set",
    bytes: &[0x83, 0xf9, 0x02, 0x7c, 0x22, 0x48, 0x8b, 0x4a, 0x08],
};

/// The switch byte and the function's epilogue.
pub const SKIP_FLAG: usize = 0x74fc238;
pub const EPILOGUE: usize = 0x2216283;

pub fn asked() -> bool {
    std::env::var_os("VICTORY_SKIP_TO_IIS").is_some_and(|v| v == "1")
}

/// `mov byte [flag], 1 ; jmp epilogue`, nine bytes like the test it replaces.
pub fn set_flag_code() -> Vec<u8> {
    let at = ARGV_TEST.at;
    let mut c = vec![0xc6, 0x05];
    c.extend(((SKIP_FLAG as i64 - (at + 7) as i64) as i32).to_le_bytes());
    c.push(1);
    c.extend([0xeb, (EPILOGUE - (at + 9)) as u8]);
    c
}

/// Sets the switch if asked; a line in the log and on standard output.
///
/// # Safety
/// The image of `size` bytes at `base` stays mapped and no thread runs its
/// code yet.
#[cfg(windows)]
pub unsafe fn apply_asked(base: usize, size: usize) {
    if !asked() {
        return;
    }
    // SAFETY: the caller's promise.
    let done = unsafe { crate::resolution::write(base, size, &ARGV_TEST, &set_flag_code()) };
    let offset = format!("{:#x}", ARGV_TEST.at);
    match done {
        Patched::Done => tracing::info!(site = %offset, "scenes: -skipToIIS set"),
        Patched::Differs => tracing::warn!(site = %offset, "scenes: -skipToIIS not set, the title's bytes there are not the ones expected (another build?)"),
        Patched::Failed(why) => tracing::warn!(site = %offset, why, "scenes: -skipToIIS not set"),
    }
    println!("scenes: -skipToIIS at title+{offset}: {done:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_store_is_the_titles_own() {
        assert_eq!(set_flag_code(), [0xc6, 0x05, 0xd5, 0x5f, 0x2e, 0x05, 0x01, 0xeb, 0x1e]);
        assert_eq!(set_flag_code().len(), ARGV_TEST.bytes.len());
        // The jl in the bytes replaced goes to the same epilogue.
        assert_eq!(ARGV_TEST.at + 5 + ARGV_TEST.bytes[4] as usize, EPILOGUE);
    }
}
