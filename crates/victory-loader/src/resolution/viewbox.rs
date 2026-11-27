//! Fit dynamic interface viewboxes to the output using the layout engine's
//! uniform stretch mode. With the stock mode, subtitle glyphs and controller
//! prompts keep their 4K pixel size at any smaller output (1280x720 and the
//! test sizes under it). On whenever a size under 3840x2160 is asked;
//! `VICTORY_UI_VIEWBOX=0` keeps the stock mode.

use super::Site;
use std::ffi::OsStr;

pub(super) fn requested(size: (u32, u32), setting: Option<&OsStr>) -> bool {
    size != (3840, 2160) && setting != Some(OsStr::new("0"))
}

// The layout reader resolves these case-sensitive names through the stretch
// enumeration. Each entry has a Jenkins name hash, value and flags. Check the
// neighboring modes as well before changing the value for dynamic viewboxes.
const MODES: [&[u8]; 4] = [
    b"STRETCH_TYPE_FILL",
    b"STRETCH_TYPE_UNIFORM",
    b"STRETCH_TYPE_UNIFORM_TO_FILL",
    b"STRETCH_TYPE_DYNAMIC",
];
const UNIFORM: u8 = 2;
const DYNAMIC: u8 = 4;
const STRIDE: usize = 12;

const fn name_hash(name: &[u8]) -> u32 {
    let mut hash = 0u32;
    let mut i = 0;
    while i < name.len() {
        hash = hash.wrapping_add(name[i] as u32);
        hash = hash.wrapping_add(hash << 10);
        hash ^= hash >> 6;
        i += 1;
    }
    hash = hash.wrapping_add(hash << 3);
    hash ^= hash >> 11;
    hash.wrapping_add(hash << 15)
}

const fn table(dynamic_mode: u8) -> [u8; STRIDE * MODES.len()] {
    let mut table = [0; STRIDE * MODES.len()];
    let mut i = 0;
    while i < MODES.len() {
        let offset = i * STRIDE;
        let hash = name_hash(MODES[i]).to_le_bytes();
        let mut j = 0;
        while j < hash.len() {
            table[offset + j] = hash[j];
            j += 1;
        }
        table[offset + 4] = if i == 3 { dynamic_mode } else { i as u8 + 1 };
        i += 1;
    }
    table
}

pub(super) const SITE: Site = Site {
    at: 0x630ff1c,
    what: "fit UI viewboxes to the output",
    bytes: &table(DYNAMIC),
};
pub(super) const FITTED: [u8; STRIDE * MODES.len()] = table(UNIFORM);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::{Patched, matches};

    #[test]
    fn every_size_under_4k_fits_unless_turned_off() {
        for size in [(426, 240), (854, 480), (1280, 720), (1920, 1080), (2560, 1440)] {
            assert!(requested(size, None));
            assert!(requested(size, Some(OsStr::new("1"))));
            assert!(!requested(size, Some(OsStr::new("0"))));
        }
        assert!(!requested((3840, 2160), None));
    }

    #[test]
    fn only_the_dynamic_value_changes_and_other_layouts_are_rejected() {
        let mut data = table(DYNAMIC);
        let index = 3 * STRIDE + 4;
        assert_eq!(data[index], DYNAMIC);
        data[index] = UNIFORM;
        assert_eq!(data, FITTED);

        let local = Site { at: 0, ..SITE };
        let mut original = table(DYNAMIC);
        let check = |data: &[u8]| {
            // SAFETY: the slice remains live throughout the checked read.
            unsafe { matches(data.as_ptr() as usize, data.len(), &local) }
        };
        assert_eq!(check(&original), Patched::Done);
        original[STRIDE + 4] += 1;
        assert_eq!(check(&original), Patched::Differs);
        assert!(matches!(check(&original[..STRIDE]), Patched::Failed(_)));
    }
}
