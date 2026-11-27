//! Byte patches to the title's code, from files named in `VICTORY_MODS`
//! (separated like `PATH`). Each line is `rva old new` in hex, `#` starts
//! a comment. A file is applied whole or not at all: one `old` that does
//! not match the image refuses every line of it, so a patch made for
//! another build never half applies.

use std::path::Path;

use victory_pe::ImageInfo;

use crate::prepare::PreparedImage;

/// Applies every file in `VICTORY_MODS` to the prepared copy. Prints one
/// line per file (its name only).
pub fn apply_asked(prepared: &mut PreparedImage) {
    let Some(list) = std::env::var_os("VICTORY_MODS") else {
        return;
    };
    let mut hasher = blake3::Hasher::new();
    let mut applied = 0;
    for path in std::env::split_paths(&list) {
        let name = file_name(&path);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                println!("mod            {name} not read: {}", e.kind());
                continue;
            }
        };
        match apply(&mut prepared.bytes, &prepared.info, &text) {
            Ok(n) => {
                println!("mod            {name}: {n} patches");
                hasher.update(text.as_bytes());
                applied += 1;
            }
            Err(why) => println!("mod            {name} refused: {why}"),
        }
    }
    if applied > 0 {
        prepared.mods = Some(hasher.finalize());
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Checks every line against `bytes`, then writes them all. Returns how
/// many lines were applied.
pub fn apply(bytes: &mut [u8], info: &ImageInfo, text: &str) -> Result<usize, String> {
    let mut patches = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let n = i + 1;
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [rva, old, new] = fields[..] else {
            return Err(format!("line {n}: want rva old new"));
        };
        let rva = u32::from_str_radix(rva.trim_start_matches("0x"), 16)
            .map_err(|_| format!("line {n}: bad rva"))?;
        let (Some(old), Some(new)) = (unhex(old), unhex(new)) else {
            return Err(format!("line {n}: bad hex"));
        };
        if old.is_empty() || old.len() != new.len() {
            return Err(format!("line {n}: old and new differ in length"));
        }
        let at = info
            .rva_to_file_offset(rva)
            .ok_or_else(|| format!("line {n}: rva {rva:x} is not in the file"))?;
        let last = rva
            .checked_add(old.len() as u32 - 1)
            .and_then(|r| info.rva_to_file_offset(r));
        if last != at.checked_add(old.len() as u32 - 1) {
            return Err(format!("line {n}: rva {rva:x} runs past its section"));
        }
        let at = at as usize;
        if bytes.get(at..at + old.len()) != Some(&old[..]) {
            return Err(format!("line {n}: bytes at {rva:x} differ, another build?"));
        }
        patches.push((at, new));
    }
    if patches.is_empty() {
        return Err("no patches".into());
    }
    for (at, new) in &patches {
        bytes[*at..*at + new.len()].copy_from_slice(new);
    }
    Ok(patches.len())
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare::prepare;

    fn own() -> (Vec<u8>, ImageInfo, u32) {
        let bytes = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let info = ImageInfo::parse(&bytes).unwrap();
        let rva = info.entry_rva;
        (bytes, info, rva)
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn applies_every_line() {
        let (mut bytes, info, rva) = own();
        let at = info.rva_to_file_offset(rva).unwrap() as usize;
        let old = bytes[at..at + 4].to_vec();
        let text = format!("# test\n{rva:x} {} 90909090  # nops\n\n", hex(&old));
        assert_eq!(apply(&mut bytes, &info, &text), Ok(1));
        assert_eq!(bytes[at..at + 4], [0x90; 4]);
    }

    #[test]
    fn one_wrong_line_refuses_the_file() {
        let (mut bytes, info, rva) = own();
        let at = info.rva_to_file_offset(rva).unwrap() as usize;
        let before = bytes.clone();
        let good = hex(&bytes[at..at + 2]);
        let mut wrong = bytes[at + 8..at + 10].to_vec();
        wrong[0] ^= 0xff;
        let text = format!("{rva:x} {good} cccc\n{:x} {} cccc\n", rva + 8, hex(&wrong));
        assert!(apply(&mut bytes, &info, &text).is_err());
        assert_eq!(bytes, before);
    }

    #[test]
    fn bad_lines_are_refused() {
        let (mut bytes, info, _) = own();
        for text in ["", "# only a comment", "10 90", "10 9 90", "10 90 9090", "zz 90 90"] {
            assert!(apply(&mut bytes, &info, text).is_err(), "{text:?}");
        }
        assert!(apply(&mut bytes, &info, &format!("{:x} 90 90", u32::MAX - 1)).is_err());
    }

    #[test]
    fn modded_copy_gets_its_own_file() {
        let (bytes, _, _) = own();
        let plain = prepare(&bytes).unwrap();
        let mut modded = plain.clone();
        modded.mods = Some(blake3::hash(b"mod"));
        assert_ne!(plain.file_name(), modded.file_name());
        assert!(modded.file_name().starts_with(&plain.file_name()[..30]));
    }
}
