//! Fingerprints the shader translator, so that cached translations
//! (`src/shader_cache.rs`) last until the translator itself changes and not
//! only until the next build of anything. Comments and blank lines are left
//! out, and of `Cargo.lock` only the packages the translator is built from:
//! an edit that changes neither used to translate every shader again. The
//! form entries are kept in has its own number, `shader_cache::FORMAT`.
//! `VICTORY_TRANSLATOR_DEPS` is the same without the source: the cache keys
//! on it and `emit::VERSION`, and on the whole only when a check finds a
//! source change gave other output.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Rust source without its comments, trailing spaces and blank lines.
/// Strings, raw strings and char literals are kept as they are.
fn code(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let mut depth = 0;
            while i < b.len() {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            out.push(b' ');
        } else if c == b'r' && matches!(b.get(i + 1), Some(b'"' | b'#')) && !ident(b, i) {
            let start = i;
            i += 1;
            let mut hashes = 0;
            while b.get(i) == Some(&b'#') {
                hashes += 1;
                i += 1;
            }
            if b.get(i) != Some(&b'"') {
                out.extend_from_slice(&b[start..i]);
                continue;
            }
            i += 1;
            while i < b.len() {
                if b[i] == b'"'
                    && b[i + 1..]
                        .iter()
                        .take(hashes)
                        .filter(|&&h| h == b'#')
                        .count()
                        == hashes
                {
                    i += 1 + hashes;
                    break;
                }
                i += 1;
            }
            out.extend_from_slice(&b[start..i.min(b.len())]);
        } else if c == b'"' {
            let start = i;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
            out.extend_from_slice(&b[start..i.min(b.len())]);
        } else if c == b'\'' {
            // A char literal ('x', '\n', '\u{..}'), or a lifetime ('a).
            let start = i;
            let end = if b.get(i + 1) == Some(&b'\\') {
                b.get(i + 3..)
                    .and_then(|r| r.iter().position(|&x| x == b'\''))
                    .map(|p| i + 4 + p)
            } else {
                let len = text[i + 1..].chars().next().map_or(1, char::len_utf8);
                (b.get(i + 1 + len) == Some(&b'\'')).then_some(i + 2 + len)
            };
            i = end.unwrap_or(i + 1);
            out.extend_from_slice(&b[start..i]);
        } else {
            out.push(c);
            i += 1;
        }
    }
    let text = String::from_utf8_lossy(&out);
    let mut kept = String::with_capacity(text.len());
    for line in text.lines().map(str::trim_end).filter(|l| !l.is_empty()) {
        kept.push_str(line);
        kept.push('\n');
    }
    kept
}

/// `code` without its test modules (`#[cfg(test)] mod name { .. }`). A block whose
/// end is not found is kept.
fn untested(code: &str) -> String {
    let lines: Vec<&str> = code.lines().collect();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while i < lines.len() {
        let open = test_cfg(lines[i].trim())
            && lines
                .get(i + 1)
                .is_some_and(|l| l.trim_start().starts_with("mod ") && l.ends_with('{'));
        if open {
            if let Some(end) = block_end(&lines[i + 1..]) {
                i += 2 + end;
                continue;
            }
        }
        out.push_str(lines[i]);
        out.push('\n');
        i += 1;
    }
    out
}

/// The line of `lines` that closes the brace the first one opens.
fn block_end(lines: &[&str]) -> Option<usize> {
    let mut depth = 0i32;
    for (n, line) in lines.iter().enumerate() {
        let b = line.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (i + 1 == b.len()).then_some(n);
                    }
                }
                // Strings and char literals may hold braces. One that runs
                // over lines or is raw is not followed: give up.
                b'"' => {
                    if i > 0 && matches!(b[i - 1], b'r' | b'#') {
                        return None;
                    }
                    i += 1;
                    while i < b.len() && b[i] != b'"' {
                        i += if b[i] == b'\\' { 2 } else { 1 };
                    }
                    if i >= b.len() {
                        return None;
                    }
                }
                b'\'' if b.get(i + 2) == Some(&b'\'') => i += 2,
                b'\'' if b.get(i + 1) == Some(&b'\\') => {
                    i += 2 + b.get(i + 3..)?.iter().position(|&x| x == b'\'')? + 1;
                }
                _ => {}
            }
            i += 1;
        }
    }
    None
}

/// Whether a `#[cfg(..)]` line keeps what follows out of every build but
/// tests (`test`, `all(test, windows)`; not `any(test, ..)` or `not(test)`).
fn test_cfg(line: &str) -> bool {
    line.starts_with("#[cfg(")
        && line.contains("test")
        && !line.contains("any(")
        && !line.contains("not(")
}

/// Whether the `mod` line of the file at `path` (in its folder's `mod.rs`
/// or `lib.rs`) is behind a `cfg` that needs `test`.
fn tests_only(path: &Path) -> bool {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = path.parent().unwrap_or(Path::new("."));
    ["mod.rs", "lib.rs"].iter().any(|parent| {
        let text = std::fs::read_to_string(dir.join(parent)).unwrap_or_default();
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        lines
            .windows(2)
            .any(|w| w[1] == format!("mod {stem};") && test_cfg(w[0]))
    })
}

/// Whether the `r` at `i` ends an identifier (`bar"`) rather than
/// starting a raw string (`r"`, `br"`).
fn ident(b: &[u8], i: usize) -> bool {
    let word = |j: usize| b[j].is_ascii_alphanumeric() || b[j] == b'_';
    match i {
        0 => false,
        1 => b[0] != b'b' && word(0),
        _ if b[i - 1] == b'b' => word(i - 2),
        _ => word(i - 1),
    }
}

/// The `Cargo.lock` entries of `root` and everything it depends on.
fn locked(lock: &str, root: &str) -> String {
    let mut packages: BTreeMap<String, String> = BTreeMap::new();
    let mut deps: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for block in lock.split("[[package]]").skip(1) {
        let block = block.split("\n[").next().unwrap_or(block).trim();
        let Some(name) = block.lines().find_map(|l| l.strip_prefix("name = ")) else {
            continue;
        };
        let name = name.trim_matches('"').to_string();
        let mut list = Vec::new();
        if let Some(rest) = block.split("dependencies = [").nth(1) {
            for dep in rest.split(']').next().unwrap_or("").split(',') {
                let dep = dep.trim().trim_matches('"');
                if let Some(dep) = dep.split(' ').next().filter(|d| !d.is_empty()) {
                    list.push(dep.to_string());
                }
            }
        }
        // A name locked twice (two versions) keeps both entries.
        packages.entry(name.clone()).or_default().push_str(block);
        deps.entry(name).or_default().extend(list);
    }
    let mut seen = BTreeSet::new();
    let mut todo = vec![root.to_string()];
    while let Some(name) = todo.pop() {
        if seen.insert(name.clone()) {
            todo.extend(deps.get(&name).into_iter().flatten().cloned());
        }
    }
    seen.iter()
        .filter_map(|n| packages.get(n))
        .map(|b| format!("{b}\n"))
        .collect()
}

fn main() {
    let here = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let crates = here.join("..");
    let root = crates.join("..");
    let mut all = vec![
        crates.join("victory-shader/Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("rust-toolchain.toml"),
    ];
    let source = crates.join("victory-shader/src");
    println!("cargo:rerun-if-changed={}", source.display());
    files(&source, &mut all);
    all.sort();
    let mut hash = blake3::Hasher::new();
    let mut deps = blake3::Hasher::new();
    for path in &all {
        println!("cargo:rerun-if-changed={}", path.display());
        // Names and line endings as they are in every checkout.
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let text =
            String::from_utf8_lossy(&std::fs::read(path).unwrap_or_default()).replace('\r', "");
        let bytes = if name == "Cargo.lock" {
            locked(&text, "victory-shader")
        } else if name.ends_with("_tests.rs") && tests_only(path) {
            continue;
        } else if path.extension().is_some_and(|e| e == "rs") {
            untested(&code(&text))
        } else {
            text
        };
        let parts = [name.as_bytes(), &(bytes.len() as u64).to_le_bytes(), bytes.as_bytes()];
        for part in parts {
            hash.update(part);
            if !path.starts_with(&source) {
                deps.update(part);
            }
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rustc-env=VICTORY_TRANSLATOR={}",
        hash.finalize().to_hex()
    );
    // What the translator is built from beside its source (its manifest,
    // locked dependencies, compiler): with `emit::VERSION` the key a source
    // change alone does not move.
    println!(
        "cargo:rustc-env=VICTORY_TRANSLATOR_DEPS={}",
        deps.finalize().to_hex()
    );
}
