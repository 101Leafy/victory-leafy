//! The F1 settings menu. F1 shows and hides it while our window has the
//! focus; Up/Down pick a setting, Left/Right change it, Enter runs the line
//! picked (save, reset, close), Esc closes. While it is shown the title sees
//! no keyboard (`crate::SETTINGS_MENU_OPEN`, read by the input module). The
//! pad in slot 0 works in it too: the D-pad picks and changes, A selects, B
//! closes. With `VICTORY_MENU_PAD=1` both stick buttons also open it, and
//! `VICTORY_MENU_PAD=0` keeps the pad out of it.
//!
//! Most settings are read once at start, so a change to one of those is
//! marked to restart. Saved, the changes go to the `[settings]` table of the
//! config file (`VICTORY_CONFIG`, else `victory.toml`), which the next start
//! reads (victory-core `Config::apply_settings`).
//!
//! In every build (the stats panel is in diagnostics builds only). Drawn as
//! that panel is: white text on a black box, copied onto the swap chain's
//! buffer, in the middle of the picture.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_DOWN, VK_ESCAPE, VK_F1, VK_LEFT, VK_RETURN, VK_RIGHT, VK_UP,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::font::{colours, glyph};
use super::{Host, HostList, transition};

const GLYPH: (u32, u32) = (5, 7);
const CELL: (u32, u32) = (6, 9);
const COLUMNS: u32 = 60;
/// Lines of the box: a title, a help line, a blank, the settings shown, a
/// blank and a note.
const ROWS: usize = 24;
const LINES: u32 = ROWS as u32 + 5;
const RIM: u32 = 4;
const UPLOADS: usize = 4;

/// A setting: the variable, its name in the menu, the values it takes
/// (`""`: not set, the build's own default), and whether a change applies at
/// once. With no variable, the name of the group the next ones are in.
struct Item {
    var: &'static str,
    label: &'static str,
    values: &'static [(&'static str, &'static str)],
    live: bool,
}

const ON_UNSET: &[(&str, &str)] = &[("1", "ON"), ("", "OFF")];
const UNSET_ON: &[(&str, &str)] = &[("", "OFF"), ("1", "ON")];
const STAGES: &[(&str, &str)] = &[("2", "2"), ("1", "1"), ("0", "0 OFF")];

/// Said under the menu while a row that makes frames up or renders fewer
/// pixels is picked (render scale now, frame generation if it comes).
const NOT_ENDORSED: &str = "NOT ENDORSED BY 101LEAFY, NEVER USED BY THE DEVELOPER";

const fn group(label: &'static str) -> Item {
    Item { var: "", label, values: &[], live: false }
}

/// The first value of each is what scripts\play.ps1 starts with.
#[rustfmt::skip]
const ITEMS: &[Item] = &[
    group("DISPLAY"),
    Item { var: "VICTORY_OUTPUT", label: "OUTPUT SIZE", values: &[("1280x720", "1280X720"), ("1664x936", "1664X936"), ("1920x1080", "1920X1080"), ("2560x1440", "2560X1440"), ("3840x2160", "3840X2160")], live: false },
    Item { var: "VICTORY_BORDERLESS", label: "BORDERLESS WINDOW", values: UNSET_ON, live: false },
    group("GRAPHICS"),
    Item { var: "VICTORY_AMBIENT", label: "LIGHT ESTIMATE", values: &[("0.8", "0.8"), ("1", "1"), ("0.5", "0.5"), ("", "OFF")], live: false },
    Item { var: "VICTORY_AMBIENT_AIR", label: "FOG CAP", values: &[("", "NONE"), ("1.5", "1.5"), ("1", "1"), ("2", "2")], live: false },
    Item { var: "VICTORY_BVH_BUILDER", label: "RAY TRACING (SLOW)", values: UNSET_ON, live: false },
    // Under 1.0 the title's own upscaler fills the output (victory-loader resolution.rs).
    Item { var: "VICTORY_RENDER_SCALE", label: "RENDER SCALE (UPSCALING)", values: &[("", "1.0 NATIVE"), ("0.83", "0.83"), ("0.67", "0.67"), ("0.5", "0.5")], live: false },
    group("PERFORMANCE"),
    Item { var: "VICTORY_TEXTURE_NO_WAIT", label: "TEXTURE NO WAIT", values: ON_UNSET, live: false },
    Item { var: "VICTORY_RELAY", label: "RELAY (SPEED, TRIANGLES)", values: STAGES, live: false },
    Item { var: "VICTORY_CARD_STEPS", label: "CARD STEPS", values: STAGES, live: false },
    Item { var: "VICTORY_INDIRECT_BATCH", label: "INDIRECT BATCH", values: STAGES, live: false },
    Item { var: "VICTORY_WC_NO_BACK", label: "SKIP BIG COPY-BACKS", values: UNSET_ON, live: false },
    Item { var: "VICTORY_PORT_CONSTANTS", label: "CONSTANTS HEAP IN PLACE", values: UNSET_ON, live: false },
    group("AUDIO"),
    Item { var: "VICTORY_DSP_CONVOLVE", label: "REVERB, WORLD SPEAKERS", values: &[("", "FULL"), ("mono", "MONO"), ("0", "OFF")], live: false },
    group("INPUT"),
    // The mouse on the stand-in pad's right stick (controls.rs): the same
    // `VICTORY_MOUSE_SENS` a player would set before the start, changeable
    // at once so the feel can be matched to the title's own slider.
    Item { var: "VICTORY_MOUSE_SENS", label: "MOUSE LOOK SENSITIVITY", values: &[("", "THE TITLE'S OWN"), ("0.5", "0.5 (SLOW)"), ("1", "1"), ("1.5", "1.5"), ("2", "2"), ("3", "3 (FAST)")], live: true },
    Item { var: "VICTORY_MENU_PAD", label: "PAD IN THIS MENU", values: &[("", "YES"), ("1", "L3+R3 OPENS"), ("0", "NO")], live: false },
    group("DEBUG"),
    Item { var: "VICTORY_FPS_LINE", label: "FPS AT THE TOP", values: &[("1", "ON"), ("0", "OFF")], live: true },
];
/// The lines after the settings.
const ACTIONS: [&str; 3] = ["SAVE TO VICTORY.TOML", "RESET ALL TO PLAY DEFAULTS", "CLOSE"];

fn get(var: &str) -> String {
    std::env::var(var).unwrap_or_default()
}

fn live(item: &Item) -> bool {
    item.live
}

fn is_group(row: usize) -> bool {
    ITEMS.get(row).is_some_and(|i| i.var.is_empty())
}

/// The config file the settings are saved in, found as victory-core finds it.
fn config_path() -> std::path::PathBuf {
    std::env::var_os("VICTORY_CONFIG").filter(|v| !v.is_empty()).map_or_else(|| "victory.toml".into(), Into::into)
}

/// `text` (a config file) with its `[settings]` table holding `changes`
/// (variable, value) over what it had, or with no such table when `changes`
/// is `None`. The other lines are kept as they were.
fn with_settings(text: &str, changes: Option<&[(&str, &str)]>) -> String {
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut kept = Vec::new();
    let mut table: Vec<(String, String)> = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let head = line.trim();
        if head.starts_with('[') {
            inside = head == "[settings]";
            if inside {
                continue;
            }
        }
        if !inside {
            kept.push(line);
        } else if let Some((k, v)) = head.split_once('=').filter(|_| !head.starts_with('#')) {
            table.push((k.trim().to_owned(), v.trim().to_owned()));
        }
    }
    while kept.last().is_some_and(|l| l.trim().is_empty()) {
        kept.pop();
    }
    let mut out: String = kept.iter().flat_map(|l| [*l, newline]).collect();
    let Some(changes) = changes else { return out };
    for (var, value) in changes {
        let key = var.trim_start_matches("VICTORY_").to_ascii_lowercase();
        table.retain(|(k, _)| *k != key);
        table.push((key, format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))));
    }
    if !out.is_empty() {
        out.push_str(newline);
    }
    out.push_str("[settings]");
    out.push_str(newline);
    out.push_str("# Saved by the F1 menu. \"\" unsets one; a variable set before the start wins.");
    out.push_str(newline);
    for (k, v) in &table {
        out.push_str(&format!("{k} = {v}{newline}"));
    }
    out
}

/// Whether the frame-rate line is shown at the top while the menu is closed
/// (`VICTORY_FPS_LINE=0`: not; the menu changes it).
fn fps_line_on() -> &'static AtomicBool {
    static ON: OnceLock<AtomicBool> = OnceLock::new();
    ON.get_or_init(|| AtomicBool::new(std::env::var_os("VICTORY_FPS_LINE").is_none_or(|v| v != "0")))
}

/// That line: the frames shown a second.
fn fps_line() -> Option<String> {
    if !fps_line_on().load(Ordering::Relaxed) {
        return None;
    }
    // The stats panel has that corner (diagnostics builds, F3).
    #[cfg(feature = "diagnostics")]
    if super::panel::shown().load(Ordering::Relaxed) {
        return None;
    }
    let fps = f64::from(super::METER_FPS_TENTHS.load(Ordering::Relaxed)) / 10.0;
    if fps <= 0.0 {
        return None;
    }
    Some(format!("FPS: {fps:.1}"))
}

/// The index of `value` among the item's values (an unknown value: shown as
/// itself, at the end).
fn index_of(item: &Item, value: &str) -> Option<usize> {
    item.values.iter().position(|(v, _)| *v == value)
}

#[derive(Clone, Copy)]
enum MenuKey {
    Keyboard(u16),
    Pad(u16),
}

static MENU_KEYS: Mutex<Vec<MenuKey>> = Mutex::new(Vec::new());

// A toggle stays pending until the frame has applied it or focus loss cancels it.
// The input sampler can run many times before a single presented frame.
static PAD_TOGGLES: AtomicUsize = AtomicUsize::new(0);
// Invalidate pad events even if a frame has already taken them out of the queue.
static PAD_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// Whether F1 is open, and whether a controller toggle is queued or being processed.
pub fn settings_pad_capture() -> (bool, bool) {
    let pending = PAD_TOGGLES.load(Ordering::Acquire) != 0;
    (crate::SETTINGS_MENU_OPEN.load(Ordering::Relaxed), pending)
}

/// Queue a controller action from GameInput's pad sampler. When the pad loses
/// focus or disconnects, cancel its unconsumed actions. Keyboard events retain
/// their original order in both cases.
pub fn settings_pad_key(key: Option<u16>) {
    let mut keys = MENU_KEYS.lock().unwrap_or_else(|e| e.into_inner());
    match key {
        Some(vk) => {
            if vk == VK_F1 {
                PAD_TOGGLES.fetch_add(1, Ordering::Relaxed);
            }
            keys.push(MenuKey::Pad(vk));
        }
        None => {
            PAD_GENERATION.fetch_add(1, Ordering::Release);
            keys.retain(|key| {
                if matches!(key, MenuKey::Pad(VK_F1)) {
                    PAD_TOGGLES.fetch_sub(1, Ordering::Release);
                }
                matches!(key, MenuKey::Keyboard(_))
            });
        }
    }
}

fn focused() -> bool {
    let mut process = 0;
    // SAFETY: plain queries; room for the answer.
    unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
    process == std::process::id()
}

/// Keys pressed, gathered on a thread of their own every 10 ms: a frame
/// lasts 50 ms and more, and a short press would fall between two frames.
fn watch_keys() {
    static WATCH: std::sync::Once = std::sync::Once::new();
    WATCH.call_once(|| {
        let _ = std::thread::Builder::new().name("settings menu keys".into()).spawn(|| {
            const WATCHED: [u16; 7] = [VK_F1, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_RETURN, VK_ESCAPE];
            let mut was = [false; WATCHED.len()];
            loop {
                std::thread::sleep(Duration::from_millis(10));
                let ours = focused();
                for (i, &vk) in WATCHED.iter().enumerate() {
                    // SAFETY: a plain query.
                    let down = ours && unsafe { GetAsyncKeyState(i32::from(vk)) } < 0;
                    if down && !was[i] {
                        MENU_KEYS.lock().unwrap_or_else(|e| e.into_inner()).push(MenuKey::Keyboard(vk));
                    }
                    was[i] = down;
                }
            }
        });
    });
}

/// The box's size in pixels.
const fn size(scale: u32) -> (u32, u32) {
    ((COLUMNS * CELL.0 + 2 * RIM) * scale, (LINES * CELL.1 + 2 * RIM) * scale)
}

fn lit(lines: &[String], x: u32, y: u32) -> bool {
    let (Some(x), Some(y)) = (x.checked_sub(RIM), y.checked_sub(RIM)) else {
        return false;
    };
    if x % CELL.0 >= GLYPH.0 || y % CELL.1 >= GLYPH.1 || x / CELL.0 >= COLUMNS {
        return false;
    }
    let Some(&c) = lines.get((y / CELL.1) as usize).and_then(|l| l.as_bytes().get((x / CELL.0) as usize)) else {
        return false;
    };
    (glyph(c.to_ascii_uppercase())[(y % CELL.1) as usize] >> (GLYPH.0 - 1 - x % CELL.0)) & 1 != 0
}

fn painted(lines: &[String], scale: u32, colours: [&[u8]; 2], pitch: usize) -> Vec<u8> {
    let (width, height) = size(scale);
    let bytes = colours[0].len();
    let mut out = vec![0; pitch * height as usize];
    for y in 0..height / scale {
        let top = pitch * (y * scale) as usize;
        for x in 0..width / scale {
            let colour = colours[usize::from(lit(lines, x, y))];
            for i in 0..scale {
                let at = top + (x * scale + i) as usize * bytes;
                out[at..at + bytes].copy_from_slice(colour);
            }
        }
        for i in 1..scale as usize {
            out.copy_within(top..top + pitch, top + i * pitch);
        }
    }
    out
}

struct Buffers {
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    uploads: Vec<(ID3D12Resource, *mut u8)>,
    card: ID3D12Resource,
    turn: usize,
    fresh: bool,
    /// The card's buffer was copied into before (it is in the copy-source state;
    /// at first it is in the common state, promoted by the first copy).
    used: bool,
}

// SAFETY: free-threaded resources, and memory only written under the presenter's lock.
unsafe impl Send for Buffers {}

impl Buffers {
    fn new(host: &Host, format: DXGI_FORMAT, scale: u32) -> Option<Buffers> {
        let (width, height) = size(scale);
        let pitch = (width * colours(format)?[0].len() as u32).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let uploads = (0..UPLOADS).map(|_| host.upload_buffer(bytes)).collect::<Option<Vec<_>>>()?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut card)
                .ok()?;
        }
        Some(Buffers { format, scale, pitch, uploads, card: card?, turn: 0, fresh: false, used: false })
    }

    fn write(&mut self, lines: &[String]) {
        let Some(colours) = colours(self.format) else { return };
        let picture = painted(lines, self.scale, colours, self.pitch as usize);
        self.turn = (self.turn + 1) % UPLOADS;
        // SAFETY: a mapped upload buffer of the picture's size, which the card read frames ago.
        unsafe { std::ptr::copy_nonoverlapping(picture.as_ptr(), self.uploads[self.turn].1, picture.len()) };
        self.fresh = true;
    }
}

/// The menu of a presenter.
pub(super) struct Menu {
    /// The values chosen, by item (as at start until changed).
    values: Vec<String>,
    /// The values at start.
    started: Vec<String>,
    /// The line picked: an item, or an action after them.
    at: usize,
    note: String,
    changed: bool,
    /// What the buffer holds: the frame-rate line, or (none) the menu's lines.
    painted: Option<String>,
    buffers: Option<Buffers>,
    failed: bool,
}

impl Menu {
    pub(super) fn new() -> Menu {
        // Start the key thread now: F1 must be seen while the menu is closed.
        watch_keys();
        // `VICTORY_MENU_OPEN=1`: shown from the start (to look at it in a run nobody presses keys in).
        if std::env::var_os("VICTORY_MENU_OPEN").is_some_and(|v| v == "1") {
            crate::SETTINGS_MENU_OPEN.store(true, Ordering::Relaxed);
        }
        let values: Vec<String> = ITEMS.iter().map(|i| get(i.var)).collect();
        Menu { started: values.clone(), values, at: 1, note: String::new(), changed: true, painted: None, buffers: None, failed: false }
    }

    fn open() -> bool {
        crate::SETTINGS_MENU_OPEN.load(Ordering::Relaxed)
    }

    fn lines(&self) -> Vec<String> {
        let total = ITEMS.len() + ACTIONS.len();
        let first = self.at.saturating_sub(ROWS / 2).min(total.saturating_sub(ROWS));
        let pad_help = "D-PAD/ARROWS PICK OR CHANGE   A/CROSS OR ENTER SELECTS";
        let (title, help) = match crate::settings_pad_mode() {
            crate::SettingsPadMode::StickShortcut => ("SETTINGS       F1/ESC OR L3+R3 OR B/CIRCLE CLOSES", pad_help),
            crate::SettingsPadMode::MenuOnly => ("SETTINGS       F1/ESC OR B/CIRCLE CLOSES", pad_help),
            crate::SettingsPadMode::Off => ("SETTINGS                       F1 OR ESC CLOSES", "UP/DOWN PICK  LEFT/RIGHT CHANGE  ENTER DOES IT"),
        };
        let mut out = vec![title.to_owned(), help.to_owned(), String::new()];
        for row in first..(first + ROWS).min(total) {
            let cursor = if row == self.at { '>' } else { ' ' };
            let line = if let Some(item) = ITEMS.get(row).filter(|i| i.var.is_empty()) {
                format!("  -- {} --", item.label)
            } else if let Some(item) = ITEMS.get(row) {
                let value = &self.values[row];
                let shown = index_of(item, value).map_or(value.to_ascii_uppercase(), |i| item.values[i].1.to_owned());
                let mark = match (*value != self.started[row], live(item)) {
                    (false, _) => "",
                    (true, true) => " *",
                    (true, false) => " *RESTART",
                };
                format!("{cursor} {:<27} < {shown} >{mark}", item.label)
            } else {
                format!("{cursor} {}", ACTIONS[row - ITEMS.len()])
            };
            out.push(line.chars().take(COLUMNS as usize).collect());
        }
        while out.len() < 3 + ROWS {
            out.push(String::new());
        }
        out.push(String::new());
        out.push(match ITEMS.get(self.at) {
            _ if !self.note.is_empty() => self.note.clone(),
            Some(item) if item.var == "VICTORY_RENDER_SCALE" => NOT_ENDORSED.to_owned(),
            _ => "* CHANGED     *RESTART: AT THE NEXT START".to_owned(),
        });
        out
    }

    /// Changes item `i` by `step` (+1 / -1) through its values.
    fn change(&mut self, i: usize, step: isize) {
        let item = &ITEMS[i];
        let n = item.values.len() as isize;
        let now = index_of(item, &self.values[i]).map_or(-1, |x| x as isize);
        let next = if now < 0 { if step > 0 { 0 } else { n - 1 } } else { (now + step).rem_euclid(n) };
        self.values[i] = item.values[next as usize].0.to_owned();
        if live(item) {
            self.apply_live(i);
        }
        self.note.clear();
    }

    /// A setting the build reads at each use: changed in this run's environment at once.
    fn apply_live(&self, i: usize) {
        let (var, value) = (ITEMS[i].var, &self.values[i]);
        // SAFETY: the variables changed here are read by this build with
        // std::env::var at each use; Windows' environment block is locked by
        // the process heap's routines, and std holds its own lock.
        unsafe {
            if value.is_empty() {
                std::env::remove_var(var);
            } else {
                std::env::set_var(var, value);
            }
        }
        match var {
            "VICTORY_FPS_LINE" => fps_line_on().store(value != "0", Ordering::Relaxed),
            // The look's deflection per pixel, from the next poll on.
            "VICTORY_MOUSE_SENS" => crate::set_kbm_mouse_sens(value.parse().unwrap_or(1.0)),
            _ => {}
        }
    }

    /// Writes the changed values into the config file's `[settings]`.
    fn save(&mut self) -> bool {
        let changes: Vec<(&str, &str)> = ITEMS
            .iter()
            .zip(self.values.iter().zip(&self.started))
            .filter(|(item, (now, then))| !item.var.is_empty() && now != then)
            .map(|(item, (now, _))| (item.var, now.as_str()))
            .collect();
        let path = config_path();
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        match std::fs::write(&path, with_settings(&text, Some(&changes))) {
            Ok(()) => {
                tracing::info!(path = %path.display(), changed = changes.len(), "settings menu: saved");
                self.started = self.values.clone();
                self.note = "SAVED TO VICTORY.TOML [SETTINGS]".to_owned();
                true
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "settings menu: not saved");
                self.note = "COULD NOT WRITE VICTORY.TOML".to_owned();
                false
            }
        }
    }

    /// Takes `[settings]` out of the config file: the launcher's own settings at the next start.
    fn reset(&mut self) {
        let path = config_path();
        if let Ok(text) = std::fs::read_to_string(&path) {
            let rest = with_settings(&text, None);
            if rest != text && std::fs::write(&path, rest).is_err() {
                self.note = "COULD NOT WRITE VICTORY.TOML".to_owned();
                return;
            }
        }
        self.note = "RESET. PLAY DEFAULTS AT THE NEXT START".to_owned();
    }

    /// A frame was presented: the keys pressed since the last one.
    pub(super) fn frame(&mut self) {
        let (pressed, generation) = {
            let mut keys = MENU_KEYS.lock().unwrap_or_else(|e| e.into_inner());
            (std::mem::take(&mut *keys), PAD_GENERATION.load(Ordering::Relaxed))
        };
        if pressed.is_empty() {
            return;
        }
        let total = ITEMS.len() + ACTIONS.len();
        let toggles = pressed.iter().filter(|key| matches!(key, MenuKey::Pad(VK_F1))).count();
        for event in pressed {
            let vk = match event {
                MenuKey::Keyboard(vk) => vk,
                MenuKey::Pad(vk) if focused() && generation == PAD_GENERATION.load(Ordering::Acquire) => vk,
                MenuKey::Pad(_) => continue,
            };
            let open = Self::open();
            if vk == VK_F1 || (open && vk == VK_ESCAPE) {
                crate::SETTINGS_MENU_OPEN.store(!open && vk == VK_F1, Ordering::Relaxed);
                self.note.clear();
                self.changed = true;
                continue;
            }
            if !open {
                continue;
            }
            match vk {
                VK_UP | VK_DOWN => {
                    let step = if vk == VK_UP { total - 1 } else { 1 };
                    self.at = (self.at + step) % total;
                    while is_group(self.at) {
                        self.at = (self.at + step) % total;
                    }
                }
                VK_LEFT if self.at < ITEMS.len() => self.change(self.at, -1),
                VK_RIGHT if self.at < ITEMS.len() => self.change(self.at, 1),
                VK_RETURN if self.at < ITEMS.len() => self.change(self.at, 1),
                VK_RETURN => match self.at - ITEMS.len() {
                    0 => {
                        self.save();
                    }
                    1 => self.reset(),
                    _ => crate::SETTINGS_MENU_OPEN.store(false, Ordering::Relaxed),
                },
                _ => {}
            }
            self.changed = true;
        }
        if toggles != 0 {
            PAD_TOGGLES.fetch_sub(toggles, Ordering::Release);
        }
    }

    /// Records the copy of the box onto `back` (in the copy-destination state).
    pub(super) fn draw(&mut self, host: &Host, list: &HostList, back: &ID3D12Resource, (width, height): (u32, u32), format: DXGI_FORMAT) {
        if self.failed {
            return;
        }
        // Closed: the frame-rate line in the top left corner, if it is wanted and there is a rate.
        let open = Self::open();
        let line = if open { None } else { fps_line() };
        if !open && line.is_none() {
            return;
        }
        // Our own copies are not part of a recorded frame (the recorder is in diagnostics builds).
        #[cfg(feature = "diagnostics")]
        let _ours = crate::replay::record::unrecorded();
        let scale = (height / 540).max(1);
        if self.buffers.as_ref().is_none_or(|b| (b.format, b.scale) != (format, scale)) {
            std::mem::forget(self.buffers.take());
            self.buffers = Buffers::new(host, format, scale);
            self.failed = self.buffers.is_none();
            self.changed = true;
        }
        if self.painted != line {
            self.painted = line.clone();
            self.changed = true;
        }
        let lines = if self.changed { Some(line.as_ref().map_or_else(|| self.lines(), |l| vec![l.clone()])) } else { None };
        let Some(buffers) = &mut self.buffers else { return };
        if let Some(lines) = lines {
            self.changed = false;
            buffers.write(&lines);
        }
        // The whole box in the middle, or of it the first line's part in the corner.
        let (w, h) = match &line {
            Some(l) => ((l.len() as u32 * CELL.0 + 2 * RIM) * scale, (CELL.1 + 2 * RIM) * scale),
            None => size(scale),
        };
        let (w, h) = (w.min(width), h.min(height));
        let (x, y) = if line.is_some() { (0, 0) } else { ((width - w) / 2, (height - h) / 2) };
        let list = &list.list;
        // SAFETY: an open list and live resources; the box lies within both.
        unsafe {
            if std::mem::take(&mut buffers.fresh) {
                if std::mem::replace(&mut buffers.used, true) {
                    list.ResourceBarrier(&[transition(&buffers.card, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)]);
                }
                list.CopyResource(&buffers.card, &buffers.uploads[buffers.turn].0);
                list.ResourceBarrier(&[transition(&buffers.card, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            }
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffers.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: size(scale).0, Height: size(scale).1, Depth: 1, RowPitch: buffers.pitch },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX { left: 0, top: 0, front: 0, right: w, bottom: h, back: 1 };
            list.CopyTextureRegion(&to, x, y, 0, &from, Some(&part));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_line_fits_and_every_character_has_a_picture() {
        let mut menu = Menu { values: ITEMS.iter().map(|_| String::new()).collect(), started: Vec::new(), at: 0, note: String::new(), changed: false, painted: None, buffers: None, failed: false };
        menu.started = menu.values.clone();
        for at in 0..ITEMS.len() + ACTIONS.len() {
            menu.at = at;
            for line in menu.lines() {
                assert!(line.len() <= COLUMNS as usize, "{line}");
                for c in line.bytes() {
                    assert_ne!(glyph(c.to_ascii_uppercase()), glyph(b'#'), "{line}: {}", c as char);
                }
            }
        }
    }

    #[test]
    fn values_cycle_both_ways() {
        let mut menu = Menu { values: ITEMS.iter().map(|i| i.values.first().map_or("", |v| v.0).to_owned()).collect(), started: Vec::new(), at: 1, note: String::new(), changed: false, painted: None, buffers: None, failed: false };
        menu.started = menu.values.clone();
        let n = ITEMS[1].values.len();
        for _ in 0..n {
            menu.change(1, 1);
        }
        assert_eq!(menu.values[1], ITEMS[1].values[0].0);
        menu.change(1, -1);
        assert_eq!(menu.values[1], ITEMS[1].values[n - 1].0);
    }

    #[test]
    fn groups_are_stepped_over() {
        assert!(is_group(0) && !is_group(1));
        let total = ITEMS.len() + ACTIONS.len();
        for at in 0..total {
            assert!(!(is_group(at) && is_group((at + 1) % total)), "two groups in a row at {at}");
        }
    }

    #[test]
    fn saves_go_into_the_settings_table_and_keep_the_rest() {
        let text = "title = \"a\"\r\n\r\n[settings]\r\nambient = \"1\"\r\ncard_steps = \"2\"\r\n\r\n[graphics]\r\nx = 1\r\n";
        let saved = with_settings(text, Some(&[("VICTORY_AMBIENT", "0.5"), ("VICTORY_MOUSE_SENS", "")]));
        assert_eq!(
            saved,
            "title = \"a\"\r\n\r\n[graphics]\r\nx = 1\r\n\r\n[settings]\r\n# Saved by the F1 menu. \"\" unsets one; a variable set before the start wins.\r\ncard_steps = \"2\"\r\nambient = \"0.5\"\r\nmouse_sens = \"\"\r\n"
        );
        assert_eq!(with_settings(&saved, None), "title = \"a\"\r\n\r\n[graphics]\r\nx = 1\r\n");
        assert_eq!(with_settings("", Some(&[("VICTORY_RELAY", "1")])), "[settings]\n# Saved by the F1 menu. \"\" unsets one; a variable set before the start wins.\nrelay = \"1\"\n");
    }
}
