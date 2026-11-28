//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! The title creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    container: Vec<u8>,
    /// Its part in the stand-in for the indirect light.
    #[cfg(windows)]
    pub(crate) ambient: Option<crate::ambient::Role>,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// For the memory lines: containers kept and their bytes, translations
/// kept and the bytes of their host code. None of them is ever let go.
static KEPT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the shader table and the pipeline library keep in memory:
/// containers and their bytes, translations and their bytes, the bytes of
/// the library's file (kept whole while the library lives) and what its
/// last save wrote.
#[cfg(windows)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn kept() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        KEPT[0].load(Relaxed),
        KEPT[1].load(Relaxed),
        KEPT[2].load(Relaxed),
        KEPT[3].load(Relaxed),
        library::BLOB.load(Relaxed),
        library::SAVED.load(Relaxed),
        library::TRIMMED_BYTES.load(Relaxed),
    ]
}

/// The shared container for `bytes`. A new one is built outside the lock
/// (the copy and the compute role scan), so other creations do not wait on
/// it; of two built at once, the first in keeps its place.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let key = *full.as_bytes();
    if let Some(known) = SHADERS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|s| s.get(&key)) {
        return known.clone();
    }
    let built = Arc::new(Shader {
        stage,
        hash: full.to_hex()[..16].to_owned(),
        container: bytes.to_vec(),
        #[cfg(windows)]
        ambient: (stage == "compute").then(|| crate::ambient::role(bytes)).flatten(),
        #[cfg(windows)]
        translated: OnceLock::new(),
    });
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(key)
        .or_insert_with(|| {
            KEPT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            KEPT[1].fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            built
        })
        .clone()
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    /// A compute program's threads in a group.
    pub(crate) threads: Option<[u32; 3]>,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    /// Whether the shader's code can store to buffer memory ([`stores`]);
    /// true when its code is not known.
    pub(crate) stores: bool,
    /// Whether it can store through raw addresses ([`stores_by_address`]).
    pub(crate) stores_by_address: bool,
    /// Whether it can store anywhere a later command may read, textures
    /// included ([`writes`]).
    pub(crate) writes: bool,
    /// Whether it stores through a view it picks from the heap by an index,
    /// once asked (`crate::picked`).
    pub(crate) picks: std::sync::atomic::AtomicU8,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use; a shader that does not
    /// translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        let asked = std::time::Instant::now();
        let mut own = false;
        let done = self
            .translated
            .get_or_init(|| {
                own = true;
                // Time spent here is part of a slow pipeline's time
                // (`slow-pipelines.txt`); this tells it from the driver's.
                let began = std::time::Instant::now();
                let done = translate(&self.container);
                let ms = began.elapsed().as_millis() as u64;
                // `VICTORY_SHADER_SAVE=<dir>`: each shader as the title gave it (`.cdxil`) and as
                // translated (`.dxil`), by the hash crash reports name it with
                if let Some(dir) = std::env::var_os("VICTORY_SHADER_SAVE") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    let stem = format!("{}.{}", self.hash, self.stage);
                    let _ = std::fs::write(dir.join(format!("{stem}.cdxil")), &self.container);
                    if let Ok(t) = &done {
                        let _ = std::fs::write(dir.join(format!("{stem}.dxil")), &t.host.dxil);
                    }
                }
                if let Ok(t) = &done {
                    KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    KEPT[3].fetch_add(t.host.dxil.len() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        ms,
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        ms,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref();
        // Another thread translating the same shader holds this one up.
        let spent = asked.elapsed();
        if own {
            spent::add(|p| {
                p.translate += spent;
                p.translated += 1;
            });
        } else if spent > std::time::Duration::from_millis(1) {
            spent::add(|p| p.shared += spent);
        }
        done
    }
}

#[cfg(windows)]
impl Shader {
    /// The console programs of the container; `None` if they are not known.
    pub(crate) fn programs(&self) -> Option<Vec<victory_shader::NativeProgram>> {
        use victory_shader::{Container, FourCC};
        let c = Container::parse(&self.container).ok()?;
        let part = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
        victory_shader::native_programs(part.data).ok()
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    // The translation depends on it (a view index past a capped heap).
    crate::descriptor::host::probe_view_limit();
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let programs = match c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) {
        Some(part) => victory_shader::native_programs(part.data).map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    let host = crate::shader_cache::host_shader(container)?;
    let stores = programs.is_empty() || programs.iter().any(stores);
    let threads = programs.iter().find_map(|p| match &p.io {
        victory_shader::native::StageIo::Cs(cs) if cs.threads.iter().all(|&n| n > 0) => Some(cs.threads),
        _ => None,
    });
    let stores_by_address = programs.is_empty() || programs.iter().any(stores_by_address);
    let writes = stores || programs.iter().any(writes);
    let used = programs
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
            )
        })
        .map(|p| p.used)
        .unwrap_or_default();
    Ok(Translated {
        host,
        threads,
        used,
        stores,
        stores_by_address,
        writes,
        picks: Default::default(),
    })
}

/// Whether `shaders` are a pipeline of the ray-tracing structure builder,
/// guessed by its programs (a spilled user-data table) while its image is
/// not known ([`builder`]).
#[cfg(windows)]
fn of_builder(shaders: &[Arc<Shader>]) -> bool {
    use victory_shader::{Container, FourCC};
    let [shader] = shaders else { return false };
    if shader.stage != "compute" {
        return false;
    }
    let Ok(c) = Container::parse(&shader.container) else {
        return false;
    };
    c.part(FourCC::DXIL)
        .and_then(|part| victory_shader::native_programs(part.data).ok())
        .is_some_and(|programs| programs.iter().any(victory_shader::spill::is_spilled))
}

/// `VICTORY_BVH_QUARANTINE=1`: what the structure builder's stages store
/// stays in the host's copies and is not copied back into title memory. A
/// check for a first run of the whole builder: a stage that stores where
/// it should not can then not reach the title's own objects (run 98), and
/// the programs that walk the structures read the host's copies anyway.
/// shortcut: not a way to run: the title also reads results back (sizes,
/// counters) and gets none, and a copy's pages are overwritten from title
/// memory when the title writes them.
#[cfg(windows)]
pub(crate) fn builder_quarantined(this: This) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    is_builder(this) && *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_QUARANTINE").is_some_and(|v| v == "1"))
}

/// Whether one of the title's pipeline `this`'s shaders has the 16-digit
/// hash `hash` (as the log names shaders).
#[cfg(windows)]
pub(crate) fn has_shader(this: This, hash: &str) -> bool {
    !this.is_null() && pipeline(this).shaders.iter().any(|s| s.hash == hash)
}

/// Whether the title's pipeline `this` is one of the structure builder's.
#[cfg(windows)]
pub(crate) fn is_builder(this: This) -> bool {
    !this.is_null() && pipeline(this).builder
}

/// A stage of the builder was dropped after all: none runs for the rest of
/// this frame and the next (what it fed is not there).
#[cfg(windows)]
pub(crate) fn builder_rests(why: &'static str) {
    builder::rests(why);
}

/// `VICTORY_BVH_BUILDER=1`.
#[cfg(windows)]
pub(crate) fn builder_allowed() -> bool {
    builder::allowed()
}

/// At a dispatch of a builder pipeline: whether the builder runs (see
/// [`builder`]).
#[cfg(windows)]
pub(crate) fn builder_runs() -> bool {
    builder::runs()
}

/// Why a dispatch of the title's pipeline `this` is held back, if it is: a
/// stage of the structure builder in a frame where the builder does not
/// run, or a program that walks its structures before it has run whole
/// ([`builder`]).
#[cfg(windows)]
pub(crate) fn held_back(this: This) -> Option<&'static str> {
    if this.is_null() {
        return None;
    }
    let p = pipeline(this);
    if p.builder {
        return (!builder::runs()).then_some("a stage of the ray-tracing structure builder, which is off");
    }
    (p.walks_bvh && !builder::built()).then_some("walks the ray-tracing structures before the builder has run whole")
}

/// A title DLL was mapped (the structure builder's pipelines are known by
/// its image from then on).
#[cfg(windows)]
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    builder::note_title_dll(file, base, size);
}

/// Whether a shader container's programs walk the ray-tracing structures.
#[cfg(windows)]
fn walks_bvh(container: &[u8]) -> bool {
    use victory_shader::{Container, FourCC};
    let Ok(c) = Container::parse(container) else {
        return false;
    };
    c.part(FourCC::DXIL)
        .and_then(|part| victory_shader::native_programs(part.data).ok())
        .is_some_and(|programs| programs.iter().any(|p| code_walks_bvh(&p.code)))
}

/// Whether console code has an `image_bvh*_intersect_ray`.
#[cfg(windows)]
fn code_walks_bvh(code: &[u8]) -> bool {
    victory_shader::gfx10::decode(code)
        .is_ok_and(|code| code.iter().any(|i| i.name().is_some_and(|n| n.starts_with("image_bvh"))))
}

/// Whether a console program can store to buffer memory: it has a buffer,
/// typed-buffer, scalar-memory, flat or global store or atomic, or code
/// that is not all known (an instruction of those kinds outside our table,
/// an indirect jump, code that does not decode). Stores to textures
/// (image instructions) and to the program's own scratch memory do not
/// count: neither reaches a buffer's host copy. After upstream #413,
/// which leaves scalar-memory stores out and counts image stores.
#[cfg(windows)]
fn stores(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        let name = i.name();
        match i.encoding {
            Encoding::Mubuf
            | Encoding::Mtbuf
            | Encoding::Smem
            | Encoding::Flat
            | Encoding::Global => name.is_none_or(|n| {
                (n.contains("store") || n.contains("atomic")) && !n.contains("scratch")
            }),
            Encoding::Sop1 => name.is_none_or(|n| matches!(n, "s_setpc_b64" | "s_swappc_b64")),
            _ => false,
        }
    })
}

/// Whether a console program can store to memory through raw addresses
/// (`crate::memory`): a flat or global store or atomic, or code that does
/// not decode. The programs that only walk ray-tracing structures do not,
/// whatever they store through views.
#[cfg(windows)]
fn stores_by_address(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        matches!(i.encoding, Encoding::Flat | Encoding::Global)
            && i.name()
                .is_none_or(|n| (n.contains("store") || n.contains("atomic")) && !n.contains("scratch"))
    })
}

/// Whether a console program can store to a texture (an image store or
/// atomic), or code that does not decode: with [`stores`], whether a later
/// command may read what it wrote.
#[cfg(windows)]
fn writes(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        i.encoding == Encoding::Mimg && i.name().is_none_or(|n| n.contains("store") || n.contains("atomic"))
    })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// One of the ray-tracing structure builder's ([`builder`]).
    #[cfg(windows)]
    builder: bool,
    /// Walks the builder's structures (waits until it has run whole).
    #[cfg(windows)]
    walks_bvh: bool,
    /// Made from loose parts: its host pipelines go by the draw's targets.
    #[cfg(windows)]
    loose: Option<Box<loose::Loose>>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    let p = pipeline(this);
    if let Some(loose) = &p.loose {
        return loose.chosen();
    }
    make::ready(&p.host)
}

/// Whether one of the containers of the title pipeline `this` is one `f`
/// picks (false for none bound). Known at the call, before any host
/// pipeline is made.
#[cfg(windows)]
pub(crate) fn shaders_any(this: This, f: impl Fn(&Shader) -> bool) -> bool {
    !this.is_null() && pipeline(this).shaders.iter().any(|s| f(s))
}

/// At a draw, before [`host_pipeline`]: a pipeline made from loose parts
/// picks its host pipeline for the targets bound (`State::set_targets`'s
/// handles). Nothing for other pipelines.
#[cfg(windows)]
pub(crate) fn draw_targets(this: This, handles: &[u64], depth: u64) {
    if this.is_null() {
        return;
    }
    if let Some(loose) = &pipeline(this).loose {
        loose.choose(handles, depth);
    }
}

/// As [`host_pipeline`], for a draw into render target and depth views of
/// the formats `bound`: if a bound view has another format than the title's
/// pipeline declares for its slot (`host::formats_differ`), the host
/// pipeline of the same shaders and state made for the bound formats
/// (`make::for_formats`). A draw whose views are as declared pays one
/// comparison of the nine words.
#[cfg(windows)]
pub(crate) fn host_pipeline_for<'a>(this: This, bound: &host::TargetFormats) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    let p = &pipeline(this).host;
    if p.formats != *bound && host::formats_differ(&p.formats, bound) {
        return make::for_formats(p, bound);
    }
    make::ready(p)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| match &p.loose {
                Some(loose) => loose.missing(),
                None => make::missing(&p.host),
            })
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    crate::capture::op(0, || format!("DROPPED {why}"));
    crate::host::panel::log_event(format!("DROP {why}"));
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    #[cfg(windows)] code: &[*const u8],
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    hand_out_loose(
        shaders,
        #[cfg(windows)]
        graphics,
        #[cfg(windows)]
        code,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

fn hand_out_loose(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    #[cfg(windows)] code: &[*const u8],
    #[cfg(windows)] loose: Option<Box<loose::Loose>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics));
    // A loose pipeline's own is never made: its host pipelines go by targets.
    #[cfg(windows)]
    if loose.is_none() {
        known::note(host.key(), &shaders, host.graphics.as_deref());
        make::early(&host);
    }
    #[cfg(windows)]
    let compute = matches!(&shaders[..], [s] if s.stage == "compute");
    #[cfg(windows)]
    let builder = compute
        && if builder::mapped() {
            builder::created_by_builder(code)
        } else {
            of_builder(&shaders)
        };
    #[cfg(windows)]
    if builder {
        builder::note(&host);
    }
    #[cfg(windows)]
    let walks_bvh = !builder && builder::allowed() && shaders.iter().any(|s| walks_bvh(&s.container));
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        #[cfg(windows)]
        walks_bvh,
        #[cfg(windows)]
        loose,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// A graphics pipeline from loose parts (device method 0x1c8). Its host
/// pipelines wait for the draws' targets ([`draw_targets`]).
#[cfg(windows)]
pub(crate) fn create_loose_graphics(g: &victory_shader::LooseGraphics, iid: *const Iid, out: *mut This) -> HRESULT {
    let mut shaders = vec![shader("vertex", &g.vs)];
    shaders.extend(g.ps.as_deref().map(|ps| shader("pixel", ps)));
    let Some(loose) = loose::Loose::new(shaders.clone(), g) else {
        return E_INVALIDARG;
    };
    hand_out_loose(shaders, None, &[], Some(Box::new(loose)), iid, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        #[cfg(windows)]
        &[],
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        #[cfg(windows)]
        &[desc.stage.code],
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while desc.size.saturating_sub(at) >= 8 {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if at.checked_add(size).is_none_or(|end| end > desc.size) {
            tracing::warn!(
                kind,
                at,
                size,
                stream = desc.size,
                "truncated pipeline subobject; the rest is ignored"
            );
            break;
        }
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8 (may be unaligned).
            let code = unsafe { desc.stream.add(at + 8).cast::<Bytecode>().read_unaligned() };
            shaders.extend(copy(stage, &code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        #[cfg(windows)]
        &[],
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
mod builder;
#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
pub mod known;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod loose;
#[cfg(windows)]
mod make;
#[cfg(windows)]
mod samplers;
#[cfg(windows)]
mod spent;

/// Lets the pipeline library's mapped file go from memory when no pipeline
/// has loaded for a while (`library::trim`); called by the host's sweep.
#[cfg(windows)]
pub(crate) fn trim_library() {
    library::trim();
}

/// The pipeline library's part of the report when the host device is
/// removed (`library::removal_note`).
#[cfg(windows)]
pub(crate) fn library_removal_note() -> String {
    library::removal_note()
}

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
    known::flush();
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    /// The same bytes give the same container, kept and counted once.
    #[test]
    fn same_bytes_share_one_container() {
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 31 + 7) as u8).collect();
        let first = shader("pixel", &bytes);
        let kept = KEPT[0].load(std::sync::atomic::Ordering::Relaxed);
        let again = shader("pixel", &bytes);
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(first.hash, blake3::hash(&bytes).to_hex()[..16]);
        assert!(KEPT[0].load(std::sync::atomic::Ordering::Relaxed) >= kept);
    }

    /// A compute pipeline the structure builder creates (its bytecode in
    /// the builder's image) is known as the builder's, and its dispatches
    /// are held back while its host pipeline is not made: here never, as
    /// its bytes are no shader container. Pipelines of other code are not
    /// held back.
    #[cfg(windows)]
    #[test]
    fn builder_dispatches_wait_for_every_builder_pipeline() {
        let _one = builder::test_lock();
        let code = [0xB0u8; 64];
        let create = |code: &[u8]| {
            let desc = ComputeHead {
                root: std::ptr::null(),
                stage: Bytecode {
                    code: code.as_ptr(),
                    size: code.len(),
                },
            };
            let mut p: This = std::ptr::null();
            assert_eq!(
                create_compute(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
                S_OK
            );
            p
        };
        builder::note_title_dll("xboxbvhbuilder.DLL", code.as_ptr() as usize, code.len());
        let p = create(&code);
        let other_code = [0xC0u8; 64];
        let other = create(&other_code);
        builder::clear_image_for_tests();
        assert!(pipeline(p).builder, "its bytecode is in the builder's image");
        assert!(!pipeline(other).builder);
        // Its host pipeline is started at once and refused.
        let refused = (0..500).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            make::state(&pipeline(p).host).is_some()
        });
        assert!(refused, "the builder's pipeline was started");
        assert_eq!(make::state(&pipeline(p).host), Some(false));
        assert!(held_back(p).is_some(), "one builder pipeline missing: the builder waits");
        assert_eq!(held_back(other), None);
        for q in [p, other] {
            slot::<extern "system" fn(This) -> u32>(q, 0x10)(q);
        }
    }

    #[cfg(windows)]
    #[test]
    fn programs_that_intersect_rays_walk_the_structures() {
        let code = |words: &[u32]| words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        const S_ENDPGM: u32 = 0xBF81_0000;
        // Result v[20:23], descriptor s[12:15] (as in victory-shader's BVH tests).
        let rest = 20 << 8 | 3 << 16;
        let narrow = code(&[0xF198_9F01, rest | 2, S_ENDPGM]);
        // NSA: pointer v2, v13, then the ray from v3 to v12.
        let wide = code(&[0xF19C_9F07, rest | 2, 0x0504_030D, 0x0908_0706, 0x000C_0B0A, S_ENDPGM]);
        assert!(code_walks_bvh(&narrow), "image_bvh_intersect_ray");
        assert!(code_walks_bvh(&wide), "image_bvh64_intersect_ray");
        assert!(!code_walks_bvh(&code(&[S_ENDPGM])));
    }

    /// What walks the structures waits until the builder has run whole: in a
    /// title with the builder, not before; in a title without it, at once.
    #[cfg(windows)]
    #[test]
    fn walks_wait_for_the_builder_to_run_whole() {
        let _one = builder::test_lock();
        builder::note_title_dll("XboxBvhBuilder.dll", 0x1000, 0x1000);
        assert!(!builder::built(), "a title with the builder: not built yet");
        builder::clear_image_for_tests();
        assert!(builder::built(), "a title without the builder: walks run");
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// The stream layout the title builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }
}

#[cfg(test)]
mod safety_tests {
    use super::*;
    use crate::object::{S_OK, slot};

    struct PipelineGuard(This);
    impl PipelineGuard {
        fn release_now(&mut self) -> u32 {
            let ptr = std::mem::replace(&mut self.0, std::ptr::null());
            if ptr.is_null() {
                0
            } else {
                slot::<extern "system" fn(This) -> u32>(ptr, 0x10)(ptr)
            }
        }
    }
    impl Drop for PipelineGuard {
        fn drop(&mut self) {
            let _ = self.release_now();
        }
    }

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn unaligned_stream_yields_its_shaders() {
        let vs = [1u8, 2, 3, 4];
        let mut backing = [0u64; 5];
        // SAFETY: `backing` is 40 contiguous bytes; slicing at +1 leaves 39
        // bytes and guarantees non-8-byte alignment.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), size_of_val(&backing))
        };
        let stream = &mut bytes[1..25];
        assert_ne!(stream.as_ptr() as usize % 8, 0);
        stream[0..4].copy_from_slice(&1u32.to_le_bytes());
        stream[8..16].copy_from_slice(&(vs.as_ptr() as usize).to_le_bytes());
        stream[16..24].copy_from_slice(&vs.len().to_le_bytes());

        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut raw = std::ptr::null();
        assert_eq!(
            create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut raw),
            S_OK
        );
        let mut p = PipelineGuard(raw);
        assert_eq!(shaders(p.0), [("vertex", vs.to_vec())]);
        assert_eq!(p.release_now(), 0);
    }

    #[test]
    fn truncated_shader_subobject_does_not_read_past_stream() {
        let mut stream = [0u8; 16];
        stream[0..4].copy_from_slice(&1u32.to_le_bytes());
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut raw = std::ptr::null();
        assert_eq!(
            create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut raw),
            S_OK
        );
        let mut p = PipelineGuard(raw);
        assert!(shaders(p.0).is_empty());
        assert_eq!(p.release_now(), 0);
    }
}
