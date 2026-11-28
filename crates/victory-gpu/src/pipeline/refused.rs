//! Pipelines the host does not make, for a report from a machine we do not
//! have: the device's shader caps once at start, each refused pipeline once
//! (its key, stage, HRESULT, the removed reason or debug layer error, and the
//! features its shaders ask for that the device lacks), and at exit one line
//! with the count by reason.
//!
//! `VICTORY_SHADER_CAPS=1` (off by default): a refusal whose shaders ask for
//! nothing the device lacks is made once more, straight from the driver
//! without the pipeline library, and the result logged.

use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::Shader;

#[derive(Debug, Default)]
pub(crate) struct Caps {
    model: u32,
    wave_min: u32,
    wave_max: u32,
    wave_ops: bool,
    native16: bool,
    int64: bool,
    doubles: bool,
    rovs: bool,
    barycentrics: bool,
    typed_loads: bool,
    binding_tier: i32,
    mesh_tier: i32,
    atomic64_typed: bool,
    atomic64_heap: bool,
    mesh_derivatives: bool,
}

static CAPS: OnceLock<Caps> = OnceLock::new();

/// Formats the translations load from typed UAVs; the ones the device cannot
/// load are named in the caps line.
const TYPED_LOADS: &[(DXGI_FORMAT, &str)] = &[
    (DXGI_FORMAT_R32G32B32A32_FLOAT, "rgba32f"),
    (DXGI_FORMAT_R32G32B32A32_UINT, "rgba32u"),
    (DXGI_FORMAT_R16G16B16A16_FLOAT, "rgba16f"),
    (DXGI_FORMAT_R16G16B16A16_UNORM, "rgba16"),
    (DXGI_FORMAT_R16G16B16A16_UINT, "rgba16u"),
    (DXGI_FORMAT_R32G32_FLOAT, "rg32f"),
    (DXGI_FORMAT_R32G32_UINT, "rg32u"),
    (DXGI_FORMAT_R10G10B10A2_UNORM, "rgb10a2"),
    (DXGI_FORMAT_R11G11B10_FLOAT, "rg11b10f"),
    (DXGI_FORMAT_R8G8B8A8_UNORM, "rgba8"),
    (DXGI_FORMAT_R8G8B8A8_UINT, "rgba8u"),
    (DXGI_FORMAT_B8G8R8A8_UNORM, "bgra8"),
    (DXGI_FORMAT_R16G16_FLOAT, "rg16f"),
    (DXGI_FORMAT_R16G16_UNORM, "rg16"),
    (DXGI_FORMAT_R16G16_UINT, "rg16u"),
    (DXGI_FORMAT_R8G8_UNORM, "rg8"),
    (DXGI_FORMAT_R16_FLOAT, "r16f"),
    (DXGI_FORMAT_R16_UNORM, "r16"),
    (DXGI_FORMAT_R16_UINT, "r16u"),
    (DXGI_FORMAT_R8_UNORM, "r8"),
    (DXGI_FORMAT_R8_UINT, "r8u"),
];

fn feature<T: Default>(device: &ID3D12Device, which: D3D12_FEATURE) -> Option<T> {
    let mut data = T::default();
    // SAFETY: `data` is the structure `which` fills, of its size.
    unsafe { device.CheckFeatureSupport(which, (&raw mut data).cast(), size_of::<T>() as u32) }
        .ok()
        .map(|()| data)
}

/// Reads the device's shader caps and logs them once.
pub(crate) fn read_caps(device: &ID3D12Device) {
    if CAPS.get().is_some() {
        return;
    }
    let mut caps = Caps::default();
    for asked in [D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_5] {
        let mut model = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: asked };
        // SAFETY: as in `feature`; asked from the newest down, as an older
        // runtime refuses a model it does not know.
        if unsafe { device.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, (&raw mut model).cast(), size_of_val(&model) as u32) }.is_ok() {
            caps.model = model.HighestShaderModel.0 as u32;
            break;
        }
    }
    if let Some(o) = feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS>(device, D3D12_FEATURE_D3D12_OPTIONS) {
        caps.doubles = o.DoublePrecisionFloatShaderOps.as_bool();
        caps.rovs = o.ROVsSupported.as_bool();
        caps.typed_loads = o.TypedUAVLoadAdditionalFormats.as_bool();
        caps.binding_tier = o.ResourceBindingTier.0;
    }
    if let Some(o) = feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>(device, D3D12_FEATURE_D3D12_OPTIONS1) {
        caps.wave_min = o.WaveLaneCountMin;
        caps.wave_max = o.WaveLaneCountMax;
        caps.wave_ops = o.WaveOps.as_bool();
        caps.int64 = o.Int64ShaderOps.as_bool();
    }
    if let Some(o) = feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS3>(device, D3D12_FEATURE_D3D12_OPTIONS3) {
        caps.barycentrics = o.BarycentricsSupported.as_bool();
    }
    if let Some(o) = feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS4>(device, D3D12_FEATURE_D3D12_OPTIONS4) {
        caps.native16 = o.Native16BitShaderOpsSupported.as_bool();
    }
    if let Some(o) = feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(device, D3D12_FEATURE_D3D12_OPTIONS7) {
        caps.mesh_tier = o.MeshShaderTier.0;
    }
    if let Some(o) = feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>(device, D3D12_FEATURE_D3D12_OPTIONS9) {
        caps.atomic64_typed = o.AtomicInt64OnTypedResourceSupported.as_bool();
        caps.mesh_derivatives = o.DerivativesInMeshAndAmplificationShadersSupported.as_bool();
    }
    if let Some(o) = feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>(device, D3D12_FEATURE_D3D12_OPTIONS11) {
        caps.atomic64_heap = o.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
    }
    let mut no_typed_load = Vec::new();
    for &(format, name) in TYPED_LOADS {
        let mut f = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
        // SAFETY: as in `feature`.
        let ok = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, (&raw mut f).cast(), size_of_val(&f) as u32) }.is_ok();
        if !ok || f.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD.0 == 0 {
            no_typed_load.push(name);
        }
    }
    tracing::info!(
        shader_model = format_args!("{}.{}", caps.model >> 4, caps.model & 15),
        wave_min = caps.wave_min,
        wave_max = caps.wave_max,
        wave_ops = caps.wave_ops,
        native16 = caps.native16,
        int64 = caps.int64,
        doubles = caps.doubles,
        rovs = caps.rovs,
        barycentrics = caps.barycentrics,
        typed_uav_loads = caps.typed_loads,
        no_typed_load = no_typed_load.join(" "),
        binding_tier = caps.binding_tier,
        mesh_tier = caps.mesh_tier,
        atomic64_typed = caps.atomic64_typed,
        atomic64_heap = caps.atomic64_heap,
        "host shader caps"
    );
    let _ = CAPS.set(caps);
}

/// The feature flags of a DXIL container (its `SFI0` part).
fn flags(dxil: &[u8]) -> u64 {
    let u32_at = |at: usize| dxil.get(at..at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    if dxil.get(..4) != Some(b"DXBC") {
        return 0;
    }
    for i in 0..u32_at(28) as usize {
        let part = u32_at(32 + 4 * i) as usize;
        if dxil.get(part..part + 4) == Some(b"SFI0") {
            return dxil.get(part + 8..part + 16).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
        }
    }
    0
}

/// What the shaders' feature flags ask for that the device lacks.
fn lacks(code: &[&[u8]]) -> Vec<&'static str> {
    let Some(caps) = CAPS.get() else {
        return Vec::new();
    };
    let asked = code.iter().fold(0, |all, dxil| all | flags(dxil));
    let wants: [(u64, &str, bool); 11] = [
        (0x1, "doubles", caps.doubles),
        (0x800, "typed UAV loads", caps.typed_loads),
        (0x1000, "ROVs", caps.rovs),
        (0x4000, "wave ops", caps.wave_ops),
        (0x8000, "int64", caps.int64),
        (0x2_0000, "barycentrics", caps.barycentrics),
        (0x4_0000, "native 16-bit", caps.native16),
        (0x40_0000, "64-bit typed atomics", caps.atomic64_typed),
        (0x200_0000, "heap indexing", caps.model >= 0x66 && caps.binding_tier >= 3),
        (0x1000_0000, "64-bit heap atomics", caps.atomic64_heap),
        (0x100_0000, "derivatives in mesh shaders", caps.mesh_derivatives),
    ];
    wants.iter().filter(|&&(bit, _, has)| asked & bit != 0 && !has).map(|&(_, name, _)| name).collect()
}

/// Refused pipelines by key, and the count by reason.
struct Seen {
    keys: HashSet<String>,
    reasons: BTreeMap<String, u64>,
    untranslated: HashSet<String>,
    retried: u64,
    retry_made: u64,
    changed: bool,
}

static SEEN: Mutex<Option<Seen>> = Mutex::new(None);

fn seen<T>(f: impl FnOnce(&mut Seen) -> T) -> T {
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    f(seen.get_or_insert_with(|| Seen {
        keys: HashSet::new(),
        reasons: BTreeMap::new(),
        untranslated: HashSet::new(),
        retried: 0,
        retry_made: 0,
        changed: false,
    }))
}

/// `VICTORY_SHADER_CAPS=1`.
fn retry_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SHADER_CAPS").is_ok_and(|v| v.trim() == "1"))
}

fn key(shaders: &[std::sync::Arc<Shader>]) -> String {
    shaders.iter().map(|s| s.hash.as_str()).collect::<Vec<_>>().join("+")
}

/// Logs a pipeline the host refused, once per key, and counts it by reason.
/// `stage` is the part that refused it (compute, graphics, mesh, stream,
/// root signature), `crashed` that the pipeline library left it out as one
/// the driver's compiler crashed on in an earlier run.
pub(crate) fn failed(
    stage: &'static str,
    shaders: &[std::sync::Arc<Shader>],
    error: &windows::core::Error,
    code: &[&[u8]],
    crashed: bool,
) {
    let key = key(shaders);
    let hr = error.code().0 as u32;
    let host = crate::host::host();
    let removed = host
        // SAFETY: a plain query of the device.
        .and_then(|h| unsafe { h.device.GetDeviceRemovedReason() }.err())
        .map(|e| e.code().0 as u32);
    let messages = host.and_then(|h| h.log_messages());
    let lacks = lacks(code);
    let reason = if crashed {
        "the driver crashed on it in an earlier run".to_owned()
    } else if !lacks.is_empty() {
        format!("device lacks {}", lacks.join(", "))
    } else if let Some(removed) = removed {
        format!("device removed {removed:#010x}")
    } else {
        format!("{stage} {hr:#010x}")
    };
    let first = seen(|s| {
        let first = s.keys.insert(key.clone());
        if first {
            *s.reasons.entry(reason.clone()).or_default() += 1;
            s.changed = true;
        }
        first
    });
    if first {
        tracing::warn!(
            key,
            stage,
            hr = format_args!("{hr:#010x}"),
            removed = removed.map(|r| format!("{r:#010x}")),
            messages,
            lacks = lacks.join(", "),
            reason,
            error = %error,
            "host pipeline failed (logged once per key)"
        );
    }
}

/// The result of a `VICTORY_SHADER_CAPS` retry.
fn retried(shaders: &[std::sync::Arc<Shader>], made: &windows::core::Result<ID3D12PipelineState>) {
    seen(|s| {
        s.retried += 1;
        s.retry_made += u64::from(made.is_ok());
        s.changed = true;
    });
    match made {
        Ok(_) => tracing::warn!(key = key(shaders), "host pipeline made on the retry without the pipeline library"),
        Err(e) => tracing::warn!(key = key(shaders), error = %e, "host pipeline refused on the retry too"),
    }
}

/// Under `VICTORY_SHADER_CAPS=1`, a refused pipeline whose shaders ask for
/// nothing the device lacks, and that did not crash the driver before, made
/// once more by `create` (straight from the driver).
pub(crate) fn retry(
    shaders: &[std::sync::Arc<Shader>],
    code: &[&[u8]],
    crashed: bool,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> Option<ID3D12PipelineState> {
    if !retry_on() || crashed || !lacks(code).is_empty() {
        return None;
    }
    let made = create();
    retried(shaders, &made);
    made.ok()
}

/// A shader that does not translate, counted once by hash.
pub(crate) fn untranslated(hash: &str) {
    seen(|s| {
        if s.untranslated.insert(hash.to_owned()) {
            s.changed = true;
        }
    });
}

/// The one line for a tester to send: refused pipelines by reason and the
/// shaders that do not translate. `always` for the line at exit; otherwise
/// only when something changed since the last one.
pub(crate) fn summary(always: bool) {
    let line = seen(|s| {
        if !always && !s.changed {
            return None;
        }
        s.changed = false;
        let reasons: Vec<String> = s.reasons.iter().map(|(r, n)| format!("{n} {r}")).collect();
        Some((s.keys.len(), reasons.join("; "), s.untranslated.len(), s.retried, s.retry_made))
    });
    if let Some((failed, reasons, untranslated, retried, retry_made)) = line {
        tracing::warn!(
            failed,
            untranslated,
            retried,
            retry_made,
            "pipeline failures: {failed} host pipelines failed ({}), {untranslated} shaders do not translate",
            if reasons.is_empty() { "none" } else { &reasons }
        );
    }
}
