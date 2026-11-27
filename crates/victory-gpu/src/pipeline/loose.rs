//! Host state for pipelines handed over as loose parts (device method
//! 0x1c8, `victory_shader::loose_graphics`).

use std::cell::RefCell;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::host::{GraphicsState, HostPipeline};
use super::{Shader, make};

/// A blend factor of CB_BLENDn_CONTROL. The alpha factors take no colour:
/// D3D12 refuses a colour factor there, which reads the same channel's
/// alpha anyway.
fn factor(f: u32, alpha: bool) -> Option<D3D12_BLEND> {
    Some(match (f, alpha) {
        (0, _) => D3D12_BLEND_ZERO,
        (1, _) => D3D12_BLEND_ONE,
        (2, false) => D3D12_BLEND_SRC_COLOR,
        (3, false) => D3D12_BLEND_INV_SRC_COLOR,
        (2, true) | (4, _) => D3D12_BLEND_SRC_ALPHA,
        (3, true) | (5, _) => D3D12_BLEND_INV_SRC_ALPHA,
        (6, _) | (8, true) => D3D12_BLEND_DEST_ALPHA,
        (7, _) | (9, true) => D3D12_BLEND_INV_DEST_ALPHA,
        (8, false) => D3D12_BLEND_DEST_COLOR,
        (9, false) => D3D12_BLEND_INV_DEST_COLOR,
        (10, _) => D3D12_BLEND_SRC_ALPHA_SAT,
        (13, false) => D3D12_BLEND_BLEND_FACTOR,
        (14, false) => D3D12_BLEND_INV_BLEND_FACTOR,
        (13, true) | (19, _) => D3D12_BLEND_ALPHA_FACTOR,
        (14, true) | (20, _) => D3D12_BLEND_INV_ALPHA_FACTOR,
        (15, false) => D3D12_BLEND_SRC1_COLOR,
        (16, false) => D3D12_BLEND_INV_SRC1_COLOR,
        (15, true) | (17, _) => D3D12_BLEND_SRC1_ALPHA,
        (16, true) | (18, _) => D3D12_BLEND_INV_SRC1_ALPHA,
        _ => return None,
    })
}

fn op(f: u32) -> Option<D3D12_BLEND_OP> {
    Some(match f {
        0 => D3D12_BLEND_OP_ADD,
        1 => D3D12_BLEND_OP_SUBTRACT,
        2 => D3D12_BLEND_OP_MIN,
        3 => D3D12_BLEND_OP_MAX,
        4 => D3D12_BLEND_OP_REV_SUBTRACT,
        _ => return None,
    })
}

/// The blend state of CB_TARGET_MASK and CB_BLEND0..7_CONTROL: per target
/// the colour source factor in bits 0-4, function in 5-7, destination in
/// 8-12, the same for alpha from bit 16, separate alpha at 29, enable at 30.
/// None for what D3D12 has no form of (the BOTH_ factors).
pub(crate) fn blend(target_mask: u32, controls: &[u32; 8]) -> Option<D3D12_BLEND_DESC> {
    let mut desc = D3D12_BLEND_DESC::default();
    for (i, (&c, t)) in controls.iter().zip(&mut desc.RenderTarget).enumerate() {
        let field = |at: u32, bits: u32| (c >> at) & ((1 << bits) - 1);
        let alpha = if c & 1 << 29 != 0 { 16 } else { 0 };
        *t = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: (c & 1 << 30 != 0).into(),
            LogicOpEnable: false.into(),
            SrcBlend: factor(field(0, 5), false)?,
            DestBlend: factor(field(8, 5), false)?,
            BlendOp: op(field(5, 3))?,
            SrcBlendAlpha: factor(field(alpha, 5), true)?,
            DestBlendAlpha: factor(field(alpha + 8, 5), true)?,
            BlendOpAlpha: op(field(alpha + 5, 3))?,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: (target_mask >> (4 * i) & 0xf) as u8,
        };
    }
    desc.IndependentBlendEnable = desc.RenderTarget[1..]
        .iter()
        .any(|t| *t != desc.RenderTarget[0])
        .into();
    Some(desc)
}

/// What a draw's targets add to a loose pipeline: the views' formats and
/// sample count, which the description leaves to the draw.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Targets {
    colour: [DXGI_FORMAT; 8],
    count: u32,
    depth: DXGI_FORMAT,
    samples: u32,
}

impl Targets {
    /// The views at `handles` and `depth` as `State::set_targets` binds
    /// them: a typed null view keeps its slot, a view it leaves out takes
    /// none. A view without a format has its resource's.
    pub(crate) fn of(handles: &[u64], depth: u64) -> Targets {
        let mut t = Targets {
            colour: [DXGI_FORMAT_UNKNOWN; 8],
            count: 0,
            depth: DXGI_FORMAT_UNKNOWN,
            samples: 1,
        };
        let described = |format: Option<DXGI_FORMAT>, texture: Option<&ID3D12Resource>| {
            // SAFETY: a live host resource.
            let desc = texture.map(|r| unsafe { r.GetDesc() });
            let format = format.filter(|f| f.0 != 0).or(desc.map(|d| d.Format))?;
            Some((format, desc.map_or(1, |d| d.SampleDesc.Count)))
        };
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                continue;
            };
            let view = record.desc();
            let texture = crate::resource::host_resource(record.resource);
            if texture.is_none() && view.is_none_or(|d| d.ViewDimension.0 == 0) {
                continue;
            }
            let Some((format, samples)) = described(view.map(|d| d.Format), texture) else {
                continue;
            };
            if let Some(slot) = t.colour.get_mut(t.count as usize) {
                *slot = format;
                t.count += 1;
                t.samples = samples;
            }
        }
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            described(record.depth_desc().map(|d| d.Format), Some(texture))
        });
        if let Some((format, samples)) = dsv {
            t.depth = format;
            t.samples = samples;
        }
        t
    }
}

/// A pipeline handed over as loose parts. It has one host pipeline per set
/// of target formats it is drawn into, made at the first draw into them.
pub(crate) struct Loose {
    id: u64,
    shaders: Vec<Arc<Shader>>,
    inputs: Vec<victory_shader::InputElement>,
    blend: D3D12_BLEND_DESC,
    made: Mutex<Vec<(Targets, Arc<make::Pending>)>>,
}

thread_local! {
    /// The host pipeline the last draw on this thread chose, and whose.
    static CHOSEN: RefCell<Option<(u64, Arc<make::Pending>)>> = const { RefCell::new(None) };
}

impl Loose {
    /// None for a blend state D3D12 has no form of.
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, g: &victory_shader::LooseGraphics) -> Option<Loose> {
        static IDS: AtomicU64 = AtomicU64::new(1);
        Some(Loose {
            id: IDS.fetch_add(1, Ordering::Relaxed),
            shaders,
            inputs: g.inputs.clone(),
            blend: blend(g.target_mask, &g.blend)?,
            made: Mutex::new(Vec::new()),
        })
    }

    /// shortcut: depth testing is off and nothing culls. The description has
    /// neither, and a guess at the title's depth convention (reversed or
    /// not) would drop whole passes; every draw is taken as triangles.
    fn state(&self, t: &Targets) -> GraphicsState {
        let mut g = GraphicsState { blend: self.blend, ..GraphicsState::default() };
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.depth.DepthEnable = false.into();
        g.depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        g.names = self.inputs.iter().map(|e| CString::new(e.name.as_str()).unwrap_or_default()).collect();
        g.inputs = self
            .inputs
            .iter()
            .zip(&g.names)
            .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: e.index,
                Format: DXGI_FORMAT(crate::resource::pc_format(e.format) as i32),
                InputSlot: e.slot,
                AlignedByteOffset: e.offset,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(e.class as i32),
                InstanceDataStepRate: e.step,
            })
            .collect();
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.targets = D3D12_RT_FORMAT_ARRAY { RTFormats: t.colour, NumRenderTargets: t.count };
        g.depth_format = t.depth;
        g.samples.Count = t.samples;
        g
    }

    /// At a draw: the host pipeline for the targets bound, queued the first
    /// time, which [`Loose::chosen`] then gives on this thread.
    pub(crate) fn choose(&self, handles: &[u64], depth: u64) {
        let targets = Targets::of(handles, depth);
        let p = {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            match made.iter().find(|(t, _)| *t == targets) {
                Some((_, p)) => p.clone(),
                None => {
                    let state = Box::new(self.state(&targets));
                    let p = Arc::new(make::Pending::new(self.shaders.clone(), Some(state)));
                    made.push((targets, p.clone()));
                    p
                }
            }
        };
        CHOSEN.with_borrow_mut(|c| *c = Some((self.id, p)));
    }

    /// The host pipeline the last [`Loose::choose`] on this thread picked,
    /// if it was for this pipeline and is made.
    pub(crate) fn chosen(&self) -> Option<&HostPipeline> {
        let p = CHOSEN.with_borrow(|c| c.as_ref().filter(|(id, _)| *id == self.id).map(|(_, p)| p.clone()))?;
        let made: *const HostPipeline = make::ready(&p)?;
        // SAFETY: `self.made` holds `p`, and so what it made, while self lives.
        Some(unsafe { &*made })
    }

    /// Why the last draw on this thread had no host pipeline (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn missing(&self) -> Option<String> {
        let p = CHOSEN.with_borrow(|c| c.as_ref().filter(|(id, _)| *id == self.id).map(|(_, p)| p.clone()));
        match p {
            Some(p) => make::missing(&p),
            None => Some("loose, no targets chosen".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_reads_the_title_controls() {
        let off = blend(u32::MAX, &[0x8000_0000; 8]).unwrap();
        let t = off.RenderTarget[0];
        assert!(!t.BlendEnable.as_bool() && !off.IndependentBlendEnable.as_bool());
        assert_eq!((t.SrcBlend, t.DestBlend, t.RenderTargetWriteMask), (D3D12_BLEND_ZERO, D3D12_BLEND_ZERO, 0xf));

        let alpha = blend(u32::MAX, &[0xe504_0504; 8]).unwrap().RenderTarget[3];
        assert!(alpha.BlendEnable.as_bool());
        assert_eq!(
            (alpha.SrcBlend, alpha.DestBlend, alpha.BlendOp, alpha.SrcBlendAlpha, alpha.DestBlendAlpha),
            (D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA)
        );

        let none = blend(0, &[0x8000_0000; 8]).unwrap();
        assert!(none.RenderTarget.iter().all(|t| t.RenderTargetWriteMask == 0));

        let mut mixed = [0x8000_0000; 8];
        mixed[1] = 0x4000_0302;
        let m = blend(0xff, &mixed).unwrap();
        assert!(m.IndependentBlendEnable.as_bool());
        assert_eq!((m.RenderTarget[1].SrcBlend, m.RenderTarget[1].SrcBlendAlpha), (D3D12_BLEND_SRC_COLOR, D3D12_BLEND_SRC_ALPHA));
        assert_eq!(m.RenderTarget[2].RenderTargetWriteMask, 0);
        assert!(blend(u32::MAX, &[0x4000_000b; 8]).is_none());
    }
}
