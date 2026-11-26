//! Host state for pipelines handed over as loose parts (device method
//! 0x1c8, `victory_shader::loose_graphics`).

use windows::Win32::Graphics::Direct3D12::*;

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
