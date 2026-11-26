//! VALU runs under one EXEC mask lowered in every lane with a select per register they leave
//! ([`super::Lower::guarded_run`]), and the float form of guard selects. Wave32 programs of 32
//! threads: v1..v7 start as lane-varying old values (NaN payloads, denormals), vcc holds the
//! lanes below s3 (`v_cmp_gt_u32`), and EXEC becomes s0 for the body. After the body EXEC is
//! all lanes again and every lane stores v0..v7 at dword 8 * lane.

use super::super::host::{Dispatch, Host, Root, View};
use super::super::{HostShader, TEST_GUARD_FLOAT, TEST_GUARD_RUNS, validator};
use super::{Lowered, Parts, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const VCC: u16 = 106;
const EXEC: u8 = 126;
const LITERAL: u16 = 255;
const S_ENDPGM: u32 = 0xBF81_0000;

/// What v1..v7 hold before the body, each XORed with the lane.
const OLD: [u32; 8] = [
    0,
    0x7FC0_1234,
    0xFF80_0001,
    0x0000_0F0F,
    0x3F80_0000,
    0x8000_0101,
    0x4049_0FDB,
    0,
];

const fn k(n: u16) -> u16 {
    128 + n
}

const fn v(n: u16) -> u16 {
    256 + n
}

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_load(op: u32, dst: u32, base: u32, byte_off: u32) -> [u32; 2] {
    [0xF400_0000 | op << 18 | dst << 6 | base >> 1, 0xFA00_0000 | byte_off]
}

const fn vop1(op: u32, vdst: u32, src0: u16) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0 as u32
}

const fn vop2(op: u32, vdst: u32, src0: u16, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0 as u32
}

const fn vopc(op: u32, src0: u16, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0 as u32
}

/// VOP3 (VOP3b when `sdst` is not 0).
const fn vop3(op: u32, vdst: u32, sdst: u32, src: [u16; 3]) -> [u32; 2] {
    [
        0xD400_0000 | op << 16 | sdst << 8 | vdst,
        src[0] as u32 | (src[1] as u32) << 9 | (src[2] as u32) << 18,
    ]
}

/// The program around `body`, with an SRV and a sampler bound when `sample`.
fn program(body: &[u32], sample: bool) -> NativeProgram {
    let mut w = vec![s_mov_b32(5, 0x80)];
    w.extend(s_load(2, 8, 4, 0)); // UAV to s[8:11]
    if sample {
        w.push(s_mov_b32(7, 0x80));
        w.extend(s_load(3, 16, 4, 32)); // SRV to s[16:23]
        w.extend(s_load(2, 24, 6, 0)); // sampler to s[24:27]
    }
    w.push(vop2(0x1A, 15, k(5), 0)); // v_lshlrev_b32 v15, 5, v0
    for (r, old) in OLD.iter().enumerate().skip(1) {
        w.extend([vop2(0x1D, r as u32, LITERAL, 0), *old]); // v_xor_b32 vr, old, v0
    }
    w.push(vopc(0xC4, 3, 0)); // v_cmp_gt_u32 vcc, s3, v0
    w.push(s_mov_b32(EXEC, 0));
    w.extend_from_slice(body);
    w.push(s_mov_b32(EXEC, 193)); // exec_lo = -1
    for r in 0..8u32 {
        w.extend([0xE070_1000 | r * 4, 0x8002_000F | r << 8]); // buffer_store_dword vr, v15 offen
    }
    w.push(S_ENDPGM);
    let table = |user_data, slot| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let range = |kind, at, dwords, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    };
    let mut layout = vec![
        RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        },
        table(4, 1),
    ];
    let mut used = vec![range(ResourceType::Uav, 0, 4, 4, 1)];
    if sample {
        layout.push(table(6, 2));
        used.push(range(ResourceType::Srv, 8, 8, 4, 1));
        used.push(range(ResourceType::Sampler, 0, 4, 6, 2));
    }
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: if sample { 7 << 1 } else { 5 << 1 },
            threads: [32, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` lowered with the two switches set, and the runs it lowered in every lane by the
/// names of their instructions.
fn lowered(native: &NativeProgram, float: bool, runs: bool) -> (Lowered, Vec<Vec<&'static str>>) {
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    TEST_GUARD_FLOAT.set(Some(float));
    TEST_GUARD_RUNS.set(Some(runs));
    let l = lower(&Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    });
    TEST_GUARD_FLOAT.set(None);
    TEST_GUARD_RUNS.set(None);
    let l = l.unwrap();
    let names = l
        .guard_runs
        .iter()
        .map(|&(b, start, end)| p.blocks[b].insts[start..end].iter().map(|i| i.name()).collect())
        .collect();
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = l.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    (l, names)
}

fn runs_of(body: &[u32], sample: bool) -> Vec<Vec<&'static str>> {
    lowered(&program(body, sample), true, true).1
}

const BEFORE: [u32; 2] = [
    vop2(0x03, 1, v(1), 3), // v_add_f32 v1, v1, v3
    vop2(0x08, 2, 1, 2),    // v_mul_f32 v2, s1, v2
];

const AFTER: [u32; 3] = {
    let fma = vop3(0x14B, 6, 0, [v(1), v(2), v(3)]); // v_fma_f32 v6, v1, v2, v3
    [vop2(0x1D, 7, v(7), 1), fma[0], fma[1]] // v_xor_b32 v7, v7, v1
};

/// The runs of BEFORE, `middle` and AFTER.
fn around(middle: &[u32], sample: bool) -> Vec<Vec<&'static str>> {
    let mut body = BEFORE.to_vec();
    body.extend_from_slice(middle);
    body.extend_from_slice(&AFTER);
    runs_of(&body, sample)
}

/// `middle` ends the run: BEFORE and AFTER are two runs and neither takes it.
fn refused(middle: &[u32], sample: bool) {
    assert_eq!(
        around(middle, sample),
        [vec!["v_add_f32", "v_mul_f32"], vec!["v_xor_b32", "v_fma_f32"]]
    );
}

#[test]
fn lane_local_instructions_make_one_run() {
    let runs = around(&[vop2(0x04, 5, v(5), 4)], false); // v_sub_f32 v5, v5, v4
    assert_eq!(
        runs,
        [vec!["v_add_f32", "v_mul_f32", "v_sub_f32", "v_xor_b32", "v_fma_f32"]]
    );
}

#[test]
fn no_runs_with_the_switch_off() {
    let mut body = BEFORE.to_vec();
    body.extend_from_slice(&AFTER);
    assert!(lowered(&program(&body, false), true, false).1.is_empty());
}

#[test]
fn a_lane_mask_picking_a_cndmask_stays_in_the_run() {
    let runs = around(&[vop2(0x01, 4, v(5), 6)], false); // v_cndmask_b32 v4, v5, v6, vcc
    assert_eq!(
        runs,
        [vec!["v_add_f32", "v_mul_f32", "v_cndmask_b32", "v_xor_b32", "v_fma_f32"]]
    );
}

#[test]
fn dpp_ends_a_run() {
    // v_mov_b32_dpp v4, v1 quad_perm:[1,0,3,2]
    refused(&[vop1(0x01, 4, 0xFA), 0xFF00_B101], false);
}

#[test]
fn v_readlane_ends_a_run() {
    let readlane = vop3(0x360, 20, 0, [v(1), k(3), 0]);
    refused(&readlane, false);
}

#[test]
fn v_readfirstlane_ends_a_run() {
    refused(&[vop1(0x02, 20, v(1))], false);
}

#[test]
fn v_writelane_ends_a_run() {
    refused(&vop3(0x361, 4, 0, [1, k(5), 0]), false);
}

#[test]
fn v_permlane16_ends_a_run() {
    refused(&vop3(0x377, 4, 0, [v(1), 1, 2]), false);
}

#[test]
fn v_mbcnt_ends_a_run() {
    refused(&vop3(0x365, 4, 0, [193, k(0), 0]), false);
}

#[test]
fn ds_swizzle_ends_a_run() {
    refused(&[0xD8D4_041F, 0x0400_0001], false); // ds_swizzle_b32 v4, v1 offset:0x41f
}

#[test]
fn ds_permute_ends_a_run() {
    refused(&[0xDAC8_0000, 0x0400_010F], false); // ds_permute_b32 v4, v15, v1
}

#[test]
fn an_implicit_lod_sample_ends_a_run() {
    // image_sample v8, v[1:2], s[16:23], s[24:27] dmask:0x1 dim:2D
    refused(&[0xF080_0108, 0x00C4_0801], true);
}

#[test]
fn reading_exec_ends_a_run() {
    refused(&[vop1(0x01, 4, EXEC as u16)], false); // v_mov_b32 v4, exec_lo
}

#[test]
fn reading_a_lane_mask_as_data_ends_a_run() {
    refused(&[vop1(0x01, 4, VCC)], false); // v_mov_b32 v4, vcc_lo
}

#[test]
fn writing_vcc_ends_a_run() {
    refused(&vop3(0x30F, 4, VCC as u32, [v(1), v(2), 0]), false); // v_add_co_u32 v4, vcc_lo
}

#[test]
fn v_cmp_ends_a_run() {
    refused(&[vopc(0xC4, 3, 1)], false); // v_cmp_gt_u32 vcc, s3, v1
}

/// The four ways to lower the same program give the same dwords in every lane, on the
/// hardware adapter and on WARP, for EXEC masks from none to all; lanes outside the mask
/// keep their old values bit for bit.
#[test]
fn guarded_runs_and_float_selects_change_no_lane() {
    let fma = vop3(0x14B, 6, 0, [v(1), v(2), v(3)]);
    let body = [
        vop2(0x03, 1, v(1), 3),  // v_add_f32 v1, v1, v3
        vop2(0x08, 2, 1, 2),     // v_mul_f32 v2, s1, v2
        vop2(0x1D, 7, v(7), 1),  // v_xor_b32 v7, v7, v1
        vop2(0x01, 4, v(5), 6),  // v_cndmask_b32 v4, v5, v6, vcc
        fma[0],
        fma[1],
        vop1(0x01, 3, v(2)),     // v_mov_b32 v3, v2
        vop1(0x02, 20, v(6)),    // v_readfirstlane_b32 s20, v6
        vop2(0x10, 5, 20, 5),    // v_max_f32 v5, s20, v5
        vop1(0x06, 1, v(7)),     // v_cvt_f32_u32 v1, v7 (v1 again)
        vop2(0x25, 2, 2, 7),     // v_add_nc_u32 v2, s2, v7
        vop2(0x08, 0, v(1), 5),  // v_mul_f32 v0, v1, v5
        vop2(0x01, 6, v(3), 1),  // v_cndmask_b32 v6, v3, v1, vcc
    ];
    let native = program(&body, false);
    let cases: [[u32; 4]; 6] = [
        [0, 0x4049_0FDB, 0x8000_0001, 13],
        [!0, 0x4049_0FDB, 0x8000_0001, 13],
        [0x5555_5555, 0xBF80_0000, 0x0000_0001, 0],
        [0xA5C3_0F96, 0x7F80_0000, 0x7FC0_0000, 32],
        [1, 0x0000_0F0F, 0xFFFF_FFFF, 7],
        [0x8000_0000, 0x3F00_0000, 0x1234_5678, 31],
    ];
    let shaders: Vec<(HostShader, usize)> = [(false, false), (true, false), (false, true), (true, true)]
        .into_iter()
        .map(|(float, runs)| {
            let (l, names) = lowered(&native, float, runs);
            assert_eq!(names.is_empty(), !runs, "runs {names:?}");
            let shader = HostShader {
                stage: HwStage::Cs,
                dxil: l.dxil,
                user_data: l.user_data,
                samplers: l.samplers,
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: Vec::new(),
            };
            (shader, names.len())
        })
        .collect();
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        ran += 1;
        for (c, inputs) in cases.iter().enumerate() {
            let mut first: Option<Vec<u32>> = None;
            for (n, (shader, _)) in shaders.iter().enumerate() {
                let (root, pso) = host.compute_pipeline(shader).unwrap();
                let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    ud[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Constants, 0) => inputs[u.dword as usize],
                        _ => 0,
                    };
                }
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &[View::RawUav(256)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap()
                    .swap_remove(0);
                for lane in 0..32u32 {
                    if inputs[0] >> lane & 1 == 0 {
                        for r in 1..8 {
                            assert_eq!(
                                out[(lane * 8) as usize + r],
                                OLD[r] ^ lane,
                                "{label} case {c} lowering {n} lane {lane} v{r}"
                            );
                        }
                    }
                }
                match &first {
                    None => first = Some(out),
                    Some(want) => {
                        for (i, (a, b)) in out.iter().zip(want).enumerate() {
                            assert_eq!(
                                a,
                                b,
                                "{label} case {c} lowering {n} lane {} v{}",
                                i / 8,
                                i % 8
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(ran > 0, "no SM 6.6 device");
}
