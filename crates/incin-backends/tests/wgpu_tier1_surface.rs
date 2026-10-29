//! Tier-1 bool-mask ops through the user-facing `Tensor` surface (#91/#100).
//!
//! `wgpu_indexing.rs` proves `masked_fill`/`where_cond` execute through
//! canonical dispatch; that is the backend half. The other half is whether
//! a statically shaped tensor can reach them at all (#100): the `Tensor`
//! methods carry generic bounds (`BroadcastShape`, `Execute<op::...>`,
//! layout threading) that an `Execute` impl alone does not satisfy. Each
//! case here builds the same statically shaped tensors on WGPU and on CPU,
//! runs the same method chain, and compares values — including a
//! rank-deficient broadcast mask, the shape the pre-broadcast path exists
//! for.
//!
//! Requires a WGPU adapter:
//! `cargo test -p incin-backends --features wgpu,cpu --test wgpu_tier1_surface`.
#![cfg(all(feature = "wgpu", feature = "cpu"))]

extern crate incin_core as incin;

use incin_backends::prelude::*;
use incin_core::prelude::*;
use incin_core::tensor::device::Cpu;

fn require_wgpu(gpu: &Wgpu) {
    gpu.zeros(shape![1]).expect(
        "no WGPU adapter, but the `wgpu` feature is enabled -- that is an explicit request \
         for this backend. Skipping here would report `ok` for a test that ran nothing.",
    );
}

#[test]
fn masked_fill_from_a_comparison_mask_matches_cpu() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    // mask = a > b = [[F, T], [F, T]]; filled keeps `a` under F, writes
    // the constant under T.
    let got = gpu
        .tensor([[1.0f32, 5.0], [3.0, 3.0]])
        .unwrap()
        .masked_fill(
            &gpu.tensor([[1.0f32, 5.0], [3.0, 3.0]])
                .unwrap()
                .gt(&gpu.tensor([[4.0f32, 2.0], [3.0, 0.0]]).unwrap())
                .unwrap(),
            -1.0,
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = Cpu
        .tensor([[1.0f32, 5.0], [3.0, 3.0]])
        .unwrap()
        .masked_fill(
            &Cpu.tensor([[1.0f32, 5.0], [3.0, 3.0]])
                .unwrap()
                .gt(&Cpu.tensor([[4.0f32, 2.0], [3.0, 0.0]]).unwrap())
                .unwrap(),
            -1.0,
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, want, "masked_fill must agree with CPU");
    assert_eq!(got, vec![1.0, -1.0, 3.0, -1.0]);
}

#[test]
fn where_cond_selects_per_element_matches_cpu() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    // Same mask as above ([[F, T], [F, T]]): out = mask ? a : b,
    // so the last element takes `a` (3.0 > 0.0 is true), not `b`.
    let gpu_mask = gpu
        .tensor([[1.0f32, 5.0], [3.0, 3.0]])
        .unwrap()
        .gt(&gpu.tensor([[4.0f32, 2.0], [3.0, 0.0]]).unwrap())
        .unwrap();
    let got = gpu_mask
        .where_cond(
            &gpu.tensor([[1.0f32, 5.0], [3.0, 3.0]]).unwrap(),
            &gpu.tensor([[4.0f32, 2.0], [3.0, 0.0]]).unwrap(),
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cpu_mask = Cpu
        .tensor([[1.0f32, 5.0], [3.0, 3.0]])
        .unwrap()
        .gt(&Cpu.tensor([[4.0f32, 2.0], [3.0, 0.0]]).unwrap())
        .unwrap();
    let want = cpu_mask
        .where_cond(
            &Cpu.tensor([[1.0f32, 5.0], [3.0, 3.0]]).unwrap(),
            &Cpu.tensor([[4.0f32, 2.0], [3.0, 0.0]]).unwrap(),
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, want, "where_cond must agree with CPU");
    assert_eq!(got, vec![4.0, 5.0, 3.0, 3.0]);
}

#[test]
fn masked_fill_with_a_broadcast_mask_matches_cpu() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    // A [2, 1] mask against a [2, 2] input: row 0 kept, row 1 filled.
    // This is the rank-deficient shape the pre-broadcast path exists for.
    // The mask is built by comparison (bool `.tensor` construction is not
    // part of this surface's contract; the comparison path is).
    let gpu_mask = gpu
        .tensor([[0.0f32], [1.0]])
        .unwrap()
        .gt(&gpu.tensor([[0.5f32], [0.5]]).unwrap())
        .unwrap();
    let got = gpu
        .tensor([[1.0f32, 2.0], [3.0, 4.0]])
        .unwrap()
        .masked_fill(&gpu_mask, 0.0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cpu_mask = Cpu
        .tensor([[0.0f32], [1.0]])
        .unwrap()
        .gt(&Cpu.tensor([[0.5f32], [0.5]]).unwrap())
        .unwrap();
    let want = Cpu
        .tensor([[1.0f32, 2.0], [3.0, 4.0]])
        .unwrap()
        .masked_fill(&cpu_mask, 0.0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, want, "broadcast masked_fill must agree with CPU");
    assert_eq!(got, vec![1.0, 2.0, 0.0, 0.0]);
}
