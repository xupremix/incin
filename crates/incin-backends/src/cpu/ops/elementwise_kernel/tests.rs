/// The AVX2 kernels must be reachable in the build users actually get.
///
/// `avx2_f32_available` was inline at four call sites and read
/// `simd_lanes::<f32>() >= 8` alone. That constant is false in a stock
/// `cargo build`, so all four branches were dead code and the CPU backend
/// fell through to a scalar loop — a 9x difference on `add_f32/65536`,
/// invisible to every test because the kernels themselves stayed correct.
///
/// This is the assertion that would have caught it. It fails if the gate is
/// ever narrowed back to a compile-time-only condition.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[test]
fn the_avx2_gate_opens_on_a_machine_that_supports_avx2() {
    // Deliberately the raw macro and not `simd::avx2_detected`: the point is
    // to ask the hardware independently of the predicate under test. Routing
    // this through the same predicate would make the assertion compare a
    // value with itself and always pass.
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    assert!(
        avx2_f32_available(),
        "this machine supports AVX2 but the f32 kernel gate is closed"
    );
    assert!(
        avx2_f64_available(),
        "this machine supports AVX2 but the f64 kernel gate is closed"
    );
}

use super::*;

#[test]
fn f32_contiguous_add_stays_typed() {
    let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2]);
    let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![3.0, 4.0]), vec![2]);
    let output = execute_binary(BinaryOp::Add, &lhs, &rhs, &[2])
        .unwrap()
        .unwrap();

    assert_eq!(&*output.buffer, &CpuBuffer::F32(vec![4.0, 6.0]));
}

#[test]
fn f64_contiguous_math_keeps_f64_precision() {
    let lhs = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0 + f64::EPSILON]), vec![1]);
    let rhs = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0]), vec![1]);
    let output = execute_binary(BinaryOp::Sub, &lhs, &rhs, &[1])
        .unwrap()
        .unwrap();

    assert_eq!(&*output.buffer, &CpuBuffer::F64(vec![f64::EPSILON]));
}

#[test]
fn vector_kernels_handle_odd_scalar_tails_for_every_operation() {
    let lhs_f32: Vec<f32> = (1..=19).map(|value| value as f32).collect();
    let rhs_f32: Vec<f32> = (1..=19).map(|value| value as f32 * 0.5).collect();
    let lhs_f64: Vec<f64> = lhs_f32.iter().map(|&value| f64::from(value)).collect();
    let rhs_f64: Vec<f64> = rhs_f32.iter().map(|&value| f64::from(value)).collect();

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        let actual_f32 = map_binary_f32(op, &lhs_f32, &rhs_f32);
        let expected_f32: Vec<_> = lhs_f32
            .iter()
            .zip(&rhs_f32)
            .map(|(&lhs, &rhs)| op.eval_f32(lhs, rhs))
            .collect();
        assert_eq!(actual_f32, expected_f32);

        let actual_f64 = map_binary_f64(op, &lhs_f64, &rhs_f64);
        let expected_f64: Vec<_> = lhs_f64
            .iter()
            .zip(&rhs_f64)
            .map(|(&lhs, &rhs)| op.eval_f64(lhs, rhs))
            .collect();
        assert_eq!(actual_f64, expected_f64);
    }
}

#[test]
fn scalar_vector_kernels_preserve_order_and_handle_tails() {
    let dense_f32: Vec<f32> = (1..=19).map(|value| value as f32).collect();
    let dense_f64: Vec<f64> = dense_f32.iter().map(|&value| f64::from(value)).collect();

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        for scalar_left in [false, true] {
            let actual_f32 = map_scalar_f32(op, &dense_f32, 3.25, scalar_left);
            let expected_f32: Vec<_> = dense_f32
                .iter()
                .map(|&dense| {
                    if scalar_left {
                        op.eval_f32(3.25, dense)
                    } else {
                        op.eval_f32(dense, 3.25)
                    }
                })
                .collect();
            assert_eq!(actual_f32, expected_f32);

            let actual_f64 = map_scalar_f64(op, &dense_f64, 3.25, scalar_left);
            let expected_f64: Vec<_> = dense_f64
                .iter()
                .map(|&dense| {
                    if scalar_left {
                        op.eval_f64(3.25, dense)
                    } else {
                        op.eval_f64(dense, 3.25)
                    }
                })
                .collect();
            assert_eq!(actual_f64, expected_f64);
        }
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[test]
// Sized to exceed SIMD_PARALLEL_CHUNK on purpose, which is also what puts it
// out of reach of the interpreter: miri needs hours for a single crossing.
// The soundness gate splits the two jobs rather than losing one of them -
// miri proves the aliasing rules on the small cases, and AddressSanitizer
// runs this one at native speed, where the chunk count is what matters.
// See tools/soundness.sh.
#[cfg_attr(miri, ignore)]
fn parallel_vector_chunks_preserve_operations_and_tails() {
    if !avx2_f32_available() {
        return;
    }

    let len = SIMD_PARALLEL_CHUNK + 3;
    let lhs_f32: Vec<f32> = (1..=len).map(|value| value as f32).collect();
    let rhs_f32: Vec<f32> = (1..=len).map(|value| value as f32 * 0.5 + 1.0).collect();
    let lhs_f64: Vec<f64> = lhs_f32.iter().map(|&value| f64::from(value)).collect();
    let rhs_f64: Vec<f64> = rhs_f32.iter().map(|&value| f64::from(value)).collect();

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        let actual_f32 = parallel_avx2_binary_f32(op, &lhs_f32, &rhs_f32);
        let expected_f32: Vec<_> = lhs_f32
            .iter()
            .zip(&rhs_f32)
            .map(|(&lhs, &rhs)| op.eval_f32(lhs, rhs))
            .collect();
        assert_eq!(actual_f32, expected_f32);

        let actual_f64 = parallel_avx2_binary_f64(op, &lhs_f64, &rhs_f64);
        let expected_f64: Vec<_> = lhs_f64
            .iter()
            .zip(&rhs_f64)
            .map(|(&lhs, &rhs)| op.eval_f64(lhs, rhs))
            .collect();
        assert_eq!(actual_f64, expected_f64);

        for scalar_left in [false, true] {
            let actual_f32 = parallel_avx2_scalar_f32(op, &lhs_f32, 3.25, scalar_left);
            let expected_f32: Vec<_> = lhs_f32
                .iter()
                .map(|&dense| {
                    if scalar_left {
                        op.eval_f32(3.25, dense)
                    } else {
                        op.eval_f32(dense, 3.25)
                    }
                })
                .collect();
            assert_eq!(actual_f32, expected_f32);

            let actual_f64 = parallel_avx2_scalar_f64(op, &lhs_f64, 3.25, scalar_left);
            let expected_f64: Vec<_> = lhs_f64
                .iter()
                .map(|&dense| {
                    if scalar_left {
                        op.eval_f64(3.25, dense)
                    } else {
                        op.eval_f64(dense, 3.25)
                    }
                })
                .collect();
            assert_eq!(actual_f64, expected_f64);
        }
    }
}

#[test]
fn half_storage_uses_f32_compute() {
    let lhs = CpuStorage::from_contiguous(
        CpuBuffer::F16(vec![f16::from_f32(1.5), f16::from_f32(2.0)]),
        vec![2],
    );
    let rhs = CpuStorage::from_contiguous(
        CpuBuffer::F16(vec![f16::from_f32(2.0), f16::from_f32(4.0)]),
        vec![2],
    );
    let output = execute_binary(BinaryOp::Mul, &lhs, &rhs, &[2])
        .unwrap()
        .unwrap();

    assert_eq!(
        &*output.buffer,
        &CpuBuffer::F16(vec![f16::from_f32(3.0), f16::from_f32(8.0)])
    );
}

#[test]
fn scalar_broadcast_preserves_operand_order() {
    let scalar = CpuStorage::from_contiguous(CpuBuffer::F32(vec![10.0]), vec![]);
    let dense = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2]);

    let left = execute_binary(BinaryOp::Sub, &scalar, &dense, &[2])
        .unwrap()
        .unwrap();
    let right = execute_binary(BinaryOp::Sub, &dense, &scalar, &[2])
        .unwrap()
        .unwrap();

    assert_eq!(&*left.buffer, &CpuBuffer::F32(vec![9.0, 8.0]));
    assert_eq!(&*right.buffer, &CpuBuffer::F32(vec![-9.0, -8.0]));
}

#[test]
fn unary_family_uses_native_float_compute() {
    let f32_input = CpuStorage::from_contiguous(CpuBuffer::F32(vec![-1.0, 0.0, 2.0]), vec![3]);
    let f32_output = execute_unary(UnaryOp::Relu, &f32_input).unwrap().unwrap();
    assert_eq!(&*f32_output.buffer, &CpuBuffer::F32(vec![0.0, 0.0, 2.0]));

    let f64_input = CpuStorage::from_contiguous(CpuBuffer::F64(vec![0.0, 1.0]), vec![2]);
    let f64_output = execute_unary(UnaryOp::Exp, &f64_input).unwrap().unwrap();
    assert_eq!(
        &*f64_output.buffer,
        &CpuBuffer::F64(vec![1.0, core::f64::consts::E])
    );
}

#[test]
fn general_broadcast_uses_typed_strided_kernel() {
    let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2, 1]);
    let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);

    let output = execute_binary(BinaryOp::Add, &lhs, &rhs, &[2, 3])
        .unwrap()
        .unwrap();
    assert_eq!(
        &*output.buffer,
        &CpuBuffer::F32(vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0])
    );
}

#[test]
fn dense_broadcast_vector_projection_preserves_order_and_odd_tails() {
    let rows_f32 = vec![2.0, 4.0, 8.0];
    let columns_f32: Vec<f32> = (1..=19).map(|value| value as f32 * 0.5).collect();
    let rows_f64: Vec<f64> = rows_f32.iter().map(|&value| f64::from(value)).collect();
    let columns_f64: Vec<f64> = columns_f32.iter().map(|&value| f64::from(value)).collect();

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        for reverse in [false, true] {
            let lhs_f32 = if reverse {
                CpuStorage::from_contiguous(CpuBuffer::F32(columns_f32.clone()), vec![1, 19])
            } else {
                CpuStorage::from_contiguous(CpuBuffer::F32(rows_f32.clone()), vec![3, 1])
            };
            let rhs_f32 = if reverse {
                CpuStorage::from_contiguous(CpuBuffer::F32(rows_f32.clone()), vec![3, 1])
            } else {
                CpuStorage::from_contiguous(CpuBuffer::F32(columns_f32.clone()), vec![1, 19])
            };
            let actual_f32 = execute_binary(op, &lhs_f32, &rhs_f32, &[3, 19])
                .unwrap()
                .unwrap();
            let expected_f32: Vec<_> = rows_f32
                .iter()
                .flat_map(|&row| {
                    columns_f32.iter().map(move |&column| {
                        if reverse {
                            op.eval_f32(column, row)
                        } else {
                            op.eval_f32(row, column)
                        }
                    })
                })
                .collect();
            assert_eq!(&*actual_f32.buffer, &CpuBuffer::F32(expected_f32));

            let lhs_f64 = if reverse {
                CpuStorage::from_contiguous(CpuBuffer::F64(columns_f64.clone()), vec![1, 19])
            } else {
                CpuStorage::from_contiguous(CpuBuffer::F64(rows_f64.clone()), vec![3, 1])
            };
            let rhs_f64 = if reverse {
                CpuStorage::from_contiguous(CpuBuffer::F64(rows_f64.clone()), vec![3, 1])
            } else {
                CpuStorage::from_contiguous(CpuBuffer::F64(columns_f64.clone()), vec![1, 19])
            };
            let actual_f64 = execute_binary(op, &lhs_f64, &rhs_f64, &[3, 19])
                .unwrap()
                .unwrap();
            let expected_f64: Vec<_> = rows_f64
                .iter()
                .flat_map(|&row| {
                    columns_f64.iter().map(move |&column| {
                        if reverse {
                            op.eval_f64(column, row)
                        } else {
                            op.eval_f64(row, column)
                        }
                    })
                })
                .collect();
            assert_eq!(&*actual_f64.buffer, &CpuBuffer::F64(expected_f64));
        }
    }
}

#[test]
// 1025 * 257 elements clears PARALLEL_GRAIN deliberately. Same reasoning as
// parallel_vector_chunks_preserve_operations_and_tails above.
#[cfg_attr(miri, ignore)]
fn parallel_dense_broadcast_projection_crosses_chunk_boundaries() {
    let rows = 1_025;
    let columns = 257;
    let row_values: Vec<f32> = (0..rows).map(|value| value as f32).collect();
    let column_values: Vec<f32> = (0..columns).map(|value| value as f32 * 0.25).collect();
    let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(row_values.clone()), vec![rows, 1]);
    let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(column_values.clone()), vec![1, columns]);

    let output = execute_binary(BinaryOp::Sub, &lhs, &rhs, &[rows, columns])
        .unwrap()
        .unwrap();
    let CpuBuffer::F32(values) = &*output.buffer else {
        panic!("expected F32 output");
    };
    for &(row, column) in &[
        (0, 0),
        (0, columns - 1),
        (PARALLEL_GRAIN / columns, PARALLEL_GRAIN % columns),
        (rows - 1, columns - 1),
    ] {
        assert_eq!(
            values[row * columns + column],
            row_values[row] - column_values[column]
        );
    }
}

#[test]
fn broadcast_strided_fast_path_matches_scalar_reference() {
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        // Case 1: [B, T, 1] vs [B, T, C] (layer_norm shape, C not multiple of 8)
        let (b, t, c) = (2, 3, 11);
        let btc_data: Vec<f32> = (1..=b * t * c).map(|x| x as f32 * 0.5 + 1.0).collect();
        let bt1_data: Vec<f32> = (1..=b * t).map(|x| x as f32 * 1.5 + 2.0).collect();

        let full = CpuStorage::from_contiguous(CpuBuffer::F32(btc_data.clone()), vec![b, t, c]);
        let broadcast =
            CpuStorage::from_contiguous(CpuBuffer::F32(bt1_data.clone()), vec![b, t, 1]);

        // Broadcast on RHS: full (op) broadcast
        let out_rhs = execute_binary(op, &full, &broadcast, &[b, t, c])
            .unwrap()
            .unwrap();
        let CpuBuffer::F32(ref vals_rhs) = *out_rhs.buffer else {
            panic!("expected F32 output");
        };
        for bi in 0..b {
            for ti in 0..t {
                for ci in 0..c {
                    let full_idx = bi * t * c + ti * c + ci;
                    let bcast_idx = bi * t + ti;
                    let expected = op.eval_f32(btc_data[full_idx], bt1_data[bcast_idx]);
                    assert!(
                        (vals_rhs[full_idx] - expected).abs() < 1e-6,
                        "mismatch at b={bi}, t={ti}, c={ci} for op {op:?}"
                    );
                }
            }
        }

        // Broadcast on LHS: broadcast (op) full
        let out_lhs = execute_binary(op, &broadcast, &full, &[b, t, c])
            .unwrap()
            .unwrap();
        let CpuBuffer::F32(ref vals_lhs) = *out_lhs.buffer else {
            panic!("expected F32 output");
        };
        for bi in 0..b {
            for ti in 0..t {
                for ci in 0..c {
                    let full_idx = bi * t * c + ti * c + ci;
                    let bcast_idx = bi * t + ti;
                    let expected = op.eval_f32(bt1_data[bcast_idx], btc_data[full_idx]);
                    assert!(
                        (vals_lhs[full_idx] - expected).abs() < 1e-6,
                        "mismatch at b={bi}, t={ti}, c={ci} for op {op:?}"
                    );
                }
            }
        }

        // Case 2: [C] vs [B, C] (bias-add shape, C not multiple of 8)
        let (b, c) = (4, 13);
        let bc_data: Vec<f32> = (1..=b * c).map(|x| x as f32 * 0.75 + 1.0).collect();
        let c_data: Vec<f32> = (1..=c).map(|x| x as f32 * 2.0 + 3.0).collect();

        let full_bc = CpuStorage::from_contiguous(CpuBuffer::F32(bc_data.clone()), vec![b, c]);
        let bcast_c = CpuStorage::from_contiguous(CpuBuffer::F32(c_data.clone()), vec![c]);

        // Broadcast on RHS: [B, C] (op) [C]
        let out_bias_rhs = execute_binary(op, &full_bc, &bcast_c, &[b, c])
            .unwrap()
            .unwrap();
        let CpuBuffer::F32(ref vals_bias_rhs) = *out_bias_rhs.buffer else {
            panic!("expected F32 output");
        };
        for bi in 0..b {
            for (ci, &c_val) in c_data.iter().enumerate().take(c) {
                let full_idx = bi * c + ci;
                let expected = op.eval_f32(bc_data[full_idx], c_val);
                assert!(
                    (vals_bias_rhs[full_idx] - expected).abs() < 1e-6,
                    "mismatch at b={bi}, c={ci} for op {op:?}"
                );
            }
        }

        // Broadcast on LHS: [C] (op) [B, C]
        let out_bias_lhs = execute_binary(op, &bcast_c, &full_bc, &[b, c])
            .unwrap()
            .unwrap();
        let CpuBuffer::F32(ref vals_bias_lhs) = *out_bias_lhs.buffer else {
            panic!("expected F32 output");
        };
        for bi in 0..b {
            for (ci, &c_val) in c_data.iter().enumerate().take(c) {
                let full_idx = bi * c + ci;
                let expected = op.eval_f32(c_val, bc_data[full_idx]);
                assert!(
                    (vals_bias_lhs[full_idx] - expected).abs() < 1e-6,
                    "mismatch at b={bi}, c={ci} for op {op:?}"
                );
            }
        }
    }
}

#[cfg(feature = "std")]
#[test]
#[ignore = "microbenchmark: run explicitly with --release --ignored --nocapture"]
fn benchmark_cpu_binary_kernels() {
    use std::hint::black_box;
    use std::time::Instant;

    println!(
        "execution,layout,dtype,elements,iterations,samples,median_ns_per_element,median_effective_gib_s"
    );
    for &(elements, iterations) in &[
        (1_024usize, 4_000usize),
        (4_096, 2_000),
        (16_384, 500),
        (65_536, 200),
        (262_144, 50),
        (1_048_576, 20),
        (2_097_152, 10),
        (4_194_304, 8),
    ] {
        let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.25; elements]), vec![elements]);
        let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(vec![2.5; elements]), vec![elements]);
        benchmark_case("contiguous", elements, iterations, || {
            execute_binary(BinaryOp::Add, black_box(&lhs), black_box(&rhs), &[elements])
        });

        let scalar = CpuStorage::from_contiguous(CpuBuffer::F32(vec![2.5]), vec![]);
        benchmark_case("scalar_broadcast", elements, iterations, || {
            execute_binary(
                BinaryOp::Add,
                black_box(&lhs),
                black_box(&scalar),
                &[elements],
            )
        });
        if elements >= DENSE_PARALLEL_GRAIN {
            benchmark_case("contiguous_rayon_reference", elements, iterations, || {
                let values = map_binary(f32_values(&lhs), f32_values(&rhs), &|a, b| a + b);
                Ok(Some(CpuStorage::from_contiguous(
                    CpuBuffer::F32(values),
                    vec![elements],
                )))
            });
            benchmark_case("scalar_rayon_reference", elements, iterations, || {
                let values = map_scalar_right(f32_values(&lhs), 2.5, &|a, b| a + b);
                Ok(Some(CpuStorage::from_contiguous(
                    CpuBuffer::F32(values),
                    vec![elements],
                )))
            });
        }

        let columns = 256;
        let rows = elements / columns;
        let row_values: Vec<f32> = (0..rows).map(|value| value as f32).collect();
        let column_values: Vec<f32> = (0..columns).map(|value| value as f32).collect();
        let rows_storage = CpuStorage::from_contiguous(CpuBuffer::F32(row_values), vec![rows, 1]);
        let columns_storage =
            CpuStorage::from_contiguous(CpuBuffer::F32(column_values), vec![1, columns]);
        benchmark_case("dense_broadcast", elements, iterations, || {
            execute_binary(
                BinaryOp::Add,
                black_box(&rows_storage),
                black_box(&columns_storage),
                &[rows, columns],
            )
        });
        let broadcast_plan = binary_iteration_plan(
            &rows_storage,
            rows,
            &columns_storage,
            columns,
            &[rows, columns],
        )
        .unwrap();
        benchmark_case(
            "dense_broadcast_odometer_reference",
            elements,
            iterations,
            || {
                let values = map_binary_strided(
                    f32_values(&rows_storage),
                    f32_values(&columns_storage),
                    &broadcast_plan,
                    &|lhs, rhs| lhs + rhs,
                );
                Ok(Some(CpuStorage::from_contiguous(
                    CpuBuffer::F32(values),
                    vec![rows, columns],
                )))
            },
        );
    }

    fn benchmark_case(
        layout: &str,
        elements: usize,
        iterations: usize,
        mut operation: impl FnMut() -> Result<Option<CpuStorage>>,
    ) {
        for _ in 0..5 {
            black_box(operation().unwrap().unwrap());
        }
        const SAMPLES: usize = 7;
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(operation().unwrap().unwrap());
            }
            samples.push(started.elapsed().as_secs_f64());
        }
        samples.sort_by(f64::total_cmp);
        let elapsed = samples[SAMPLES / 2];
        let ns_per_element = elapsed * 1e9 / (elements * iterations) as f64;
        let bytes = (elements * iterations * 3 * size_of::<f32>()) as f64;
        let effective_gib_s = bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{},{layout},f32,{elements},{iterations},{SAMPLES},{ns_per_element:.4},{effective_gib_s:.3}",
            selected_execution(layout, elements)
        );
    }

    fn selected_execution(layout: &str, elements: usize) -> &'static str {
        if layout.ends_with("_rayon_reference") {
            return "rayon_autovec";
        }
        if layout == "dense_broadcast" {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            if avx2_f32_available() {
                return if elements >= PARALLEL_GRAIN {
                    "rayon_avx2_broadcast"
                } else {
                    "avx2_broadcast"
                };
            }
            return if elements >= PARALLEL_GRAIN {
                "rayon_iterator"
            } else {
                "serial_odometer"
            };
        }
        if layout == "dense_broadcast_odometer_reference" {
            return if elements >= PARALLEL_GRAIN {
                "rayon_iterator"
            } else {
                "serial_odometer"
            };
        }
        if elements >= DENSE_PARALLEL_GRAIN {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            if avx2_f32_available() {
                return "rayon_avx2";
            }
            return "rayon";
        }
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        if avx2_f32_available() {
            return "avx2";
        }
        "scalar"
    }

    fn f32_values(storage: &CpuStorage) -> &[f32] {
        match &*storage.buffer {
            CpuBuffer::F32(values) => values,
            _ => unreachable!("benchmark storage is F32"),
        }
    }
}
