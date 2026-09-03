//! CUDA embedding lookup and its gradient scatter.
//!
//! The previous version of this file was named for mixed-precision embeddings
//! and asserted that `f32`, `f64`, `f16` and `bf16` report `is_float()`. It
//! launched nothing and touched no embedding code. Its one green result counted
//! as coverage for an operation it never ran.
//!
//! Rewriting the two sibling suites that were vacuous in the same way found a
//! real defect in each: the CUDA optimizers never launched at all, and
//! `argmax`/`argmin` computed only their first output row.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_embedding -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::cuda::testing::{
    download_f32, embedding, embedding_backward, require_cuda, upload_f32_shaped, upload_i64,
};

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-5 * left.abs().max(right.abs()).max(1.0)
}

/// A 4-row vocabulary of width 3, each row distinguishable from the others.
fn weight() -> (Vec<usize>, Vec<f32>) {
    (
        vec![4, 3],
        vec![
            0.0, 0.1, 0.2, // row 0
            1.0, 1.1, 1.2, // row 1
            2.0, 2.1, 2.2, // row 2
            3.0, 3.1, 3.2, // row 3
        ],
    )
}

/// A lookup must copy the named rows, in the order named.
///
/// The indices are deliberately out of order and include a repeat, so a kernel
/// that returned the first rows of the table, or that dropped duplicates, would
/// be caught.
#[test]
#[ignore = "requires CUDA hardware"]
fn embedding_gathers_the_rows_its_indices_name() {
    require_cuda();
    let (shape, values) = weight();
    let w = upload_f32_shaped(&shape, &values);
    let ids = [2_i64, 0, 3, 2];
    let indices = upload_i64(&[ids.len()], &ids);

    let out = embedding(&w, &indices).expect("embedding must launch");
    assert_eq!(&out.shape[..], &[4, 3], "one row of width 3 per index");
    let got = download_f32(&out);

    for (position, id) in ids.iter().enumerate() {
        for column in 0..3usize {
            let expected = values[usize::try_from(*id).unwrap() * 3 + column];
            assert!(
                close(f64::from(got[position * 3 + column]), f64::from(expected)),
                "row {position} (index {id}) column {column}: kernel gave {}, table has {expected}",
                got[position * 3 + column]
            );
        }
    }
}

/// The gradient scatter must accumulate where an index repeats.
///
/// This is the property that makes embedding backward more than a permutation:
/// index 2 appears twice above, so its vocabulary row must receive the sum of
/// both contributions. A kernel that assigned rather than accumulated would
/// return only the last one, and a single-occurrence test could not tell.
#[test]
#[ignore = "requires CUDA hardware"]
fn embedding_backward_accumulates_repeated_indices() {
    require_cuda();
    let (vocab_size, hidden_size) = (4usize, 3usize);
    let ids = [2_i64, 0, 3, 2];
    let indices = upload_i64(&[ids.len()], &ids);

    // A distinct gradient per position, so contributions are distinguishable.
    let grad: Vec<f32> = (0..ids.len() * hidden_size)
        .map(|index| index as f32 + 1.0)
        .collect();
    let grad_out = upload_f32_shaped(&[ids.len(), hidden_size], &grad);

    let out = embedding_backward(&grad_out, &indices, vocab_size, hidden_size)
        .expect("embedding_backward must launch");
    assert_eq!(&out.shape[..], &[vocab_size, hidden_size]);
    let got = download_f32(&out);

    // Reference: scatter-add each position's gradient row into its index's row.
    let mut expected = vec![0.0_f64; vocab_size * hidden_size];
    for (position, id) in ids.iter().enumerate() {
        let row = usize::try_from(*id).unwrap();
        for column in 0..hidden_size {
            expected[row * hidden_size + column] +=
                f64::from(grad[position * hidden_size + column]);
        }
    }

    for index in 0..expected.len() {
        assert!(
            close(f64::from(got[index]), expected[index]),
            "grad at {index}: kernel gave {}, scatter-add gives {}",
            got[index],
            expected[index]
        );
    }

    // And specifically: row 2 received both contributions, not just one.
    let row2: Vec<f64> = (0..hidden_size)
        .map(|column| f64::from(got[2 * hidden_size + column]))
        .collect();
    let first = (0..hidden_size)
        .map(|column| f64::from(grad[column]))
        .collect::<Vec<_>>();
    assert!(
        row2.iter().zip(&first).all(|(total, one)| *total > *one),
        "row 2 is named twice; it must hold more than a single contribution, got {row2:?}"
    );
}

/// A vocabulary row that no index names must stay zero.
#[test]
#[ignore = "requires CUDA hardware"]
fn embedding_backward_leaves_unreferenced_rows_at_zero() {
    require_cuda();
    let (vocab_size, hidden_size) = (4usize, 3usize);
    // Row 1 is never named.
    let ids = [0_i64, 2, 3];
    let indices = upload_i64(&[ids.len()], &ids);
    let grad: Vec<f32> = vec![1.0; ids.len() * hidden_size];
    let grad_out = upload_f32_shaped(&[ids.len(), hidden_size], &grad);

    let got = download_f32(
        &embedding_backward(&grad_out, &indices, vocab_size, hidden_size)
            .expect("embedding_backward must launch"),
    );

    for column in 0..hidden_size {
        assert!(
            close(f64::from(got[hidden_size + column]), 0.0),
            "row 1 is unreferenced and must stay zero, got {}",
            got[hidden_size + column]
        );
    }
}
