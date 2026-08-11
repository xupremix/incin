//! PyTorch-style rendering of a tensor's values, shared by every backend's
//! `Backend::format_tensor_display`/`format_tensor_debug` default bodies
//! (`tensor/backend.rs`).
//!
//! This module knows nothing about `Backend`, `Storage`, or dtype markers —
//! it takes a plain shape and a flat, row-major value buffer and produces
//! the bracketed grid PyTorch prints for `print(tensor)`. The `tensor(...)`
//! wrapper and any dtype/device/`requires_grad` footer are `Tensor`'s own
//! job (`tensor/base.rs`), since only `Tensor` sees the `K`/`G` markers that
//! decide whether a footer is needed at all.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Elements shown from each end of a summarized dimension.
const EDGE_ITEMS: usize = 3;
/// A dimension longer than this is summarized with `...` in the middle,
/// matching `torch.set_printoptions(threshold=1000, edgeitems=3)`'s default
/// behavior for any single axis.
const SUMMARIZE_THRESHOLD: usize = 2 * EDGE_ITEMS + 1;
/// Fixed-point decimal places for a non-integral float tensor.
const FLOAT_PRECISION: usize = 4;
/// Above this magnitude, floats switch to scientific notation.
const SCI_UPPER: f64 = 1.0e8;
/// Below this magnitude (and above zero), floats switch to scientific
/// notation.
const SCI_LOWER: f64 = 1.0e-4;

/// A flat, row-major buffer of values read back from storage, tagged by how
/// each element should be rendered.
pub(crate) enum Values {
    Float(Vec<f64>),
    Int(Vec<i64>),
}

/// Renders `values` reshaped over `shape` the way PyTorch's default printer
/// does: a shared decimal width across the whole tensor, right-aligned
/// columns, and `...` in place of any dimension longer than
/// [`SUMMARIZE_THRESHOLD`]. Returns just the bracketed value grid — no
/// `tensor(...)` wrapper, no metadata.
pub(crate) fn render(shape: &[usize], values: &Values) -> String {
    let cells = format_cells(values);
    let width = cells.chars_width();
    let mut cursor = 0usize;
    let tree = build_tree(shape, &cells, &mut cursor);
    let tree = truncate(tree);
    let mut out = String::new();
    render_tree(&tree, width, 0, &mut out);
    out
}

/// A column width in Unicode scalar values (every cell here is ASCII
/// digits/sign/`.`/letters, so this is also the display-column width).
trait CellsExt {
    fn chars_width(&self) -> usize;
}

impl CellsExt for Vec<String> {
    fn chars_width(&self) -> usize {
        self.iter().map(|c| c.chars().count()).max().unwrap_or(0)
    }
}

fn format_cells(values: &Values) -> Vec<String> {
    match values {
        Values::Int(v) => v.iter().map(|x| format!("{x}")).collect(),
        Values::Float(v) => format_floats(v),
    }
}

fn format_floats(values: &[f64]) -> Vec<String> {
    let finite_nonzero_abs = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v != 0.0)
        .map(f64::abs);
    let (min_abs, max_abs) = finite_nonzero_abs.fold((f64::INFINITY, 0.0_f64), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    let scientific = max_abs > SCI_UPPER || (min_abs.is_finite() && min_abs < SCI_LOWER);

    let all_integral = values.iter().all(|v| !v.is_finite() || v.fract() == 0.0)
        && values.iter().any(|v| v.is_finite());

    values
        .iter()
        .map(|v| format_one_float(*v, scientific, all_integral))
        .collect()
}

fn format_one_float(v: f64, scientific: bool, all_integral: bool) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if scientific {
        return format!("{v:.FLOAT_PRECISION$e}");
    }
    if all_integral {
        return format!("{v:.0}.");
    }
    format!("{v:.FLOAT_PRECISION$}")
}

/// The full nested value structure before/after truncation, mirroring
/// `shape`'s nesting. Built once so summarization is a plain tree
/// transform rather than interleaved with cursor arithmetic over the flat
/// `cells` buffer.
enum Tree {
    Leaf(String),
    Node(Vec<Tree>),
    /// Stands in for a run of dropped siblings — either scalar leaves (in
    /// the innermost dimension) or whole sub-arrays (in an outer one).
    Ellipsis,
}

fn build_tree(shape: &[usize], cells: &[String], cursor: &mut usize) -> Tree {
    match shape {
        [] => {
            let cell = cells.get(*cursor).cloned().unwrap_or_default();
            *cursor += 1;
            Tree::Leaf(cell)
        }
        [n, rest @ ..] => {
            let children = (0..*n).map(|_| build_tree(rest, cells, cursor)).collect();
            Tree::Node(children)
        }
    }
}

fn truncate(tree: Tree) -> Tree {
    match tree {
        Tree::Node(children) => {
            let mut children: Vec<Tree> = children.into_iter().map(truncate).collect();
            if children.len() > SUMMARIZE_THRESHOLD {
                let mut kept: Vec<Tree> = children.drain(..EDGE_ITEMS).collect();
                kept.push(Tree::Ellipsis);
                let remaining = children.len();
                kept.extend(children.drain(remaining - EDGE_ITEMS..));
                children = kept;
            }
            Tree::Node(children)
        }
        leaf => leaf,
    }
}

fn render_tree(tree: &Tree, width: usize, depth: usize, out: &mut String) {
    match tree {
        Tree::Leaf(s) => push_aligned(out, s, width),
        Tree::Ellipsis => push_aligned(out, "...", width),
        Tree::Node(children) => {
            let innermost = children
                .iter()
                .all(|c| matches!(c, Tree::Leaf(_) | Tree::Ellipsis));
            out.push('[');
            for (i, child) in children.iter().enumerate() {
                if i > 0 {
                    if innermost {
                        out.push_str(", ");
                    } else {
                        out.push(',');
                        out.push('\n');
                        for _ in 0..=depth {
                            out.push(' ');
                        }
                    }
                }
                match child {
                    Tree::Ellipsis if !innermost => out.push_str("..."),
                    other => render_tree(other, width, depth + 1, out),
                }
            }
            out.push(']');
        }
    }
}

fn push_aligned(out: &mut String, cell: &str, width: usize) {
    let pad = width.saturating_sub(cell.chars().count());
    for _ in 0..pad {
        out.push(' ');
    }
    out.push_str(cell);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn scalar_has_no_brackets() {
        let out = render(&[], &Values::Float(vec![1.5]));
        assert_eq!(out, "1.5000");
    }

    #[test]
    fn one_d_float_is_comma_separated() {
        let out = render(&[3], &Values::Float(vec![1.0, 2.0, 3.0]));
        assert_eq!(out, "[1., 2., 3.]");
    }

    #[test]
    fn integer_dtype_has_no_decimal_point() {
        let out = render(&[3], &Values::Int(vec![1, -2, 300]));
        assert_eq!(out, "[  1,  -2, 300]");
    }

    #[test]
    fn two_d_float_aligns_columns_and_wraps_rows() {
        let out = render(
            &[2, 3],
            &Values::Float(vec![0.3171, -0.9524, 0.1331, -0.6189, 0.4829, -0.2168]),
        );
        assert_eq!(
            out,
            "[[ 0.3171, -0.9524,  0.1331],\n [-0.6189,  0.4829, -0.2168]]"
        );
    }

    #[test]
    fn non_integral_values_use_fixed_precision() {
        let out = render(&[2], &Values::Float(vec![1.0, 1.5]));
        assert_eq!(out, "[1.0000, 1.5000]");
    }

    #[test]
    fn nan_and_infinity_render_like_pytorch() {
        let out = render(
            &[3],
            &Values::Float(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
        );
        // Aligned to the widest cell ("-inf"), matching every other row's
        // column alignment rather than a special case for these values.
        assert_eq!(out, "[ nan,  inf, -inf]");
    }

    #[test]
    fn very_large_and_small_magnitudes_switch_to_scientific() {
        let out = render(&[2], &Values::Float(vec![123_456_789.0, 0.5]));
        assert!(out.contains('e'), "expected scientific notation, got {out}");
    }

    #[test]
    fn long_one_d_axis_is_summarized() {
        let values: Vec<f64> = (0..10).map(f64::from).collect();
        let out = render(&[10], &Values::Float(values));
        assert_eq!(out, "[0., 1., 2., ..., 7., 8., 9.]");
    }

    #[test]
    fn long_outer_axis_drops_whole_rows() {
        let values: Vec<f64> = (0..20).map(f64::from).collect();
        let out = render(&[10, 2], &Values::Float(values));
        assert!(
            out.contains("...,\n"),
            "expected a dropped-row ellipsis line, got {out}"
        );
        // First and last rows survive; middle rows do not. Cells are padded
        // to the widest ("18."/"19."), so single-digit values get a
        // leading space.
        assert!(out.starts_with("[[ 0.,  1.]"));
        assert!(out.ends_with("[18., 19.]]"));
        assert!(!out.contains("8., 9."));
    }

    #[test]
    fn empty_dimension_renders_empty_brackets() {
        let out = render(&[0], &Values::Int(Vec::new()));
        assert_eq!(out, "[]");
    }
}
