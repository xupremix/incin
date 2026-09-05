// The crate is `no_std` without the `std` feature, so `vec!` is not in the
// prelude for either configuration of this module. Importing it from
// `alloc` is what makes the tests compile under both.
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use super::*;

#[test]
fn test_parse_single_typenum() {
    assert_eq!(parse_single_typenum("UTerm"), Some(0));
    assert_eq!(parse_single_typenum("UInt<UTerm, B1>"), Some(1));
    assert_eq!(parse_single_typenum("UInt<UInt<UTerm, B1>, B0>"), Some(2));
    assert_eq!(parse_single_typenum("UInt<UInt<UTerm, B1>, B1>"), Some(3));
    assert_eq!(
        parse_single_typenum("UInt<UInt<UInt<UTerm, B1>, B0>, B0>"),
        Some(4)
    );
}

#[test]
fn test_translate_typenum_text_with_hints() {
    let input = "Cannot concatenate shape (UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)";
    let (translated, hints) = translate_typenum_text(input);
    assert_eq!(translated, "Cannot concatenate shape (2, 3)");
    assert_eq!(hints.len(), 2);
    assert_eq!(
        hints[0],
        ("2".to_string(), "UInt<UInt<UTerm, B1>, B0>".to_string())
    );
    assert_eq!(
        hints[1],
        ("3".to_string(), "UInt<UInt<UTerm, B1>, B1>".to_string())
    );
}

/// A smaller nested expression's translation (e.g. `2`) must not
/// clobber the same literal substring where it recurs inside a
/// larger, separately-translated expression later in the text -
/// regression test for a real corrupted-diagnostic bug found while
/// reviewing this file.
#[test]
fn test_translate_typenum_text_does_not_corrupt_nested_reoccurring_expressions() {
    let input = "expected `UInt<UInt<UInt<UTerm, B1>, B1>, B0>`, found `UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>`";
    let (translated, _hints) = translate_typenum_text(input);
    assert_eq!(translated, "expected `6`, found `8`");
}

#[test]
fn test_humanize_diagnostic_wraps_translate_typenum_text() {
    let input = "UInt<UInt<UTerm, B1>, B0>";
    let translated = humanize_diagnostic(input);
    assert_eq!(translated.text, "2");
    assert_eq!(translated.hints, vec![("2".to_string(), input.to_string())]);
}

#[test]
fn test_humanize_inlay_label_keeps_backend_by_default() {
    let label =
        "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>";
    assert_eq!(
        humanize_inlay_label(label, false),
        "Tensor<[2, 3], CpuBackendImpl<Cpu>>"
    );
}

#[test]
fn test_humanize_inlay_label_shortens_backend_when_requested() {
    let label =
        "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>";
    assert_eq!(humanize_inlay_label(label, true), "Tensor<[2, 3]>");
}

/// Realistic case: all four `Tensor<S, B, K, G>` type params resolved
/// and shown (rust-analyzer often renders defaulted params explicitly),
/// with nested `<...>` inside the backend param - the balanced-bracket
/// scan must not stop at the first `>` it sees.
#[test]
fn test_humanize_inlay_label_handles_nested_angle_brackets_in_backend_param() {
    let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>), CpuBackendImpl<Cpu>, f32, Grad>";
    assert_eq!(
        humanize_inlay_label(label, false),
        "Tensor<[2, 4], CpuBackendImpl<Cpu>, f32, Grad>"
    );
    assert_eq!(humanize_inlay_label(label, true), "Tensor<[2, 4]>");
}

/// Regression test: Rust's 1-tuple syntax (`(U8,)`) leaves a trailing
/// comma inside the parens that must not end up inside the `[...]`.
#[test]
fn test_humanize_inlay_label_strips_trailing_comma_on_rank_one_shape() {
    let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>,), CpuBackendImpl<Cpu>>";
    assert_eq!(
        humanize_inlay_label(label, false),
        "Tensor<[2], CpuBackendImpl<Cpu>>"
    );
    assert_eq!(humanize_inlay_label(label, true), "Tensor<[2]>");
}

/// Regression test for a real, reported case: a `let` binding of a
/// layer/module (not a bare `Tensor`) shows its full inferred type,
/// which is just as typenum-heavy but has no `Tensor<(...` shell -
/// previously passed through completely raw.
#[test]
fn test_humanize_inlay_label_rewrites_non_tensor_struct_types_generically() {
    let label = "Conv2d<(usize, usize, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>), UInt<UTerm, B1>>, CpuBackendImpl>";
    assert_eq!(
        humanize_inlay_label(label, false),
        "Conv2d<(usize, usize, 3, 3, 3), 1>, CpuBackendImpl>"
    );
}

#[test]
fn test_humanize_inlay_label_handles_compound_signatures_with_multiple_tensors() {
    let label = "(Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>, Tensor<(UInt<UInt<UInt<UTerm, B1>, B0>, B0>,), CpuBackendImpl<Cpu>>)";
    assert_eq!(
        humanize_inlay_label(label, false),
        "(Tensor<[2, 3], CpuBackendImpl<Cpu>>, Tensor<[4], CpuBackendImpl<Cpu>>)"
    );
    assert_eq!(
        humanize_inlay_label(label, true),
        "(Tensor<[2, 3]>, Tensor<[4]>)"
    );
}

#[test]
fn test_humanize_inlay_label_handles_colon_prefix_from_rust_analyzer() {
    let label =
        ": Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>";
    assert_eq!(
        humanize_inlay_label(label, false),
        ": Tensor<[2, 3], CpuBackendImpl<Cpu>>"
    );
    assert_eq!(humanize_inlay_label(label, true), ": Tensor<[2, 3]>");
}

#[test]
fn test_strip_path_qualifiers_collapses_qualified_paths() {
    assert_eq!(strip_path_qualifiers("typenum::B1"), "B1");
    assert_eq!(strip_path_qualifiers("incin::cpu::Tensor"), "Tensor");
    assert_eq!(strip_path_qualifiers("f32"), "f32");
    assert_eq!(
        strip_path_qualifiers(
            "incin::cpu::Tensor<(typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B1>,), incin::cpu::CpuBackendImpl, f32>"
        ),
        "Tensor<(UInt<UInt<UTerm, B1>, B1>,), CpuBackendImpl, f32>"
    );
}

/// Regression test for the real bug this was written to fix: rust-analyzer
/// truncates deeply-nested inlay-hint `label`s with a `…` ellipsis once
/// they exceed its default nesting depth, which loses the B0/B1 bits
/// entirely - no rewrite of `label` can recover data that was never sent.
/// The full type only survives in `textEdits[0].newText`, fully
/// path-qualified; `strip_path_qualifiers` normalizes that back into the
/// same shape `humanize_inlay_label` already parses.
#[test]
fn test_humanize_inlay_label_recovers_truncated_hint_via_stripped_text_edit() {
    let full_text_edit = "incin::cpu::Tensor<(typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B1>, typenum::UInt<typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B0>, typenum::B1>), incin::cpu::CpuBackendImpl, f32>";
    let normalized = strip_path_qualifiers(full_text_edit);
    assert_eq!(
        humanize_inlay_label(&normalized, false),
        "Tensor<[3, 5], CpuBackendImpl, f32>"
    );
    assert_eq!(humanize_inlay_label(&normalized, true), "Tensor<[3, 5]>");
}

#[test]
fn test_humanize_inlay_label_passes_through_non_tensor_labels_unchanged() {
    assert_eq!(humanize_inlay_label("i32", false), "i32");
    assert_eq!(
        humanize_inlay_label("Tensor<Dyn, CpuBackendImpl<Cpu>>", false),
        "Tensor<Dyn, CpuBackendImpl<Cpu>>"
    );
}

/// Regression test for a real, reported request: `cargo incin --explain`
/// only printed a generic static rule for matmul errors. This verifies
/// the parser pulls out the actual conflicting values from the trait's
/// fixed `on_unimplemented` message - the exact scenario requested:
/// multiplying a `(2, 4)` shape by a `(5, 6)` shape.
#[test]
fn test_parse_matmul_mismatch_extracts_conflicting_inner_dims() {
    let text = "Cannot matrix-multiply shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)` with `(UInt<UInt<UInt<UTerm, B1>, B0>, B1>, UInt<UInt<UInt<UTerm, B1>, B1>, B0>)`";
    let mismatch = parse_matmul_mismatch(text).unwrap();
    assert_eq!(mismatch.lhs, vec!["2".to_string(), "4".to_string()]);
    assert_eq!(mismatch.rhs, vec!["5".to_string(), "6".to_string()]);
    assert_eq!(mismatch.lhs_inner_index, 1);
    assert_eq!(mismatch.rhs_inner_index, 0);
}

#[test]
fn test_matmul_mismatch_render_points_at_conflicting_dims_and_suggests_fix() {
    let mismatch = MatMulMismatch {
        lhs: vec!["2".to_string(), "4".to_string()],
        rhs: vec!["5".to_string(), "6".to_string()],
        lhs_inner_index: 1,
        rhs_inner_index: 0,
    };
    let rendered = mismatch.render();
    let lines: Vec<&str> = rendered.lines().collect();
    assert_eq!(lines.len(), 5);
    assert_eq!(lines[0], "      lhs shape = (2, 4)");
    assert_eq!(lines[2], "      rhs shape = (5, 6)");

    // The `^` marker must sit directly under the first character of the
    // conflicting element on the shape line above it, not just "close".
    let lhs_col = lines[0].find('4').unwrap();
    let rhs_col = lines[2].find('5').unwrap();
    assert_eq!(lines[1].find('^').unwrap(), lhs_col);
    assert_eq!(lines[3].find('^').unwrap(), rhs_col);
    assert_eq!(lines[1].trim(), "^ inner dim = 4");
    assert_eq!(lines[3].trim(), "^ inner dim = 5");

    assert_eq!(
        lines[4],
        "      4 \u{2260} 5 \u{2192} change the lhs inner dim from 4 to 5, or the rhs inner dim from 5 to 4."
    );
}

#[test]
fn test_parse_matmul_mismatch_returns_none_when_inner_dims_already_match() {
    // lhs=(2,3), rhs=(3,4): inner dims are both 3, so this text isn't
    // actually describing an inner-dim mismatch - some other rule must
    // have failed, and inventing a bogus "3 != 3" explanation would be
    // actively misleading.
    let text = "Cannot matrix-multiply shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)` with `(UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)`";
    assert!(parse_matmul_mismatch(text).is_none());
}

#[test]
fn test_parse_matmul_mismatch_returns_none_for_unrelated_text() {
    assert!(parse_matmul_mismatch("some other diagnostic entirely").is_none());
}

/// Regression test for a real bug: `cargo incin --explain` passes the
/// *entire* rendered rustc diagnostic ("error[E0277]: Cannot
/// matrix-multiply shape `...` with `...`\n   --> file:line:col\n...",
/// with `help`/`note` lines following), not just the message in
/// isolation - an earlier version used `strip_prefix`, which silently
/// returned `None` for anything but the bare message on its own.
#[test]
fn test_parse_matmul_mismatch_finds_message_embedded_in_full_rendered_diagnostic() {
    let rendered = "error[E0277]: Cannot matrix-multiply shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)` with `(UInt<UInt<UInt<UTerm, B1>, B0>, B1>, UInt<UInt<UInt<UTerm, B1>, B1>, B0>)`\n   --> src/main.rs:19:24\n    |\n = help: the trait `MatMulShape<(...)>` is not implemented for `(...)`\n";
    let mismatch = parse_matmul_mismatch(rendered).unwrap();
    assert_eq!(mismatch.lhs, vec!["2".to_string(), "4".to_string()]);
    assert_eq!(mismatch.rhs, vec!["5".to_string(), "6".to_string()]);
}

#[test]
fn test_parse_matmul_mismatch_handles_batched_lhs_shape() {
    let text = "Cannot matrix-multiply shape `(usize, UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)` with `(UInt<UInt<UInt<UTerm, B1>, B0>, B1>, UInt<UInt<UInt<UTerm, B1>, B1>, B0>)`";
    let mismatch = parse_matmul_mismatch(text).unwrap();
    assert_eq!(
        mismatch.lhs,
        vec!["usize".to_string(), "2".to_string(), "4".to_string()]
    );
    assert_eq!(mismatch.lhs_inner_index, 2);
    assert_eq!(mismatch.rhs_inner_index, 0);
}

#[test]
fn test_parse_concat_mismatch() {
    let text = "Cannot concatenate shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` with `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)` along axis `0`";
    let mismatch = parse_concat_mismatch(text).unwrap();
    assert_eq!(mismatch.lhs, vec!["2".to_string(), "2".to_string()]);
    assert_eq!(mismatch.rhs, vec!["2".to_string(), "3".to_string()]);
    assert_eq!(mismatch.axis, 0);
    assert_eq!(mismatch.mismatch_index, 1);
    let rendered = mismatch.render();
    assert!(rendered.contains("2 \u{2260} 3"));
}

#[test]
fn test_parse_broadcast_mismatch() {
    let text = "Cannot broadcast shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` to `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)`";
    let mismatch = parse_broadcast_mismatch(text).unwrap();
    assert_eq!(mismatch.lhs, vec!["2".to_string(), "2".to_string()]);
    assert_eq!(mismatch.rhs, vec!["2".to_string(), "3".to_string()]);
    let rendered = mismatch.render();
    assert!(rendered.contains("2 \u{2260} 3"));
}

#[test]
fn test_parse_reshape_mismatch() {
    let text = "Cannot reshape from `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` to `(UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>)`";
    let mismatch = parse_reshape_mismatch(text).unwrap();
    assert_eq!(mismatch.src_count, 4);
    assert_eq!(mismatch.target_count, 9);
    let rendered = mismatch.render();
    assert!(rendered.contains("4 \u{2260} 9"));
}

#[test]
fn test_parse_conv2d_mismatch() {
    let text = "Cannot apply Conv2D: input shape `(UInt<UTerm, B1>, UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` is incompatible with kernel shape `(UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>)`";
    let mismatch = parse_conv2d_mismatch(text).unwrap();
    assert_eq!(mismatch.input[1], "2");
    assert_eq!(mismatch.kernel[1], "3");
    let rendered = mismatch.render();
    assert!(rendered.contains("2 \u{2260} 3"));
}

#[test]
fn test_parse_transpose_mismatch() {
    let text = "Cannot transpose dimensions `2` and `3` on shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)`";
    let mismatch = parse_transpose_mismatch(text).unwrap();
    assert_eq!(mismatch.rank, 2);
    assert_eq!(mismatch.d1, 2);
    assert_eq!(mismatch.d2, 3);
    let rendered = mismatch.render();
    assert!(rendered.contains("must be < rank (2)"));
}

#[test]
fn test_parse_reduce_dim_mismatch() {
    let text = "Cannot reduce dimension `3` on shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)`";
    let mismatch = parse_reduce_dim_mismatch(text).unwrap();
    assert_eq!(mismatch.rank, 2);
    assert_eq!(mismatch.dim, 3);
    let rendered = mismatch.render();
    assert!(rendered.contains("3 \u{2265} 2"));
}

#[test]
fn test_parse_flatten_mismatch() {
    let text = "Cannot flatten shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)` from dimension `1` to `3`";
    let mismatch = parse_flatten_mismatch(text).unwrap();
    assert_eq!(mismatch.rank, 2);
    assert_eq!(mismatch.start_dim, 1);
    assert_eq!(mismatch.end_dim, 3);
    let rendered = mismatch.render();
    assert!(rendered.contains("invalid range"));
}

#[test]
fn test_parse_module_forward_mismatch() {
    let text = "the trait bound `MyLayer<Cpu>: Module<Tensor<(UInt<UTerm, B1>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>), Cpu>>` is not satisfied\nthe following other types implement trait `Module<Input>`:\n  `MyLayer<Cpu>` implements `Module<Tensor<(UInt<UTerm, B1>, UInt<UInt<UInt<UTerm, B1>, B0>, B1>), Cpu>>`";
    let mismatch = parse_module_forward_mismatch(text).unwrap();
    assert_eq!(mismatch.actual_input, "Tensor<(1, 4), Cpu>");
    assert_eq!(mismatch.expected_input, "Tensor<(1, 5), Cpu>");
    let rendered = mismatch.render();
    assert!(rendered.contains("provided input shape = Tensor<(1, 4), Cpu>"));
}

#[test]
fn test_parse_slice_mismatch() {
    let text =
        "Cannot slice dimension with `Slice<U1, U10, U9>` for shape `(UInt<UInt<UTerm, B1>, B0>,)`";
    let mismatch = parse_slice_mismatch(text).unwrap();
    assert_eq!(mismatch.in_shape, "(2,)");
    let rendered = mismatch.render();
    assert!(rendered.contains("invalid slice"));
}

#[test]
fn test_parse_conv1d_mismatch() {
    let text = "Cannot apply 1D convolution to shape `(UInt<UInt<UTerm, B1>, B0>,)`";
    let mismatch = parse_conv1d_mismatch(text).unwrap();
    assert_eq!(mismatch.input_shape, "(2,)");
    let rendered = mismatch.render();
    assert!(rendered.contains("Conv1D requires a 2D or 3D tensor"));
}

#[test]
fn test_parse_pool2d_mismatch() {
    let text = "Cannot apply 2D pooling to shape `(UInt<UInt<UTerm, B1>, B0>,)`";
    let mismatch = parse_pool2d_mismatch(text).unwrap();
    assert_eq!(mismatch.input_shape, "(2,)");
    let rendered = mismatch.render();
    assert!(rendered.contains("Pool2D requires a 3D or 4D tensor"));
}

#[test]
fn test_parse_shape_eq_mismatch() {
    let text = "evaluation of constant value failed\nShape Mismatch: Attempted to operate on tensors of incompatible shapes.";
    let mismatch = parse_shape_eq_mismatch(text).unwrap();
    assert!(
        mismatch
            .message
            .contains("Attempted to operate on tensors of incompatible shapes")
    );
    let rendered = mismatch.render();
    assert!(rendered.contains("shape mismatch"));
}

#[test]
fn test_parse_medium_prio_diagnostics() {
    let text_bmm = "bmm error";
    assert!(parse_bmm_mismatch(text_bmm).is_some());

    let text_unfold = "unfold size cannot exceed dimension length";
    assert!(parse_unfold_mismatch(text_unfold).is_some());

    let text_pixel = "pixel_shuffle channels must be divisible";
    assert!(parse_pixel_shuffle_mismatch(text_pixel).is_some());

    let text_group = "group_norm: channels must be divisible by groups";
    assert!(parse_group_norm_mismatch(text_group).is_some());

    let text_domain = "out of domain error";
    assert!(parse_math_domain_error(text_domain).is_some());
}

#[test]
#[cfg(feature = "std")]
fn test_expand_type_file_notes_reads_and_humanizes_file() {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("test_type_file_notes.txt");
    let sample_type =
        "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>";
    std::fs::write(&temp_file, sample_type).unwrap();

    let diagnostic = format!(
        "note: the full type name was written to '{}'",
        temp_file.display()
    );

    let translated = humanize_diagnostic(&diagnostic);
    assert!(translated.text.contains("📄 [Expanded Full Type]"));
    assert!(
        translated
            .text
            .contains("Tensor<[2, 3], CpuBackendImpl<Cpu>>")
    );
    assert!(
        translated
            .hints
            .contains(&("2".to_string(), "UInt<UInt<UTerm, B1>, B0>".to_string()))
    );
    assert!(
        translated
            .hints
            .contains(&("3".to_string(), "UInt<UInt<UTerm, B1>, B1>".to_string()))
    );

    let _ = std::fs::remove_file(temp_file);
}

#[test]
fn test_collapse_dimcons_chains_bare_pair() {
    assert_eq!(
        collapse_dimcons_chains("DimCons<4, DimCons<8, Nil>>"),
        "[4, 8]"
    );
}

#[test]
fn test_collapse_dimcons_chains_single_element() {
    assert_eq!(collapse_dimcons_chains("DimCons<4, Nil>"), "[4]");
}

#[test]
fn test_collapse_dimcons_chains_rank_three() {
    assert_eq!(
        collapse_dimcons_chains("DimCons<2, DimCons<3, DimCons<4, Nil>>>"),
        "[2, 3, 4]"
    );
}

/// The real shape it was written for: a `MatMulShape` trait-bound error,
/// where the `DimCons` chain appears bare rather than wrapped in
/// `Tensor<(...)>` and so never reaches `humanize_type_signature`'s
/// tuple special case at all.
#[test]
fn test_collapse_dimcons_chains_in_a_real_matmul_mismatch() {
    let input = "Cannot contract dimension `DimCons<4, DimCons<8, Nil>>` with \
                  `MatMulShape<DimCons<3, DimCons<8, Nil>>>`";
    assert_eq!(
        collapse_dimcons_chains(input),
        "Cannot contract dimension `[4, 8]` with `MatMulShape<[3, 8]>`"
    );
}

/// A head that is itself a bracketed type (a named dimension) must not
/// be split on its own internal comma.
#[test]
fn test_collapse_dimcons_chains_head_with_internal_comma() {
    assert_eq!(
        collapse_dimcons_chains("DimCons<NamedDim<Batch, 4>, DimCons<8, Nil>>"),
        "[NamedDim<Batch, 4>, 8]"
    );
}

/// Anything that isn't a clean cons list all the way to `Nil` is left
/// exactly as written rather than partially or incorrectly collapsed.
#[test]
fn test_collapse_dimcons_chains_leaves_unrecognized_shapes_untouched() {
    let input = "DimCons<4, SomeOtherTail>";
    assert_eq!(collapse_dimcons_chains(input), input);
}

#[test]
fn test_humanize_diagnostic_collapses_dimcons_after_typenum_translation() {
    let input = "Cannot contract dimension `DimCons<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, DimCons<UInt<UInt<UInt<UTerm, B1>, B0>, B1>, Nil>>` with something";
    let translated = humanize_diagnostic(input);
    assert!(
        translated.text.contains("[4, 5]"),
        "got: {}",
        translated.text
    );
}

// --- truncated-span substitution ---
//
// `replace_truncated_spans` had no coverage at all, which is how it shipped
// overwriting complete typenum expressions with an unrelated long type. The
// three below pin the rule it now follows: substitute a truncated span when
// exactly one candidate matches, and otherwise leave the span alone.

/// A complete typenum expression is not truncated and must survive untouched,
/// so the translator downstream can render it as the decimal it is.
///
/// This is the matmul regression. A `[2, 3]` against `[4, 5]` product reported
/// `Cannot contract dimension `MatMulShape<[4, 5]>` with `MatMulShape<[4, 5]>``
/// -- two different dimensions, three and four, both replaced by one unrelated
/// type, which then read as "X is not implemented for X".
#[test]
fn a_complete_typenum_span_is_not_treated_as_truncated() {
    let text = "Cannot contract dimension `UInt<UInt<UTerm, B1>, B1>` \
                with `UInt<UInt<UInt<UTerm, B1>, B0>, B0>`";
    let lines = alloc::vec![String::from("MatMulShape<[4, 5]>")];

    assert_eq!(replace_truncated_spans(text, &lines), text);
}

/// A span rustc actually truncated is substituted when its head names exactly
/// one of the full types read back from the long-type file.
#[test]
fn a_truncated_span_takes_the_one_full_type_that_shares_its_head() {
    let text = "required for `Foo` to implement `MatMulShape<DimCons<UInt<UTerm, B1>, ...>>`";
    let lines = alloc::vec![String::from("MatMulShape<[4, 5]>")];

    assert_eq!(
        replace_truncated_spans(text, &lines),
        "required for `Foo` to implement `MatMulShape<[4, 5]>`"
    );
}

/// Two candidates sharing a head is an ambiguous match, and an ambiguous match
/// is left alone. Picking one would read as authoritative while being a coin
/// flip, which is the failure the rotating fallback used to produce.
#[test]
fn an_ambiguous_truncated_span_is_left_exactly_as_it_came_in() {
    let text = "required for `Foo` to implement `MatMulShape<DimCons<UInt<UTerm, B1>, ...>>`";
    let lines = alloc::vec![
        String::from("MatMulShape<[4, 5]>"),
        String::from("MatMulShape<[2, 3]>"),
    ];

    assert_eq!(replace_truncated_spans(text, &lines), text);
}

/// A truncated span whose head names no full type at all is also left alone,
/// rather than borrowing whichever line happened to be next.
#[test]
fn an_unmatched_truncated_span_is_left_exactly_as_it_came_in() {
    let text = "required for `Foo` to implement `ConcatShape<DimCons<UInt<UTerm, B1>, ...>>`";
    let lines = alloc::vec![String::from("MatMulShape<[4, 5]>")];

    assert_eq!(replace_truncated_spans(text, &lines), text);
}

// --- the contraction message ---

/// `matmul` carries two `on_unimplemented` messages and rustc reports the
/// innermost failing bound, so it is `ContractsWith`'s that reaches a reader.
/// Keying the explanation only on `MatMulShape`'s is why `--explain` printed a
/// rule sentence and no diagram for every real matmul mismatch.
#[test]
fn the_contraction_message_yields_the_two_axes_that_disagree() {
    let text = "error[E0277]: Cannot contract dimension `3` with `4`\n  --> src/main.rs:6:22";

    let mismatch = parse_contraction_mismatch(text).expect("the message parses");
    assert_eq!(mismatch.lhs_inner, "3");
    assert_eq!(mismatch.rhs_inner, "4");

    let rendered = mismatch.render();
    assert!(rendered.contains("last axis of the left operand"));
    assert!(rendered.contains("second-to-last axis of the right operand"));
    assert!(rendered.contains("transpose"));
}

/// Equal axes mean the contraction was not what failed -- a rank disagreement
/// reaches the same bound -- so explaining one would misdirect.
#[test]
fn equal_axes_are_not_reported_as_a_contraction_mismatch() {
    let text = "Cannot contract dimension `3` with `3`";
    assert!(parse_contraction_mismatch(text).is_none());
}

// -- layout parameter elision --------------------------------------------

/// The default layout says nothing and should not be shown.
///
/// `Tensor` gained a sixth parameter that defaults to `Dyn`, meaning the
/// compiler settled nothing about where the elements live. Printing it costs a
/// reader attention and returns no information.
#[test]
fn a_default_layout_is_elided_from_a_tensor_type() {
    let translated = humanize_type_signature(
        "Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Local, incin_core::shapes::Dyn>",
        false,
    );
    assert_eq!(
        translated.text,
        "Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Local>"
    );
}

/// A layout that is not the default is a real claim and must survive.
///
/// Eliding it would misrepresent the type: `RowMajor` is the difference between
/// a tensor that can call `reshape_view` and one that cannot.
#[test]
fn a_proven_layout_is_never_elided() {
    let label = "Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Local, RowMajor>";
    assert_eq!(humanize_type_signature(label, false).text, label);
}

/// A `Dyn` shape must survive, even though the layout slot spells its default
/// the same way.
///
/// This is the property that the layout marker sharing `Dyn` puts at risk: an
/// unproven *layout* is noise, a dynamic *shape* is information. They are told
/// apart by position, so both `Dyn`s here are handled correctly in one pass --
/// the sixth is dropped, the first is kept.
#[test]
fn a_dynamic_shape_is_not_confused_with_a_default_layout() {
    let translated = humanize_type_signature(
        "Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Local, Dyn>",
        false,
    );
    assert_eq!(
        translated.text,
        "Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Local>"
    );
}

/// Only the outermost argument list is considered.
///
/// A nested type that happens to end in `Dyn` is not a layout and must be left
/// alone, whatever its depth.
#[test]
fn a_nested_dyn_is_not_mistaken_for_the_layout_slot() {
    let label = "Tensor<DimCons<A, Dyn>, CpuBackendImpl, f32, NoGrad, Local, RowMajor>";
    assert_eq!(humanize_type_signature(label, false).text, label);
}

/// A tensor written without the layout argument is unchanged.
#[test]
fn a_tensor_without_a_layout_argument_is_untouched() {
    let label = "Tensor<Dyn, CpuBackendImpl>";
    assert_eq!(humanize_type_signature(label, false).text, label);
}

/// A trailing `Dyn` in a list too short to reach the layout slot is kept.
///
/// This is the case the old name-based test could not express, because the
/// marker used to be a spelling no other slot could produce. With one shared
/// marker the arity is the only thing separating a layout from a placement, so
/// a five-argument list must fail closed rather than drop its last argument.
#[test]
fn a_trailing_dyn_outside_the_layout_slot_is_kept() {
    let label = "Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Dyn>";
    assert_eq!(humanize_type_signature(label, false).text, label);
}

/// An argument list rustc has abbreviated is left alone.
///
/// `...` collapses an unknown number of arguments, so the position of the
/// trailing one is no longer knowable and eliding it could hide a shape.
#[test]
fn an_abbreviated_argument_list_is_not_elided() {
    let label = "Tensor<Dyn, ..., Dyn>";
    assert_eq!(humanize_type_signature(label, false).text, label);
}
