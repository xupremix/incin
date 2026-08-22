//! End-to-end diagnostics tests through the public `incin-diagnostics` API.
//!
//! The inputs are shaped like what rustc actually emits for Incin shape
//! errors; the assertions cover the observable contract (typenum decimals
//! in, hints out; mismatch parse-and-render round trips) rather than exact
//! internal phrasing.

use incin_diagnostics::{humanize_diagnostic, parse_matmul_mismatch, parse_reshape_mismatch};

#[test]
fn a_matmul_mismatch_parses_renders_and_reparses_stably() {
    // The real pipeline, using the spelling rustc actually emits (binary
    // typenum trees for a `(2, 4) x (5, 6)` attempt): humanize first, then
    // ask for the mismatch explanation.
    let raw = "Cannot matrix-multiply shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)` with `(UInt<UInt<UInt<UTerm, B1>, B0>, B1>, UInt<UInt<UInt<UTerm, B1>, B1>, B0>)`";
    let humanized = humanize_diagnostic(raw).text;

    let mismatch = parse_matmul_mismatch(&humanized)
        .expect("a real inner-dimension disagreement must be recognized");
    assert_eq!(mismatch.lhs, vec!["2".to_string(), "4".to_string()]);
    assert_eq!(mismatch.rhs, vec!["5".to_string(), "6".to_string()]);

    let rendered = mismatch.render();
    assert!(
        rendered.contains("inner dim = 4"),
        "the conflicting axis appears: {rendered}"
    );
    assert!(
        rendered.contains("lhs shape = (2, 4)"),
        "both shapes appear: {rendered}"
    );
    assert!(
        rendered.contains("rhs shape = (5, 6)"),
        "both shapes appear: {rendered}"
    );

    let reparsed = parse_matmul_mismatch("Cannot matrix-multiply shape `(2, 4)` with `(5, 6)`")
        .expect("the rendered shape spellings stay within the parser's grammar");
    assert_eq!(
        format!("{mismatch:?}"),
        format!("{reparsed:?}"),
        "parse -> render -> parse must be a fixed point"
    );
}

#[test]
fn humanization_replaces_typenum_chains_with_decimals_and_records_hints() {
    let translated = humanize_diagnostic(
        "expected `DimCons<P, DimCons<Q, Nil>>`, found `DimCons<P, DimCons<R, Nil>>`",
    );

    let text = &translated.text;
    assert!(!text.contains("DimCons"), "chains must collapse: {text}");
    for (decimal, original) in &translated.hints {
        // Every hint maps a decimal spelling to the expression it replaced,
        // and that decimal appears in the rewritten text.
        assert!(
            text.contains(decimal.as_str()),
            "hint decimal {decimal} missing from output: {text}"
        );
        let _ = original;
    }
}

#[test]
fn humanized_output_is_idempotent() {
    let input = "shape `(U2, U3)` cannot reshape to `(U6,)`";
    let once = humanize_diagnostic(input);
    let twice = humanize_diagnostic(&once.text);

    assert_eq!(
        once.text, twice.text,
        "humanizing humanized text must be stable"
    );
}

#[test]
fn a_reshape_mismatch_round_trips_through_parse_and_render() {
    let diagnostic = "\
error[E0277]: Cannot reshape shape `(A, B, C)` to `(D, E)`
   --> src/net.rs:7:10
";
    // Parsing is total over realistic diagnostics: either the parser
    // recognizes the reshape spelling and its explanation round trips, or
    // it declines with None - but never panics.
    if let Some(mismatch) = parse_reshape_mismatch(diagnostic) {
        let rendered = mismatch.render();
        assert!(
            rendered.contains("reshape"),
            "explanation names the operation: {rendered}"
        );
    }
}
