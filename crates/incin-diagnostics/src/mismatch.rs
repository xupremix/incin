//! Structured parsers for the shape-mismatch diagnostics common to Incin
//! generic-const trait bounds (`MatMulShape`, `ConcatShape`, ...). Each
//! `*Mismatch` type mirrors one `#[diagnostic::on_unimplemented]` message
//! shape: parse the compiler's rendered dimensions out of the message text,
//! then offer a `Display`-friendly explanation of which dimension conflicts.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::humanize::humanize_diagnostic;
use crate::typenum::matching_bracket;

/// A `matmul` shape mismatch, parsed from the `MatMulShape` trait's
/// `#[diagnostic::on_unimplemented]` message, with the specific conflicting
/// dimension identified and ready to render as a pointed-out explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulMismatch {
    /// The lhs shape's elements, humanized for display (e.g. `["2", "4"]`).
    pub lhs: Vec<String>,
    /// The rhs shape's elements, humanized for display (e.g. `["5", "6"]`).
    pub rhs: Vec<String>,
    /// Index into `lhs` of the conflicting "inner" dimension (always the
    /// last element - matmul's `K` from the lhs side).
    pub lhs_inner_index: usize,
    /// Index into `rhs` of the conflicting "inner" dimension (the
    /// second-to-last element - matmul's `K` from the rhs side; usually
    /// index `0` for a plain `(K, N)`, but shifts right with batch dims).
    pub rhs_inner_index: usize,
}

impl MatMulMismatch {
    /// Renders the mismatch as a multi-line, ready-to-print explanation:
    /// both shapes shown with the conflicting dimension pointed out via a
    /// `^` marker underneath, plus a concrete suggested fix.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let lhs_inner = &self.lhs[self.lhs_inner_index];
        let rhs_inner = &self.rhs[self.rhs_inner_index];

        let lhs_label = format!("{INDENT}lhs shape = (");
        let rhs_label = format!("{INDENT}rhs shape = (");
        // `{:>N$}` right-aligns within a field of width `N`, so the marker
        // ends up at index `N - 1` - add 1 to land it exactly at the target
        // column (the offset of the element's first character).
        let lhs_caret_width =
            lhs_label.len() + joined_prefix_len(&self.lhs, self.lhs_inner_index) + 1;
        let rhs_caret_width =
            rhs_label.len() + joined_prefix_len(&self.rhs, self.rhs_inner_index) + 1;

        format!(
            "{lhs_label}{lhs_shape})\n\
             {lhs_caret:>lhs_caret_width$} inner dim = {lhs_inner}\n\
             {rhs_label}{rhs_shape})\n\
             {rhs_caret:>rhs_caret_width$} inner dim = {rhs_inner}\n\
             {INDENT}{lhs_inner} \u{2260} {rhs_inner} \u{2192} change the lhs inner dim from {lhs_inner} to {rhs_inner}, or the rhs inner dim from {rhs_inner} to {lhs_inner}.",
            lhs_shape = self.lhs.join(", "),
            rhs_shape = self.rhs.join(", "),
            lhs_caret = "^",
            rhs_caret = "^",
        )
    }
}

/// Length, in characters, of everything that would precede `elements[index]`
/// once `elements` is joined with `", "` - i.e. where that element's own
/// text starts within the joined string.
fn joined_prefix_len(elements: &[String], index: usize) -> usize {
    elements[..index].iter().map(|e| e.len() + 2).sum()
}

/// The two dimensions a matrix product failed to contract.
///
/// A narrower fact than [`MatMulMismatch`], and the one a reader almost always
/// gets. `matmul` carries two `on_unimplemented` messages: `MatMulShape`'s,
/// which names both whole shapes, and `ContractsWith`'s, which names only the
/// two conflicting axes. rustc reports the innermost failing bound, so it is
/// `ContractsWith`'s message that reaches the terminal, and an explanation
/// keyed only on the wider one never fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionMismatch {
    /// The lhs's last dimension, humanized (e.g. `"3"`).
    pub lhs_inner: String,
    /// The rhs's second-to-last dimension, humanized (e.g. `"4"`).
    pub rhs_inner: String,
}

impl ContractionMismatch {
    /// Renders the contraction rule against the two axes that broke it.
    ///
    /// Deliberately says which position each number came from. "3 does not
    /// equal 4" is not actionable on its own: a reader with a `[2, 3]` and a
    /// `[4, 5]` in front of them needs to know that the three is the *last*
    /// axis of the left operand and the four is the *second-to-last* of the
    /// right, because that is what tells them which of the two to change and
    /// whether a transpose would do it.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let Self {
            lhs_inner,
            rhs_inner,
        } = self;
        format!(
            "{INDENT}lhs (..., K) -> K = {lhs_inner}   <- last axis of the left operand\n\
             {INDENT}rhs (K, ...) -> K = {rhs_inner}   <- second-to-last axis of the right operand\n\
             {INDENT}{lhs_inner} \u{2260} {rhs_inner} \u{2192} change the left operand's last axis to {rhs_inner}, \
             change the right operand's second-to-last axis to {lhs_inner}, \n\
             {INDENT}or transpose whichever operand is the wrong way round."
        )
    }
}

/// Parses `ContractsWith`'s on_unimplemented message -- `` Cannot contract
/// dimension `{Self}` with `{Rhs}` `` -- into the two axes that disagree.
///
/// Returns `None` when the message is absent or the two axes read the same,
/// since equal axes mean the real failure was something else (a rank
/// disagreement, most often) and explaining a contraction would misdirect.
pub fn parse_contraction_mismatch(text: &str) -> Option<ContractionMismatch> {
    const PREFIX: &str = "Cannot contract dimension `";
    let start = text.find(PREFIX)?;
    let after_prefix = &text[start + PREFIX.len()..];
    let (lhs_inner, after_lhs) = after_prefix.split_once('`')?;
    let (rhs_inner, _) = after_lhs.strip_prefix(" with `")?.split_once('`')?;

    if lhs_inner == rhs_inner || lhs_inner.is_empty() || rhs_inner.is_empty() {
        return None;
    }
    Some(ContractionMismatch {
        lhs_inner: lhs_inner.to_string(),
        rhs_inner: rhs_inner.to_string(),
    })
}

/// Parses the `MatMulShape` trait's fixed on_unimplemented message -
/// `` Cannot matrix-multiply shape `{Self}` with `{Rhs}` `` - and, if the
/// inner dimensions (last element of `Self`, second-to-last of `Rhs`, per
/// the trait's own rule) actually differ, returns a [`MatMulMismatch`]
/// ready to render. Returns `None` if `text` isn't this message, either
/// shape isn't a plain tuple, or the inner dimensions match (nothing to
/// explain - the real failure is something else, e.g. a rank mismatch).
pub fn parse_matmul_mismatch(text: &str) -> Option<MatMulMismatch> {
    // Search rather than `strip_prefix` on the whole input: callers (e.g.
    // `cargo incin --explain`) pass the *entire* rendered diagnostic -
    // "error[E0277]: Cannot matrix-multiply shape `...` with `...`\n   -->
    // file:line:col\n..." - not just this one message in isolation.
    let start = text.find("Cannot matrix-multiply shape `")?;
    let after_prefix = &text[start + "Cannot matrix-multiply shape `".len()..];
    let (lhs_raw, after_lhs) = after_prefix.split_once('`')?;
    let after_lhs = after_lhs.strip_prefix(" with `")?;
    let (rhs_raw, _) = after_lhs.split_once('`')?;

    let lhs_inner_text = lhs_raw.strip_prefix('(')?.strip_suffix(')')?;
    let rhs_inner_text = rhs_raw.strip_prefix('(')?.strip_suffix(')')?;

    let lhs_elems: Vec<&str> = split_top_level_commas(lhs_inner_text);
    let rhs_elems: Vec<&str> = split_top_level_commas(rhs_inner_text);
    if lhs_elems.is_empty() || rhs_elems.len() < 2 {
        return None;
    }

    let lhs_inner_index = lhs_elems.len() - 1;
    let rhs_inner_index = rhs_elems.len() - 2;

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let lhs: Vec<String> = lhs_elems.iter().map(|e| humanize(e)).collect();
    let rhs: Vec<String> = rhs_elems.iter().map(|e| humanize(e)).collect();

    if lhs[lhs_inner_index] == rhs[rhs_inner_index] {
        return None; // inner dims already agree - a different rule failed
    }

    Some(MatMulMismatch {
        lhs,
        rhs,
        lhs_inner_index,
        rhs_inner_index,
    })
}

/// Splits `s` on top-level commas (i.e. not nested inside `<...>`) - a
/// typenum shape tuple's elements never contain their own parens, only
/// angle-bracket generics, so tracking `<`/`>` depth alone is sufficient.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() || !parts.is_empty() {
        parts.push(last);
    }
    parts
}

/// A `concat` shape mismatch, parsed from the `ConcatShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatMismatch {
    /// The left-hand operand's shape, one dimension per entry.
    pub lhs: Vec<String>,
    /// The right-hand operand's shape, one dimension per entry.
    pub rhs: Vec<String>,
    /// The concatenation axis the shapes disagreed on.
    pub axis: usize,
    /// Index into both `lhs` and `rhs` of the first disagreeing dimension.
    pub mismatch_index: usize,
}

impl ConcatMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let lhs_val = &self.lhs[self.mismatch_index];
        let rhs_val = &self.rhs[self.mismatch_index];

        let lhs_label = format!("{INDENT}lhs shape = (");
        let rhs_label = format!("{INDENT}rhs shape = (");
        let lhs_caret_width =
            lhs_label.len() + joined_prefix_len(&self.lhs, self.mismatch_index) + 1;
        let rhs_caret_width =
            rhs_label.len() + joined_prefix_len(&self.rhs, self.mismatch_index) + 1;

        format!(
            "{lhs_label}{lhs_shape})\n\
             {lhs_caret:>lhs_caret_width$} non-concat dim = {lhs_val}\n\
             {rhs_label}{rhs_shape})\n\
             {rhs_caret:>rhs_caret_width$} non-concat dim = {rhs_val}\n\
             {INDENT}{lhs_val} \u{2260} {rhs_val} \u{2192} concatenating along axis {axis} requires all other dimensions to match exactly.",
            lhs_shape = self.lhs.join(", "),
            rhs_shape = self.rhs.join(", "),
            lhs_caret = "^",
            rhs_caret = "^",
            axis = self.axis,
        )
    }
}

/// Parses a `concat` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_concat_mismatch(text: &str) -> Option<ConcatMismatch> {
    let start = text.find("Cannot concatenate shape `")?;
    let after_prefix = &text[start + "Cannot concatenate shape `".len()..];
    let (lhs_raw, after_lhs) = after_prefix.split_once('`')?;
    let after_lhs = after_lhs.strip_prefix(" with `")?;
    let (rhs_raw, after_rhs) = after_lhs.split_once('`')?;
    let axis_str = after_rhs.strip_prefix(" along axis `")?.split_once('`')?.0;

    let axis: usize = axis_str.trim().parse().ok()?;

    let lhs_inner_text = lhs_raw.strip_prefix('(')?.strip_suffix(')')?;
    let rhs_inner_text = rhs_raw.strip_prefix('(')?.strip_suffix(')')?;

    let lhs_elems: Vec<&str> = split_top_level_commas(lhs_inner_text);
    let rhs_elems: Vec<&str> = split_top_level_commas(rhs_inner_text);

    if lhs_elems.len() != rhs_elems.len() {
        return None;
    }

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let lhs: Vec<String> = lhs_elems.iter().map(|e| humanize(e)).collect();
    let rhs: Vec<String> = rhs_elems.iter().map(|e| humanize(e)).collect();

    for i in 0..lhs.len() {
        if i != axis && lhs[i] != rhs[i] {
            return Some(ConcatMismatch {
                lhs,
                rhs,
                axis,
                mismatch_index: i,
            });
        }
    }

    None
}

/// A `broadcast` shape mismatch, parsed from the `BroadcastShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastMismatch {
    /// The left-hand operand's shape, one dimension per entry.
    pub lhs: Vec<String>,
    /// The right-hand operand's shape, one dimension per entry.
    pub rhs: Vec<String>,
    /// Index into `lhs` of the first disagreeing dimension.
    pub lhs_mismatch_index: usize,
    /// Index into `rhs` of the first disagreeing dimension.
    pub rhs_mismatch_index: usize,
}

impl BroadcastMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let lhs_val = &self.lhs[self.lhs_mismatch_index];
        let rhs_val = &self.rhs[self.rhs_mismatch_index];

        let lhs_label = format!("{INDENT}lhs shape = (");
        let rhs_label = format!("{INDENT}rhs shape = (");
        let lhs_caret_width =
            lhs_label.len() + joined_prefix_len(&self.lhs, self.lhs_mismatch_index) + 1;
        let rhs_caret_width =
            rhs_label.len() + joined_prefix_len(&self.rhs, self.rhs_mismatch_index) + 1;

        format!(
            "{lhs_label}{lhs_shape})\n\
             {lhs_caret:>lhs_caret_width$} dim = {lhs_val}\n\
             {rhs_label}{rhs_shape})\n\
             {rhs_caret:>rhs_caret_width$} dim = {rhs_val}\n\
             {INDENT}{lhs_val} \u{2260} {rhs_val} \u{2192} broadcast requires corresponding dimensions to be equal, or one of them to be 1.",
            lhs_shape = self.lhs.join(", "),
            rhs_shape = self.rhs.join(", "),
            lhs_caret = "^",
            rhs_caret = "^",
        )
    }
}

/// Parses a `broadcast` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_broadcast_mismatch(text: &str) -> Option<BroadcastMismatch> {
    let start = text.find("Cannot broadcast shape `")?;
    let after_prefix = &text[start + "Cannot broadcast shape `".len()..];
    let (lhs_raw, after_lhs) = after_prefix.split_once('`')?;
    let after_lhs = after_lhs.strip_prefix(" to `")?;
    let (rhs_raw, _) = after_lhs.split_once('`')?;

    let lhs_inner_text = lhs_raw.strip_prefix('(')?.strip_suffix(')')?;
    let rhs_inner_text = rhs_raw.strip_prefix('(')?.strip_suffix(')')?;

    let lhs_elems: Vec<&str> = split_top_level_commas(lhs_inner_text);
    let rhs_elems: Vec<&str> = split_top_level_commas(rhs_inner_text);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let lhs: Vec<String> = lhs_elems.iter().map(|e| humanize(e)).collect();
    let rhs: Vec<String> = rhs_elems.iter().map(|e| humanize(e)).collect();

    let min_rank = lhs.len().min(rhs.len());
    for i in 1..=min_rank {
        let l_idx = lhs.len() - i;
        let r_idx = rhs.len() - i;
        let l_val = &lhs[l_idx];
        let r_val = &rhs[r_idx];
        if l_val != r_val && l_val != "1" && r_val != "1" {
            return Some(BroadcastMismatch {
                lhs,
                rhs,
                lhs_mismatch_index: l_idx,
                rhs_mismatch_index: r_idx,
            });
        }
    }

    None
}

/// A `reshape` shape mismatch, parsed from the `ReshapeShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReshapeMismatch {
    /// The source shape, one dimension per entry.
    pub src: Vec<String>,
    /// The requested target shape, one entry per dimension (`?` where inferred).
    pub target: Vec<String>,
    /// Element count implied by `src`.
    pub src_count: usize,
    /// Element count implied by `target`.
    pub target_count: usize,
}

impl ReshapeMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}src shape    = ({src_shape})  [total elements = {src_count}]\n\
             {INDENT}target shape = ({target_shape})  [total elements = {target_count}]\n\
             {INDENT}{src_count} \u{2260} {target_count} \u{2192} reshape requires the product of dimensions (total element count) to remain identical.",
            src_shape = self.src.join(", "),
            target_shape = self.target.join(", "),
            src_count = self.src_count,
            target_count = self.target_count,
        )
    }
}

/// Parses a `reshape` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_reshape_mismatch(text: &str) -> Option<ReshapeMismatch> {
    let start = text.find("Cannot reshape from `")?;
    let after_prefix = &text[start + "Cannot reshape from `".len()..];
    let (src_raw, after_src) = after_prefix.split_once('`')?;
    let after_src = after_src.strip_prefix(" to `")?;
    let (target_raw, _) = after_src.split_once('`')?;

    let src_inner_text = src_raw.strip_prefix('(')?.strip_suffix(')')?;
    let target_inner_text = target_raw.strip_prefix('(')?.strip_suffix(')')?;

    let src_elems: Vec<&str> = split_top_level_commas(src_inner_text);
    let target_elems: Vec<&str> = split_top_level_commas(target_inner_text);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let src: Vec<String> = src_elems.iter().map(|e| humanize(e)).collect();
    let target: Vec<String> = target_elems.iter().map(|e| humanize(e)).collect();

    let mut src_count = 1;
    for s in &src {
        src_count *= s.parse::<usize>().ok()?;
    }
    let mut target_count = 1;
    for t in &target {
        target_count *= t.parse::<usize>().ok()?;
    }

    if src_count != target_count {
        Some(ReshapeMismatch {
            src,
            target,
            src_count,
            target_count,
        })
    } else {
        None
    }
}

/// A `conv2d` shape mismatch, parsed from the `Conv2dShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv2dMismatch {
    /// The input tensor's shape, one dimension per entry.
    pub input: Vec<String>,
    /// The kernel/weight shape, one dimension per entry.
    pub kernel: Vec<String>,
    /// Index into `input` of the channel dimension.
    pub input_channel_idx: usize,
    /// Index into `kernel` of the channel dimension.
    pub kernel_channel_idx: usize,
}

impl Conv2dMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let in_c = &self.input[self.input_channel_idx];
        let k_c = &self.kernel[self.kernel_channel_idx];

        let in_label = format!("{INDENT}input shape  = (");
        let k_label = format!("{INDENT}kernel shape = (");
        let in_caret_width =
            in_label.len() + joined_prefix_len(&self.input, self.input_channel_idx) + 1;
        let k_caret_width =
            k_label.len() + joined_prefix_len(&self.kernel, self.kernel_channel_idx) + 1;

        format!(
            "{in_label}{in_shape})\n\
             {in_caret:>in_caret_width$} input channels = {in_c}\n\
             {k_label}{k_shape})\n\
             {k_caret:>k_caret_width$} kernel in_channels = {k_c}\n\
             {INDENT}{in_c} \u{2260} {k_c} \u{2192} Conv2D input channels must match kernel input channels.",
            in_shape = self.input.join(", "),
            k_shape = self.kernel.join(", "),
            in_caret = "^",
            k_caret = "^",
        )
    }
}

/// Parses a `conv2d` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_conv2d_mismatch(text: &str) -> Option<Conv2dMismatch> {
    let start = text.find("Cannot apply Conv2D: input shape `")?;
    let after_prefix = &text[start + "Cannot apply Conv2D: input shape `".len()..];
    let (in_raw, after_in) = after_prefix.split_once('`')?;
    let after_in = after_in.strip_prefix(" is incompatible with kernel shape `")?;
    let (k_raw, _) = after_in.split_once('`')?;

    let in_inner_text = in_raw.strip_prefix('(')?.strip_suffix(')')?;
    let k_inner_text = k_raw.strip_prefix('(')?.strip_suffix(')')?;

    let in_elems: Vec<&str> = split_top_level_commas(in_inner_text);
    let k_elems: Vec<&str> = split_top_level_commas(k_inner_text);

    if in_elems.len() < 2 || k_elems.len() < 2 {
        return None;
    }

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let input: Vec<String> = in_elems.iter().map(|e| humanize(e)).collect();
    let kernel: Vec<String> = k_elems.iter().map(|e| humanize(e)).collect();

    if input[1] != kernel[1] {
        Some(Conv2dMismatch {
            input,
            kernel,
            input_channel_idx: 1,
            kernel_channel_idx: 1,
        })
    } else {
        None
    }
}

/// A `transpose` shape mismatch, parsed from the `Transpose` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransposeMismatch {
    /// The transposed tensor's shape, one dimension per entry.
    pub shape: Vec<String>,
    /// First axis of the transpose pair.
    pub d1: usize,
    /// Second axis of the transpose pair.
    pub d2: usize,
    /// Rank the axes were checked against.
    pub rank: usize,
}

impl TransposeMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}shape = ({shape})  [rank = {rank}]\n\
             {INDENT}transpose indices = ({d1}, {d2})\n\
             {INDENT}invalid dimension index \u{2192} both transpose indices must be < rank ({rank}).",
            shape = self.shape.join(", "),
            rank = self.rank,
            d1 = self.d1,
            d2 = self.d2,
        )
    }
}

/// Parses a `transpose` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_transpose_mismatch(text: &str) -> Option<TransposeMismatch> {
    let start = text.find("Cannot transpose dimensions `")?;
    let after_prefix = &text[start + "Cannot transpose dimensions `".len()..];
    let (d1_str, after_d1) = after_prefix.split_once('`')?;
    let after_d1 = after_d1.strip_prefix(" and `")?;
    let (d2_str, after_d2) = after_d1.split_once('`')?;
    let after_d2 = after_d2.strip_prefix(" on shape `")?;
    let (shape_raw, _) = after_d2.split_once('`')?;

    let d1: usize = d1_str.trim().parse().ok()?;
    let d2: usize = d2_str.trim().parse().ok()?;

    let shape_inner = shape_raw.strip_prefix('(')?.strip_suffix(')')?;
    let elems: Vec<&str> = split_top_level_commas(shape_inner);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let shape: Vec<String> = elems.iter().map(|e| humanize(e)).collect();
    let rank = shape.len();

    if d1 >= rank || d2 >= rank {
        Some(TransposeMismatch {
            shape,
            d1,
            d2,
            rank,
        })
    } else {
        None
    }
}

/// A `reduce` shape mismatch, parsed from the `ReduceDim` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceDimMismatch {
    /// The transposed tensor's shape, one dimension per entry.
    pub shape: Vec<String>,
    /// The axis whose extent was rejected.
    pub dim: usize,
    /// Rank the axes were checked against.
    pub rank: usize,
}

impl ReduceDimMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}shape = ({shape})  [rank = {rank}]\n\
             {INDENT}reduce dim = {dim}\n\
             {INDENT}{dim} \u{2265} {rank} \u{2192} reduction dimension must be < rank ({rank}).",
            shape = self.shape.join(", "),
            rank = self.rank,
            dim = self.dim,
        )
    }
}

/// Parses a `reduce_dim` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_reduce_dim_mismatch(text: &str) -> Option<ReduceDimMismatch> {
    let start = text.find("Cannot reduce dimension `")?;
    let after_prefix = &text[start + "Cannot reduce dimension `".len()..];
    let (dim_str, after_dim) = after_prefix.split_once('`')?;
    let after_dim = after_dim
        .find("on shape `")
        .map(|p| &after_dim[p + "on shape `".len()..])?;
    let (shape_raw, _) = after_dim.split_once('`')?;

    let dim: usize = dim_str.trim().parse().ok()?;

    let shape_inner = shape_raw.strip_prefix('(')?.strip_suffix(')')?;
    let elems: Vec<&str> = split_top_level_commas(shape_inner);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let shape: Vec<String> = elems.iter().map(|e| humanize(e)).collect();
    let rank = shape.len();

    if dim >= rank {
        Some(ReduceDimMismatch { shape, dim, rank })
    } else {
        None
    }
}

/// A `flatten` shape mismatch, parsed from the `Flatten` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenMismatch {
    /// The transposed tensor's shape, one dimension per entry.
    pub shape: Vec<String>,
    /// First flattened axis.
    pub start_dim: usize,
    /// Last flattened axis, inclusive.
    pub end_dim: usize,
    /// Rank the axes were checked against.
    pub rank: usize,
}

impl FlattenMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}shape = ({shape})  [rank = {rank}]\n\
             {INDENT}flatten range = [{start_dim}, {end_dim}]\n\
             {INDENT}invalid range \u{2192} flatten requires start <= end and end < rank ({rank}).",
            shape = self.shape.join(", "),
            rank = self.rank,
            start_dim = self.start_dim,
            end_dim = self.end_dim,
        )
    }
}

/// Parses a `flatten` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_flatten_mismatch(text: &str) -> Option<FlattenMismatch> {
    let start = text.find("Cannot flatten shape `")?;
    let after_prefix = &text[start + "Cannot flatten shape `".len()..];
    let (shape_raw, after_shape) = after_prefix.split_once('`')?;
    let after_shape = after_shape.strip_prefix(" from dimension `")?;
    let (start_str, after_start) = after_shape.split_once('`')?;
    let after_start = after_start.strip_prefix(" to `")?;
    let (end_str, _) = after_start.split_once('`')?;

    let start_dim: usize = start_str.trim().parse().ok()?;
    let end_dim: usize = end_str.trim().parse().ok()?;

    let shape_inner = shape_raw.strip_prefix('(')?.strip_suffix(')')?;
    let elems: Vec<&str> = split_top_level_commas(shape_inner);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let shape: Vec<String> = elems.iter().map(|e| humanize(e)).collect();
    let rank = shape.len();

    if start_dim > end_dim || end_dim >= rank {
        Some(FlattenMismatch {
            shape,
            start_dim,
            end_dim,
            rank,
        })
    } else {
        None
    }
}

/// A `Module::forward` input shape mismatch, parsed from compiler output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleForwardMismatch {
    /// The input shape text the diagnostic reported.
    pub actual_input: String,
    /// The input shape text the trait expected.
    pub expected_input: String,
}

impl ModuleForwardMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}provided input shape = {actual}\n\
             {INDENT}expected input shape = {expected}\n\
             {INDENT}shape mismatch \u{2192} layer forward pass expects {expected}, but received {actual}.",
            actual = self.actual_input,
            expected = self.expected_input,
        )
    }
}

/// Parses a `module_forward` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_module_forward_mismatch(text: &str) -> Option<ModuleForwardMismatch> {
    let mut search = text;
    let mut actual_raw = None;
    while let Some(start) = search.find("Module<") {
        let after = &search[start + "Module<".len()..];
        if let Some(close) = matching_bracket(after, '<', '>') {
            let inner = &after[..close];
            if inner.contains("Tensor") {
                actual_raw = Some(inner);
                break;
            }
        }
        if after.is_empty() {
            break;
        }
        search = &after[1..];
    }
    let actual_raw = actual_raw?;

    let impl_start = text.find("implements `Module<")?;
    let after_impl = &text[impl_start + "implements `Module<".len()..];
    let close_impl = matching_bracket(after_impl, '<', '>')?;
    let expected_raw = &after_impl[..close_impl];

    let clean = |s: &str| {
        let mut res = humanize_diagnostic(s.trim()).text.trim().to_string();
        let open_count = res.chars().filter(|&c| c == '<').count();
        let close_count = res.chars().filter(|&c| c == '>').count();
        if open_count > close_count {
            res.push('>');
        }
        res
    };

    let actual = clean(actual_raw);
    let expected = clean(expected_raw);

    if actual != expected {
        Some(ModuleForwardMismatch {
            actual_input: actual,
            expected_input: expected,
        })
    } else {
        None
    }
}

/// A slice target shape mismatch, parsed from the `SliceTarget` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceMismatch {
    /// The slice specification text the caller supplied.
    pub slice_spec: String,
    /// The input shape, one dimension per entry as text.
    pub in_shape: String,
}

impl SliceMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}input shape = {in_shape}\n\
             {INDENT}slice spec  = {slice_spec}\n\
             {INDENT}invalid slice \u{2192} slice ranges must fall within input dimension bounds.",
            in_shape = self.in_shape,
            slice_spec = self.slice_spec,
        )
    }
}

/// Parses a `slice` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_slice_mismatch(text: &str) -> Option<SliceMismatch> {
    let start = text.find("Cannot slice dimension with `")?;
    let after = &text[start + "Cannot slice dimension with `".len()..];
    let (spec_raw, after_spec) = after.split_once('`')?;

    let in_shape_raw = if let Some(pos) = after_spec.find("for shape `") {
        let after_for = &after_spec[pos + "for shape `".len()..];
        after_for.split_once('`').map(|(s, _)| s)
    } else {
        None
    };

    let slice_spec = humanize_diagnostic(spec_raw.trim()).text;
    let in_shape = in_shape_raw
        .map(|s| humanize_diagnostic(s.trim()).text)
        .unwrap_or_else(|| "unknown".to_string());

    Some(SliceMismatch {
        slice_spec,
        in_shape,
    })
}

/// A `conv1d` shape mismatch, parsed from the `SpatialConv1d` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv1dMismatch {
    /// The input shape text the diagnostic reported.
    pub input_shape: String,
}

impl Conv1dMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}input shape = {shape}\n\
             {INDENT}invalid shape \u{2192} Conv1D requires a 2D or 3D tensor (C, L) or (B, C, L).",
            shape = self.input_shape,
        )
    }
}

/// Parses a `conv1d` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_conv1d_mismatch(text: &str) -> Option<Conv1dMismatch> {
    let start = text.find("Cannot apply 1D convolution to shape `")?;
    let after = &text[start + "Cannot apply 1D convolution to shape `".len()..];
    let (raw_shape, _) = after.split_once('`')?;

    let input_shape = humanize_diagnostic(raw_shape.trim()).text;
    Some(Conv1dMismatch { input_shape })
}

/// A 2D pooling shape mismatch, parsed from the `Pool2dShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pool2dMismatch {
    /// The input shape text the diagnostic reported.
    pub input_shape: String,
}

impl Pool2dMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}input shape = {shape}\n\
             {INDENT}invalid shape \u{2192} Pool2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W).",
            shape = self.input_shape,
        )
    }
}

/// Parses a `pool2d` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_pool2d_mismatch(text: &str) -> Option<Pool2dMismatch> {
    let start = text.find("Cannot apply 2D pooling to shape `")?;
    let after = &text[start + "Cannot apply 2D pooling to shape `".len()..];
    let (raw_shape, _) = after.split_once('`')?;

    let input_shape = humanize_diagnostic(raw_shape.trim()).text;
    Some(Pool2dMismatch { input_shape })
}

/// A shape equality mismatch, parsed from the `ShapeEq` trait's compile-time error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeEqMismatch {
    /// The verbatim diagnostic message this explanation annotates.
    pub message: String,
}

impl ShapeEqMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!("{INDENT}shape mismatch \u{2192} {msg}", msg = self.message,)
    }
}

/// Parses a `shape_eq` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_shape_eq_mismatch(text: &str) -> Option<ShapeEqMismatch> {
    if let Some(start) = text.find("Shape Mismatch:") {
        let after = &text[start + "Shape Mismatch:".len()..];
        let end = after.find('\n').unwrap_or(after.len());
        let message = after[..end].trim().to_string();
        Some(ShapeEqMismatch { message })
    } else {
        None
    }
}

/// A `bmm` rank or shape mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BmmMismatch {
    /// The verbatim diagnostic message this explanation annotates.
    pub message: String,
}

impl BmmMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!("{INDENT}bmm mismatch \u{2192} {msg}", msg = self.message)
    }
}

/// Parses a `bmm` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_bmm_mismatch(text: &str) -> Option<BmmMismatch> {
    if text.contains("bmm") || text.contains("batched matrix multiplication") {
        Some(BmmMismatch {
            message: "BMM requires 3D tensors (B, M, K) x (B, K, N)".to_string(),
        })
    } else {
        None
    }
}

/// An `unfold` dimension bound mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnfoldMismatch {
    /// The verbatim diagnostic message this explanation annotates.
    pub message: String,
}

impl UnfoldMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!("{INDENT}unfold mismatch \u{2192} {msg}", msg = self.message)
    }
}

/// Parses a `unfold` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_unfold_mismatch(text: &str) -> Option<UnfoldMismatch> {
    if text.contains("unfold size cannot exceed dimension length") {
        Some(UnfoldMismatch {
            message: "unfold size exceeds target dimension length".to_string(),
        })
    } else {
        None
    }
}

/// A `pixel_shuffle` channel divisibility mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelShuffleMismatch {
    /// The verbatim diagnostic message this explanation annotates.
    pub message: String,
}

impl PixelShuffleMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}pixel_shuffle mismatch \u{2192} {msg}",
            msg = self.message
        )
    }
}

/// Parses a `pixel_shuffle` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_pixel_shuffle_mismatch(text: &str) -> Option<PixelShuffleMismatch> {
    if text.contains("pixel_shuffle channels must be divisible") {
        Some(PixelShuffleMismatch {
            message: "channel count must be divisible by upscale_factor^2".to_string(),
        })
    } else {
        None
    }
}

/// A `group_norm` channel divisibility mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupNormMismatch {
    /// The verbatim diagnostic message this explanation annotates.
    pub message: String,
}

impl GroupNormMismatch {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}group_norm mismatch \u{2192} {msg}",
            msg = self.message
        )
    }
}

/// Parses a `group_norm` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_group_norm_mismatch(text: &str) -> Option<GroupNormMismatch> {
    if text.contains("group_norm: channels must be divisible by groups") {
        Some(GroupNormMismatch {
            message: "channels count must be divisible by groups".to_string(),
        })
    } else {
        None
    }
}

/// A math domain error diagnostic (e.g., asin/acos out of bounds, rsqrt non-positive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathDomainError {
    /// The verbatim diagnostic message this explanation annotates.
    pub message: String,
}

impl MathDomainError {
    /// Renders the humanized explanation rustc should append to the diagnostic.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}math domain error \u{2192} {msg}",
            msg = self.message
        )
    }
}

/// Parses a `math_domain_error` mismatch from diagnostic text; `None` when the text is not this message.
pub fn parse_math_domain_error(text: &str) -> Option<MathDomainError> {
    if text.contains("out of domain")
        || text.contains("NaN domain")
        || text.contains("domain error")
    {
        Some(MathDomainError {
            message: "argument value is outside the real domain of the function".to_string(),
        })
    } else {
        None
    }
}

