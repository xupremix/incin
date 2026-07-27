//! The single rank ceiling, and the sweep that generates every rank ladder
//! from it.
//!
//! Before `SHP-006` each shape rule carried its own ceiling, spelled as a
//! hand-written list of macro invocations. There were eighteen such lists and
//! they disagreed: `Shape` reached rank 8, `ElementCount` reached 4,
//! `HasChannels1D` reached 3, and `ReplaceLastDim` overshot at 12. A rank
//! between a rule's ceiling and `Shape`'s is one where a tensor type is
//! expressible but its operations cannot resolve — the frontend accepts the
//! shape and then has no proof to offer.
//!
//! [`MAX_RANK`] is now the only place a ceiling is written. `rank_sweep!`
//! expands a rule's invocation ladder from it, so raising the ceiling is a
//! one-line change and no rule can drift below it.

use proc_macro::TokenStream;
use quote::format_ident;
use std::str::FromStr;

/// The highest tuple rank every shape rule is implemented for.
///
/// Raising this raises every rule together. The cost is monomorphization: each
/// rank adds one impl per rule, and the spatial rules add a `typenum` type-level
/// arithmetic chain per axis, so compile time grows faster than linearly.
pub(crate) const MAX_RANK: usize = 8;

/// The shape of one invocation in a rank ladder.
///
/// Each variant corresponds to the argument form a family's `macro_rules!`
/// already accepts; the sweep does not change those, it only generates the
/// list that drives them.
enum Form {
    /// `D0, D1, …, D{n-1}` — the common "one type parameter per axis" form.
    Names,
    /// `D1, D2, …, Dn` — same, but 1-based, as the index rules spell it.
    NamesFrom1,
    /// `n, D0 0, D1 1, …` — a leading rank followed by name/index pairs.
    RankedPairs,
    /// `A, B, C, …` — single-letter parameters, as the broadcast rules spell it.
    Letters,
    /// `L0, L1, …; R0, R1, …` — two independent parameter lists of the same
    /// length, one per operand. For rules that relate a pair of equal-rank
    /// shapes axis by axis rather than requiring the axes to be identical.
    OperandPairs,
    /// `B, C, D, …` — letters from `B`, for rules whose first axis is a fixed
    /// `usize` and so is not a parameter.
    LettersFromB,
    /// `n+1; B0: 0, …` — the conv1d form: the length axis's tuple index,
    /// then colon-separated batch name/index pairs.
    Conv1d,
    /// `n+1, n+2; B0: 0, …` — the conv2d form, with two spatial indices.
    Conv2d,
    /// `n, n+1, n+2; B0: 0, …` — the pool2d form. Unlike conv, pooling
    /// *preserves* the channel axis rather than replacing it, so the rule needs
    /// that axis's tuple index too.
    Pool2d,
    /// `(A); (B, C)` — every way to split `n` letters into a non-empty left
    /// operand and a non-empty right one. Broadcasting a shorter shape against
    /// a longer one is a rule over a *pair* of ranks, so this form emits
    /// several invocations per rank rather than one.
    Prepend,
    /// `(); (B, C)` — the same split, lettered from `B` and allowing an empty
    /// left operand, for the rules whose leading axis is a fixed `usize`.
    UsizePrepend,
    /// `(P0); (L0, L1); (R0, R1)` — the prepend split with the shared suffix
    /// given twice, once per operand, for rules that relate the overlapping
    /// axes pairwise rather than requiring them to be identical.
    OperandPairsPrepend,
    /// `D0 ; D1 ; D2 ; U1` — the names before the target axis, the axis itself,
    /// the names after it, and the axis index. For per-axis rules like
    /// concatenation, which rewrite one axis and pass the rest through.
    AxisSplit,
    /// `D0 ; D1, D2 ; U1` — the names before an *inserted* axis and those
    /// after, plus its index. For rules like stacking, which add an axis rather
    /// than rewriting one, so the index may equal the rank.
    AxisInsert,
}

impl FromStr for Form {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "names" => Self::Names,
            "names_from1" => Self::NamesFrom1,
            "ranked_pairs" => Self::RankedPairs,
            "letters" => Self::Letters,
            "operand_pairs" => Self::OperandPairs,
            "letters_from_b" => Self::LettersFromB,
            "conv1d" => Self::Conv1d,
            "conv2d" => Self::Conv2d,
            "pool2d" => Self::Pool2d,
            "prepend" => Self::Prepend,
            "usize_prepend" => Self::UsizePrepend,
            "operand_pairs_prepend" => Self::OperandPairsPrepend,
            "axis_split" => Self::AxisSplit,
            "axis_insert" => Self::AxisInsert,
            other => {
                return Err(format!(
                    "unknown rank_sweep form `{other}`; expected one of names, \
                     names_from1, ranked_pairs, letters, operand_pairs, letters_from_b, conv1d, conv2d, pool2d, prepend, usize_prepend, operand_pairs_prepend, axis_split, axis_insert"
                ));
            }
        })
    }
}

impl Form {
    /// The argument lists for one rank, as source text.
    ///
    /// Most forms yield exactly one invocation per rank. The prepend forms
    /// yield one per way of splitting that rank across two operands.
    fn arguments(&self, rank: usize) -> Vec<String> {
        let names = |prefix: &str, start: usize| {
            (0..rank)
                .map(|i| format!("{prefix}{}", i + start))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let letters = |start: u8| {
            (0..rank)
                .map(|i| ((start + i as u8) as char).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let colon_pairs = || {
            (0..rank)
                .map(|i| format!("B{i}: {i}"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        // Every way to split `rank` sequentially-lettered parameters across
        // two parenthesised operand lists.
        let splits = |start: u8, min_lhs: usize| {
            (min_lhs..rank)
                .map(|lhs| {
                    let take = |from: usize, count: usize| {
                        (0..count)
                            .map(|i| ((start + (from + i) as u8) as char).to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    format!("({}); ({})", take(0, lhs), take(lhs, rank - lhs))
                })
                .collect::<Vec<_>>()
        };

        match self {
            Self::Names => vec![names("D", 0)],
            Self::NamesFrom1 => vec![names("D", 1)],
            Self::RankedPairs => {
                let pairs = (0..rank)
                    .map(|i| format!("D{i} {i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                vec![format!("{rank}, {pairs}")]
            }
            Self::Letters => vec![letters(b'A')],
            Self::OperandPairs => vec![format!("{}; {}", names("L", 0), names("R", 0))],
            Self::LettersFromB => vec![letters(b'B')],
            Self::Conv1d => vec![format!("{}; {}", rank + 1, colon_pairs())],
            Self::Conv2d => vec![format!("{}, {}; {}", rank + 1, rank + 2, colon_pairs())],
            Self::Pool2d => {
                vec![format!("{}, {}, {}; {}", rank, rank + 1, rank + 2, colon_pairs())]
            }
            // A non-empty left operand: `(A); (B)` but never `(); (A)`.
            Self::Prepend => splits(b'A', 1),
            // The same splits, but the shared suffix is listed once per
            // operand so the rule can relate the two axis by axis.
            Self::OperandPairsPrepend => (1..rank)
                .map(|lead| {
                    let run = |prefix: &str, from: usize, count: usize| {
                        (0..count)
                            .map(|i| format!("{prefix}{}", from + i))
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    format!(
                        "({}); ({}); ({})",
                        run("P", 0, lead),
                        run("L", 0, rank - lead),
                        run("R", 0, rank - lead)
                    )
                })
                .collect(),
            // An empty left operand is meaningful here, because the rule's
            // leading `usize` axis already stands outside the letter list.
            Self::UsizePrepend => splits(b'B', 0),
            // One invocation per axis of a rank-`rank` shape.
            Self::AxisSplit => (0..rank)
                .map(|axis| {
                    let run = |from: usize, to: usize| {
                        (from..to).map(|i| format!("D{i}")).collect::<Vec<_>>().join(", ")
                    };
                    format!("{} ; D{axis} ; {} ; U{axis}", run(0, axis), run(axis + 1, rank))
                })
                .collect(),
            // One per *insertion point*, of which a rank-`rank` shape has
            // `rank + 1` — an axis can be added before the first or after the
            // last, which is one more position than the shape has axes.
            Self::AxisInsert => (0..=rank)
                .map(|axis| {
                    let run = |from: usize, to: usize| {
                        (from..to).map(|i| format!("D{i}")).collect::<Vec<_>>().join(", ")
                    };
                    format!("{} ; {} ; U{axis}", run(0, axis), run(axis, rank))
                })
                .collect(),
        }
    }
}

/// Parse `form => macro_name` or `form => macro_name, min = K` or
/// `form => macro_name, min = K, max = M`.
struct Sweep {
    form: Form,
    macro_name: String,
    min: usize,
    max: usize,
}

fn parse(input: &str) -> Result<Sweep, String> {
    let (form, rest) = input
        .split_once("=>")
        .ok_or_else(|| "expected `form => macro_name`".to_string())?;
    let mut parts = rest.split(',').map(str::trim);
    let macro_name = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "expected a macro name after `=>`".to_string())?
        .to_string();

    let (mut min, mut max) = (1usize, MAX_RANK);
    for part in parts {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| format!("expected `key = value`, found `{part}`"))?;
        let value: usize = value
            .trim()
            .parse()
            .map_err(|_| format!("`{}` is not a rank", value.trim()))?;
        match key.trim() {
            "min" => min = value,
            "max" => max = value,
            other => return Err(format!("unknown rank_sweep option `{other}`")),
        }
    }

    if min > max {
        return Err(format!("min {min} exceeds max {max}"));
    }
    if max > MAX_RANK {
        return Err(format!(
            "max {max} exceeds MAX_RANK ({MAX_RANK}); raise MAX_RANK in \
             incin-macros/src/rank.rs instead of overriding it here"
        ));
    }

    Ok(Sweep {
        form: form.trim().parse()?,
        macro_name,
        min,
        max,
    })
}

/// Expand a rank ladder. See [`crate::rank_sweep`].
pub(crate) fn rank_sweep(input: TokenStream) -> TokenStream {
    let text = input.to_string();
    let sweep = match parse(&text) {
        Ok(sweep) => sweep,
        Err(message) => {
            return TokenStream::from_str(&format!("compile_error!({message:?});"))
                .expect("compile_error! is valid Rust");
        }
    };

    let name = format_ident!("{}", sweep.macro_name);
    let lines: String = (sweep.min..=sweep.max)
        .flat_map(|rank| sweep.form.arguments(rank))
        .map(|arguments| format!("{name}!({arguments});"))
        .collect::<Vec<_>>()
        .join("\n");

    TokenStream::from_str(&lines).unwrap_or_else(|_| {
        TokenStream::from_str(&format!(
            "compile_error!({:?});",
            format!("rank_sweep produced invalid Rust for `{}`", sweep.macro_name)
        ))
        .expect("compile_error! is valid Rust")
    })
}

/// Expand to `MAX_RANK` as a literal. See [`crate::max_rank`].
pub(crate) fn max_rank() -> TokenStream {
    TokenStream::from_str(&format!("{MAX_RANK}usize")).expect("a usize literal is valid Rust")
}
