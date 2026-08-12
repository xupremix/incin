//! Small compiler-facing symbolic shape language.
//!
//! The typed shape system proves frontend operations. This module preserves
//! the facts that remain relevant after type erasure, including obligations
//! that must be checked when a compiled graph is called.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SymbolId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DimExpr {
    Const(usize),
    Symbol(SymbolId),
    Add(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    ExactDiv(Box<DimExpr>, Box<DimExpr>),
    Broadcast(Box<DimExpr>, Box<DimExpr>),
    Unknown,
}

impl DimExpr {
    #[must_use]
    pub fn simplify(self) -> Self {
        match self {
            Self::Add(lhs, rhs) => match (*lhs, *rhs) {
                (Self::Const(lhs), Self::Const(rhs)) => lhs
                    .checked_add(rhs)
                    .map(Self::Const)
                    .unwrap_or(Self::Unknown),
                (Self::Const(0), rhs) | (rhs, Self::Const(0)) => rhs,
                (lhs, rhs) => Self::Add(Box::new(lhs), Box::new(rhs)),
            },
            Self::Mul(lhs, rhs) => match (*lhs, *rhs) {
                (Self::Const(lhs), Self::Const(rhs)) => lhs
                    .checked_mul(rhs)
                    .map(Self::Const)
                    .unwrap_or(Self::Unknown),
                (Self::Const(0), _) | (_, Self::Const(0)) => Self::Const(0),
                (Self::Const(1), rhs) | (rhs, Self::Const(1)) => rhs,
                (lhs, rhs) => Self::Mul(Box::new(lhs), Box::new(rhs)),
            },
            Self::ExactDiv(lhs, rhs) => match (*lhs, *rhs) {
                (Self::Const(lhs), Self::Const(rhs)) if rhs != 0 => lhs
                    .checked_div(rhs)
                    .filter(|_| lhs % rhs == 0)
                    .map(Self::Const)
                    .unwrap_or(Self::Unknown),
                (lhs, rhs) => Self::ExactDiv(Box::new(lhs), Box::new(rhs)),
            },
            Self::Broadcast(lhs, rhs) => match (*lhs, *rhs) {
                (Self::Const(1), rhs) | (rhs, Self::Const(1)) => rhs,
                (Self::Const(lhs), Self::Const(rhs)) if lhs == rhs => Self::Const(lhs),
                (lhs, rhs) => Self::Broadcast(Box::new(lhs), Box::new(rhs)),
            },
            other => other,
        }
    }

    #[must_use]
    pub fn evaluate(&self, symbols: &[(SymbolId, usize)]) -> Option<usize> {
        match self {
            Self::Const(value) => Some(*value),
            Self::Symbol(id) => symbols
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, value)| *value),
            Self::Add(lhs, rhs) => lhs.evaluate(symbols)?.checked_add(rhs.evaluate(symbols)?),
            Self::Mul(lhs, rhs) => lhs.evaluate(symbols)?.checked_mul(rhs.evaluate(symbols)?),
            Self::ExactDiv(lhs, rhs) => {
                let lhs = lhs.evaluate(symbols)?;
                let rhs = rhs.evaluate(symbols)?;
                (rhs != 0 && lhs % rhs == 0).then_some(lhs / rhs)
            }
            Self::Broadcast(lhs, rhs) => {
                let lhs = lhs.evaluate(symbols)?;
                let rhs = rhs.evaluate(symbols)?;
                if lhs == rhs {
                    Some(lhs)
                } else if lhs == 1 {
                    Some(rhs)
                } else if rhs == 1 {
                    Some(lhs)
                } else {
                    None
                }
            }
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RankExpr {
    Static(usize),
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Constraint {
    Equal { lhs: DimExpr, rhs: DimExpr },
    LowerBound { value: DimExpr, bound: usize },
    UpperBound { value: DimExpr, bound: usize },
    Divisible { value: DimExpr, divisor: usize },
    BroadcastCompatible { lhs: DimExpr, rhs: DimExpr },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ShapeExpr {
    pub rank: RankExpr,
    pub dims: Vec<DimExpr>,
    pub constraints: Vec<Constraint>,
}

impl ShapeExpr {
    #[must_use]
    pub fn concrete(dims: &[usize]) -> Self {
        Self {
            rank: RankExpr::Static(dims.len()),
            dims: dims.iter().copied().map(DimExpr::Const).collect(),
            constraints: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn validate(&self, actual: &[usize]) -> Result<(), String> {
        if let RankExpr::Static(rank) = self.rank {
            if rank != actual.len() {
                return Err(alloc::format!(
                    "expected rank {}, got {}",
                    rank,
                    actual.len()
                ));
            }
        }
        let mut symbols = Vec::new();
        for (expr, value) in self.dims.iter().zip(actual.iter().copied()) {
            match expr {
                DimExpr::Const(expected) if *expected != value => {
                    return Err(alloc::format!(
                        "expected dimension {}, got {}",
                        expected,
                        value
                    ));
                }
                DimExpr::Symbol(id) => {
                    if let Some((_, previous)) =
                        symbols.iter().find(|(candidate, _)| candidate == id)
                    {
                        if *previous != value {
                            return Err(alloc::format!(
                                "symbol {:?} was {}, got {}",
                                id,
                                previous,
                                value
                            ));
                        }
                    } else {
                        symbols.push((*id, value));
                    }
                }
                _ => {}
            }
        }
        for constraint in &self.constraints {
            let valid = match constraint {
                Constraint::Equal { lhs, rhs } => lhs.evaluate(&symbols) == rhs.evaluate(&symbols),
                Constraint::LowerBound { value, bound } => value
                    .evaluate(&symbols)
                    .is_some_and(|value| value >= *bound),
                Constraint::UpperBound { value, bound } => value
                    .evaluate(&symbols)
                    .is_some_and(|value| value <= *bound),
                Constraint::Divisible { value, divisor } => value
                    .evaluate(&symbols)
                    .is_some_and(|value| *divisor != 0 && value % divisor == 0),
                Constraint::BroadcastCompatible { lhs, rhs } => {
                    match (lhs.evaluate(&symbols), rhs.evaluate(&symbols)) {
                        (Some(lhs), Some(rhs)) => lhs == rhs || lhs == 1 || rhs == 1,
                        _ => false,
                    }
                }
            };
            if !valid {
                return Err(alloc::format!("shape constraint failed: {:?}", constraint));
            }
        }
        Ok(())
    }
}
