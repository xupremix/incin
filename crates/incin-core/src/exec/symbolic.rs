//! Small compiler-facing symbolic shape language.
//!
//! The typed shape system proves frontend operations. This module preserves
//! the facts that remain relevant after type erasure, including obligations
//! that must be checked when a compiled graph is called.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
/// Identifier of one symbolic dimension variable.
pub struct SymbolId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// Registration record for a symbol.
pub struct SymbolInfo {
    /// Symbol identifier.
    pub id: SymbolId,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Producer identity string, when known.
    pub identity: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Table of symbols and their constraints.
pub struct SymbolTable {
    /// Registered symbols in insertion order.
    pub symbols: Vec<SymbolInfo>,
    /// Constraints binding symbols.
    pub constraints: Vec<Constraint>,
}

impl SymbolTable {
    /// Registers a symbol with optional naming.
    pub fn register(&mut self, id: SymbolId, name: Option<String>, identity: Option<String>) {
        if let Some(existing) = self.symbols.iter_mut().find(|symbol| symbol.id == id) {
            if existing.name.is_none() {
                existing.name = name;
            }
            if existing.identity.is_none() {
                existing.identity = identity;
            }
        } else {
            self.symbols.push(SymbolInfo { id, name, identity });
        }
    }

    #[must_use]
    /// Snapshot of bound symbol values.
    pub fn environment(&self) -> SymbolEnvironment {
        SymbolEnvironment::default()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Environment binding symbols to concrete extents.
pub struct SymbolEnvironment {
    bindings: BTreeMap<SymbolId, usize>,
}

impl SymbolEnvironment {
    /// Binds one symbol, validating against constraints.
    pub fn bind(&mut self, id: SymbolId, value: usize) -> Result<(), String> {
        if let Some(previous) = self.bindings.insert(id, value)
            && previous != value
        {
            return Err(alloc::format!(
                "symbol {:?} was {}, got {}",
                id,
                previous,
                value
            ));
        }
        Ok(())
    }

    /// Bound value of one symbol, when present.
    pub fn get(&self, id: SymbolId) -> Option<usize> {
        self.bindings.get(&id).copied()
    }

    /// Checks one expression against the actual extent.
    pub fn validate_expr(&self, expr: &DimExpr, actual: usize) -> Result<(), String> {
        let value = expr
            .evaluate_env(self)
            .ok_or_else(|| alloc::format!("symbolic dimension {:?} remains unresolved", expr))?;
        if value == actual {
            Ok(())
        } else {
            Err(alloc::format!(
                "expected expression {:?} = {}, got {}",
                expr,
                value,
                actual
            ))
        }
    }

    #[must_use]
    /// Whether the symbol has a binding.
    pub fn is_bound(&self, id: SymbolId) -> bool {
        self.bindings.contains_key(&id)
    }

    /// Checks every constraint in order.
    pub fn validate_constraints(&self, constraints: &[Constraint]) -> Result<(), String> {
        for (index, constraint) in constraints.iter().enumerate() {
            let result = match constraint {
                Constraint::Equal { lhs, rhs } => lhs
                    .evaluate_env(self)
                    .zip(rhs.evaluate_env(self))
                    .map(|(l, r)| l == r),
                Constraint::LowerBound { value, bound } => {
                    value.evaluate_env(self).map(|value| value >= *bound)
                }
                Constraint::UpperBound { value, bound } => {
                    value.evaluate_env(self).map(|value| value <= *bound)
                }
                Constraint::Divisible { value, divisor } => value
                    .evaluate_env(self)
                    .map(|value| *divisor != 0 && value % divisor == 0),
                Constraint::BroadcastCompatible { lhs, rhs } => lhs
                    .evaluate_env(self)
                    .zip(rhs.evaluate_env(self))
                    .map(|(lhs, rhs)| lhs == rhs || lhs == 1 || rhs == 1),
            };
            match result {
                Some(true) => {}
                Some(false) => {
                    return Err(alloc::format!(
                        "shape constraint {} failed: {:?}",
                        index,
                        constraint
                    ));
                }
                None => {
                    return Err(alloc::format!(
                        "shape constraint {} remains unresolved: {:?}",
                        index,
                        constraint
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// One dimension extent: constant, symbol, alias, or arithmetic combination.
pub enum DimExpr {
    /// A fixed extent known at capture time.
    Const(usize),
    /// A frontend-local symbol token. It must be allocated by graph capture
    /// before this expression is used as compiled metadata.
    Fresh(u32),
    /// Fresh axis minted by a producer during capture.
    NamedFresh {
        /// Capture-time id of the producing operation.
        source: u32,
        /// Human-readable name for diagnostics.
        name: String,
        /// Canonical identity binding this axis to its producer.
        identity: String,
    },
    /// A named dimension variable resolved at run time.
    Symbol(SymbolId),
    /// Reference to a registered named symbol.
    NamedSymbol {
        /// Symbol identifier.
        id: SymbolId,
        /// Human-readable name for diagnostics.
        name: String,
        /// Canonical identity binding this axis to its producer.
        identity: String,
    },
    /// A derived dimension that retains the semantic identity of its axis.
    NamedExpr {
        /// The aliased dimension expression.
        expr: Box<DimExpr>,
        /// Symbol identifier.
        id: SymbolId,
        /// Human-readable name for diagnostics.
        name: String,
        /// Canonical identity binding this axis to its producer.
        identity: String,
    },
    /// Sum of two dimension expressions.
    Add(Box<DimExpr>, Box<DimExpr>),
    /// Difference of two dimension expressions.
    Sub(Box<DimExpr>, Box<DimExpr>),
    /// Product of two dimension expressions.
    Mul(Box<DimExpr>, Box<DimExpr>),
    /// Division that must divide exactly to be legal.
    ExactDiv(Box<DimExpr>, Box<DimExpr>),
    /// Right-aligned broadcast combination of two extents.
    Broadcast(Box<DimExpr>, Box<DimExpr>),
    /// Minimum of two extents.
    Min(Box<DimExpr>, Box<DimExpr>),
    /// Maximum of two extents.
    Max(Box<DimExpr>, Box<DimExpr>),
    /// Extent could not be determined symbolically.
    Unknown,
}

impl DimExpr {
    #[must_use]
    /// Algebraic simplification to a canonical form.
    pub fn simplify(self) -> Self {
        match self {
            Self::NamedExpr {
                expr,
                id,
                name,
                identity,
            } => Self::NamedExpr {
                expr: Box::new(expr.simplify()),
                id,
                name,
                identity,
            },
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
            Self::Sub(lhs, rhs) => match (*lhs, *rhs) {
                (Self::Const(lhs), Self::Const(rhs)) => lhs
                    .checked_sub(rhs)
                    .map(Self::Const)
                    .unwrap_or(Self::Unknown),
                (lhs, Self::Const(0)) => lhs,
                (lhs, rhs) => Self::Sub(Box::new(lhs), Box::new(rhs)),
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
            Self::Min(lhs, rhs) => match (*lhs, *rhs) {
                (Self::Const(lhs), Self::Const(rhs)) => Self::Const(lhs.min(rhs)),
                (lhs, rhs) if lhs == rhs => lhs,
                (lhs, rhs) => Self::Min(Box::new(lhs), Box::new(rhs)),
            },
            Self::Max(lhs, rhs) => match (*lhs, *rhs) {
                (Self::Const(lhs), Self::Const(rhs)) => Self::Const(lhs.max(rhs)),
                (lhs, rhs) if lhs == rhs => lhs,
                (lhs, rhs) => Self::Max(Box::new(lhs), Box::new(rhs)),
            },
            other => other,
        }
    }

    /// Evaluates to a concrete extent given bindings.
    pub fn evaluate(&self, symbols: &[(SymbolId, usize)]) -> Option<usize> {
        match self {
            Self::Const(value) => Some(*value),
            Self::Symbol(id) | Self::NamedSymbol { id, .. } => symbols
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, value)| *value),
            Self::Fresh(_) | Self::NamedFresh { .. } => None,
            Self::NamedExpr { expr, id, .. } => symbols
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, value)| *value)
                .or_else(|| expr.evaluate(symbols)),
            Self::Add(lhs, rhs) => lhs.evaluate(symbols)?.checked_add(rhs.evaluate(symbols)?),
            Self::Sub(lhs, rhs) => lhs.evaluate(symbols)?.checked_sub(rhs.evaluate(symbols)?),
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
            Self::Min(lhs, rhs) => Some(lhs.evaluate(symbols)?.min(rhs.evaluate(symbols)?)),
            Self::Max(lhs, rhs) => Some(lhs.evaluate(symbols)?.max(rhs.evaluate(symbols)?)),
            Self::Unknown => None,
        }
    }

    fn evaluate_env(&self, environment: &SymbolEnvironment) -> Option<usize> {
        match self {
            Self::Const(value) => Some(*value),
            Self::Symbol(id) | Self::NamedSymbol { id, .. } => environment.get(*id),
            Self::Fresh(_) | Self::NamedFresh { .. } => None,
            Self::NamedExpr { expr, id, .. } => environment
                .get(*id)
                .or_else(|| expr.evaluate_env(environment)),
            Self::Add(lhs, rhs) => lhs
                .evaluate_env(environment)?
                .checked_add(rhs.evaluate_env(environment)?),
            Self::Sub(lhs, rhs) => lhs
                .evaluate_env(environment)?
                .checked_sub(rhs.evaluate_env(environment)?),
            Self::Mul(lhs, rhs) => lhs
                .evaluate_env(environment)?
                .checked_mul(rhs.evaluate_env(environment)?),
            Self::ExactDiv(lhs, rhs) => {
                let lhs = lhs.evaluate_env(environment)?;
                let rhs = rhs.evaluate_env(environment)?;
                (rhs != 0 && lhs % rhs == 0).then_some(lhs / rhs)
            }
            Self::Broadcast(lhs, rhs) => {
                let lhs = lhs.evaluate_env(environment)?;
                let rhs = rhs.evaluate_env(environment)?;
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
            Self::Min(lhs, rhs) => Some(
                lhs.evaluate_env(environment)?
                    .min(rhs.evaluate_env(environment)?),
            ),
            Self::Max(lhs, rhs) => Some(
                lhs.evaluate_env(environment)?
                    .max(rhs.evaluate_env(environment)?),
            ),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// Symbolic rank: static or fully dynamic.
pub enum RankExpr {
    /// Extent fixed at capture time.
    Static(usize),
    /// Extent resolved only when data arrives.
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// Predicates binding symbolic extents at validation time.
pub enum Constraint {
    /// Two expressions must evaluate equal.
    Equal {
        /// Left-hand expression.
        lhs: DimExpr,
        /// Right-hand expression.
        rhs: DimExpr,
    },
    /// Expression must be at least the bound.
    LowerBound {
        /// Expression being constrained.
        value: DimExpr,
        /// Bound it must respect.
        bound: usize,
    },
    /// Expression must not exceed the bound.
    UpperBound {
        /// Expression being constrained.
        value: DimExpr,
        /// Bound it must respect.
        bound: usize,
    },
    /// Expression must divide exactly.
    Divisible {
        /// Expression being constrained.
        value: DimExpr,
        /// Divisor that must divide exactly.
        divisor: usize,
    },
    /// Pair must combine under broadcasting.
    BroadcastCompatible {
        /// Left-hand expression.
        lhs: DimExpr,
        /// Right-hand expression.
        rhs: DimExpr,
    },
}

impl Constraint {
    #[must_use]
    /// Builds an equality constraint.
    pub fn equal(lhs: DimExpr, rhs: DimExpr) -> Self {
        Self::Equal { lhs, rhs }
    }

    #[must_use]
    /// Builds a broadcast-compatibility constraint.
    pub fn broadcast(lhs: DimExpr, rhs: DimExpr) -> Self {
        Self::BroadcastCompatible { lhs, rhs }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// Symbolic shape: rank, dims, and constraints.
pub struct ShapeExpr {
    /// Rank expression.
    pub rank: RankExpr,
    /// One expression per axis.
    pub dims: Vec<DimExpr>,
    /// Constraints binding symbols.
    pub constraints: Vec<Constraint>,
}

impl ShapeExpr {
    #[must_use]
    /// Builds a fully concrete shape expression.
    pub fn concrete(dims: &[usize]) -> Self {
        Self {
            rank: RankExpr::Static(dims.len()),
            dims: dims.iter().copied().map(DimExpr::Const).collect(),
            constraints: Vec::new(),
        }
    }

    #[must_use]
    /// Builds a shape whose tail dims become fresh symbols.
    pub fn symbolic(dims: &[usize], base: u32) -> Self {
        Self {
            rank: RankExpr::Static(dims.len()),
            dims: (0..dims.len())
                .map(|axis| DimExpr::Fresh(base.saturating_add(axis as u32)))
                .collect(),
            constraints: Vec::new(),
        }
    }

    #[must_use]
    /// Attaches constraints.
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Validates actual runtime dimensions against this shape.
    pub fn validate(&self, actual: &[usize]) -> Result<(), String> {
        let mut environment = SymbolEnvironment::default();
        self.bind_and_validate(actual, &mut environment)?;
        environment.validate_constraints(&self.constraints)
    }

    /// Binds symbols from actuals and validates in one step.
    pub fn bind_and_validate(
        &self,
        actual: &[usize],
        environment: &mut SymbolEnvironment,
    ) -> Result<(), String> {
        if let RankExpr::Static(rank) = self.rank
            && rank != actual.len()
        {
            return Err(alloc::format!(
                "expected rank {}, got {}",
                rank,
                actual.len()
            ));
        }
        for (axis, (expr, value)) in self.dims.iter().zip(actual.iter().copied()).enumerate() {
            match expr {
                DimExpr::Const(expected) if *expected != value => {
                    return Err(alloc::format!(
                        "axis {} expected dimension {}, got {}",
                        axis,
                        expected,
                        value
                    ));
                }
                DimExpr::Symbol(id) => {
                    environment.bind(*id, value)?;
                }
                DimExpr::NamedSymbol { id, .. } => {
                    environment.bind(*id, value)?;
                }
                DimExpr::NamedExpr { id, .. } => {
                    environment.bind(*id, value)?;
                }
                DimExpr::Fresh(_) | DimExpr::NamedFresh { .. } => {
                    return Err(alloc::format!(
                        "unallocated frontend symbol in compiled shape expression: {:?}",
                        expr
                    ));
                }
                _ => {}
            }
        }
        for (axis, (expr, value)) in self.dims.iter().zip(actual.iter().copied()).enumerate() {
            match expr {
                DimExpr::Const(expected) if *expected != value => {
                    return Err(alloc::format!(
                        "axis {} expected dimension {}, got {}",
                        axis,
                        expected,
                        value
                    ));
                }
                DimExpr::Symbol(_) | DimExpr::NamedSymbol { .. } => {}
                DimExpr::NamedExpr { expr, .. } => {
                    environment.validate_expr(expr, value)?;
                }
                DimExpr::Fresh(_) | DimExpr::NamedFresh { .. } => {
                    return Err(alloc::format!(
                        "unallocated frontend symbol in compiled shape expression: {:?}",
                        expr
                    ));
                }
                _ => environment.validate_expr(expr, value)?,
            }
        }
        Ok(())
    }
}
