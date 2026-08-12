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
pub struct SymbolId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SymbolInfo {
    pub id: SymbolId,
    pub name: Option<String>,
    pub identity: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SymbolTable {
    pub symbols: Vec<SymbolInfo>,
    pub constraints: Vec<Constraint>,
}

impl SymbolTable {
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
    pub fn environment(&self) -> SymbolEnvironment {
        SymbolEnvironment::default()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymbolEnvironment {
    bindings: BTreeMap<SymbolId, usize>,
}

impl SymbolEnvironment {
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

    #[must_use]
    pub fn get(&self, id: SymbolId) -> Option<usize> {
        self.bindings.get(&id).copied()
    }

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
    pub fn is_bound(&self, id: SymbolId) -> bool {
        self.bindings.contains_key(&id)
    }

    pub fn validate_constraints(&self, constraints: &[Constraint]) -> Result<(), String> {
        for constraint in constraints {
            let valid = match constraint {
                Constraint::Equal { lhs, rhs } => lhs
                    .evaluate_env(self)
                    .zip(rhs.evaluate_env(self))
                    .is_some_and(|(l, r)| l == r),
                Constraint::LowerBound { value, bound } => value
                    .evaluate_env(self)
                    .is_some_and(|value| value >= *bound),
                Constraint::UpperBound { value, bound } => value
                    .evaluate_env(self)
                    .is_some_and(|value| value <= *bound),
                Constraint::Divisible { value, divisor } => value
                    .evaluate_env(self)
                    .is_some_and(|value| *divisor != 0 && value % divisor == 0),
                Constraint::BroadcastCompatible { lhs, rhs } => lhs
                    .evaluate_env(self)
                    .zip(rhs.evaluate_env(self))
                    .is_some_and(|(lhs, rhs)| lhs == rhs || lhs == 1 || rhs == 1),
            };
            if !valid {
                return Err(alloc::format!("shape constraint failed: {:?}", constraint));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DimExpr {
    Const(usize),
    Symbol(SymbolId),
    NamedSymbol {
        id: SymbolId,
        name: String,
        identity: String,
    },
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
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
            other => other,
        }
    }

    #[must_use]
    pub fn evaluate(&self, symbols: &[(SymbolId, usize)]) -> Option<usize> {
        match self {
            Self::Const(value) => Some(*value),
            Self::Symbol(id) | Self::NamedSymbol { id, .. } => symbols
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, value)| *value),
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
            Self::Unknown => None,
        }
    }

    fn evaluate_env(&self, environment: &SymbolEnvironment) -> Option<usize> {
        match self {
            Self::Const(value) => Some(*value),
            Self::Symbol(id) | Self::NamedSymbol { id, .. } => environment.get(*id),
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

impl Constraint {
    #[must_use]
    pub fn equal(lhs: DimExpr, rhs: DimExpr) -> Self {
        Self::Equal { lhs, rhs }
    }

    #[must_use]
    pub fn broadcast(lhs: DimExpr, rhs: DimExpr) -> Self {
        Self::BroadcastCompatible { lhs, rhs }
    }
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
    pub fn symbolic(dims: &[usize], base: u32) -> Self {
        Self {
            rank: RankExpr::Static(dims.len()),
            dims: (0..dims.len())
                .map(|axis| DimExpr::Symbol(SymbolId(base.saturating_add(axis as u32))))
                .collect(),
            constraints: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn validate(&self, actual: &[usize]) -> Result<(), String> {
        let mut environment = SymbolEnvironment::default();
        self.bind_and_validate(actual, &mut environment)?;
        environment.validate_constraints(&self.constraints)
    }

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
                    environment.bind(*id, value)?;
                }
                DimExpr::NamedSymbol { id, .. } => {
                    environment.bind(*id, value)?;
                }
                _ => {}
            }
        }
        for (expr, value) in self.dims.iter().zip(actual.iter().copied()) {
            match expr {
                DimExpr::Const(expected) if *expected != value => {
                    return Err(alloc::format!(
                        "expected dimension {}, got {}",
                        expected,
                        value
                    ));
                }
                DimExpr::Symbol(_) | DimExpr::NamedSymbol { .. } => {}
                _ => environment.validate_expr(expr, value)?,
            }
        }
        Ok(())
    }
}
