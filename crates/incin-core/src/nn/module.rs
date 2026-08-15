use crate::backend_authoring::VariableBackend;
use crate::err::Result;
use crate::shapes::{DynShape, Shape};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::transfer::ToDevice;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Receives trainable parameter leaves without exposing backend handle maps.
pub trait ParameterVisitor<B: VariableBackend> {
    fn visit_param<S, K, Train>(
        &mut self,
        path: &crate::nn::StatePath,
        param: &crate::nn::param::Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: crate::nn::param::TrainState;
}

/// Traverses trainable parameter leaves using a typed visitor.
pub trait VisitParameters<B: VariableBackend> {
    /// Number of flat module slots occupied by this subtree.
    fn flat_width() -> usize
    where
        Self: Sized,
    {
        1
    }

    fn visit_parameters<V: ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()>;

    /// Visits this subtree at a flat positional index under `parent`.
    fn visit_parameters_flat<V: ParameterVisitor<B>>(
        &self,
        parent: &crate::nn::StatePath,
        base_index: usize,
        visitor: &mut V,
    ) -> Result<()>
    where
        Self: Sized,
    {
        self.visit_parameters(&parent.index(base_index), visitor)
    }
}

/// Recursively switches a module (and every submodule reachable through
/// `#[module]`-derived fields) between training and evaluation behavior —
/// `#[module]` auto-implements this for derived modules.
/// typed state visitors, walking every field and delegating to its `TrainMode`
/// implementation, so calling `.eval()` on a top-level model
/// propagates all the way down to every nested [`crate::nn::dropout::Dropout`]
/// without the caller needing to reach into the tree by hand.
///
/// `set_training` defaults to a no-op, so any leaf layer with no
/// training-dependent behavior (`Linear`, `ReLU`, `Conv2d`, ...) can opt in
/// with a bare `impl TrainMode for X {}` — this is what makes it possible
/// for [`Sequential`] to require `L1: TrainMode, L2: TrainMode` (see its own
/// impl below) without forcing every existing layer type to implement real
/// logic. Containers use explicit trait bounds and direct field calls.
///
/// ## What this currently affects
///
/// Only [`crate::nn::dropout::Dropout`] has training-dependent behavior
/// today (`is_training` gates whether it randomly zeroes elements or acts as
/// an identity function — see its own doc). **`BatchNorm2d` does not
/// currently respond to this call** — its own `forward` always normalizes
/// using the supplied running statistics regardless of mode (a deliberate,
/// already-documented "inference-mode-only" scope carried forward from an
/// earlier design decision — see the `_momentum` parameter's doc comment in
/// `cpu/ops/norm.rs::batch_norm_impl`), so it opts into `TrainMode` as a
/// harmless no-op via the macro's default rather than silently claiming a
/// behavior change that isn't actually implemented. Reversing that
/// BatchNorm scope decision is a separate, larger, cross-backend proposal,
/// not part of this one.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::tensor::device::Cpu>;
/// use incin::prelude::*;
///
/// let mut model = seq!(
///     Linear::<s![4, 4], DefaultBackend>::build(())?,
///     Dropout::new(0.5)
/// );
/// let x = Tensor::<Dyn, DefaultBackend>::zeros(vec![1, 4])?;
///
/// model.eval();   // nested Dropout layers become identity functions
/// let out = model.forward(x)?;
/// model.train();  // back to normal (randomized) training behavior
/// # Ok(()) }
/// ```
pub trait TrainMode {
    /// Sets training mode on this module and recursively on every
    /// submodule. `train()`/`eval()` are the ergonomic entry points — most
    /// callers should use those instead of calling this directly. Defaults
    /// to a no-op so leaf layers with no training-dependent behavior can
    /// opt in with a bare `impl TrainMode for X {}`.
    #[inline]
    fn set_training(&mut self, _training: bool) {}

    /// Switches to training mode (the default after construction).
    #[inline]
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Switches to evaluation/inference mode.
    #[inline]
    fn eval(&mut self) {
        self.set_training(false);
    }
}

/// Explicit field contract used by module traversal and diagnostics.
pub trait ShapeInfo {
    /// Returns a compact shape description when the field owns tensor-shaped state.
    fn shape_info(&self) -> Option<String>;
}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: crate::nn::param::TrainState,
> ShapeInfo for crate::nn::param::Param<S, B, K, Train>
{
    fn shape_info(&self) -> Option<String> {
        Some(format!("{:?}", self.shape_dims()))
    }
}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend, K: DType> ShapeInfo
    for crate::nn::param::Buffer<S, B, K>
{
    fn shape_info(&self) -> Option<String> {
        Some(format!("{:?}", self.shape_dims()))
    }
}

impl<T: ShapeInfo> ShapeInfo for Option<T> {
    fn shape_info(&self) -> Option<String> {
        self.as_ref().and_then(ShapeInfo::shape_info)
    }
}

/// A generic Neural Network Layer or Module.
/// Capable of taking an input and returning an output or error.
///
/// `Module` is the fundamental building block of neural networks in Incin.
/// Any layer, from a simple ReLU to an entire ResNet architecture, implements `Module`.
///
/// ## Deriving Modules
///
/// While you can implement `Module` manually, the recommended approach is to use the `#[module]` attribute macro
/// provided by `incin-macros`. This automatically derives typed visitors and structural boilerplate.
///
/// ```rust
/// # extern crate incin_core as incin;
/// use incin::prelude::*;
///
/// #[module]
/// pub struct MyLayer<B: VariableBackend> {
///     weight: Param<s![128, 128], B>,
///     bias: Param<s![128], B>,
/// }
///
/// impl<B: VariableBackend> Module<Tensor<s![1, 128], B>> for MyLayer<B> {
///     type Output = Tensor<s![1, 128], B>;
///     type Error = Error;
///
///     fn forward(&self, x: Tensor<s![1, 128], B>) -> Result<Self::Output> {
///         // Custom logic here
///         Ok(x)
///     }
/// }
/// ```
pub trait Module<Input> {
    /// The output tensor type produced by this module's forward pass.
    type Output;
    /// The error type returned if the forward pass fails.
    type Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, input: Input) -> core::result::Result<Self::Output, Self::Error>;
}

/// A sequential container for composing two modules.
/// `Sequential` automatically implements `Module` if the inner modules are compatible.
#[derive(Debug, Clone)]
pub struct Sequential<L1, L2>(pub L1, pub L2);

impl<I, L1, L2> Module<I> for Sequential<L1, L2>
where
    L1: Module<I>,
    L2: Module<L1::Output, Error = L1::Error>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = L2::Output;
    /// The error type returned if the forward pass fails.
    type Error = L1::Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, input: I) -> core::result::Result<Self::Output, Self::Error> {
        let out1 = self.0.forward(input)?;
        self.1.forward(out1)
    }
}

impl<B: crate::tensor::backend::VariableBackend, NewD: Device, L1, L2>
    crate::tensor::transfer::ToDevice<B, NewD> for Sequential<L1, L2>
where
    L1: ToDevice<B, NewD>,
    L2: ToDevice<B, NewD>,
{
    /// The same `Sequential` with each inner module transferred to `NewD`.
    type Output = Sequential<L1::Output, L2::Output>;
    /// Transfers both inner modules to the new device.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(Sequential(self.0.to_device(arg)?, self.1.to_device(arg)?))
    }
}

// Explicit bounds + direct calls for every leaf layer with no
// training-dependent behavior
// (`Linear`, `ReLU`, `Conv2d`, pooling layers, ...) implements `TrainMode`
// via its default no-op body specifically so this bound is satisfiable for
// real `seq!`-built chains, not just chains containing `Dropout`.
impl<L1: TrainMode, L2: TrainMode> TrainMode for Sequential<L1, L2> {
    /// Recursively sets training mode on both inner modules.
    fn set_training(&mut self, training: bool) {
        self.0.set_training(training);
        self.1.set_training(training);
    }
}

impl<B, L1, L2> crate::nn::VisitState<B> for Sequential<L1, L2>
where
    B: crate::tensor::backend::VariableBackend,
    L1: crate::nn::VisitState<B>,
    L2: crate::nn::VisitState<B>,
{
    fn flat_width() -> usize {
        L1::flat_width() + L2::flat_width()
    }

    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.visit_state_flat(path, 0, visitor)
    }

    fn visit_state_flat<V: crate::nn::StateVisitor<B>>(
        &self,
        parent: &crate::nn::StatePath,
        base_index: usize,
        visitor: &mut V,
    ) -> Result<()> {
        self.0.visit_state_flat(parent, base_index, visitor)?;
        self.1
            .visit_state_flat(parent, base_index + L1::flat_width(), visitor)
    }
}

impl<B, L1, L2> crate::nn::VisitStateMut<B> for Sequential<L1, L2>
where
    B: crate::tensor::backend::VariableBackend,
    L1: crate::nn::VisitStateMut<B>,
    L2: crate::nn::VisitStateMut<B>,
{
    fn flat_width() -> usize {
        L1::flat_width() + L2::flat_width()
    }

    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.visit_state_mut_flat(path, 0, visitor)
    }

    fn visit_state_mut_flat<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        parent: &crate::nn::StatePath,
        base_index: usize,
        visitor: &mut V,
    ) -> Result<()> {
        self.0.visit_state_mut_flat(parent, base_index, visitor)?;
        self.1
            .visit_state_mut_flat(parent, base_index + L1::flat_width(), visitor)
    }
}

impl<B, L1, L2> crate::nn::VisitParameters<B> for Sequential<L1, L2>
where
    B: crate::tensor::backend::VariableBackend,
    L1: crate::nn::VisitParameters<B>,
    L2: crate::nn::VisitParameters<B>,
{
    fn flat_width() -> usize {
        L1::flat_width() + L2::flat_width()
    }

    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.visit_parameters_flat(path, 0, visitor)
    }

    fn visit_parameters_flat<V: crate::nn::ParameterVisitor<B>>(
        &self,
        parent: &crate::nn::StatePath,
        base_index: usize,
        visitor: &mut V,
    ) -> Result<()> {
        self.0.visit_parameters_flat(parent, base_index, visitor)?;
        self.1
            .visit_parameters_flat(parent, base_index + L1::flat_width(), visitor)
    }
}

impl<T, B: crate::tensor::backend::VariableBackend> crate::nn::VisitStateMut<B>
    for core::marker::PhantomData<T>
where
    T: crate::tensor::dtype::DType,
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        _path: &crate::nn::StatePath,
        _visitor: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl<L, B> crate::nn::VisitState<B> for Option<L>
where
    L: crate::nn::VisitState<B>,
    B: crate::tensor::backend::VariableBackend,
{
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        if let Some(value) = self {
            value.visit_state(path, visitor)?;
        }
        Ok(())
    }
}

impl<L, B> crate::nn::VisitStateMut<B> for Option<L>
where
    L: crate::nn::VisitStateMut<B>,
    B: crate::tensor::backend::VariableBackend,
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        if let Some(value) = self {
            value.visit_state_mut(path, visitor)?;
        }
        Ok(())
    }
}

impl<L, B> crate::nn::VisitParameters<B> for Option<L>
where
    L: crate::nn::VisitParameters<B>,
    B: crate::tensor::backend::VariableBackend,
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        if let Some(value) = self {
            value.visit_parameters(path, visitor)?;
        }
        Ok(())
    }
}

/// Represents a node in the neural network layer structure metadata tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LayerNode {
    /// The display name of this layer node.
    pub name: String,
    /// The type name of this layer node.
    pub type_name: String,
    /// Human-readable string describing the layer's shape for display purposes.
    pub shape_info: String,
    /// Nested child layer nodes for hierarchical visualization.
    pub children: Vec<LayerNode>,
}

/// A trait implemented by all Neural Network modules to report their structural architecture.
pub trait NamedLayers {
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<LayerNode>;

    /// Returns a formatted human-readable architecture summary table.
    fn summary(&self) -> String {
        format_layer_summary(&self.layer_structure(""))
    }

    /// [`Self::summary`] plus a params/MACs/FLOPs totals footer at the given
    /// batch size — see [`format_layer_summary_with_stats`]. Only callable
    /// when `Self` also implements
    /// [`crate::nn::stats::ComputeStats`] (every `#[module]`
    /// struct does, automatically).
    fn summary_with_stats(&self, batch: u64) -> String
    where
        Self: crate::nn::stats::ComputeStats,
    {
        format_layer_summary_with_stats(&self.layer_structure(""), self.stats(batch))
    }
}

/// Formats a slice of `LayerNode` items into a human-readable printable tree table.
pub fn format_layer_summary(nodes: &[LayerNode]) -> String {
    let mut out = String::new();
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&format!(
        "{:<32} {:<28} {:<16}\n",
        "Layer (Name: Type)", "Shape / Spec", "Details"
    ));
    out.push_str(
        "================================================================================\n",
    );

    fn print_node(node: &LayerNode, indent: usize, out: &mut String) {
        let indent_str = "  ".repeat(indent);
        let clean_type = clean_type_name(&node.type_name);
        let name_type = if node.name.is_empty() {
            format!("{}{}", indent_str, clean_type)
        } else {
            format!("{}{}: {}", indent_str, node.name, clean_type)
        };
        let shape = if node.shape_info.is_empty() {
            "-"
        } else {
            &node.shape_info
        };
        out.push_str(&format!("{:<32} {:<28} {:<16}\n", name_type, shape, "-"));
        for child in &node.children {
            print_node(child, indent + 1, out);
        }
    }

    for node in nodes {
        print_node(node, 0, &mut out);
    }
    out.push_str(
        "================================================================================\n",
    );
    out
}

/// [`format_layer_summary`] plus a totals footer (params / MACs / FLOPs at
/// the given `stats`' batch size) — the "optional stats column" from
/// `docs/growth/04-compile-time-stats.md`'s v1 acceptance criteria. This is
/// a **totals row**, not a per-layer breakdown: threading per-node stats
/// through `LayerNode` itself would mean adding a batch parameter to
/// `NamedLayers::layer_structure`, a breaking signature change to a widely
/// implemented trait that a "ships this week" v1 shouldn't take on. Extends
/// [`format_layer_summary`] by composition rather than duplicating its tree
/// printer.
pub fn format_layer_summary_with_stats(
    nodes: &[LayerNode],
    stats: crate::nn::stats::ModelStats,
) -> String {
    let mut out = format_layer_summary(nodes);
    out.push_str(&format!(
        "Total params: {}    MACs: {}    FLOPs: {}\n",
        stats.params, stats.macs, stats.flops
    ));
    out.push_str(
        "================================================================================\n",
    );
    out
}

/// Cleans a fully qualified Rust type path into a simple name (e.g. `incin::nn::linear::Linear<...>` -> `Linear`).
pub fn clean_type_name(raw: &str) -> String {
    let without_generics = match raw.find('<') {
        Some(idx) => &raw[..idx],
        None => raw,
    };
    let name = match without_generics.rfind("::") {
        Some(idx) => &without_generics[idx + 2..],
        None => without_generics,
    };
    name.trim().to_string()
}

/// Recursively updates child node prefixes when a parent node is renamed.
pub fn update_node_name_prefix(node: &mut LayerNode, new_name: &str) {
    let old_name = node.name.clone();
    node.name = new_name.to_string();
    for child in &mut node.children {
        let child_suffix = if old_name.is_empty() {
            &child.name
        } else if child.name.starts_with(&old_name) {
            let prefix_len = old_name.len();
            if child.name.len() > prefix_len && child.name.as_bytes()[prefix_len] == b'.' {
                &child.name[prefix_len + 1..]
            } else {
                &child.name[prefix_len..]
            }
        } else {
            &child.name
        };
        let new_child_name = if new_name.is_empty() {
            child_suffix.to_string()
        } else {
            format!("{}.{}", new_name, child_suffix)
        };
        update_node_name_prefix(child, &new_child_name);
    }
}

/// Flatten helper to assign sequential names (e.g. Linear1, ReLU1, Linear2) to a slice of child nodes.
pub fn assign_sequential_names(nodes: &mut [LayerNode], prefix: &str) {
    let mut type_counts = alloc::collections::BTreeMap::new();
    for (i, node) in nodes.iter_mut().enumerate() {
        let clean_type = clean_type_name(&node.type_name);
        let count = type_counts.entry(clean_type.clone()).or_insert(0);
        *count += 1;

        let seq_name = if !clean_type.is_empty() && clean_type != "LayerNode" {
            format!("{}{}", clean_type, count)
        } else {
            format!("{}", i)
        };

        let new_name = if prefix.is_empty() {
            seq_name
        } else {
            format!("{}.{}", prefix, seq_name)
        };

        update_node_name_prefix(node, &new_name);
    }
}

impl<L1: NamedLayers, L2: NamedLayers> NamedLayers for Sequential<L1, L2> {
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<LayerNode> {
        let mut nodes = Vec::new();
        nodes.extend(self.0.layer_structure(""));
        nodes.extend(self.1.layer_structure(""));

        assign_sequential_names(&mut nodes, prefix);
        nodes
    }
}

impl<T: NamedLayers> NamedLayers for Option<T> {
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<LayerNode> {
        if let Some(layer) = self {
            layer.layer_structure(prefix)
        } else {
            Vec::new()
        }
    }
}

impl<T: TrainMode> TrainMode for Option<T> {
    fn set_training(&mut self, training: bool) {
        if let Some(value) = self {
            value.set_training(training);
        }
    }
}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: crate::nn::param::TrainState,
> NamedLayers for crate::nn::param::Param<S, B, K, Train>
{
    fn layer_structure(&self, _prefix: &str) -> Vec<LayerNode> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend, K: DType> NamedLayers
    for crate::nn::param::Buffer<S, B, K>
{
    fn layer_structure(&self, _prefix: &str) -> Vec<LayerNode> {
        Vec::new()
    }
}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: crate::nn::param::TrainState,
> TrainMode for crate::nn::param::Param<S, B, K, Train>
{
}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend, K: DType> TrainMode
    for crate::nn::param::Buffer<S, B, K>
{
}

/// A macro to easily build Sequential models with many layers.
/// `seq!(L1, L2, L3)` expands to `Sequential(L1, Sequential(L2, L3))`.
///
/// Naming the *type* of that value (e.g. for a `#[module]` struct field)
/// still requires hand-nesting `Sequential<L1, Sequential<L2, L3>>` — see
/// `SeqTy!`, which generates that same nesting from the same
/// flat layer list so it never has to be written out by hand.
#[macro_export]
macro_rules! seq {
    ($l1:expr) => {
        $l1
    };
    ($l1:expr, $($tail:expr),+ $(,)?) => {
        $crate::nn::Sequential($l1, $crate::seq!($($tail),+))
    };
}

/// Names the nested [`Sequential`] *type* that [`seq!`] would build a
/// *value* of, from the same flat list of layer types.
///
/// `Sequential<L1, L2>` only composes two layers at a time, so a
/// three-or-more-layer model's field type has to be hand-nested
/// (`Sequential<A, Sequential<B, C>>`) even though `seq!(a, b, c)` already
/// builds the matching value without that nesting spelled out. This macro
/// mirrors `seq!`'s exact right-nesting rule at the type level:
/// `SeqTy!(L1, L2, L3)` expands to `Sequential<L1, Sequential<L2, L3>>`,
/// so a layer list only needs to be written once per meaning (the type via
/// this macro, the value via `seq!`) instead of the type being re-derived by
/// hand every time the layer list changes.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::tensor::device::Cpu>;
/// use incin::prelude::*;
///
/// type Net = SeqTy!(
///     Linear<s![768, 256], DefaultBackend>,
///     ReLU,
///     Linear<s![256, 10], DefaultBackend>
/// );
///
/// let net: Net = seq!(
///     Linear::<s![768, 256], DefaultBackend>::build(())?,
///     ReLU,
///     Linear::<s![256, 10], DefaultBackend>::build(())?
/// );
/// # Ok(()) }
/// ```
#[macro_export]
macro_rules! SeqTy {
    ($l1:ty) => {
        $l1
    };
    ($l1:ty, $($tail:ty),+ $(,)?) => {
        $crate::nn::Sequential<$l1, $crate::SeqTy!($($tail),+)>
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! seq_ty {
    ($($tokens:tt)*) => {
        $crate::SeqTy!($($tokens)*)
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! seq_type {
    ($($tokens:tt)*) => {
        $crate::SeqTy!($($tokens)*)
    };
}
