use crate::prelude::{Backend, DType, Device, Dim, DynShape, Error, ErrorMessage, Result, Shape, Tensor, ToDevice};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Typed traversal and portable persistence contract for module state.
///
/// The public state representation is [`StateSnapshot`]. Backend storage is
/// reached only at typed `Param`/`Buffer` leaves during prepare/commit.
pub trait StateDict<B: Backend> {
    /// Collects exact, owned state without retaining a live backend tensor.
    /// Implementations for state-bearing leaves override this; the default is
    /// appropriate for stateless modules and legacy hand-written markers.
    fn collect_state(
        &self,
        _prefix: &crate::nn::StatePath,
        _snapshot: &mut crate::nn::StateSnapshot,
    ) -> Result<()> {
        Ok(())
    }

    /// Prepares all replacements. No live module state may be mutated here.
    fn prepare_state(
        &self,
        _prefix: &crate::nn::StatePath,
        _snapshot: &crate::nn::StateSnapshot,
        _plan: &mut crate::nn::StateLoadPlan,
    ) -> Result<()> {
        Ok(())
    }

    /// Commits a previously prepared replacement.
    fn commit_state(
        &mut self,
        _prefix: &crate::nn::StatePath,
        _plan: &mut crate::nn::StateLoadPlan,
    ) -> Result<()> {
        Ok(())
    }

    /// Returns an owned heterogeneous snapshot of this module.
    fn state_snapshot(&self) -> Result<crate::nn::StateSnapshot> {
        let mut snapshot = crate::nn::StateSnapshot::new();
        self.collect_state(&crate::nn::StatePath::root(), &mut snapshot)?;
        Ok(snapshot)
    }

    /// Strictly stages and commits a complete snapshot.
    fn load_state_snapshot(&mut self, snapshot: &crate::nn::StateSnapshot) -> Result<()> {
        let current = self.state_snapshot()?;
        let expected: alloc::collections::BTreeSet<_> =
            current.iter().map(|(path, _)| path).collect();
        let provided: alloc::collections::BTreeSet<_> =
            snapshot.iter().map(|(path, _)| path).collect();
        if expected != provided {
            let missing = expected
                .difference(&provided)
                .map(ToString::to_string)
                .collect::<alloc::vec::Vec<_>>();
            let unexpected = provided
                .difference(&expected)
                .map(ToString::to_string)
                .collect::<alloc::vec::Vec<_>>();
            return Err(Error::InvalidModuleState {
                operation: "load state",
                reason: ErrorMessage::new(format!(
                    "state paths differ: missing {:?}, unexpected {:?}",
                    missing, unexpected
                )),
            });
        }
        let mut plan = crate::nn::StateLoadPlan::new();
        self.prepare_state(&crate::nn::StatePath::root(), snapshot, &mut plan)?;
        self.commit_state(&crate::nn::StatePath::root(), &mut plan)
    }

    /// Number of flat positional slots occupied inside `Sequential`.
    fn flat_width() -> usize
    where
        Self: Sized,
    {
        1
    }

    /// Extracts all state as an owned heterogeneous snapshot.
    fn state_dict(&self) -> Result<crate::nn::StateSnapshot> {
        self.state_snapshot()
    }

    /// Strictly prepares and commits a complete snapshot.
    fn load_state_dict(&mut self, snapshot: &crate::nn::StateSnapshot) -> Result<()> {
        self.load_state_snapshot(snapshot)
    }
}

/// A trait implemented by all Neural Network modules.
/// Usually automatically derived via `#[incin::module]`.
pub trait Parameters<B: Backend> {
    /// Recursively extract all trainable parameters from this module into a named map.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    );

    /// Helper to retrieve all parameters as a new map.
    fn parameters(&self) -> alloc::collections::BTreeMap<String, B::RawVar> {
        let mut map = alloc::collections::BTreeMap::new();
        self.named_parameters("", &mut map);
        map
    }

    /// Number of flat positional slots this module occupies when nested
    /// inside a [`Sequential`] chain — `1` for any ordinary layer (this
    /// default, and every existing layer type inherits it automatically
    /// since `Parameters` is already implemented everywhere — no per-type
    /// opt-in needed, unlike `TrainMode`'s default-no-op design). Overridden
    /// only by [`Sequential`]'s own impl, which sums its two children's
    /// widths.
    ///
    /// `seq!(a, b, c)` builds the right-nested value
    /// `Sequential(a, Sequential(b, c))`, so a naive recursive
    /// `named_parameters` (literally prepending `"0."`/`"1."` at each level)
    /// produces keys that encode that nesting structure —
    /// `0.weight, 1.0.weight, 1.1.weight` — rather than PyTorch
    /// `nn.Sequential`'s flat `0.weight, 1.weight, 2.weight`. `flat_width`
    /// plus [`named_parameters_flat`](Self::named_parameters_flat) is the
    /// mechanism that fixes this: each level needs to know how many flat
    /// positional slots its *left* child consumed before it can correctly
    /// number its *right* child's first slot, and that width has to be
    /// known independent of how many parameters (if any) each layer
    /// actually has — a parameter-less layer like `ReLU` still occupies
    /// exactly one position, exactly like PyTorch reserves index `1` for a
    /// parameter-less middle layer in `nn.Sequential(Linear(3,3), ReLU(),
    /// Linear(3,3))`'s state dict (keys `0.weight/0.bias`, nothing at index
    /// `1`, then `2.weight/2.bias`).
    fn flat_width() -> usize
    where
        Self: Sized,
    {
        1
    }

    /// Collects named parameters using FLAT positional numbering relative
    /// to `base_index` (PyTorch `nn.Sequential` semantics), unlike
    /// `named_parameters`'s own literal prefix-string recursion. Default:
    /// treat `self` as one flat slot at `base_index`, delegating to
    /// `named_parameters`. [`Sequential`] overrides this to recurse with the
    /// correct running offset instead — see `flat_width`'s doc for why this
    /// exists.
    fn named_parameters_flat(
        &self,
        outer_prefix: &str,
        base_index: usize,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) where
        Self: Sized,
    {
        // `named_parameters`'s own `#[module]`-generated body ALREADY
        // appends a trailing `.` to a non-empty incoming prefix before
        // using it (`let prefix = if prefix.is_empty() { .. } else {
        // format!("{}.", prefix) }`) — passing an already-dotted prefix
        // here would double it up (`"0..weight"` instead of `"0.weight"`),
        // a real bug caught by `test_sequential_state_dict_keys_are_flat_like_pytorch`
        // actually asserting on key *strings* rather than just a count.
        let joined = if outer_prefix.is_empty() {
            format!("{base_index}")
        } else {
            format!("{outer_prefix}.{base_index}")
        };
        self.named_parameters(&joined, map);
    }
}

/// Recursively switches a module (and every submodule reachable through
/// `#[module]`-derived fields) between training and evaluation behavior —
/// `#[module]` auto-implements this exactly like it does `Parameters`/
/// `StateDict`, walking every field and delegating to whichever ones
/// implement `TrainMode` themselves (via the same autoref-specialization
/// pattern those two use), so calling `.eval()` on a top-level model
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
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
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

impl<S: Shape + DynShape, B: Backend, K: DType, Train: crate::nn::param::TrainState>
    ShapeInfo for crate::nn::param::Param<S, B, K, Train>
{
    fn shape_info(&self) -> Option<String> {
        Some(format!("{:?}", self.shape_dims()))
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType> ShapeInfo
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
/// provided by `incin-macros`. This automatically derives `Parameters`, `StateDict`, and generates structural boilerplate.
///
/// ```rust
/// # extern crate incin_core as incin;
/// use incin::prelude::*;
///
/// #[module]
/// pub struct MyLayer<B: Backend> {
///     weight: Param<s![128, 128], B>,
///     bias: Param<s![128], B>,
/// }
///
/// impl<B: Backend> Module<Tensor<s![1, 128], B>> for MyLayer<B> {
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

impl<B: Backend, NewD: Device, L1, L2> crate::tensor::transfer::ToDevice<B, NewD>
    for Sequential<L1, L2>
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

impl<B: Backend, L1, L2> Parameters<B> for Sequential<L1, L2>
where
    L1: Parameters<B>,
    L2: Parameters<B>,
{
    /// Entry point: flat-numbers from index `0`, matching PyTorch
    /// `nn.Sequential`'s state-dict key scheme — see `flat_width`'s doc.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        self.named_parameters_flat(prefix, 0, map);
    }

    /// Sums both children's widths — see `flat_width`'s doc.
    fn flat_width() -> usize {
        L1::flat_width() + L2::flat_width()
    }

    /// Recurses with a running index offset: the left child starts at
    /// `base_index`, the right child starts `L1::flat_width()` positions
    /// later. Each child's OWN `named_parameters_flat` (its default if it's
    /// an ordinary layer, or this same override if it's itself a
    /// `Sequential`) handles turning that starting index into real keys.
    fn named_parameters_flat(
        &self,
        outer_prefix: &str,
        base_index: usize,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        self.0.named_parameters_flat(outer_prefix, base_index, map);
        self.1
            .named_parameters_flat(outer_prefix, base_index + L1::flat_width(), map);
    }
}

// Explicit bounds + direct calls, matching `Parameters`/`StateDict`'s own
// impls for `Sequential` immediately below/above — NOT the autoref-
// specialization trick `#[module]`'s generated code uses for its fields.
// That trick only resolves to the "real" impl when the compiler can PROVE
// the bound holds while checking the generic code, which is impossible for
// `L1`/`L2` here: they're bare, unconstrained type parameters with no
// `TrainMode` bound, so autoref would *always* silently pick the no-op
// fallback regardless of what `L1`/`L2` are eventually monomorphized to —
// verified empirically while building this (a `Sequential<Linear<..>,
// Dropout>`'s `.eval()` call still ran `Dropout` in training mode with the
// autoref version). Every leaf layer with no training-dependent behavior
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

impl<B: Backend, L1, L2> StateDict<B> for Sequential<L1, L2>
where
    L1: StateDict<B>,
    L2: StateDict<B>,
{
    fn collect_state(
        &self,
        path: &crate::nn::StatePath,
        snapshot: &mut crate::nn::StateSnapshot,
    ) -> Result<()> {
        self.0.collect_state(&path.index(0), snapshot)?;
        self.1
            .collect_state(&path.index(L1::flat_width()), snapshot)
    }

    fn prepare_state(
        &self,
        path: &crate::nn::StatePath,
        snapshot: &crate::nn::StateSnapshot,
        plan: &mut crate::nn::StateLoadPlan,
    ) -> Result<()> {
        self.0.prepare_state(&path.index(0), snapshot, plan)?;
        self.1
            .prepare_state(&path.index(L1::flat_width()), snapshot, plan)
    }

    fn commit_state(
        &mut self,
        path: &crate::nn::StatePath,
        plan: &mut crate::nn::StateLoadPlan,
    ) -> Result<()> {
        self.0.commit_state(&path.index(0), plan)?;
        self.1.commit_state(&path.index(L1::flat_width()), plan)
    }

    /// Sums both children's widths.
    fn flat_width() -> usize {
        L1::flat_width() + L2::flat_width()
    }
}

// Dummy implementations for primitive/marker types that are often fields in modules.
macro_rules! impl_dummy_state {
    ($($t:ty),+) => {
        $(
            impl<B: Backend> Parameters<B> for $t {
                /// Collects named trainable parameters into `map` under the given `prefix`.
                fn named_parameters(&self, _prefix: &str, _map: &mut alloc::collections::BTreeMap<String, B::RawVar>) {}
            }

            impl<B: Backend> StateDict<B> for $t {
            }
        )+
    };
}

impl_dummy_state!(usize, f32);

impl<T, B: Backend> Parameters<B> for core::marker::PhantomData<T>
where
    T: crate::prelude::DType,
{
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}
impl<T, B: Backend> StateDict<B> for core::marker::PhantomData<T> where T: crate::prelude::DType {}

impl<T: Parameters<B>, B: Backend> Parameters<B> for Option<T> {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        if let Some(v) = self {
            v.named_parameters(prefix, map);
        }
    }
}

impl<L: StateDict<B>, B: Backend> StateDict<B> for Option<L> {
    fn collect_state(
        &self,
        path: &crate::nn::StatePath,
        snapshot: &mut crate::nn::StateSnapshot,
    ) -> Result<()> {
        if let Some(value) = self {
            value.collect_state(path, snapshot)?;
        }
        Ok(())
    }
    fn prepare_state(
        &self,
        path: &crate::nn::StatePath,
        snapshot: &crate::nn::StateSnapshot,
        plan: &mut crate::nn::StateLoadPlan,
    ) -> Result<()> {
        if let Some(value) = self {
            value.prepare_state(path, snapshot, plan)?;
        }
        Ok(())
    }
    fn commit_state(
        &mut self,
        path: &crate::nn::StatePath,
        plan: &mut crate::nn::StateLoadPlan,
    ) -> Result<()> {
        if let Some(value) = self {
            value.commit_state(path, plan)?;
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

impl<S: Shape + DynShape, B: Backend, K: DType, Train: crate::nn::param::TrainState>
    NamedLayers for crate::nn::param::Param<S, B, K, Train>
{
    fn layer_structure(&self, _prefix: &str) -> Vec<LayerNode> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType> NamedLayers
    for crate::nn::param::Buffer<S, B, K>
{
    fn layer_structure(&self, _prefix: &str) -> Vec<LayerNode> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend, K: DType, Train: crate::nn::param::TrainState> TrainMode
    for crate::nn::param::Param<S, B, K, Train>
{
}

impl<S: Shape + DynShape, B: Backend, K: DType> TrainMode
    for crate::nn::param::Buffer<S, B, K>
{
}

/// A macro to easily build Sequential models with many layers.
/// `seq!(L1, L2, L3)` expands to `Sequential(L1, Sequential(L2, L3))`.
///
/// Naming the *type* of that value (e.g. for a `#[module]` struct field)
/// still requires hand-nesting `Sequential<L1, Sequential<L2, L3>>` — see
/// [`SeqTy!`], which generates that same nesting from the same
/// flat layer list so it never has to be written out by hand.
#[macro_export]
macro_rules! seq {
    ($l1:expr) => {
        $l1
    };
    ($l1:expr, $($tail:expr),+ $(,)?) => {
        $crate::prelude::Sequential($l1, $crate::seq!($($tail),+))
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
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
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
        $crate::prelude::Sequential<$l1, $crate::SeqTy!($($tail),+)>
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
