use crate::prelude::*;
use alloc::collections::BTreeMap;

/// A trait implemented by all Neural Network modules to manage their state (weights).
/// Usually automatically derived via `#[kindle::module]`.
pub trait StateDict<B: Backend> {
    /// Loads the module's state from a dictionary of dynamic tensors.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()>;

    /// Collects the module's state into a dictionary of dynamic tensors.
    fn state_dict(&self, prefix: &str, tensors: &mut BTreeMap<String, Tensor<Dyn, B>>);

    /// Helper: Serializes this module's state to a given serializer.
    fn save_to<S: crate::serialize::Serializer>(
        &self,
        serializer: &mut S,
    ) -> core::result::Result<(), S::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        let mut map = BTreeMap::new();
        self.state_dict("", &mut map);
        serializer.serialize(&map)
    }

    /// Helper: Deserializes this module's state from a given deserializer.
    fn load_from<D: crate::serialize::Deserializer>(
        &mut self,
        deserializer: &mut D,
        device: &DeviceId,
    ) -> Result<()>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        let map = deserializer
            .deserialize(device)
            .map_err(|e| Error::ShapeMismatch {
                op: "Deserialization",
                expected: vec![],
                got: vec![],
                msg: format!("{:?}", e),
            })?;
        self.load_state_dict("", &map)
    }
}

/// A trait implemented by all Neural Network modules.
/// Usually automatically derived via `#[kindle::module]`.
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
}

/// A trait to transfer ownership of a module to a new device.
pub trait ToDevice<B: Backend, NewD: Device> {
    /// The same module type, but rebuilt on backend `NewD` — e.g.
    /// `Linear<S, KindleBackend<T, Cpu>>` transferred to `Wgpu` becomes
    /// `Linear<S, KindleBackend<T, Wgpu>>`.
    type Output;
    /// Moves every parameter/buffer this module owns onto device `arg`,
    /// returning the module rebuilt on the new backend.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output>;
}

impl<T: ToDevice<B, NewD>, B: Backend, NewD: Device> ToDevice<B, NewD> for Option<T> {
    /// `None` stays `None`; `Some(t)` becomes `Some(t.to_device(..))`.
    type Output = Option<T::Output>;
    /// Transfers the wrapped value if present; a no-op for `None`.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        self.map(|t| t.to_device(arg)).transpose()
    }
}

impl<B: Backend, NewD: Device> ToDevice<B, NewD> for () {
    /// `()` has no per-device state, so transferring it is a no-op.
    type Output = ();

    /// No-op: `()` carries nothing to transfer.
    fn to_device(self, _arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(())
    }
}

#[doc(hidden)]
/// Autoref-specialization fallback: the `&&T` blanket impl every type gets
/// "for free," used so `#[module]`-derived fields that don't implement
/// `Parameters` (plain scalars, markers) contribute nothing instead of
/// failing to compile. Method resolution prefers `AutorefParameters`'s `&T`
/// impl when it applies, falling back to this `&&T` impl otherwise.
pub trait AutorefParametersFallback<B: Backend> {
    /// No-op: contributes no parameters.
    fn maybe_parameters(
        &self,
        _phantom: core::marker::PhantomData<B>,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}
impl<T, B: Backend> AutorefParametersFallback<B> for &&T {}

#[doc(hidden)]
/// The preferred (non-fallback) half of the autoref-specialization pair:
/// picked over `AutorefParametersFallback` for any type that actually
/// implements `Parameters`.
pub trait AutorefParameters<B: Backend> {
    /// Delegates to `Parameters::named_parameters`.
    fn maybe_parameters(
        &self,
        _phantom: core::marker::PhantomData<B>,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    );
}
impl<T: Parameters<B>, B: Backend> AutorefParameters<B> for &T {
    #[inline]
    /// Delegates to `Parameters::named_parameters`.
    fn maybe_parameters(
        &self,
        _marker: core::marker::PhantomData<B>,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        self.named_parameters(prefix, map);
    }
}

#[doc(hidden)]
/// Autoref-specialization fallback for `StateDict`: fields that don't
/// implement it (plain scalars, markers) silently contribute nothing to
/// save/load instead of failing to compile.
pub trait AutorefStateDictFallback<B: Backend> {
    /// No-op: nothing to load.
    fn maybe_load_state_dict(
        &mut self,
        _phantom: core::marker::PhantomData<B>,
        _prefix: &str,
        _tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        Ok(())
    }
    /// No-op: nothing to save.
    fn maybe_state_dict(
        &self,
        _phantom: core::marker::PhantomData<B>,
        _prefix: &str,
        _tensors: &mut BTreeMap<String, Tensor<Dyn, B>>,
    ) {
    }
}
impl<T, B: Backend> AutorefStateDictFallback<B> for &mut &mut T {}
impl<T, B: Backend> AutorefStateDictFallback<B> for &&T {}

#[doc(hidden)]
/// The preferred (non-fallback) half of the autoref-specialization pair:
/// picked over `AutorefStateDictFallback` for any type that actually
/// implements `StateDict`.
pub trait AutorefStateDict<B: Backend> {
    /// Delegates to `StateDict::load_state_dict`.
    fn maybe_load_state_dict(
        &mut self,
        _phantom: core::marker::PhantomData<B>,
        prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()>;
    /// Delegates to `StateDict::state_dict`.
    fn maybe_state_dict(
        &self,
        _phantom: core::marker::PhantomData<B>,
        prefix: &str,
        tensors: &mut BTreeMap<String, Tensor<Dyn, B>>,
    );
}

// For mutable operations
impl<T: StateDict<B>, B: Backend> AutorefStateDict<B> for &mut T {
    #[inline]
    /// Delegates to `StateDict::load_state_dict`.
    fn maybe_load_state_dict(
        &mut self,
        _phantom: core::marker::PhantomData<B>,
        prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        (*self).load_state_dict(prefix, tensors)
    }
    #[inline]
    /// Delegates to `StateDict::state_dict`.
    fn maybe_state_dict(
        &self,
        _phantom: core::marker::PhantomData<B>,
        prefix: &str,
        tensors: &mut BTreeMap<String, Tensor<Dyn, B>>,
    ) {
        (**self).state_dict(prefix, tensors)
    }
}

// For immutable operations (state_dict uses &self)
impl<T: StateDict<B>, B: Backend> AutorefStateDict<B> for &T {
    #[inline]
    /// Unreachable in practice: loading requires `&mut`, so this
    /// shared-reference impl only exists to satisfy the autoref-resolution
    /// pair's shape; it never actually gets called for loading.
    fn maybe_load_state_dict(
        &mut self,
        _phantom: core::marker::PhantomData<B>,
        _prefix: &str,
        _tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        Ok(()) // Should not be called
    }
    #[inline]
    /// Delegates to `StateDict::state_dict`.
    fn maybe_state_dict(
        &self,
        _phantom: core::marker::PhantomData<B>,
        prefix: &str,
        tensors: &mut BTreeMap<String, Tensor<Dyn, B>>,
    ) {
        (*self).state_dict(prefix, tensors)
    }
}

/// A generic Neural Network Layer or Module.
/// Capable of taking an input and returning an output or error.
///
/// `Module` is the fundamental building block of neural networks in Kindle.
/// Any layer, from a simple ReLU to an entire ResNet architecture, implements `Module`.
///
/// ## Deriving Modules
///
/// While you can implement `Module` manually, the recommended approach is to use the `#[module]` attribute macro
/// provided by `kindle-macros`. This automatically derives `Parameters`, `StateDict`, and generates structural boilerplate.
///
/// ```rust,ignore
/// use kindle::prelude::*;
///
/// #[module]
/// pub struct MyLayer<B: Backend> {
///     weight: Param<Tensor<s![128, 128], B>>,
///     bias: Param<Tensor<s![128], B>>,
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

impl<B: Backend, NewD: Device, L1, L2> ToDevice<B, NewD> for Sequential<L1, L2>
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
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        self.0.named_parameters(&format!("{}0.", prefix), map);
        self.1.named_parameters(&format!("{}1.", prefix), map);
    }
}

impl<B: Backend, L1, L2> StateDict<B> for Sequential<L1, L2>
where
    L1: StateDict<B>,
    L2: StateDict<B>,
{
    /// Loads parameters from a flat name→tensor map, in-place.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        self.0.load_state_dict(&format!("{}0.", prefix), tensors)?;
        self.1.load_state_dict(&format!("{}1.", prefix), tensors)?;
        Ok(())
    }

    /// Returns a flat map from parameter name to its raw tensor value.
    fn state_dict(&self, prefix: &str, tensors: &mut BTreeMap<String, Tensor<Dyn, B>>) {
        self.0.state_dict(&format!("{}0.", prefix), tensors);
        self.1.state_dict(&format!("{}1.", prefix), tensors);
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
                /// Loads parameters from a flat name→tensor map, in-place.
                fn load_state_dict(&mut self, _prefix: &str, _tensors: &BTreeMap<String, Tensor<Dyn, B>>) -> Result<()> {
                    Ok(())
                }

                /// Returns a flat map from parameter name to its raw tensor value.
                fn state_dict(&self, _prefix: &str, _tensors: &mut BTreeMap<String, Tensor<Dyn, B>>) {
                }
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
impl<T, B: Backend> StateDict<B> for core::marker::PhantomData<T>
where
    T: crate::prelude::DType,
{
    /// Loads parameters from a flat name→tensor map, in-place.
    fn load_state_dict(
        &mut self,
        _prefix: &str,
        _tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        Ok(())
    }
    /// Returns a flat map from parameter name to its raw tensor value.
    fn state_dict(&self, _prefix: &str, _tensors: &mut BTreeMap<String, Tensor<Dyn, B>>) {}
}

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
    /// Loads parameters from a flat name→tensor map, in-place.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        if let Some(v) = self {
            v.load_state_dict(prefix, tensors)?;
        }
        Ok(())
    }

    /// Returns a flat map from parameter name to its raw tensor value.
    fn state_dict(&self, prefix: &str, tensors: &mut BTreeMap<String, Tensor<Dyn, B>>) {
        if let Some(v) = self {
            v.state_dict(prefix, tensors);
        }
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
}

/// Cleans a fully qualified Rust type path into a simple name (e.g. `kindle::nn::linear::Linear<...>` -> `Linear`).
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

#[doc(hidden)]
/// Autoref-specialization fallback for `NamedLayers`: fields that don't
/// implement it report no structure (`None`) instead of failing to compile.
pub trait AutorefNamedLayersFallback {
    /// `None`: this field contributes no layer-structure node.
    fn maybe_layer_structure(&self, _prefix: &str) -> Option<Vec<LayerNode>> {
        None
    }
}
impl<T> AutorefNamedLayersFallback for &&T {}

#[doc(hidden)]
/// The preferred (non-fallback) half of the autoref-specialization pair:
/// picked over `AutorefNamedLayersFallback` for any type that actually
/// implements `NamedLayers`.
pub trait AutorefNamedLayers {
    /// Delegates to `NamedLayers::layer_structure`, wrapped in `Some`.
    fn maybe_layer_structure(&self, prefix: &str) -> Option<Vec<LayerNode>>;
}
impl<T: NamedLayers> AutorefNamedLayers for &T {
    #[inline]
    /// Delegates to `NamedLayers::layer_structure`, wrapped in `Some`.
    fn maybe_layer_structure(&self, prefix: &str) -> Option<Vec<LayerNode>> {
        Some(self.layer_structure(prefix))
    }
}

#[doc(hidden)]
/// Autoref-specialization fallback for shape-info reporting: fields with
/// no known shape (not a `Param`/`Buffer`) report `None` instead of
/// failing to compile.
pub trait AutorefShapeInfoFallback {
    /// `None`: this field has no shape to report.
    fn maybe_shape_info(&self) -> Option<String> {
        None
    }
}
impl<T> AutorefShapeInfoFallback for &&T {}

#[doc(hidden)]
/// The preferred (non-fallback) half of the autoref-specialization pair:
/// picked over `AutorefShapeInfoFallback` for `Param`/`Buffer` fields (and
/// anything else with a known shape).
pub trait AutorefShapeInfo {
    /// Renders this field's shape as a debug string, if it has one.
    fn maybe_shape_info(&self) -> Option<String>;
}
impl<S: Shape + DynShape, B: Backend> AutorefShapeInfo for &crate::nn::param::Param<S, B> {
    #[inline]
    /// Renders the parameter's dimensions, e.g. `[128, 256]`.
    fn maybe_shape_info(&self) -> Option<String> {
        Some(format!("{:?}", self.shape_dims()))
    }
}
impl<S: Shape + DynShape, B: Backend> AutorefShapeInfo for &crate::nn::param::Buffer<S, B> {
    #[inline]
    /// Renders the buffer's dimensions, e.g. `[128, 256]`.
    fn maybe_shape_info(&self) -> Option<String> {
        Some(format!("{:?}", self.shape_dims()))
    }
}
impl<T> AutorefShapeInfo for &Option<T>
where
    for<'a> &'a T: AutorefShapeInfo,
{
    #[inline]
    /// Delegates to the wrapped value's shape info; `None` for `None`.
    fn maybe_shape_info(&self) -> Option<String> {
        if let Some(val) = self {
            (&val).maybe_shape_info()
        } else {
            None
        }
    }
}

/// A macro to easily build Sequential models with many layers.
/// `seq!(L1, L2, L3)` expands to `Sequential(L1, Sequential(L2, L3))`.
///
/// Naming the *type* of that value (e.g. for a `#[module]` struct field)
/// still requires hand-nesting `Sequential<L1, Sequential<L2, L3>>` — see
/// [`seq_type!`], which generates that same nesting from the same
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
/// `seq_type!(L1, L2, L3)` expands to `Sequential<L1, Sequential<L2, L3>>`,
/// so a layer list only needs to be written once per meaning (the type via
/// this macro, the value via `seq!`) instead of the type being re-derived by
/// hand every time the layer list changes.
///
/// ## Examples
/// ```rust,ignore
/// use kindle::prelude::*;
///
/// type Backend = KindleBackend<f32, Cpu>;
///
/// // Instead of writing:
/// //   Sequential<Linear<s![768, 256], Backend>, Sequential<ReLU, Linear<s![256, 10], Backend>>>
/// type Net = seq_type!(
///     Linear<s![768, 256], Backend>,
///     ReLU,
///     Linear<s![256, 10], Backend>
/// );
///
/// let net: Net = seq!(
///     Linear::<s![768, 256], Backend>::build(())?,
///     ReLU,
///     Linear::<s![256, 10], Backend>::build(())?
/// );
/// # Ok::<(), Error>(())
/// ```
#[macro_export]
macro_rules! seq_type {
    ($l1:ty) => {
        $l1
    };
    ($l1:ty, $($tail:ty),+ $(,)?) => {
        $crate::prelude::Sequential<$l1, $crate::seq_type!($($tail),+)>
    };
}
