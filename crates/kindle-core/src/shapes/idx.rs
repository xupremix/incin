use crate::prelude::*;

/// An inferred dimension in a reshape or slice operation (`-1`).
pub struct InferDim;

/// Represents all remaining dimensions (`..`).
pub struct Ellipsis;

/// A slice spanning a range `[START, END)`.
pub struct Slice<const START: usize, const END: usize, const DIFF: usize>;

/// A trait for types that can appear in an index macro `idx![...]` for reshape.
pub trait DimIdx {
    /// The resolved `Dim` type in the output shape. `InferDim` becomes `usize`.
    type Resolved: Dim;
    
    /// Returns the specified size if known, or `None` if it's `InferDim`.
    fn size() -> Option<usize>;
}

impl<const N: usize> DimIdx for Const<N> {
    type Resolved = Const<N>;
    fn size() -> Option<usize> { Some(N) }
}

impl DimIdx for InferDim {
    type Resolved = usize;
    fn size() -> Option<usize> { None }
}

// Implement for NamedDyn
impl<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> DimIdx for NamedDyn<Tag> {
    type Resolved = NamedDyn<Tag>;
    fn size() -> Option<usize> { None } // Not a static known size, but handled specially
}

/// Computes the new shape resulting from a reshape.
pub trait ReshapeTarget<In: Shape> {
    type Output: Shape;
    fn calculate_shape(in_shape_vec: &[usize]) -> Vec<usize>;
}

// Generate implementations for tuples up to 4D for now to match the user's common cases
macro_rules! impl_reshape_target {
    ($($D:ident),*) => {
        impl<In: Shape, $($D: DimIdx),*> ReshapeTarget<In> for ($($D,)*)
        where
            ($($D::Resolved,)*): Shape,
        {
            type Output = ($($D::Resolved,)*);
            
            fn calculate_shape(in_shape_vec: &[usize]) -> Vec<usize> {
                let total_elements: usize = in_shape_vec.iter().product();
                let mut resolved_sizes = vec![];
                let mut infer_idx = None;
                
                // Collect specified sizes and find the InferDim
                let mut _current_idx = 0usize;
                $(
                    let size = $D::size();
                    if size.is_none() {
                        // Wait, NamedDyn also has None! We need a way to differentiate.
                        // Actually, NamedDyn shouldn't be None if we want to infer it.
                        // But we don't have access to runtime values of NamedDyn here.
                        // Let's just say InferDim is the only one we infer for Reshape.
                        // For NamedDyn in Reshape, we must either pass its value, OR if it's the 
                        // ONLY unknown, we can infer it!
                        
                        if infer_idx.is_some() {
                            panic!("Only one inferred dimension (-1 or NamedDyn) is allowed in reshape");
                        }
                        infer_idx = Some(_current_idx);
                        resolved_sizes.push(0); // placeholder
                    } else {
                        resolved_sizes.push(size.unwrap());
                    }
                    _current_idx += 1;
                )*
                
                if let Some(idx) = infer_idx {
                    let known_product: usize = resolved_sizes.iter().filter(|&&s| s != 0).product();
                    if known_product > 0 {
                        resolved_sizes[idx] = total_elements / known_product;
                    } else {
                        resolved_sizes[idx] = 0;
                    }
                }
                
                resolved_sizes
            }
        }
    };
}

impl_reshape_target!(D1);
impl_reshape_target!(D1, D2);
impl_reshape_target!(D1, D2, D3);
impl_reshape_target!(D1, D2, D3, D4);



/// A trait for indexing/slicing dimensions in the `idx!` macro
pub trait SliceIdx {
    /// The resolved `Dim` type in the output shape.
    type Resolved: Dim;
    
    /// Extract the slice bounds (start, end) for a given dimension size.
    /// If the index spans the whole dimension (e.g. `..`), it returns (0, size).
    fn bounds(size: usize) -> (usize, usize);
}

impl<const START: usize, const END: usize, const DIFF: usize> SliceIdx for Slice<START, END, DIFF> {
    type Resolved = Const<DIFF>;
    fn bounds(_size: usize) -> (usize, usize) {
        (START, END)
    }
}

impl<const N: usize> SliceIdx for Const<N> {
    type Resolved = Const<1>;
    fn bounds(_size: usize) -> (usize, usize) {
        (N, N + 1)
    }
}

// Ellipsis represents taking the full dimension for all remaining dims.
// We can't easily implement a variadic Ellipsis in simple tuple traits without
// advanced macro work. For now, we assume `..` is a single full dimension slice.
impl SliceIdx for Ellipsis {
    type Resolved = usize; // Dyn size since it depends on the input
    fn bounds(size: usize) -> (usize, usize) {
        (0, size)
    }
}

pub trait SliceTarget<In: Shape> {
    type Output: Shape;
    /// Returns the bounds (start, end) for each dimension in `in_shape_vec`
    fn calculate_bounds(in_shape_vec: &[usize]) -> Vec<(usize, usize)>;
}

macro_rules! impl_slice_target {
    ($($D:ident),*) => {
        impl<In: Shape, $($D: SliceIdx),*> SliceTarget<In> for ($($D,)*)
        where
            ($($D::Resolved,)*): Shape,
        {
            type Output = ($($D::Resolved,)*);
            
            fn calculate_bounds(in_shape_vec: &[usize]) -> Vec<(usize, usize)> {
                let mut bounds = vec![];
                let mut _current_idx = 0usize;
                $(
                    let size = in_shape_vec.get(_current_idx).copied().unwrap_or(0);
                    bounds.push($D::bounds(size));
                    _current_idx += 1;
                )*
                bounds
            }
        }
    };
}

impl_slice_target!(D1);
impl_slice_target!(D1, D2);
impl_slice_target!(D1, D2, D3);
impl_slice_target!(D1, D2, D3, D4);
