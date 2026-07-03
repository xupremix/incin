//! Flattening with compile-time shape verification.

use crate::prelude::*;
use typenum::{Prod, U1, U0};

pub trait FlattenShape: Shape {
    type Output: Shape;
    fn output_shape(lhs: &<Self as Shape>::Field) -> <Self::Output as Shape>::Field;
}

// Dyn -> Dyn
impl FlattenShape for Dyn {
    type Output = Dyn;
    fn output_shape(_lhs: &<Dyn as Shape>::Field) -> <Dyn as Shape>::Field {
        alloc::vec![]
    }
}

// Static 2D -> 1D
impl<D0, D1> FlattenShape for (D0, D1)
where
    D0: StaticDim + core::ops::Mul<D1>,
    D1: StaticDim,
    Prod<D0, D1>: StaticDim,
{
    type Output = (Prod<D0, D1>,);
    #[inline(always)]
    fn output_shape(_: &<Self as Shape>::Field) -> <Self::Output as Shape>::Field {
        (Default::default(),)
    }
}

// Static 3D -> 2D (Keep batch dim)
impl<B, D1, D2> FlattenShape for (B, D1, D2)
where
    B: StaticDim,
    D1: StaticDim + core::ops::Mul<D2>,
    D2: StaticDim,
    Prod<D1, D2>: StaticDim,
{
    type Output = (B, Prod<D1, D2>);
    #[inline(always)]
    fn output_shape(_: &<Self as Shape>::Field) -> <Self::Output as Shape>::Field {
        (Default::default(), Default::default())
    }
}

// Static 4D -> 2D (Keep batch dim)
impl<B, D1, D2, D3> FlattenShape for (B, D1, D2, D3)
where
    B: StaticDim,
    D1: StaticDim + core::ops::Mul<Prod<D2, D3>>,
    D2: StaticDim + core::ops::Mul<D3>,
    D3: StaticDim,
    Prod<D2, D3>: StaticDim,
    Prod<D1, Prod<D2, D3>>: StaticDim,
{
    type Output = (B, Prod<D1, Prod<D2, D3>>);
    #[inline(always)]
    fn output_shape(_: &<Self as Shape>::Field) -> <Self::Output as Shape>::Field {
        (Default::default(), Default::default())
    }
}

// Mixed dynamic 3D -> 2D
impl<D1, D2> FlattenShape for (usize, D1, D2)
where
    D1: StaticDim + core::ops::Mul<D2>,
    D2: StaticDim,
    Prod<D1, D2>: StaticDim,
{
    type Output = (usize, Prod<D1, D2>);
    fn output_shape(lhs: &<Self as Shape>::Field) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}

// Mixed dynamic 4D -> 2D
impl<D1, D2, D3> FlattenShape for (usize, D1, D2, D3)
where
    D1: StaticDim + core::ops::Mul<Prod<D2, D3>>,
    D2: StaticDim + core::ops::Mul<D3>,
    D3: StaticDim,
    Prod<D2, D3>: StaticDim,
    Prod<D1, Prod<D2, D3>>: StaticDim,
{
    type Output = (usize, Prod<D1, Prod<D2, D3>>);
    fn output_shape(lhs: &<Self as Shape>::Field) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}

impl<S1, B: Backend<S1>, T, D, G> Tensor<S1, B, T, D, G>
where
    S1: Shape + DynShape,
    T: DType,
    D: Device,
    G: RequiresGrad,
{
    pub fn flatten(&self) -> Result<Tensor<S1::Output, B, T, D, G>>
    where
        S1: FlattenShape,
        B: Backend<S1::Output, RawTensor = <B as Backend<S1>>::RawTensor>,
    {
        let current_dims = S1::dims(&self._shape);
        // By default flatten logic keeps batch dim (if rank >= 2) and flattens the rest.
        // For rank 2, it becomes 1D.
        let mut new_dims = alloc::vec![];
        if current_dims.as_ref().len() == 2 {
            new_dims.push(current_dims.as_ref()[0] * current_dims.as_ref()[1]);
        } else if current_dims.as_ref().len() >= 3 {
            new_dims.push(current_dims.as_ref()[0]);
            let rest: usize = current_dims.as_ref()[1..].iter().product();
            new_dims.push(rest);
        } else {
            return Err(Error::ShapeMismatch { expected: alloc::vec![], got: current_dims.as_ref().to_vec() });
        }
        
        let inner = <B as Backend<S1>>::reshape(&self.inner, &new_dims)?;
        let output_shape = S1::output_shape(&self._shape);
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
