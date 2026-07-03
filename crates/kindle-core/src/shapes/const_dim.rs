use crate::prelude::Dim;

pub trait ConstDim: Default + Dim {
    const SIZE: usize;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Const<const N: usize>;

// use core::cmp::Ordering;
// pub const fn dim_cmp<N: ConstDim, M: ConstDim>(ord: Ordering) -> bool {
//     match ord {
//         Ordering::Less => N::SIZE < M::SIZE,
//         Ordering::Equal => N::SIZE == M::SIZE,
//         Ordering::Greater => N::SIZE > M::SIZE,
//     }
// }

impl<const N: usize> Dim for Const<N> {
    type Arg = ();

    #[inline(always)]
    fn size(&self) -> usize {
        N
    }

    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == N { Some(Const) } else { None }
    }

    #[inline(always)]
    fn from_arg(_: Self::Arg) -> Self {
        Const
    }
}

impl<const N: usize> ConstDim for Const<N> {
    const SIZE: usize = N;
}

macro_rules! impl_op_for_const {
    ($($op:ident $name:ident $tok:tt)*) => {
        $(
            #[cfg(feature = "nightly")]
            impl<const N: usize, const M: usize> core::ops::$op<Const<N>> for Const<M>
            where
                Const<{ M $tok N }>: Sized,
            {
                type Output = Const<{ M $tok N }>;
                fn $name(self, _: Const<N>) -> Self::Output {
                    Const
                }
            }

            impl<const N: usize> core::ops::$op<usize> for Const<N> {
                type Output = usize;
                fn $name(self, rhs: usize) -> Self::Output {
                    N $tok rhs
                }
            }

            impl<const N: usize> core::ops::$op<Const<N>> for usize {
                type Output = usize;
                fn $name(self, _: Const<N>) -> Self::Output {
                    self $tok N
                }
            }
        )*
    };
}

impl_op_for_const! {
    Add add +
    Sub sub -
    Mul mul *
    Div div /
}

impl<U, B> ConstDim for typenum::UInt<U, B>
where
    U: typenum::Unsigned + Dim,
    B: typenum::Bit + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
    typenum::UInt<U, B>: typenum::Unsigned + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
{
    const SIZE: usize = <Self as typenum::Unsigned>::USIZE;
}

impl ConstDim for typenum::UTerm {
    const SIZE: usize = 0;
}
