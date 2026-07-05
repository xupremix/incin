use crate::prelude::*;
use typenum::{U0, U1, U2, U3, U4, U5, U6};

pub trait StackShape<Axis> {
    type Output: Shape;
}


impl<D0> StackShape<U0> for (D0,)
where
    D0: Dim,
{
    type Output = (U2, D0,);
}

impl<D0> StackShape<U1> for (D0,)
where
    D0: Dim,
{
    type Output = (D0, U2,);
}

impl<D0, D1> StackShape<U0> for (D0, D1,)
where
    D0: Dim, D1: Dim,
{
    type Output = (U2, D0, D1,);
}

impl<D0, D1> StackShape<U1> for (D0, D1,)
where
    D0: Dim, D1: Dim,
{
    type Output = (D0, U2, D1,);
}

impl<D0, D1> StackShape<U2> for (D0, D1,)
where
    D0: Dim, D1: Dim,
{
    type Output = (D0, D1, U2,);
}

impl<D0, D1, D2> StackShape<U0> for (D0, D1, D2,)
where
    D0: Dim, D1: Dim, D2: Dim,
{
    type Output = (U2, D0, D1, D2,);
}

impl<D0, D1, D2> StackShape<U1> for (D0, D1, D2,)
where
    D0: Dim, D1: Dim, D2: Dim,
{
    type Output = (D0, U2, D1, D2,);
}

impl<D0, D1, D2> StackShape<U2> for (D0, D1, D2,)
where
    D0: Dim, D1: Dim, D2: Dim,
{
    type Output = (D0, D1, U2, D2,);
}

impl<D0, D1, D2> StackShape<U3> for (D0, D1, D2,)
where
    D0: Dim, D1: Dim, D2: Dim,
{
    type Output = (D0, D1, D2, U2,);
}

impl<D0, D1, D2, D3> StackShape<U0> for (D0, D1, D2, D3,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim,
{
    type Output = (U2, D0, D1, D2, D3,);
}

impl<D0, D1, D2, D3> StackShape<U1> for (D0, D1, D2, D3,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim,
{
    type Output = (D0, U2, D1, D2, D3,);
}

impl<D0, D1, D2, D3> StackShape<U2> for (D0, D1, D2, D3,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim,
{
    type Output = (D0, D1, U2, D2, D3,);
}

impl<D0, D1, D2, D3> StackShape<U3> for (D0, D1, D2, D3,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim,
{
    type Output = (D0, D1, D2, U2, D3,);
}

impl<D0, D1, D2, D3> StackShape<U4> for (D0, D1, D2, D3,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim,
{
    type Output = (D0, D1, D2, D3, U2,);
}

impl<D0, D1, D2, D3, D4> StackShape<U0> for (D0, D1, D2, D3, D4,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim,
{
    type Output = (U2, D0, D1, D2, D3, D4,);
}

impl<D0, D1, D2, D3, D4> StackShape<U1> for (D0, D1, D2, D3, D4,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim,
{
    type Output = (D0, U2, D1, D2, D3, D4,);
}

impl<D0, D1, D2, D3, D4> StackShape<U2> for (D0, D1, D2, D3, D4,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim,
{
    type Output = (D0, D1, U2, D2, D3, D4,);
}

impl<D0, D1, D2, D3, D4> StackShape<U3> for (D0, D1, D2, D3, D4,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim,
{
    type Output = (D0, D1, D2, U2, D3, D4,);
}

impl<D0, D1, D2, D3, D4> StackShape<U4> for (D0, D1, D2, D3, D4,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim,
{
    type Output = (D0, D1, D2, D3, U2, D4,);
}

impl<D0, D1, D2, D3, D4> StackShape<U5> for (D0, D1, D2, D3, D4,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim,
{
    type Output = (D0, D1, D2, D3, D4, U2,);
}

impl<D0, D1, D2, D3, D4, D5> StackShape<U0> for (D0, D1, D2, D3, D4, D5,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim,
{
    type Output = (U2, D0, D1, D2, D3, D4, D5,);
}

impl<D0, D1, D2, D3, D4, D5> StackShape<U1> for (D0, D1, D2, D3, D4, D5,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim,
{
    type Output = (D0, U2, D1, D2, D3, D4, D5,);
}

impl<D0, D1, D2, D3, D4, D5> StackShape<U2> for (D0, D1, D2, D3, D4, D5,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim,
{
    type Output = (D0, D1, U2, D2, D3, D4, D5,);
}

impl<D0, D1, D2, D3, D4, D5> StackShape<U3> for (D0, D1, D2, D3, D4, D5,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim,
{
    type Output = (D0, D1, D2, U2, D3, D4, D5,);
}

impl<D0, D1, D2, D3, D4, D5> StackShape<U4> for (D0, D1, D2, D3, D4, D5,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim,
{
    type Output = (D0, D1, D2, D3, U2, D4, D5,);
}

impl<D0, D1, D2, D3, D4, D5> StackShape<U5> for (D0, D1, D2, D3, D4, D5,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim,
{
    type Output = (D0, D1, D2, D3, D4, U2, D5,);
}

impl<D0, D1, D2, D3, D4, D5> StackShape<U6> for (D0, D1, D2, D3, D4, D5,)
where
    D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim,
{
    type Output = (D0, D1, D2, D3, D4, D5, U2,);
}

impl StackShape<U0> for () {
    type Output = (U2,);
}
