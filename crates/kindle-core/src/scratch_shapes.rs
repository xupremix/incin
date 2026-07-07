pub trait Shape {
    fn dims() -> std::vec::Vec<isize>;
}

pub struct Shape1D<const A: isize>;
impl<const A: isize> Shape for Shape1D<A> {
    fn dims() -> std::vec::Vec<isize> { vec![A] }
}

pub struct Shape2D<const A: isize, const B: isize>;
impl<const A: isize, const B: isize> Shape for Shape2D<A, B> {
    fn dims() -> std::vec::Vec<isize> { vec![A, B] }
}

pub struct Tensor<S: Shape>(std::marker::PhantomData<S>);

impl<S: Shape> Tensor<S> {
    pub fn new() -> Self { Tensor(std::marker::PhantomData) }
}

fn main() {
    let t: Tensor<Shape2D<10, 20>> = Tensor::new();
    let dyn_t: Tensor<Shape2D<-1, 20>> = Tensor::new(); // -1 represents Dyn
}
