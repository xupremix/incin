use kindle::prelude::*;
use kindle_backends::candle::CandleBackend;

pub struct MyNetwork;

#[kindle::module]
impl MyNetwork {
    // 1. Fully Static Auto-Resolution
    #[kindle::forward]
    pub fn forward_static(&self, x: Tensor<s![10, 20], CandleBackend>) -> Result<Tensor<s![10, 20], CandleBackend>> {
        let b = x.abs()?;
        Ok(b)
    }

    // 2. Dynamic Type Boundaries
    #[kindle::forward]
    pub fn forward_dynamic(&self, x: Tensor<Dyn, CandleBackend>) -> Result<Tensor<s![dyn, 224, 224], CandleBackend>> {
        let activated = x.relu()?;
        Ok(activated)
    }

    // 3. Generics and Named Tensors
    #[kindle::forward]
    pub fn forward_generic<B: Dim, C: Dim>(&self, x: Tensor<s![B, C, 224, 224], CandleBackend>) -> Result<Tensor<s![B, C, 224, 224], CandleBackend>> {
        let b = x.relu()?;
        Ok(b)
    }
}

type BatchSize = typenum::U32;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NamedBatchTag;

fn main() -> Result<()> {
    println!("--- Testing macros with generics and named dimensions ---");
    
    // Testing static aliases
    let x: Tensor<s![BatchSize, 3, 224, 224], CandleBackend> = Tensor::<Dyn, CandleBackend>::zeros([32, 3, 224, 224])?.into_shape()?;
    println!("Created tensor with aliased BatchSize (U32) shape: {:?}", x.dims());

    // Testing NamedDyn generic markers
    let y: Tensor<s![NamedDyn<NamedBatchTag>, 3, 224, 224], CandleBackend> = Tensor::<Dyn, CandleBackend>::zeros([16, 3, 224, 224])?.into_shape()?;
    println!("Created tensor with NamedDyn<NamedBatchTag> shape: {:?}", y.dims());

    println!("Macros and generic shape validation successful!");
    
    let net = MyNetwork;
    
    let t_static: Tensor<s![10, 20], CandleBackend> = Tensor::<Dyn, CandleBackend>::zeros([10, 20]).unwrap().into_shape().unwrap();
    let out1 = net.forward_static(t_static).unwrap();
    println!("✅ Static Output: {:?}", out1.dims());
    
    let t_dyn: Tensor<Dyn, CandleBackend> = Tensor::<Dyn, CandleBackend>::zeros([3, 224, 224]).unwrap();
    let out2 = net.forward_dynamic(t_dyn).unwrap();
    println!("✅ Dynamic Output (Good Shape): {:?}", out2.dims());
    
    let t_dyn_bad: Tensor<Dyn, CandleBackend> = Tensor::<Dyn, CandleBackend>::zeros([3, 50, 50]).unwrap();
    match net.forward_dynamic(t_dyn_bad) {
        Ok(out3) => println!("✅ Dynamic Output: {:?}", out3.dims()),
        Err(e) => {
            println!("✅ Correctly caught Dynamic Shape Error: {:?}", e);
        }
    }

    Ok(())
}
