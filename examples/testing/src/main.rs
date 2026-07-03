use kindle::prelude::*;

#[kindle::module]
struct MyNetwork {}

impl MyNetwork {
    // 1. Fully Static Auto-Resolution
    #[kindle::forward]
    fn forward_static(&self, x: Tensor<s![10, 20]>) -> Result<Tensor<s![10, 20]>> {
        // Rust fully resolves the intermediate ops statically
        let a = x.relu()?;
        let b = a.abs()?;
        // Macro enforces Ok(b.into_shape().unwrap())
        Ok(b)
    }

    // 2. Dynamic Type Boundaries
    #[kindle::forward]
    fn forward_dynamic(&self, x: Tensor<Dyn>) -> Result<Tensor<s![dyn, 224, 224]>> {
        // x is completely dynamic. The return type expects the last two dimensions to be 224
        let activated = x.relu()?;
        // The macro writes: Ok(activated.into_shape().unwrap())
        // At compile-time this is fine. At runtime, if `activated` isn't [..., 224, 224], unwrap() panics!
        Ok(activated)
    }

    // 3. Deliberate Compile Error Scenario (Commented out to allow compilation)
    /*
    #[kindle::forward]
    fn forward_error(&self, x: Tensor<s![10, 20]>) -> Result<Tensor<s![10, 50]>> {
        let bad = x.add(&x)?; // shape is still [10, 20]
        
        // The macro injects: Ok(bad.into_shape().unwrap())
        // Since `bad` is statically known as [10, 20], the compiler sees we are asking 
        // to cast it to `s![10, 50]`. The Rust compiler will fail here with a Type Mismatch Error!
        Ok(bad) 
    }
    */
}

fn main() {
    let net = MyNetwork {};

    // Run Static Scenario
    let t_static: Tensor<s![10, 20]> = Tensor::<Dyn>::zeros([10, 20]).unwrap().into_shape().unwrap();
    let out1 = net.forward_static(t_static).unwrap();
    println!("✅ Static Output: {:?}", out1.dims());

    // Run Dynamic Scenario (Passing)
    let t_dyn: Tensor<Dyn> = Tensor::zeros([3, 224, 224]).unwrap();
    let out2 = net.forward_dynamic(t_dyn).unwrap();
    println!("✅ Dynamic Output (Good Shape): {:?}", out2.dims());

    // Run Dynamic Scenario (Failing Gracefully)
    let t_dyn_bad: Tensor<Dyn> = Tensor::zeros([3, 50, 50]).unwrap();
    
    // Instead of panicking, the macro's `?` operator cleanly passes the Error back to us
    match net.forward_dynamic(t_dyn_bad) {
        Ok(out3) => println!("✅ Dynamic Output: {:?}", out3.dims()),
        Err(e) => {
            println!("❌ Dynamic Output gracefully failed!");
            println!("   Reason: {}", e);
            // Here the user can execute completely different code!
            println!("   Running fallback behavior instead...");
        }
    }
}
