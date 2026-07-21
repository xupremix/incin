use kindle_backends::ndarray_backend::NdarrayBackend;
use kindle_core::prelude::*;

#[test]
/// Auto-generated documentation for test_ndarray_interior_mutability.
fn test_ndarray_interior_mutability() {
    let backend_tensor = ndarray::ArrayD::zeros(ndarray::IxDyn(&[2, 2]));

    // Create tensor
    let t: Tensor<Dyn, NdarrayBackend<f32, Cpu>> =
        Tensor::from_raw(backend_tensor.clone(), [2, 2]).unwrap();

    // Convert to variable
    let params = vec![NdarrayBackend::<f32, Cpu>::var_from_tensor::<f32>(t.inner()).unwrap()];

    // Mutate the variable directly through the Arc<RwLock>
    {
        let mut array = params[0].0.write();
        array[[0, 0]] = 42.0;
    }

    // Read it back through var_as_tensor
    let new_backend_tensor = NdarrayBackend::<f32, Cpu>::var_as_tensor::<f32>(&params[0]).unwrap();

    // Assert the original array inside the variable was mutated in-place
    assert_eq!(new_backend_tensor[[0, 0]], 42.0);
}
