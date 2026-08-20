//! `AutogradBackend` and `VariableBackend` implementations for
//! `CudaBackendImpl`.

use super::*;

impl<D: Device> incin_core::backend_authoring::AutogradBackend for CudaBackendImpl<D> {
    type Grads = CudaGrads;

    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        let loss: &CudaStorage = loss;
        crate::cuda::tape::backward(loss)
    }

    fn backward_with<K: DType>(
        loss: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        let loss: &CudaStorage = loss;
        let seed: &CudaStorage = seed;
        crate::cuda::tape::backward_with(loss, seed)
    }

    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        let t: &CudaStorage = t;
        Ok(grads.get(t.id).cloned())
    }

    fn set_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &mut Self::Grads,
        value: Self::Storage<K>,
    ) -> Result<()> {
        let t: &CudaStorage = t;
        grads.set(t.id, value);
        Ok(())
    }
}

impl<D: Device> VariableBackend for CudaBackendImpl<D> {
    type Var<K: DType> = CudaVar;

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }

    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::Var<K>> {
        let t: &CudaStorage = t;
        Ok(CudaVar { storage: t.clone() })
    }

    fn assign_var<K: DType>(var: &mut Self::Var<K>, tensor: &Self::Storage<K>) -> Result<()> {
        let tensor: &CudaStorage = tensor;
        var.storage = tensor.clone();
        Ok(())
    }
}
