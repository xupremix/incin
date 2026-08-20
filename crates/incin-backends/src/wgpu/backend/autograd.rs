//! `AutogradBackend` and `VariableBackend` implementations for
//! `WgpuBackendImpl`.

use super::*;

impl<D: Device> incin_core::backend_authoring::AutogradBackend for WgpuBackendImpl<D> {
    type Grads = WgpuGrads;

    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::wgpu::tape::backward(loss)
    }

    fn backward_with<K: DType>(
        loss: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        crate::wgpu::tape::backward_with(loss, seed)
    }

    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }

    fn set_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &mut Self::Grads,
        value: Self::Storage<K>,
    ) -> Result<()> {
        grads.set(t.id, value);
        Ok(())
    }
}
impl<D: Device> VariableBackend for WgpuBackendImpl<D> {
    /// `Var<K>`.
    type Var<K: DType> = WgpuVar;

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        Ok(var.value())
    }

    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::Var<K>> {
        let t: &WgpuStorage = t;
        Ok(WgpuVar::new(t.clone()))
    }

    fn assign_var<K: DType>(var: &mut Self::Var<K>, tensor: &Self::Storage<K>) -> Result<()> {
        var.assign(tensor.clone());
        Ok(())
    }
}
