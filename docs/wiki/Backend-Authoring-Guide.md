# Backend Authoring Guide

This guide explains how to implement a custom hardware backend for Incin (e.g. for custom NPUs, FPGAs, or novel accelerators).

---

## 1. Core Traits

A backend must implement the `Backend` trait and its associated storage:

```rust
pub trait Backend: Clone + Send + Sync + 'static {
    type Storage<D: DType>: StorageBackend<D>;
    type Device: DeviceHandle;
    
    fn device(&self) -> &Self::Device;
    fn name(&self) -> &'static str;
}
```

---

## 2. StorageBackend Implementation

`StorageBackend<D>` manages raw device allocations and view slicing:

```rust
pub trait StorageBackend<D: DType>: Sized + Send + Sync + 'static {
    fn allocate(shape: &[usize], strides: &[usize], device: &Device) -> Result<Self, BackendError>;
    fn from_host_slice(slice: &[D], shape: &[usize], strides: &[usize]) -> Result<Self, BackendError>;
    fn to_host_vec(&self) -> Result<Vec<D>, BackendError>;
    
    fn metadata(&self) -> &TensorMeta;
    fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self, BackendError>;
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError>;
}
```

---

## 3. Operation Dispatch via `Execute<Op>`

Individual operations are implemented using the `Execute<OpDescriptor>` pattern:

```rust
pub struct MatMulOp;

impl<D: FloatDType> Execute<MatMulOp> for MyCustomBackend {
    type Input = (Self::Storage<D>, Self::Storage<D>);
    type Output = Self::Storage<D>;
    type Error = BackendError;

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, Self::Error> {
        // 1. Validate device synchronization
        // 2. Launch accelerator compute kernel
        // 3. Return output storage wrapper
    }
}
```

---

## 4. Capability Registration

Register the backend's supported dtypes and operations in `crates/incin-backends/src/catalog.rs` so that compile-time and runtime feature probes accurately reflect its capabilities.
