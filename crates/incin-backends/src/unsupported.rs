//! Explicit declarations for operations a backend does not implement.
//!
//! `FloatOps` used to give every method a default body returning
//! [`Error::UnsupportedBackendOperation`], so a backend covering sixteen of its
//! forty-two operations looked exactly like one covering all forty-two. The
//! refusal was real but invisible: it lived in the trait, not in the backend
//! that was actually refusing, and nothing failed when a backend's coverage
//! silently shrank.
//!
//! `EXE-009` removes those defaults, which makes the omission a compile error.
//! This macro is how a backend answers it: the gap is written down where the
//! backend is defined, greppable and reviewable, and the compiler adds any
//! newly-declared operation to the list a backend must consciously answer for.
//!
//! The error a caller sees is unchanged. What changes is that a reader can see
//! which operations a backend does not have without running it.
//!
//! [`Error::UnsupportedBackendOperation`]: incin_core::prelude::Error

/// Declares float operations a backend does not implement, grouped by
/// signature: `unary` takes one tensor, `exponent` a tensor and an exponent,
/// `bounds` a tensor and a min/max pair, and `binary` two tensors.
macro_rules! unsupported_float_ops {
    (
        unary: $($unary:ident),* $(,)?;
        exponent: $($exponent:ident),* $(,)?;
        bounds: $($bounds:ident),* $(,)?;
        binary: $($binary:ident),* $(,)?;
    ) => {
        $(
            fn $unary<K: DType>(
                _t: &<Self as Backend>::Storage<K>,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($unary)))
            }
        )*
        $(
            fn $exponent<K: DType>(
                _t: &<Self as Backend>::Storage<K>,
                _exponent: f64,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($exponent)))
            }
        )*
        $(
            fn $bounds<K: DType>(
                _t: &<Self as Backend>::Storage<K>,
                _min: f64,
                _max: f64,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($bounds)))
            }
        )*
        $(
            fn $binary<K: DType>(
                _lhs: &<Self as Backend>::Storage<K>,
                _rhs: &<Self as Backend>::Storage<K>,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($binary)))
            }
        )*
    };
}

pub(crate) use unsupported_float_ops;

/// Declares the value-filled and sequence creation operations a backend has no
/// kernel for. `zeros`/`ones`/`rand`/`randn` are deliberately not covered:
/// every backend implements those, and a gap there is a defect, not a
/// declaration.
///
/// `fill` takes a single value; `sequence` takes a pair (`start`/`step` for
/// `arange`, `start`/`end` for `linspace`).
///
/// CUDA and Metal both still invoke this; WGPU no longer does, now that it
/// has real kernels for `full`/`arange`/`linspace`. That makes the macro
/// provably unused only under feature combinations that build WGPU without
/// CUDA or Metal (e.g. CI's WGPU-only clippy job), which is a feature-gating
/// artifact rather than dead code.
#[allow(unused_macros)]
macro_rules! unsupported_creation_ops {
    (fill: $($fill:ident),* $(,)?; sequence: $($sequence:ident),* $(,)?;) => {
        $(
            fn $fill<K: DType>(
                _value: f64,
                _shape: &[usize],
                _dtype: DTypeId,
                _device: &DeviceId,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($fill)))
            }
        )*
        $(
            fn $sequence<K: DType>(
                _from: f64,
                _to: f64,
                _shape: &[usize],
                _dtype: DTypeId,
                _device: &DeviceId,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($sequence)))
            }
        )*
    };
}

#[allow(unused_imports)]
pub(crate) use unsupported_creation_ops;

/// Declares reductions a backend has no kernel for, grouped by whether they
/// collapse the whole tensor or act along one axis.
macro_rules! unsupported_reduction_ops {
    (all: $($all:ident),* $(,)?; dim: $($dim:ident),* $(,)?;) => {
        $(
            fn $all<K: DType>(
                _t: &<Self as Backend>::Storage<K>,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($all)))
            }
        )*
        $(
            fn $dim<K: DType>(
                _t: &<Self as Backend>::Storage<K>,
                _dim: usize,
            ) -> Result<<Self as Backend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($dim)))
            }
        )*
    };
}

pub(crate) use unsupported_reduction_ops;

/// Declares shape, comparison, and indexing operations a backend has no kernel
/// for, one operation name per entry.
///
/// The three macros above group by signature because their traits have only a
/// handful of shapes. `TensorOps` has nearly as many distinct signatures as it
/// has methods, so grouping would mean twenty groups, most with one member, and
/// every call site spelling out the empty ones. Matching on the operation name
/// instead keeps the declaration a flat list of exactly the gap:
///
/// ```text
/// crate::unsupported::unsupported_tensor_ops! {
///     gather, scatter, triu, tril,
/// }
/// ```
///
/// A name with no arm below is a compile error, so this cannot drift from the
/// trait without someone noticing.
///
/// CUDA and Metal both still invoke this; WGPU no longer does, now that
/// every `TensorOps` method has a real implementation. That makes the macro
/// provably unused only under feature combinations that build WGPU without
/// CUDA or Metal (e.g. CI's WGPU-only clippy job), a feature-gating artifact
/// rather than dead code — see `unsupported_creation_ops`'s identical note
/// above for the first time this happened.
#[allow(unused_macros)]
macro_rules! unsupported_tensor_ops {
    ($($op:ident),* $(,)?) => {
        $( $crate::unsupported::unsupported_tensor_ops!(@op $op); )*
    };

    (@op where_cond) => {
        fn where_cond<K: DType, KMask: DType>(
            _mask: &<Self as Backend>::Storage<KMask>,
            _on_true: &<Self as Backend>::Storage<K>,
            _on_false: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("where_cond"))
        }
    };
    (@op scatter) => {
        fn scatter<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: usize,
            _index: &<Self as Backend>::Storage<KInt>,
            _src: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("scatter"))
        }
    };
    (@op masked_fill) => {
        fn masked_fill<K: DType, KMask: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _mask: &<Self as Backend>::Storage<KMask>,
            _value: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("masked_fill"))
        }
    };
    (@op pad) => {
        fn pad<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _padding: &[(usize, usize)],
            _val: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("pad"))
        }
    };
    (@op repeat) => {
        fn repeat<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _repeats: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("repeat"))
        }
    };
    (@op lerp) => {
        fn lerp<K: DType>(
            _start: &<Self as Backend>::Storage<K>,
            _end: &<Self as Backend>::Storage<K>,
            _weight: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("lerp"))
        }
    };
    (@op addmm) => {
        fn addmm<K: DType>(
            _mat: &<Self as Backend>::Storage<K>,
            _mat1: &<Self as Backend>::Storage<K>,
            _mat2: &<Self as Backend>::Storage<K>,
            _beta: f64,
            _alpha: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("addmm"))
        }
    };
    (@op scaled_dot_product_attention) => {
        fn scaled_dot_product_attention<K: DType>(
            _q: &<Self as Backend>::Storage<K>,
            _k: &<Self as Backend>::Storage<K>,
            _v: &<Self as Backend>::Storage<K>,
            _mask: Option<&<Self as Backend>::Storage<K>>,
            _scale: Option<f64>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>(
                "scaled_dot_product_attention",
            ))
        }
    };
    (@op unfold) => {
        fn unfold<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: usize,
            _size: usize,
            _step: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("unfold"))
        }
    };
    (@op group_norm) => {
        fn group_norm<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _groups: usize,
            _eps: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>("group_norm"))
        }
    };
    (@op tensor_to_dtype) => {
        fn tensor_to_dtype<K: DType, K2: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dtype: DTypeId,
        ) -> Result<<Self as Backend>::Storage<K2>> {
            Err($crate::unsupported::unsupported::<Self>("tensor_to_dtype"))
        }
    };
    (@op float_to_scalar) => {
        fn float_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<f64> {
            Err($crate::unsupported::unsupported::<Self>("float_to_scalar"))
        }
    };
    (@op float_to_vec1) => {
        fn float_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<f64>> {
            Err($crate::unsupported::unsupported::<Self>("float_to_vec1"))
        }
    };
    (@op int_to_scalar) => {
        fn int_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<i64> {
            Err($crate::unsupported::unsupported::<Self>("int_to_scalar"))
        }
    };
    (@op int_to_vec1) => {
        fn int_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<i64>> {
            Err($crate::unsupported::unsupported::<Self>("int_to_vec1"))
        }
    };

    // Operations sharing a signature with at least one sibling. The repeated
    // arms are spelled out rather than generated so each one names the exact
    // operation the caller sees in the error.
    (@op logical_not) => {
        $crate::unsupported::unsupported_tensor_ops!(@unary logical_not);
    };
    (@op gather) => { $crate::unsupported::unsupported_tensor_ops!(@indexed gather); };
    (@op index_select) => { $crate::unsupported::unsupported_tensor_ops!(@indexed index_select); };
    (@op unsqueeze) => { $crate::unsupported::unsupported_tensor_ops!(@dim unsqueeze); };
    (@op pixel_shuffle) => { $crate::unsupported::unsupported_tensor_ops!(@dim pixel_shuffle); };
    (@op triu) => { $crate::unsupported::unsupported_tensor_ops!(@diagonal triu); };
    (@op tril) => { $crate::unsupported::unsupported_tensor_ops!(@diagonal tril); };
    (@op diag) => { $crate::unsupported::unsupported_tensor_ops!(@diagonal diag); };
    (@op sub_scalar) => { $crate::unsupported::unsupported_tensor_ops!(@scalar sub_scalar); };
    (@op div_scalar) => { $crate::unsupported::unsupported_tensor_ops!(@scalar div_scalar); };
    (@op instance_norm) => { $crate::unsupported::unsupported_tensor_ops!(@scalar instance_norm); };
    (@op cmp_eq) => { $crate::unsupported::unsupported_tensor_ops!(@binary cmp_eq); };
    (@op cmp_ne) => { $crate::unsupported::unsupported_tensor_ops!(@binary cmp_ne); };
    (@op cmp_lt) => { $crate::unsupported::unsupported_tensor_ops!(@binary cmp_lt); };
    (@op cmp_le) => { $crate::unsupported::unsupported_tensor_ops!(@binary cmp_le); };
    (@op cmp_gt) => { $crate::unsupported::unsupported_tensor_ops!(@binary cmp_gt); };
    (@op cmp_ge) => { $crate::unsupported::unsupported_tensor_ops!(@binary cmp_ge); };
    (@op logical_and) => { $crate::unsupported::unsupported_tensor_ops!(@binary logical_and); };
    (@op logical_or) => { $crate::unsupported::unsupported_tensor_ops!(@binary logical_or); };
    (@op maximum) => { $crate::unsupported::unsupported_tensor_ops!(@binary maximum); };
    (@op minimum) => { $crate::unsupported::unsupported_tensor_ops!(@binary minimum); };
    (@op abs_diff) => { $crate::unsupported::unsupported_tensor_ops!(@binary abs_diff); };
    (@op bmm) => { $crate::unsupported::unsupported_tensor_ops!(@binary bmm); };

    (@unary $op:ident) => {
        fn $op<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>(stringify!($op)))
        }
    };
    (@binary $op:ident) => {
        fn $op<K: DType>(
            _lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>(stringify!($op)))
        }
    };
    (@dim $op:ident) => {
        fn $op<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>(stringify!($op)))
        }
    };
    (@scalar $op:ident) => {
        fn $op<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _val: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>(stringify!($op)))
        }
    };
    (@diagonal $op:ident) => {
        fn $op<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _k: i64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>(stringify!($op)))
        }
    };
    (@indexed $op:ident) => {
        fn $op<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: usize,
            _index: &<Self as Backend>::Storage<KInt>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Err($crate::unsupported::unsupported::<Self>(stringify!($op)))
        }
    };
}

#[allow(unused_imports)]
pub(crate) use unsupported_tensor_ops;

/// The error a declared gap reports, identical to the removed default body's.
pub(crate) fn unsupported<B>(op: &'static str) -> incin_core::prelude::Error {
    incin_core::prelude::Error::UnsupportedBackendOperation {
        op,
        backend: core::any::type_name::<B>(),
    }
}
