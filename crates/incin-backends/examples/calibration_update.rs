//! A custom operation with seven operands across three element types.
//!
//! Per-channel calibration is a real pass: before quantizing activations to
//! `q8_0` you need each channel's count, sum, sum of squares, minimum and
//! maximum, accumulated over a batch. That needs the activations, the channel
//! each element belongs to, and the five running statistics -- seven operands,
//! and they cannot share an element type. The activations are `f32` because
//! that is what the network produced, the channel index is `u32` because it is
//! an index and not a number to do arithmetic on, and the statistics are `f64`
//! because summing squares over a batch in `f32` loses the precision the whole
//! pass exists to establish.
//!
//! The catalog cannot express that. Its rules force a built-in operation's
//! operands to share a dtype, and the only heterogeneity it admits is a single
//! designated integer index operand, named in a hardcoded match over six
//! operations. This is a *custom* operation, and the custom path calls
//! `Operation::infer_outputs` and nothing else: the per-operand contract below
//! is the whole contract, and it is enforced before any backend is consulted.
//!
//! What a custom operation still cannot do is *advertise* that arrangement.
//! `CapabilityQuery` carries one dtype with no operand index, so a capability
//! row can only name the union of the three. The arrangement is enforced here,
//! in inference, rather than declared -- the same split the built-in
//! `cross_entropy_loss` lives with.
//!
//! Run it with:
//! `cargo run -p incin-backends --features cpu --example calibration_update`

use std::borrow::Cow;

extern crate incin_core as incin;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};

/// The CPU backend the kernel below is written for.
type CpuBackend = CpuBackendImpl;
use incin_core::backend_authoring::{
    DescriptorError, Execute, ExecutionContext, ExecutionRequest, LogicalTensorMeta, Operation,
    OperationKey, ShapeBuf, execute_shaped,
};
use incin_core::prelude::{BackendError, DTypeId, DeviceId, Result, ShapeValue};
use incin_core::shapes::error::OperationKind;

/// The five statistics this pass maintains, in the row order it writes them.
const STATS: [&str; 5] = ["count", "sum", "sum_sq", "min", "max"];

/// Operand positions, named once so the contract and the kernel cannot drift.
const VALUES: usize = 0;
const CHANNEL: usize = 1;
const FIRST_STAT: usize = 2;
const OPERANDS: usize = 7;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct CalibrationAttributes {
    /// How many channels the statistics are kept per.
    channels: usize,
}

#[derive(Debug, Clone)]
struct CalibrationUpdate;

fn invalid(attribute: &'static str, reason: &'static str) -> DescriptorError {
    DescriptorError::InvalidAttribute {
        operation: OperationKind::Reduction,
        attribute,
        reason,
    }
}

impl Operation for CalibrationUpdate {
    type Attributes = CalibrationAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("example.org"),
        name: Cow::Borrowed("calibration_update"),
        version: 1,
    };

    /// The whole per-operand contract. Nothing else validates a custom
    /// operation, so anything this accepts is what the kernel will be handed.
    fn infer_outputs(
        attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if inputs.len() != OPERANDS {
            return Err(invalid("inputs", "calibration_update takes seven operands"));
        }
        if attributes.channels == 0 {
            return Err(invalid(
                "channels",
                "calibration needs at least one channel",
            ));
        }

        // Operand 0 is f32, operand 1 is u32, operands 2..7 are f64. This is
        // the arrangement a capability row cannot state.
        let expect = |index: usize,
                      want: DTypeId,
                      attribute: &'static str|
         -> core::result::Result<(), DescriptorError> {
            match inputs[index].dtype {
                Some(actual) if actual == want.descriptor() => Ok(()),
                Some(_) => Err(invalid(attribute, "operand has the wrong element type")),
                None => Err(invalid(attribute, "operand element type is unknown")),
            }
        };
        expect(VALUES, DTypeId::F32, "values")?;
        expect(CHANNEL, DTypeId::U32, "channel")?;
        for (index, _) in inputs.iter().enumerate().take(OPERANDS).skip(FIRST_STAT) {
            expect(index, DTypeId::F64, "statistic")?;
        }

        // The channel index accompanies every value, one for one.
        let values = inputs[VALUES]
            .shape
            .as_ref()
            .ok_or_else(|| invalid("values", "activation shape is unknown"))?;
        let channel = inputs[CHANNEL]
            .shape
            .as_ref()
            .ok_or_else(|| invalid("channel", "channel shape is unknown"))?;
        if values.dims() != channel.dims() {
            return Err(invalid(
                "channel",
                "one channel index is required per activation",
            ));
        }

        // Each statistic carries one entry per channel.
        for input in inputs.iter().take(OPERANDS).skip(FIRST_STAT) {
            let stat = input
                .shape
                .as_ref()
                .ok_or_else(|| invalid("statistic", "statistic shape is unknown"))?;
            if stat.dims() != [attributes.channels] {
                return Err(invalid(
                    "statistic",
                    "each statistic holds one entry per channel",
                ));
            }
        }

        // One output: the five statistics stacked, so the typed dispatch path
        // has the single concrete output shape it requires.
        Ok(vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[STATS.len(), attributes.channels])),
            dtype: Some(DTypeId::F64.descriptor()),
            device: Some(DeviceId::cpu()),
        }])
    }
}

impl Execute<CalibrationUpdate> for CpuBackend {
    /// The five statistic rows, in `STATS` order.
    type Output = Vec<f64>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, CalibrationUpdate, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let read = |index: usize| -> core::result::Result<&CpuStorage, BackendError> {
            request.inputs[index]
                .downcast_ref::<CpuStorage>()
                .ok_or(BackendError::InvalidInput {
                    operation: OperationKind::Reduction,
                    reason: "calibration_update expects CPU storage",
                })
        };

        let values = read(VALUES)?;
        let channel = read(CHANNEL)?;
        let channels = request.operation.descriptor().attributes().channels;

        // Seed from the running statistics that came in as operands 2..7.
        let mut out = vec![0.0_f64; STATS.len() * channels];
        for (row, _) in STATS.iter().enumerate() {
            let prior = read(FIRST_STAT + row)?;
            for c in 0..channels {
                out[row * channels + c] = prior.get(&[c]);
            }
        }

        // `get` reads any element type back as f64, so one loop serves all
        // three: the f32 activation, the u32 channel, and the f64 running
        // statistics it folds into.
        let dims: Vec<usize> = values.metadata().shape.dims().to_vec();
        let count = dims.iter().product::<usize>();
        for flat in 0..count {
            let mut index = Vec::with_capacity(dims.len());
            let mut rest = flat;
            for &extent in dims.iter().rev() {
                index.push(rest % extent);
                rest /= extent;
            }
            index.reverse();

            let value = values.get(&index);
            let c = channel.get(&index) as usize;
            if c >= channels {
                return Err(BackendError::InvalidInput {
                    operation: OperationKind::Reduction,
                    reason: "channel index is outside the declared channel count",
                });
            }
            let seen = out[c] == 0.0;
            out[c] += 1.0;
            out[channels + c] += value;
            out[2 * channels + c] += value * value;
            out[3 * channels + c] = if seen {
                value
            } else {
                out[3 * channels + c].min(value)
            };
            out[4 * channels + c] = if seen {
                value
            } else {
                out[4 * channels + c].max(value)
            };
        }
        Ok(out)
    }
}

fn main() -> Result<()> {
    let channels = 2;
    let backend = CpuBackend::default();

    // Six activations over two channels: channel 0 gets 1, 3, 5 and channel 1
    // gets 2, 4, 6, so every statistic below is checkable by hand.
    let values =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [6])?;
    let channel = CpuStorage::try_from_contiguous(CpuBuffer::U32(vec![0, 1, 0, 1, 0, 1]), [6])?;
    let zeros = || CpuStorage::try_from_contiguous(CpuBuffer::F64(vec![0.0; channels]), [channels]);

    let stats: Vec<CpuStorage> = (0..STATS.len()).map(|_| zeros()).collect::<Result<_>>()?;
    let mut handles = vec![incin_core::exec::TensorHandle::from_storage::<
        CpuBackend,
        f32,
        _,
    >(&values)];
    handles.push(incin_core::exec::TensorHandle::from_storage::<
        CpuBackend,
        u32,
        _,
    >(&channel));
    for stat in &stats {
        handles.push(incin_core::exec::TensorHandle::from_storage::<
            CpuBackend,
            f64,
            _,
        >(stat));
    }

    type Out = incin_core::prelude::s![5, 2];
    let expected = ShapeValue::<Out>::try_new(ShapeBuf::from_slice(&[STATS.len(), channels]))?;
    let result = execute_shaped::<CalibrationUpdate, CpuBackend, Out>(
        &ExecutionContext::new(backend),
        CalibrationAttributes { channels },
        &handles,
        &expected,
    )?;

    for (row, name) in STATS.iter().enumerate() {
        let slice = &result[row * channels..(row + 1) * channels];
        println!("{name:>7}: {slice:?}");
    }

    // Channel 0 saw 1, 3, 5 and channel 1 saw 2, 4, 6, so every figure here is
    // checkable by hand. Asserting them keeps this example from becoming a
    // program that prints something plausible.
    assert_eq!(result[0..2], [3.0, 3.0], "count");
    assert_eq!(result[2..4], [9.0, 12.0], "sum");
    assert_eq!(result[4..6], [35.0, 56.0], "sum of squares");
    assert_eq!(result[6..8], [1.0, 2.0], "min");
    assert_eq!(result[8..10], [5.0, 6.0], "max");

    // The point of a custom operation is that the per-operand contract is
    // enforced before any backend is consulted. These are the refusals a
    // capability row could not have expressed.
    let meta = |dims: &[usize], dtype: DTypeId| LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(dims)),
        dtype: Some(dtype.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let good = || {
        let mut operands = vec![meta(&[6], DTypeId::F32), meta(&[6], DTypeId::U32)];
        operands.extend((0..5).map(|_| meta(&[channels], DTypeId::F64)));
        operands
    };
    let attributes = CalibrationAttributes { channels };

    assert!(
        CalibrationUpdate::infer_outputs(&attributes, &good()).is_ok(),
        "the well-formed operand list must be accepted"
    );

    let mut swapped = good();
    swapped.swap(VALUES, CHANNEL);
    assert!(
        CalibrationUpdate::infer_outputs(&attributes, &swapped).is_err(),
        "activations and channel indices must not be interchangeable"
    );

    let mut narrowed = good();
    narrowed[FIRST_STAT] = meta(&[channels], DTypeId::F32);
    assert!(
        CalibrationUpdate::infer_outputs(&attributes, &narrowed).is_err(),
        "a statistic accumulated in f32 must be refused"
    );

    let mut short = good();
    short.pop();
    assert!(
        CalibrationUpdate::infer_outputs(&attributes, &short).is_err(),
        "six operands must be refused"
    );

    let mut ragged = good();
    ragged[CHANNEL] = meta(&[5], DTypeId::U32);
    assert!(
        CalibrationUpdate::infer_outputs(&attributes, &ragged).is_err(),
        "one channel index per activation must be required"
    );

    println!(
        "\nper-operand contract: swapped, narrowed, short and ragged operand lists all refused"
    );
    Ok(())
}
