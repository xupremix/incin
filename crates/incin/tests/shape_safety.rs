#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin_core::prelude::ShapeError;

type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

#[test]
fn chunk_and_split_reject_invalid_axes_without_panicking() -> Result<()> {
    let tensor = Tensor::<s![2, 3], CpuBackendImpl>::zeros(())?;

    assert!(matches!(
        tensor.chunk(2, 2),
        Err(Error::Shape(ShapeError::InvalidAxis { axis: 2, rank: 2 }))
    ));
    assert!(matches!(
        tensor.split(1, 2),
        Err(Error::Shape(ShapeError::InvalidAxis { axis: 2, rank: 2 }))
    ));
    Ok(())
}
