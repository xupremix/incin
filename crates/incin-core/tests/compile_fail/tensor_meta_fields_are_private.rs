use incin_core::exec::{Alignment, TensorMeta};
use incin_core::prelude::{DTypeId, DeviceId, ShapeBuf};

fn main() {
    let valid = TensorMeta::contiguous(
        ShapeBuf::scalar(),
        DTypeId::F32.descriptor(),
        DeviceId::cpu(),
        Alignment::BYTE,
        1,
    )
    .unwrap();
    let fields = (*valid).clone();
    let _forged = TensorMeta { fields };
}
