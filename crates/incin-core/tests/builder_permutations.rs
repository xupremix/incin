#![allow(unused_imports)]
extern crate incin_core as incin;

use incin_core::prelude::*;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

/// B.
type B = DummyBackend<f32, incin_core::prelude::Cpu>;

#[test]
/// Test linear permutations.
fn test_linear_permutations() {
    // 1. Fully static, Bias = True (default)
    let _l1 = Linear::<s![10, 20], B>::build(()).unwrap();

    // 2. Fully static, Bias = False
    let _l2 = Linear::<s![10, 20], B, incin_core::prelude::False>::build(()).unwrap();

    // 3. Fully static, Bias = Dyn (true)
    let _l3 = Linear::<s![10, 20], B, Dyn>::build(true).unwrap();

    // 4. Fully static, Bias = Dyn (false)
    let _l3b = Linear::<s![10, 20], B, Dyn>::build(false).unwrap();

    // 5. Partially static (In static, Out dynamic), Bias = True
    let _l4 = Linear::<s![10, dyn], B>::build(20).unwrap();

    // 6. Partially static (In dynamic, Out static), Bias = True
    let _l5 = Linear::<s![dyn, 20], B>::build(10).unwrap();

    // 7. Fully dynamic, Bias = True
    let _l6 = Linear::<Dyn, B>::build((10, 20)).unwrap();

    // 8. Fully dynamic, Bias = False
    let _l7 = Linear::<Dyn, B, incin_core::prelude::False>::build((10, 20)).unwrap();

    // 9. Fully dynamic, Bias = Dyn (true)
    let _l8 = Linear::<Dyn, B, Dyn>::build((10, 20, true)).unwrap();

    // 10. Fully dynamic, Bias = Dyn (false)
    let _l9 = Linear::<Dyn, B, Dyn>::build((10, 20, false)).unwrap();
}

#[test]
/// Test conv1d permutations.
fn test_conv1d_permutations() {
    // Conv1dShape: (OutC, InC, K, S, P, D)
    // 1. Fully static, Bias = True (default)
    let _c1 = Conv1d::<s![16, 3, 3, 1, 1, 1], B>::build(()).unwrap();

    // 2. Fully static, Bias = False
    let _c2 = Conv1d::<s![16, 3, 3, 1, 1, 1], B, incin_core::prelude::False>::build(()).unwrap();

    // 3. Fully static, Bias = Dyn
    let _c3b = Conv1d::<s![16, 3, 3, 1, 1, 1], B, Dyn>::build(true).unwrap();

    // 4. Dynamic channels, Bias = True
    let _c3 = Conv1d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((16, 3)).unwrap();

    // 5. Dynamic channels, Bias = False
    let _c4 =
        Conv1d::<s![dyn, dyn, 3, 1, 1, 1], B, incin_core::prelude::False>::build((16, 3)).unwrap();

    // 6. Dynamic channels, Bias = Dyn
    let _c5 = Conv1d::<s![dyn, dyn, 3, 1, 1, 1], B, Dyn>::build((16, 3, true)).unwrap();

    // 7. Partially dynamic (Out dynamic, In static)
    let _c6 = Conv1d::<s![dyn, 3, 3, 1, 1, 1], B>::build(16).unwrap();
}

#[test]
/// Test conv2d permutations.
fn test_conv2d_permutations() {
    // Conv2dShape: (OutC, InC, K, S, P, D)
    // 1. Fully static, Bias = True (default)
    let _c1 = Conv2d::<s![16, 3, 3, 1, 1, 1], B>::build(()).unwrap();

    // 2. Fully static, Bias = False
    let _c2 = Conv2d::<s![16, 3, 3, 1, 1, 1], B, incin_core::prelude::False>::build(()).unwrap();

    // 3. Fully static, Bias = Dyn
    let _c3b = Conv2d::<s![16, 3, 3, 1, 1, 1], B, Dyn>::build(true).unwrap();

    // 4. Dynamic channels, Bias = True
    let _c3 = Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((16, 3)).unwrap();

    // 5. Dynamic channels, Bias = False
    let _c4 =
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B, incin_core::prelude::False>::build((16, 3)).unwrap();

    // 6. Dynamic channels, Bias = Dyn
    let _c5 = Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B, Dyn>::build((16, 3, true)).unwrap();

    // 7. Partially dynamic (Out dynamic, In static)
    let _c6 = Conv2d::<s![dyn, 3, 3, 1, 1, 1], B>::build(16).unwrap();
}

#[test]
/// Test norm permutations.
fn test_norm_permutations() {
    // LayerNorm — static
    let _ln1 = LayerNorm::<s![10], B>::build(1e-5).unwrap();
    // LayerNorm — dynamic
    let _ln2 = LayerNorm::<(usize,), B>::build((10, 1e-5)).unwrap();

    // BatchNorm2d — static
    let _bn1 = BatchNorm2d::<s![16], B>::build((1e-5, 0.1)).unwrap();
    // BatchNorm2d — dynamic
    let _bn2 = BatchNorm2d::<(usize,), B>::build((16, 1e-5, 0.1)).unwrap();
}

#[test]
/// Test rnn permutations.
fn test_rnn_permutations() {
    // RNNCell — static
    let _wi = Linear::<s![10, 20], B>::build(()).unwrap();
    let _wh = Linear::<s![20, 20], B>::build(()).unwrap();
    let _cell = RNNCell::<s![10, 20], B>::new(_wi, _wh);
    let _rnn1 = RNN::<s![10, 20], B>::new(_cell.clone());

    // RNNCell — dynamic
    let _wi2 = Linear::<(usize, usize), B>::build((10, 20)).unwrap();
    let _wh2 = Linear::<(usize, usize), B>::build((20, 20)).unwrap();
    let _cell2 = RNNCell::<Dyn, B>::new(_wi2, _wh2);
    let _rnn2 = RNN::<Dyn, B>::new(_cell2);
}
