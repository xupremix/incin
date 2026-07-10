use kindle_core::nn::*;
use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_macros::s;

type B = DummyBackend<f32, kindle_core::tensor::device::Cpu>;

#[test]
fn test_linear_permutations() {
    // 1. Fully static, Bias = True (default)
    let _l1 = Linear::<s![U10, U20], B>::new().unwrap();

    // 2. Fully static, Bias = False
    let _l2 = Linear::<s![U10, U20], B, kindle_core::nn::optional::False>::new().unwrap();

    // 3. Fully static, Bias = Dyn (true)
    let _l3 = Linear::<s![U10, U20], B, Dyn>::new_with(((), ()), true).unwrap();

    // 4. Fully static, Bias = Dyn (false)
    let _l3b = Linear::<s![U10, U20], B, Dyn>::new_with(((), ()), false).unwrap();

    // 5. Partially static (In static, Out dynamic), Bias = True
    let _l4 = Linear::<s![U10, dyn], B>::new_with(((), 20)).unwrap();

    // 6. Partially static (In dynamic, Out static), Bias = True
    let _l5 = Linear::<s![dyn, U20], B>::new_with((10, ())).unwrap();

    // 7. Fully dynamic, Bias = True
    let _l6 = Linear::<Dyn, B>::new_with((10, 20)).unwrap();

    // 8. Fully dynamic, Bias = False
    let _l7 = Linear::<Dyn, B, kindle_core::nn::optional::False>::new_with((10, 20)).unwrap();

    // 9. Fully dynamic, Bias = Dyn (true)
    let _l8 = Linear::<Dyn, B, Dyn>::new_with((10, 20), true).unwrap();

    // 10. Fully dynamic, Bias = Dyn (false)
    let _l9 = Linear::<Dyn, B, Dyn>::new_with((10, 20), false).unwrap();
}

#[test]
fn test_conv1d_permutations() {
    // Conv1dShape: (OutC, InC, K, S, P, D)
    // 1. Fully static, Bias = True (default)
    let _c1 = Conv1d::<s![U16, U3, U3, U1, U1, U1], B>::new().unwrap();

    // 2. Fully static, Bias = False
    let _c2 = Conv1d::<s![U16, U3, U3, U1, U1, U1], B, kindle_core::nn::optional::False>::new().unwrap();

    // 3. Fully static, Bias = Dyn
    let _c3b = Conv1d::<s![U16, U3, U3, U1, U1, U1], B, Dyn>::new(true).unwrap();

    // 4. Dynamic channels, Bias = True
    let _c3 = Conv1d::<s![dyn, dyn, U3, U1, U1, U1], B>::new_with((16, 3)).unwrap();

    // 5. Dynamic channels, Bias = False
    let _c4 = Conv1d::<s![dyn, dyn, U3, U1, U1, U1], B, kindle_core::nn::optional::False>::new_with((16, 3)).unwrap();

    // 6. Dynamic channels, Bias = Dyn
    let _c5 = Conv1d::<s![dyn, dyn, U3, U1, U1, U1], B, Dyn>::new_with((16, 3), true).unwrap();

    // 7. Partially dynamic (Out dynamic, In static)
    let _c6 = Conv1d::<s![dyn, U3, U3, U1, U1, U1], B>::new_with((16, ())).unwrap();
}

#[test]
fn test_conv2d_permutations() {
    // Conv2dShape: (OutC, InC, K, S, P, D)
    // 1. Fully static, Bias = True (default)
    let _c1 = Conv2d::<s![U16, U3, U3, U1, U1, U1], B>::new().unwrap();

    // 2. Fully static, Bias = False
    let _c2 = Conv2d::<s![U16, U3, U3, U1, U1, U1], B, kindle_core::nn::optional::False>::new().unwrap();

    // 3. Fully static, Bias = Dyn
    let _c3b = Conv2d::<s![U16, U3, U3, U1, U1, U1], B, Dyn>::new(true).unwrap();

    // 4. Dynamic channels, Bias = True
    let _c3 = Conv2d::<s![dyn, dyn, U3, U1, U1, U1], B>::new_with((16, 3)).unwrap();

    // 5. Dynamic channels, Bias = False
    let _c4 = Conv2d::<s![dyn, dyn, U3, U1, U1, U1], B, kindle_core::nn::optional::False>::new_with((16, 3)).unwrap();

    // 6. Dynamic channels, Bias = Dyn
    let _c5 = Conv2d::<s![dyn, dyn, U3, U1, U1, U1], B, Dyn>::new_with((16, 3), true).unwrap();

    // 7. Partially dynamic (Out dynamic, In static)
    let _c6 = Conv2d::<s![dyn, U3, U3, U1, U1, U1], B>::new_with((16, ())).unwrap();
}

#[test]
fn test_norm_permutations() {
    // LayerNorm — static
    let _ln1 = LayerNorm::<s![U10], B>::new(1e-5).unwrap();
    // LayerNorm — dynamic
    let _ln2 = LayerNorm::<(usize,), B>::new_with((10,), 1e-5).unwrap();

    // BatchNorm2d — static
    let _bn1 = BatchNorm2d::<s![U16], B>::new(1e-5, 0.1).unwrap();
    // BatchNorm2d — dynamic
    let _bn2 = BatchNorm2d::<(usize,), B>::new_with((16,), 1e-5, 0.1).unwrap();
}

#[test]
fn test_rnn_permutations() {
    // RNNCell — static
    let _wi = Linear::<s![U10, U20], B>::new().unwrap();
    let _wh = Linear::<s![U20, U20], B>::new().unwrap();
    let _cell = RNNCell::<s![U10, U20], B>::new(_wi, _wh);
    let _rnn1 = RNN::<s![U10, U20], B>::new(_cell.clone());

    // RNNCell — dynamic
    let _wi2 = Linear::<(usize, usize), B>::new_with((10, 20)).unwrap();
    let _wh2 = Linear::<(usize, usize), B>::new_with((20, 20)).unwrap();
    let _cell2 = RNNCell::<Dyn, B>::new(_wi2, _wh2);
    let _rnn2 = RNN::<Dyn, B>::new(_cell2);
}
