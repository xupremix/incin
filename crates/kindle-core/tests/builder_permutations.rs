use kindle_core::nn::*;
use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_macros::s;

type B = DummyBackend<f32>;

#[test]
fn test_linear_permutations() {
    // 1. Fully static, Bias = True (default)
    let _l1 = Linear::<s![U10, U20], B>::new().unwrap();

    // 2. Fully static, Bias = False
    let _l2 = Linear::<s![U10, U20], B, kindle_core::nn::optional::False>::new().unwrap();

    // 3. Fully static, Bias = DynParam
    let _l3 = Linear::<s![U10, U20], B, kindle_core::nn::optional::DynParam>::new_with(((), ()), true).unwrap();

    // 4. Partially static (In static, Out dynamic), Bias = True
    let _l4 = Linear::<s![U10, dyn], B>::new_with(((), 20), ()).unwrap();

    // 5. Partially static (In dynamic, Out static), Bias = True
    let _l5 = Linear::<s![dyn, U20], B>::new_with((10, ()), ()).unwrap();

    // 6. Fully dynamic, Bias = True
    let _l6 = Linear::<Dyn, B>::new_with((10, 20), ()).unwrap();

    // 7. Fully dynamic, Bias = False
    let _l7 = Linear::<Dyn, B, kindle_core::nn::optional::False>::new_with((10, 20), ()).unwrap();

    // 8. Fully dynamic, Bias = DynParam
    let _l8 = Linear::<Dyn, B, kindle_core::nn::optional::DynParam>::new_with((10, 20), true).unwrap();
}

#[test]
fn test_conv1d_permutations() {
    // Conv1dShape: (OutC, InC, K, S, P, D)
    // 1. Fully static, Bias = True (default)
    let _c1 = Conv1d::<s![U16, U3, U3, U1, U1, U1], B>::new().unwrap();

    // 2. Fully static, Bias = False
    let _c2 = Conv1d::<s![U16, U3, U3, U1, U1, U1], B, kindle_core::nn::optional::False>::new().unwrap();

    // 3. Fully dynamic channels, Bias = True
    let _c3 = Conv1d::<s![dyn, dyn, U3, U1, U1, U1], B>::new_with((16, 3), ()).unwrap();
    
    // 4. Partially dynamic (Out dynamic, In static)
    let _c4 = Conv1d::<s![dyn, U3, U3, U1, U1, U1], B>::new_with((16, ()), ()).unwrap();
}

#[test]
fn test_conv2d_permutations() {
    // Conv2dShape: (OutC, InC, K, S, P, D)
    // 1. Fully static, Bias = True (default)
    let _c1 = Conv2d::<s![U16, U3, U3, U1, U1, U1], B>::new().unwrap();

    // 2. Fully static, Bias = False
    let _c2 = Conv2d::<s![U16, U3, U3, U1, U1, U1], B, kindle_core::nn::optional::False>::new().unwrap();

    // 3. Fully dynamic channels, Bias = True
    let _c3 = Conv2d::<s![dyn, dyn, U3, U1, U1, U1], B>::new_with((16, 3), ()).unwrap();
    
    // 4. Partially dynamic (Out dynamic, In static)
    let _c4 = Conv2d::<s![dyn, U3, U3, U1, U1, U1], B>::new_with((16, ()), ()).unwrap();
}

#[test]
fn test_norm_permutations() {
    // LayerNorm
    let _ln1 = LayerNorm::<s![U10], B>::new(1e-5).unwrap();
    let _ln2 = LayerNorm::<(usize,), B>::new_with((10,), 1e-5).unwrap();

    // BatchNorm2d
    let _bn1 = BatchNorm2d::<s![U16], B>::new(1e-5, 0.1).unwrap();
    let _bn2 = BatchNorm2d::<(usize,), B>::new_with((16,), 1e-5, 0.1).unwrap();
}

#[test]
fn test_rnn_permutations() {
    // RNNCell
    let _wi = Linear::<s![U10, U20], B>::new().unwrap();
    let _wh = Linear::<s![U20, U20], B>::new().unwrap();
    let _cell = RNNCell::<s![U10, U20], B>::new(_wi, _wh);
    
    // RNN
    let _rnn1 = RNN::<s![U10, U20], B>::new(_cell.clone());
    
    let _wi2 = Linear::<(usize, usize), B>::new_with((10, 20), ()).unwrap();
    let _wh2 = Linear::<(usize, usize), B>::new_with((20, 20), ()).unwrap();
    let _cell2 = RNNCell::<Dyn, B>::new(_wi2, _wh2);
    let _rnn2 = RNN::<Dyn, B>::new(_cell2);
}
