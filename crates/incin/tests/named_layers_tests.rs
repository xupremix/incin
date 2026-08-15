#![cfg(feature = "cpu")]

use incin::nn::NamedLayers;
use incin::prelude::*;
/// B.
type B = incin::DefaultBackend;

#[module(no_stats, no_parameters, no_state, no_train_mode, no_to_device)]
/// Sub module.
struct SubModule {
    fc: Linear<s![100, 50], B>,
    act: ReLU,
}

#[module(
    no_stats,
    no_parameters,
    no_state,
    no_shape_info,
    no_train_mode,
    no_to_device
)]
/// Test mlp.
struct TestMLP {
    sub: SubModule,
    fc_out: Linear<s![50, 10], B>,
}

#[test]
/// Test named layers derivation.
fn test_named_layers_derivation() {
    let sub = SubModule {
        fc: Linear::build(()).unwrap(),
        act: ReLU,
    };
    let mlp = TestMLP {
        sub,
        fc_out: Linear::build(()).unwrap(),
    };

    let structure = mlp.layer_structure("model");
    assert_eq!(structure.len(), 1);

    let root = &structure[0];
    assert_eq!(root.name, "model");
    assert_eq!(root.type_name, "TestMLP");
    assert_eq!(root.children.len(), 2);

    let sub_node = &root.children[0];
    assert_eq!(sub_node.name, "model.sub");
    assert_eq!(sub_node.type_name, "SubModule");
    assert_eq!(sub_node.children.len(), 2);

    // sub.fc is a Linear layer
    let fc_node = &sub_node.children[0];
    assert_eq!(fc_node.name, "model.sub.fc");
    assert_eq!(fc_node.type_name, "Linear");
    // Linear has weight and bias parameters
    assert!(fc_node.shape_info.contains("weight: [50, 100]"));
    assert!(fc_node.shape_info.contains("bias: [50]"));

    let act_node = &sub_node.children[1];
    assert_eq!(act_node.name, "model.sub.act");
    assert_eq!(act_node.type_name, "ReLU");
    assert_eq!(act_node.shape_info, "");

    let fc_out_node = &root.children[1];
    assert_eq!(fc_out_node.name, "model.fc_out");
    assert_eq!(fc_out_node.type_name, "Linear");
    assert!(fc_out_node.shape_info.contains("weight: [10, 50]"));
}

#[test]
/// Test sequential named layers.
fn test_sequential_named_layers() {
    let net = seq!(
        Linear::<s![768, 256], B>::build(()).unwrap(),
        ReLU,
        Linear::<s![256, 10], B>::build(()).unwrap()
    );

    let structure = net.layer_structure("net");
    // Sequential structure returns flattened child nodes with sequential suffixes
    assert_eq!(structure.len(), 3);

    assert_eq!(structure[0].name, "net.Linear1");
    assert_eq!(structure[0].type_name, "Linear");
    assert!(structure[0].shape_info.contains("weight: [256, 768]"));

    assert_eq!(structure[1].name, "net.ReLU1");
    assert_eq!(structure[1].type_name, "ReLU");

    assert_eq!(structure[2].name, "net.Linear2");
    assert_eq!(structure[2].type_name, "Linear");
    assert!(structure[2].shape_info.contains("weight: [10, 256]"));

    let summary_text = net.summary();
    assert!(summary_text.contains("Linear"));
    assert!(summary_text.contains("ReLU"));
    assert!(summary_text.contains("weight: [256, 768]"));
}
