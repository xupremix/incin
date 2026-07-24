use incin::prelude::*;
use incin::{ConstShape, DynShape, Shape};

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

#[test]
/// Test s macro.
fn test_s_macro() {
    // Static dimensions
    /// Static shape.
    type StaticShape = s![10, 20];
    assert_eq!(<StaticShape as ConstShape>::DIMS, [10, 20]);

    // Dynamic dimensions
    /// Dyn shape type.
    type DynShapeType = s![dyn, dyn];
    let arg = (10, 20); // Arg depends on the dynamic fields
    let field = <DynShapeType as Shape>::init(arg);
    assert_eq!(<DynShapeType as DynShape>::dims(&field), [10, 20]);

    // Mixed dimensions
    /// Mixed shape.
    type MixedShape = s![2, dyn, 5, dyn];
    let mixed_arg = ((), 3, (), 7);
    let mixed_field = <MixedShape as Shape>::init(mixed_arg);
    assert_eq!(<MixedShape as DynShape>::dims(&mixed_field), [2, 3, 5, 7]);

    // Wildcard '_' dimensions
    type WildcardShape = s![_, _];
    let wildcard_field = <WildcardShape as Shape>::init((15, 30));
    assert_eq!(<WildcardShape as DynShape>::dims(&wildcard_field), [15, 30]);

    // Array repetition ';' syntax
    type RepetitionShape = s![64; 3];
    assert_eq!(<RepetitionShape as ConstShape>::DIMS, [64, 64, 64]);

    // Symbolic dim with doc comments
    symbolic_dim! {
        /// Batch size dimension
        DocBatch,
        /// Sequence length dimension
        DocSeq,
    }
    type SymDocShape = s![DocBatch, DocSeq];
    let sym_field = <SymDocShape as Shape>::init((32, 128));
    assert_eq!(<SymDocShape as DynShape>::dims(&sym_field), [32, 128]);

    // Tail Ellipsis '..' syntax (s![.., 128])
    type TailFeatureShape = s![.., 128];
    let tail_field = <TailFeatureShape as Shape>::init(vec![32, 16, 128]);
    assert_eq!(
        <TailFeatureShape as DynShape>::dims(&tail_field),
        [32, 16, 128]
    );
    assert_eq!(<TailFeatureShape as DynShape>::rank(&tail_field), 3);

    // Head Ellipsis '..' syntax (s![128, ..])
    type HeadFeatureShape = s![128, ..];
    let head_field = <HeadFeatureShape as Shape>::init(vec![128, 64, 32]);
    assert_eq!(
        <HeadFeatureShape as DynShape>::dims(&head_field),
        [128, 64, 32]
    );
    assert_eq!(<HeadFeatureShape as DynShape>::rank(&head_field), 3);

    // Span Ellipsis '..' syntax (s![32, .., 128])
    type SpanFeatureShape = s![32, .., 128];
    let span_field = <SpanFeatureShape as Shape>::init(vec![32, 16, 8, 128]);
    assert_eq!(
        <SpanFeatureShape as DynShape>::dims(&span_field),
        [32, 16, 8, 128]
    );
    assert_eq!(<SpanFeatureShape as DynShape>::rank(&span_field), 4);

    // Direct Tensor creation with Ellipsis shapes
    let t_tail = Tensor::<TailFeatureShape, CpuBackendImpl>::zeros([32, 16, 128]).unwrap();
    assert_eq!(t_tail.dims(), vec![32, 16, 128]);

    let t_head = Tensor::<HeadFeatureShape, CpuBackendImpl>::zeros([128, 64, 32]).unwrap();
    assert_eq!(t_head.dims(), vec![128, 64, 32]);

    let t_span = Tensor::<SpanFeatureShape, CpuBackendImpl>::zeros([32, 16, 8, 128]).unwrap();
    assert_eq!(t_span.dims(), vec![32, 16, 8, 128]);
}

#[test]
/// Test idx macro.
fn test_idx_macro() {
    // Basic indexing
    #[allow(dead_code)]
    /// Ranges.
    type Ranges = idx![0..5, 2..10, 0, ..];
    // Compiling is enough to verify type parsing
}

#[module]
/// My custom layer.
pub struct MyCustomLayer<B: Backend> {
    /// Linear.
    pub linear: Linear<s![10, 5], B>,
    /// Ln.
    pub ln: LayerNorm<s![5], B>,
}

#[test]
/// Test module macro.
fn test_module_macro() -> Result<()> {
    // Verify that #[module] derived Parameters and StateDict automatically
    let layer = MyCustomLayer::<CpuBackendImpl> {
        linear: Linear::build(())?,
        ln: LayerNorm::build(1e-5)?,
    };

    // Since #[module] implements Parameters, this should compile:
    let params = layer.parameters();
    assert_eq!(params.len(), 4); // linear.weight, linear.bias, ln.weight, ln.bias

    Ok(())
}
