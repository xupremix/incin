import re

with open("crates/kindle-backends/src/lib.rs", "r") as f:
    code = f.read()

# Strip lines added at bottom
code = "\n".join([line for line in code.split("\n") if "OptimizerOps<Self> for CandleBackend" not in line and "OptimizerOps<Self> for NdarrayBackend" not in line])

# Insert CandleBackend OptimizerOps inside candle_backend cfg
code = re.sub(
    r'(impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>\s*kindle_core::prelude::LossOps<Self> for CandleBackend<T, D> \{[\s\S]*?\n    \})',
    r'\1\n\n    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device> kindle_core::tensor::backend::OptimizerOps<Self> for CandleBackend<T, D> {}',
    code
)

# Insert NdarrayBackend OptimizerOps inside ndarray_backend cfg
code = re.sub(
    r'(impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>\s*kindle_core::prelude::LossOps<Self> for NdarrayBackend<T, D> \{[\s\S]*?\n    \})',
    r'\1\n\n    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device> kindle_core::tensor::backend::OptimizerOps<Self> for NdarrayBackend<T, D> {}',
    code
)

with open("crates/kindle-backends/src/lib.rs", "w") as f:
    f.write(code)

