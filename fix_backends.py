import re

with open("crates/kindle-backends/src/lib.rs", "r") as f:
    code = f.read()

# Remove the bad lines at the end
code = "\n".join([line for line in code.split("\n") if "OptimizerOps<Self> for CandleBackend" not in line and "OptimizerOps<Self> for NdarrayBackend" not in line])

code += """
impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device> kindle_core::tensor::backend::OptimizerOps<Self> for CandleBackend<T, D> {}
impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device> kindle_core::tensor::backend::OptimizerOps<Self> for NdarrayBackend<T, D> {}
"""

with open("crates/kindle-backends/src/lib.rs", "w") as f:
    f.write(code)

with open("crates/kindle-core/src/tensor/tracing.rs", "r") as f:
    tracing_code = f.read()

tracing_code += """
impl<B: crate::tensor::backend::Backend> crate::tensor::backend::OptimizerOps<Self> for TracingBackend<B> {}
"""

with open("crates/kindle-core/src/tensor/tracing.rs", "w") as f:
    f.write(tracing_code)

