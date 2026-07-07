import os

files_to_fix = [
    "crates/kindle-core/src/nn/linear.rs",
    "crates/kindle-core/src/nn/conv1d.rs",
    "crates/kindle-core/src/nn/conv2d.rs",
]

for f in files_to_fix:
    with open(f, "r") as file:
        content = file.read()
    
    # We replaced `new_dyn` with `new` in the previous step.
    # The signature looks like:
    # pub fn new<A: crate::tensor::arg_into::ArgInto<S::Target>>(args: A) -> Result<Self> {
    
    new_fn_str = """
    pub fn new() -> Result<Self>
    where
        (): crate::tensor::arg_into::ArgInto<S::Target>,
    {
        Self::new_dyn(())
    }

    pub fn new_dyn<A: crate::tensor::arg_into::ArgInto<S::Target>>(args: A) -> Result<Self> {"""

    content = content.replace(
        "pub fn new<A: crate::tensor::arg_into::ArgInto<S::Target>>(args: A) -> Result<Self> {",
        new_fn_str
    )
    
    with open(f, "w") as file:
        file.write(content)

