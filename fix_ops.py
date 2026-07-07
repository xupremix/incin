import os

files = ["binary.rs", "unary.rs", "reduce.rs", "manipulation.rs", "loss.rs"]
for f in files:
    path = f"crates/kindle-core/src/tensor/ops/{f}"
    with open(path, "r") as file:
        content = file.read()
    
    # prepend use crate::tensor::ops::*;
    content = "use crate::tensor::ops::*;\n" + content
    
    with open(path, "w") as file:
        file.write(content)
