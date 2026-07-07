import os

files = ["binary.rs", "unary.rs", "reduce.rs", "manipulation.rs", "loss.rs"]
for f in files:
    path = f"crates/kindle-core/src/tensor/ops/{f}"
    with open(path, "r") as file:
        lines = file.readlines()
    
    # remove the wrong use statement
    if "use crate::tensor::ops::*;\n" in lines:
        lines.remove("use crate::tensor::ops::*;\n")
    
    # find the end of //! comments
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("//!") or line.strip() == "":
            insert_idx = i + 1
        else:
            break
            
    lines.insert(insert_idx, "use crate::tensor::ops::*;\n")
    
    with open(path, "w") as file:
        file.write("".join(lines))
        
# Fix base.rs DummyBackend
with open("crates/kindle-core/src/tensor/base.rs", "r") as file:
    content = file.read()

get_grad_str = """
        fn get_grad(_var: &Self::RawVar, _grads: &Self::Grads) -> Result<Option<Self::RawTensor>> {
            unimplemented!()
        }
"""
if "fn get_grad" not in content and "unimplemented!()" in content:
    content = content.replace("unimplemented!()\n        }", "unimplemented!()\n        }" + get_grad_str)

with open("crates/kindle-core/src/tensor/base.rs", "w") as file:
    file.write(content)
