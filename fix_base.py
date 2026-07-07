import re

with open("crates/kindle-core/src/tensor/base.rs", "r") as f:
    content = f.read()

# remove all `fn get_grad` and `fn assign_var` and `unimplemented!()` inside DummyBackend
# actually, just regex replace them.
# There are multiple occurrences of:
#        fn get_grad(_var: &Self::RawVar, _grads: &Self::Grads) -> Result<Option<Self::RawTensor>> {
#            unimplemented!()
#        }

content = re.sub(r'        fn get_grad[^{]+\{\s*unimplemented!\(\)\s*\}', '', content)
content = re.sub(r'        fn assign_var[^{]+\{\s*unimplemented!\(\)\s*\}', '', content)

# Now add exactly one assign_var and get_grad to DummyBackend.
# Find `fn conv1d` and insert before it.

insert = """
        fn get_grad(_var: &Self::RawVar, _grads: &Self::Grads) -> Result<Option<Self::RawTensor>> {
            unimplemented!()
        }
        fn assign_var(_var: &mut Self::RawVar, _tensor: &Self::RawTensor) -> Result<()> {
            unimplemented!()
        }
"""

content = content.replace("fn conv1d(", insert + "        fn conv1d(")

with open("crates/kindle-core/src/tensor/base.rs", "w") as f:
    f.write(content)

with open("crates/kindle-core/src/tensor/ops/loss.rs", "r") as f:
    content = f.read()

content = re.sub(r'        fn step_sgd[^{]+\{\s*Ok\(\(\)\)\s*\}', '', content)
content = re.sub(r'        fn step_adam[^{]+\{\s*(unimplemented!\(\)|Ok\(\(\)\))\s*\}', '', content)
content = re.sub(r'        fn step_adamw[^{]+\{\s*Ok\(\(\)\)\s*\}', '', content)

with open("crates/kindle-core/src/tensor/ops/loss.rs", "w") as f:
    f.write(content)
