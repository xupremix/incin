import re

with open("crates/kindle-core/src/tensor/base.rs", "r") as f:
    content = f.read()

content = re.sub(r'        fn step_sgd[^{]+\{\s*(Ok\(\(\)\)|unimplemented!\(\))\s*\}', '', content)
content = re.sub(r'        fn step_adam[^{]+\{\s*(Ok\(\(\)\)|unimplemented!\(\))\s*\}', '', content)
content = re.sub(r'        fn step_adamw[^{]+\{\s*(Ok\(\(\)\)|unimplemented!\(\))\s*\}', '', content)

with open("crates/kindle-core/src/tensor/base.rs", "w") as f:
    f.write(content)
