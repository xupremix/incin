import re

with open("crates/kindle-core/src/tensor/tracing.rs", "r") as f:
    text = f.read()

# Replace } followed by optional whitespace and TracingTensor { or Ok(TracingTensor {
# where the } is from a block like let value_id = { ... }
text = re.sub(r"}\n(\s*(?:Ok\()?TracingTensor\s*\{)", r"};\n\g<1>", text)

with open("crates/kindle-core/src/tensor/tracing.rs", "w") as f:
    f.write(text)
