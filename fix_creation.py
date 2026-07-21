import re
with open('crates/kindle-backends/src/cpu/creation.rs', 'r') as f:
    text = f.read()

# Remove Cuda arms
text = re.sub(r'kindle_core::prelude::DeviceVariant::Cuda\(id\) => \{[\s\S]*?Ok\(CpuBuffer::Cuda\([\s\S]*?\}\)\)\s*\},', '', text)
text = re.sub(r'kindle_core::prelude::DeviceVariant::Cuda\(id\) => \{[\s\S]*?CpuBuffer::Cuda\([\s\S]*?\}\)\s*\},', '', text)
text = re.sub(r'\(true, kindle_core::prelude::DeviceVariant::Cuda\(id\)\) => \{[\s\S]*?CpuBuffer::Cuda\([\s\S]*?\}\)\)\s*\},', '', text)
text = re.sub(r'if let CpuBuffer::Cuda\(b\) = t\.buffer\.as_ref\(\) \{[\s\S]*?return Ok\(\(b\.device_id, dev\)\);\s*\}', '', text)

with open('crates/kindle-backends/src/cpu/creation.rs', 'w') as f:
    f.write(text)
