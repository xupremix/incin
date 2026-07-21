import re
with open('crates/kindle-backends/src/cpu/tape.rs', 'r') as f:
    text = f.read()

text = re.sub(r'if matches!\(\(\&\*a\.buffer, \&\*b\.buffer\), \(CpuBuffer::Cuda\(_\), CpuBuffer::Cuda\(_\)\)\) \{[\s\S]*?return vec!\[\];\s*\}', '', text)
text = re.sub(r'if matches!\(\&\*storage\.buffer, CpuBuffer::Cuda\(_\)\) \{[\s\S]*?return;\s*\}', '', text)
text = re.sub(r'CpuBuffer::Cuda\(_\) => panic!\("sum_dim_keepdim CUDA unreachable"\),', '', text)
text = re.sub(r'CpuBuffer::Metal\(_\) => panic!\("sum_dim_keepdim not supported on Metal buffer"\),', '', text)

with open('crates/kindle-backends/src/cpu/tape.rs', 'w') as f:
    f.write(text)
