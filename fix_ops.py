import os
import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    text = re.sub(r'if let CpuBuffer::Cuda\(_\) = \&\*_t\.buffer \{[\s\S]*?\}\n', '', text)
    text = re.sub(r'if let CpuBuffer::Cuda\(b\) = t\.buffer\.as_ref\(\) \{[\s\S]*?\}\n', '', text)
    text = re.sub(r'CpuBuffer::Cuda\(_\) => \{[\s\S]*?\}', '', text)
    text = re.sub(r'CpuBuffer::Cuda\(b\) => \{[\s\S]*?\}', '', text)
    text = re.sub(r'CpuBuffer::Cuda\(_\) => panic!\([^)]+\),', '', text)
    text = re.sub(r'CpuBuffer::Metal\(_\) => panic!\([^)]+\),', '', text)
    text = re.sub(r'if matches!\(\&\*tensors\[0\]\.buffer, CpuBuffer::Cuda\(_\)\) \{[\s\S]*?return vec!\[\];\s*\}', '', text)
    text = re.sub(r'if matches!\(\&\*storage\.buffer, CpuBuffer::Cuda\(_\)\) \{[\s\S]*?return crate::cpu::ops::cuda_reduce::launch_reduce_op\([\s\S]*?\}\n', '', text)
    text = re.sub(r'if matches!\(\&\*t\.buffer, CpuBuffer::Cuda\(_\)\) \{[\s\S]*?return crate::cpu::ops::cuda_reduce::launch_reduce_op\([\s\S]*?\}\n', '', text)
    text = re.sub(r'if let CpuBuffer::Cuda\(b\) = \&\*grad_out\.buffer \{[\s\S]*?\}\n', '', text)
    
    with open(filepath, 'w') as f:
        f.write(text)

fix_file('crates/kindle-backends/src/cpu/ops/quant.rs')
fix_file('crates/kindle-backends/src/cpu/ops/reduce.rs')
fix_file('crates/kindle-backends/src/cpu/ops/shape_ops.rs')
