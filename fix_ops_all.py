import os
import re

def fix_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r') as f:
        text = f.read()

    # Generic remove blocks starting with if let CpuBuffer::Cuda or if matches!(..., CpuBuffer::Cuda)
    # This might be tricky with regex if there are nested braces.
    # Instead, let's use a simple brace-matching parser.
    def remove_cuda_blocks(code):
        out = ""
        i = 0
        while i < len(code):
            # Look for #[cfg(feature = "cuda")] followed by something with CpuBuffer::Cuda
            # Actually, simpler: find "CpuBuffer::Cuda"
            idx = code.find("CpuBuffer::Cuda", i)
            if idx == -1:
                out += code[i:]
                break
            
            # Find the start of the line or the start of the #[cfg(feature = "cuda")] block
            cfg_idx = code.rfind('#[cfg(feature = "cuda")]', i, idx)
            start_remove = idx
            if cfg_idx != -1 and (idx - cfg_idx) < 200: # heuristic
                start_remove = cfg_idx
            else:
                # just start from the line containing CpuBuffer::Cuda
                line_start = code.rfind('\n', i, idx)
                if line_start != -1:
                    start_remove = line_start + 1
                    
            # find the end of the block or statement
            # If it's a block like `if let ... {`, find the matching closing brace
            brace_start = code.find('{', idx)
            semicolon_idx = code.find(';', idx)
            arrow_idx = code.find('=>', idx)
            
            if brace_start != -1 and (semicolon_idx == -1 or brace_start < semicolon_idx) and (arrow_idx == -1 or brace_start < arrow_idx):
                # block
                open_braces = 1
                j = brace_start + 1
                while j < len(code) and open_braces > 0:
                    if code[j] == '{': open_braces += 1
                    elif code[j] == '}': open_braces -= 1
                    j += 1
                
                # Check for return or panic followed by comma
                if j < len(code) and code[j] == ',':
                    j += 1
                    
                out += code[i:start_remove]
                i = j
                # remove trailing newline if any
                if i < len(code) and code[i] == '\n':
                    i += 1
            elif arrow_idx != -1 and (semicolon_idx == -1 or arrow_idx < semicolon_idx):
                # match arm like CpuBuffer::Cuda(...) => ... ,
                comma_idx = code.find(',', arrow_idx)
                if comma_idx != -1:
                    out += code[i:start_remove]
                    i = comma_idx + 1
                    if i < len(code) and code[i] == '\n':
                        i += 1
                else:
                    # just skip line
                    line_end = code.find('\n', idx)
                    if line_end == -1: line_end = len(code)
                    out += code[i:start_remove]
                    i = line_end + 1
            else:
                # simple statement, find semicolon or newline
                end = semicolon_idx if semicolon_idx != -1 else code.find('\n', idx)
                if end == -1: end = len(code)
                else: end += 1
                out += code[i:start_remove]
                i = end

        return out

    # We also have panic!("grad_out must be Cuda"); -> remove that line if needed
    text = remove_cuda_blocks(text)
    
    # Also clean up any lingering Metal/Cuda panic arms in match statements
    text = re.sub(r'.*CpuBuffer::Metal.*=>.*panic!.*,?\n?', '', text)
    text = re.sub(r'.*CpuBuffer::Cuda.*=>.*panic!.*,?\n?', '', text)
    text = re.sub(r'.*panic!\("grad_out must be Cuda"\);\n?', '', text)
    text = re.sub(r'.*if let CpuBuffer::Cuda.*\{\n?', '', text) # leftover if any
    
    with open(filepath, 'w') as f:
        f.write(text)

files_to_fix = [
    'crates/kindle-backends/src/cpu/ops/pool.rs',
    'crates/kindle-backends/src/cpu/ops/norm.rs',
    'crates/kindle-backends/src/cpu/ops/loss.rs',
    'crates/kindle-backends/src/cpu/ops/matmul.rs',
    'crates/kindle-backends/src/cpu/ops/optimizer.rs',
    'crates/kindle-backends/src/cpu/ops/elementwise.rs',
]

for f in files_to_fix:
    fix_file(f)

