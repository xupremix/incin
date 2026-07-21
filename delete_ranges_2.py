import sys

def delete_lines(filepath, ranges):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    out = []
    for i, line in enumerate(lines, 1):
        keep = True
        for (start, end) in ranges:
            if start <= i <= end:
                keep = False
                break
        if keep:
            out.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(out)

delete_lines('crates/kindle-backends/src/cpu/ops/loss.rs', [(70, 81)])
delete_lines('crates/kindle-backends/src/cpu/ops/matmul.rs', [(195, 275)])
