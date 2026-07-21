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

# We can specify ranges directly for norm and elementwise, and check others
# norm.rs: 52-55, 121-124, 134-149
delete_lines('crates/kindle-backends/src/cpu/ops/norm.rs', [(52, 55), (121, 124), (133, 149)])
# elementwise.rs: 74-78, 105-108
delete_lines('crates/kindle-backends/src/cpu/ops/elementwise.rs', [(74, 78), (105, 108)])
