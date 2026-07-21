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

delete_lines('crates/kindle-backends/src/cpu/ops/conv.rs', [(114, 185), (187, 259), (261, 345), (347, 432), (441, 444), (480, 483), (524, 527), (582, 585)])
delete_lines('crates/kindle-backends/src/cpu/ops/embedding.rs', [(51, 74)])
