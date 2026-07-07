import os

def main():
    with open("crates/kindle-core/src/tensor/ops.rs", "r") as f:
        lines = f.readlines()
        
    os.makedirs("crates/kindle-core/src/tensor/ops", exist_ok=True)
    
    header = lines[0:9] # Get comments and imports
    header_str = "".join(header)
    header_str += "use alloc::vec::Vec;\nuse alloc::format;\n"

    def write_mod(name, content):
        with open(f"crates/kindle-core/src/tensor/ops/{name}.rs", "w") as out:
            out.write(header_str + "\n" + "".join(content))

    # index.rs
    write_mod("index", lines[9:112])
    
    # binary.rs
    # macros 112 to 173
    # plus std ops at end: 1258 to end
    write_mod("binary", lines[112:173] + lines[1258:])
    
    # unary.rs
    write_mod("unary", lines[173:235])
    
    # reduce.rs
    write_mod("reduce", lines[235:294])
    
    # manipulation.rs
    # 294 to 761 ?
    write_mod("manipulation", lines[294:761])
    
    # loss.rs
    # 761 to 1258
    write_mod("loss", lines[761:1258])
    
    # create mod.rs
    with open("crates/kindle-core/src/tensor/ops/mod.rs", "w") as out:
        out.write("""pub mod index;
pub mod binary;
pub mod unary;
pub mod reduce;
pub mod manipulation;
pub mod loss;

pub use index::*;
pub use binary::*;
pub use unary::*;
pub use reduce::*;
pub use manipulation::*;
pub use loss::*;
""")
        
    print("Split completed successfully")

if __name__ == "__main__":
    main()
