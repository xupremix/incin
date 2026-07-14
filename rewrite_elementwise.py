import re

with open("crates/kindle-native/src/ops/elementwise.rs", "r") as f:
    code = f.read()

# Replace elementwise_binary definition
code = code.replace("""fn elementwise_binary(
    lhs: &NativeStorage,
    rhs: &NativeStorage,
    out_shape: &[usize],
    f: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> NativeStorage {""", """pub(crate) fn elementwise_binary(
    op_name: &str,
    op_expr: &str,
    lhs: &NativeStorage,
    rhs: &NativeStorage,
    out_shape: &[usize],
    f: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> Result<NativeStorage> {
    #[cfg(feature = "cuda")]
    if let (NativeBuffer::Cuda(_), NativeBuffer::Cuda(_)) = (&*lhs.buffer, &*rhs.buffer) {
        return crate::ops::cuda_elementwise::launch_binary_op(op_name, op_expr, lhs, rhs, out_shape);
    }
""")

# Replace elementwise_unary definition
code = code.replace("""fn elementwise_unary(
    t: &NativeStorage,
    f: impl Fn(f64) -> f64 + Send + Sync,
) -> NativeStorage {""", """pub(crate) fn elementwise_unary(
    op_name: &str,
    op_expr: &str,
    t: &NativeStorage,
    f: impl Fn(f64) -> f64 + Send + Sync,
) -> Result<NativeStorage> {
    #[cfg(feature = "cuda")]
    if let NativeBuffer::Cuda(_) = &*t.buffer {
        return crate::ops::cuda_elementwise::launch_unary_op(op_name, op_expr, t);
    }
""")

# Fix the end of the CPU functions to return Ok(...)
code = re.sub(r'NativeStorage::from_contiguous\(NativeBuffer::F32\(out\), out_shape\.to_vec\(\)\)\n\}', r'Ok(NativeStorage::from_contiguous(NativeBuffer::F32(out), out_shape.to_vec()))\n}', code)
code = re.sub(r'NativeStorage::from_contiguous\(NativeBuffer::F32\(out\), t\.shape\.clone\(\)\)\n\}', r'Ok(NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone()))\n}', code)

# Now, replace all calls!
# For elementwise_binary calls:
code = re.sub(r'elementwise_binary\((lhs, rhs, &out_shape, \|a, b\| a \+ b)\)', r'elementwise_binary("add", "a + b", \1)?', code)
code = re.sub(r'elementwise_binary\((lhs, rhs, &out_shape, \|a, b\| a - b)\)', r'elementwise_binary("sub", "a - b", \1)?', code)
code = re.sub(r'elementwise_binary\((lhs, rhs, &out_shape, \|a, b\| a \* b)\)', r'elementwise_binary("mul", "a * b", \1)?', code)
code = re.sub(r'elementwise_binary\((grad_out, &rhs_capture, &grad_out\.shape, \|g, r\| g / r)\)', r'elementwise_binary("div_g_r", "a / b", \1).unwrap()', code)
code = re.sub(r'elementwise_binary\((lhs, rhs, &out_shape, \|a, b\| a / b)\)', r'elementwise_binary("div", "a / b", \1)?', code)

code = code.replace("""let grad_rhs = elementwise_binary(
                    grad_out,
                    &elementwise_binary(&lhs_capture, &rhs_capture, &grad_out.shape, |l, r| {
                        -l / (r * r)
                    }),
                    &grad_out.shape,
                    |g, dr| g * dr,
                );""", """let grad_rhs = elementwise_binary(
                    "mul_g_dr", "a * b",
                    grad_out,
                    &elementwise_binary("div_grad_rhs_inner", "-a / (b * b)", &lhs_capture, &rhs_capture, &grad_out.shape, |l, r| {
                        -l / (r * r)
                    }).unwrap(),
                    &grad_out.shape,
                    |g, dr| g * dr,
                ).unwrap();""")

code = re.sub(r'elementwise_binary\((grad_out, &t_capture, &grad_out\.shape, \|g, x\| \{\n\s*let x = x;\n\s*let deriv = if x > 0\.0 \{ 1\.0 \} else \{ 0\.0 \};\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("relu_grad", "b > 0.0 ? a : 0.0", \1).unwrap()', code)

code = re.sub(r'elementwise_binary\((grad_out, &t_capture, &grad_out\.shape, \|g, x\| \{\n\s*let x = x;\n\s*let cdf = 0\.5 \* \(1\.0 \+ erf_approx\(x / core::f64::consts::SQRT_2\)\);\n\s*let pdf = \(1\.0 / \(2\.0 \* core::f64::consts::PI\)\.sqrt\(\)\) \* \(-x \* x / 2\.0\)\.exp\(\);\n\s*let deriv = cdf \+ x \* pdf;\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("gelu_grad", "a * (0.5 * (1.0 + erff(b / 1.41421356)) + b * (0.39894228 * expf(-0.5 * b * b)))", \1).unwrap()', code)

code = re.sub(r'elementwise_binary\((grad_out, &t_capture, &grad_out\.shape, \|g, x\| \{\n\s*let x = x;\n\s*let deriv = if x > 0\.0 \{\n\s*1\.0\n\s*\} else if x < 0\.0 \{\n\s*-1\.0\n\s*\} else \{\n\s*0\.0\n\s*\};\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("abs_grad", "b > 0.0 ? a : (b < 0.0 ? -a : 0.0)", \1).unwrap()', code)

code = re.sub(r'elementwise_binary\((grad_out, &out_capture, &grad_out\.shape, \|g, o\| \{\n\s*let deriv = o;\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("exp_grad", "a * b", \1).unwrap()', code)

code = re.sub(r'elementwise_binary\((grad_out, &out_capture, &grad_out\.shape, \|g, o\| \{\n\s*let deriv = 1\.0 / \(2\.0 \* o\);\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("sqrt_grad", "a / (2.0 * b)", \1).unwrap()', code)

code = re.sub(r'elementwise_binary\((grad_out, &t_capture, &grad_out\.shape, \|g, x\| \{\n\s*let deriv = 1\.0 / x;\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("log_grad", "a / b", \1).unwrap()', code)

code = re.sub(r'elementwise_binary\((grad_out, &out_capture, &grad_out\.shape, \|g, o\| \{\n\s*let deriv = 1\.0 - o \* o;\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("tanh_grad", "a * (1.0 - b * b)", \1).unwrap()', code)

code = re.sub(r'elementwise_binary\((grad_out, &out_capture, &grad_out\.shape, \|g, o\| \{\n\s*let deriv = o \* \(1\.0 - o\);\n\s*g \* deriv\n\s*\})\)', r'elementwise_binary("sigmoid_grad", "a * b * (1.0 - b)", \1).unwrap()', code)

code = re.sub(r'mul_elementwise_broadcast\((grad_out, &other)\)', r'elementwise_binary("mul", "a * b", \1, &grad_out.shape, |a, b| a * b).unwrap()', code)
code = re.sub(r'mul_elementwise_broadcast\(grad_out, &rhs_capture\)', r'elementwise_binary("mul", "a * b", grad_out, &rhs_capture, &grad_out.shape, |a, b| a * b).unwrap()', code)
code = re.sub(r'mul_elementwise_broadcast\(grad_out, &lhs_capture\)', r'elementwise_binary("mul", "a * b", grad_out, &lhs_capture, &grad_out.shape, |a, b| a * b).unwrap()', code)

# Unary
code = re.sub(r'elementwise_unary\((t, \|x\| -x)\)', r'elementwise_unary("neg", "-x", \1)?', code)
code = code.replace("""fn negate(t: &NativeStorage) -> NativeStorage {
    elementwise_unary(t, |x| -x)
}""", """fn negate(t: &NativeStorage) -> NativeStorage {
    elementwise_unary("neg", "-x", t, |x| -x).unwrap()
}""")
code = re.sub(r'elementwise_unary\((t, \|x\| x \+ scalar)\)', r'elementwise_unary("add_scalar", "x + {SCALAR}", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x \* scalar)\)', r'elementwise_unary("mul_scalar", "x * {SCALAR}", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x\.max\(0\.0\))\)', r'elementwise_unary("relu", "x > 0.0 ? x : 0.0", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x \* 0\.5 \* \(1\.0 \+ erf_approx\(x / core::f64::consts::SQRT_2\)\))\)', r'elementwise_unary("gelu", "x * 0.5f * (1.0f + erff(x / 1.41421356f))", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x\.abs\(\))\)', r'elementwise_unary("abs", "fabsf(x)", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x\.exp\(\))\)', r'elementwise_unary("exp", "expf(x)", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x\.sqrt\(\))\)', r'elementwise_unary("sqrt", "sqrtf(x)", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x\.ln\(\))\)', r'elementwise_unary("log", "logf(x)", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| x\.tanh\(\))\)', r'elementwise_unary("tanh", "tanhf(x)", \1)?', code)
code = re.sub(r'elementwise_unary\((t, \|x\| 1\.0 / \(1\.0 \+ \(-x\)\.exp\(\)\))\)', r'elementwise_unary("sigmoid", "1.0f / (1.0f + expf(-x))", \1)?', code)

code = code.replace("""        let out = elementwise_unary(t, |x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        });""", """        let out = elementwise_unary("swish", "x / (1.0f + expf(-x))", t, |x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        })?;""")

# Inject scalar formatting
code = code.replace('elementwise_unary("add_scalar", "x + {SCALAR}",', 'elementwise_unary(&format!("add_scalar_{}", scalar), &format!("x + {}", scalar),')
code = code.replace('elementwise_unary("mul_scalar", "x * {SCALAR}",', 'elementwise_unary(&format!("mul_scalar_{}", scalar), &format!("x * {}", scalar),')

with open("crates/kindle-native/src/ops/elementwise.rs", "w") as f:
    f.write(code)

