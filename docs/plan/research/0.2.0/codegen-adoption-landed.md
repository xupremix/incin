codegen: the IR now executes, and adoption is a body swap not a path swap

Follow-up to `codegen-adoption.md`, which recommended adopting one path end to
end and picking it for verifiability. That has now happened on CUDA pointwise,
on real hardware, and the shape of the seam is different from what that note
predicted.

What that note got wrong. It said both emitters "end at the same NVRTC
dispatcher, so routing an operation through `codegen` instead of `kernel` is a
swap of the source producer". They do end at the same dispatcher, but they do not
produce comparable kernels. `KernelDefinition::render_forward_cuda` emits a whole
kernel: fixed `(in0, .., out, numel)` signature, contiguous indexing, no strides,
no packed loads, no launch configuration, no cache key. `crate::kernel` emits a
kernel through templates that carry strided iteration, four layout classes,
2/4-wide packing, occupancy pruning and autotune candidates. Swapping one whole
emitter for the other would have been a large regression wearing the word
"adoption".

The seam that works is one level lower. The templates in `kernel::scalar` place
the operation body in an rvalue slot with operands already loaded as
compute-typed scalars (`x`, or `a` and `b`). So the unit to replace is not the
kernel, it is the *expression*. `codegen::fragment::lower_scalar` renders an
`IrExpr` against caller-named operands, and `ScalarFragment` became the shared
body type: a hand-written literal is a fragment with an empty prologue, an IR
expression is a fragment with SSA bindings. Both then travel the identical
rendering, keying, tuning and launch path, which is why the IR path cannot
differ in *how* a kernel is scheduled, only in what arithmetic it performs.

SSA is load-bearing, not stylistic. `render_cuda_expr` inlines each use of a
subexpression's text, and `Square`, `Silu` and `Gelu` each use their operand more
than once -- `Gelu` four times. Nesting them grows the emitted source
exponentially in depth. Binding every interior node once makes emission linear,
and memoising bindings on structural identity makes CSE fall out of the same
pass. This is the property that makes the fragment renderer usable as the
fused-kernel representation `compiled-fusion-lowering.md` wants.

Three defects surfaced, all of which had survived because the IR had never
produced a number and the CUDA suites had never run.

1. `IrUnaryOp::Gelu`'s derivative was incomplete. `gelu(x) = x * Phi(x)` is a
   product, and `diff` returned only `Phi(x)`, dropping `x * Phi'(x)`. The
   comment called it "approximate"; it was a missing product-rule term, and it
   understates the gradient everywhere. Fixed, and now checked against a central
   difference on the GPU.

2. **The CUDA module cache could serve the wrong kernel.** `KernelKey::cache_id`
   is built from the operation *name*, dtypes, layout and access -- never the
   source. Several callers format a runtime value into their expression while
   passing a constant name: `cuda_powf_storage` renders `powf(x, <exp>)` as
   `"powf"`, `cuda_clamp_storage` renders its bounds as `"clamp"`, and `mean`'s
   backward renders `x * <1/axis_len>` as `"mul_scalar"`. The first variant
   compiled in a process was served to every later one. Verified on hardware:
   `powf(2, 3)` returned `4`. This is a silent wrong answer, and `mean` backward
   makes it reachable from any model that reduces over two different axis
   lengths. Fixed by mixing an FNV-1a digest of the source into the cache key;
   `tuning_problem_id` is untouched, so autotuning still groups the variants.

3. `prod` was accepted by the CUDA reduction renderer and present in its
   warp-shuffle `combine` arm but missing from its `fast_update` arm, so a
   product over a contiguous last axis hit `unreachable!()`. The two arms exist
   separately only because the shuffle path names its operand `other` and the
   load path names it `value`.

Separately, `Execute<Conv2dExact>` for CUDA destructured exactly two inputs while
the CPU binder and the CUDA kernel itself both accept an optional bias, so biased
conv2d was executable on CPU and unreachable on CUDA. Metal has the same binder
but its `conv2d` is an unimplemented stub, so nothing was reachable there to
break.

Fusion is wired into the shipped path. Every unary backward used to be two
launches and a full-size temporary: one kernel evaluates `f'(x)`, a second
multiplies it by the incoming gradient. Both are pointwise over one shape.
`catalog::unary_fused_backward` differentiates the forward symbolically and
performs the multiply inside the IR, giving a single binary kernel -- removing
one launch and one `numel` allocation per operation per backward pass.
`cuda_pointwise!`'s `unary_wrt_input` arm now takes that path for the twelve
operations the catalog covers and falls back to its hand-written pair for the
rest, so the change is incremental per operation rather than a cutover.

This is the optimisation the string representation cannot express, because you
cannot differentiate a string. It is also why the fragment renderer had to be SSA
rather than inlining: `grad_out * f'(x)` for `gelu` reuses the same `tanh` subterm
in both product-rule halves, and the inlining renderer would emit it four times.

The `unary_wrt_output` family (`exp`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`) is
deliberately excluded. Those operations capture their *output* rather than their
input, to avoid keeping the input alive across the backward pass, and their
hand-written derivatives are written in terms of that output. `diff` produces a
derivative in terms of the input, so fusing them would change what the tape has
to retain. That is a memory-lifetime decision, not a codegen one, and it should
be made deliberately rather than fallen into.

What is verified, on hardware. `ir_conformance_tests`: 19 unary and 7 binary
operations agree with the shipped literals and with an `f64` host reference; 17
symbolic derivatives agree with a central difference; the fused
`grad_out * f'(x)` kernel agrees with the two-launch backward across 15
operations; and the real autograd tape produces correct gradients for all twelve
fused operations, which is what catches a transposed operand pair that would
still be a valid kernel returning plausible numbers.

`mish` is the useful case there. Its hand-written derivative literal is a
180-character expression repeating `tanhf(log1pf(expf(x)))` three times, and it
now agrees with the symbolic one -- so it was correct, and it no longer has to be
maintained by hand to stay that way.

Vocabulary note: `elu`, `mish`, `log2` and `log10` needed no new `IrUnaryOp`.
Each is a composition of operators the IR already had (`elu` is a `Select`,
`mish` is `x * tanh(log(1 + exp(x)))`, the logs are scaled natural logs), which
is a cheaper way to grow coverage than adding enum variants, since every variant
costs five match arms across `fold`, `diff`, `eval`, `render_cuda_expr` and
`fragment`.

Not done. The inverse and hyperbolic transcendentals (`asin`, `atan`, `sinh`,
`asinh`, `erf`, `tan`) and the rounding family have no IR spelling and keep their
literals. `tan` was considered and skipped: `sin(x)/cos(x)` is not close enough
to `tanf(x)` near the poles to hold a 1e-5 relative tolerance, so it wants a real
operator rather than a composition. Binary backward is still entirely
hand-written -- `binary_forward` exists but no fused binary backward is wired.
`PRF-007` step 5 is still checked and still contradicted by the tree.
