# Changelog

> Dated headings below `0.1.0` are development snapshots retained for
> traceability, not published releases.

All notable changes to the Incin framework will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- **BREAKING: the layout parameter moved from the marker onto the trait.**
  `Layout` is now `Layout<S>`, and `RowMajor` and `ChannelsLast` are unit
  structs. The shape is written once:

  ```text
  // was
  let x: Tensor<s![1,2,2,2], B, f32, NoGrad, Local, ChannelsLast<s![1,2,2,2]>> = ..;
  // now
  let x: Tensor<s![1,2,2,2], B, f32, NoGrad, Local, ChannelsLast> = ..;
  ```

  This is not only ergonomics. `LayoutOf<S>` was the trait that stated a layout
  describes a tensor of shape `S`, and nothing in the crate bounded anything on
  it -- it appeared in two tests and no production code, while `Tensor` bounded
  its layout on the shape-free `Layout`. So
  `Tensor<s![2, 3], B, f32, NoGrad, Local, RowMajor<s![9, 9, 9]>>` -- a rank-two
  tensor carrying a rank-three row-major claim -- was a well-formed type that
  compiled. `Layout<S>` closes that by construction: the marker has no shape of
  its own to disagree with. `tests/compile_fail/layout_must_match_its_shape.rs`
  pins it.

  `LayoutOf<S>` is gone, folded into `Layout<S>`. `RestateFor<S2>` is now
  `Restatable`: with no shape on the marker there is no `Restated` type to
  project, so what remains is the permission itself, and the congruence with
  the destination shape becomes an ordinary bound the compiler checks --
  `into_shape` now reads `L: Restatable + Layout<S2>`.

  Two consequences worth knowing. A layout parameter shared by tensors of
  different shapes must now describe each of them, which `LSTM`'s input and
  hidden states made explicit for the first time. And rejections read far
  better: `ChannelsLast: Restatable is not satisfied` in place of a line of
  typenum.

- **`ChannelsLast` is claimable only at rank four.** It previously implemented
  `Layout<S>` for every `S` and reported an empty stride list when the rank was
  wrong, so `Tensor<s![2, 3], .., ChannelsLast>` was a nameable type whose
  layout claimed nothing -- an impossible state the compiler had no reason to
  reject, because the rank test lived in a constant rather than in a bound.
  Rank is not directly bindable (`Shape::RANK` is an associated `Option<usize>`,
  and a constant cannot gate an impl) but it is expressible structurally, which
  is what the new `Rank4` marker does: four `DimCons` cells terminated by `Nil`,
  or a `Ranked<U4>`. `Dyn` is excluded deliberately, since a runtime rank cannot
  prove it is four.

### Added

- **`Nhwc<S, B>`, the channels-last spelling that writes the shape once.**
  `Dense<S, B>` has always existed for the row-major case, and its own
  documentation names the reason: the layout is congruent with the shape it
  describes, so repeating it is noise. Channels-last had no such alias, which
  left the long form as the only way to say it:

  ```text
  // was
  let x: Tensor<s![1, 2, 2, 2], B, f32, NoGrad, Local, ChannelsLast<s![1, 2, 2, 2]>>
      = Cpu.tensor_in(nchw)?;
  // now
  let x: Nhwc<s![1, 2, 2, 2], B> = Cpu.tensor_in(nchw)?;
  ```

  The layout still carries the shape. That parameter is load-bearing rather
  than incidental: `Layout::STATIC_STRIDES` and `Layout::PROOF` are computed
  from `S::STATIC_EXTENTS` and `S::RANK`, and it is what stops a channels-last
  claim riding through a shape change onto extents whose strides differ --
  `RestateFor` is the deliberate way across. The alias removes the repetition
  at the call site without removing the check, and a test pins that it names
  the same type as the long form.

### Fixed

- **CUDA launch parameters no longer truncate silently.** The grid dimensions
  and kernel ABI arguments are `u32`/`i32`; several launch paths narrowed a
  `usize` with a bare `as`, which wraps past the type's maximum and launches an
  undersized grid while the caller believes the whole tensor was covered. The
  optimizer step was worse than truncation: it saturated with
  `unwrap_or(i32::MAX)`, so a step past the ABI's range pinned Adam's bias
  correction at a constant and went on training with a frozen correction rather
  than reporting anything. Every remaining narrowing in the CUDA operation
  paths -- the optimizer launch grid, the normalization grid, and the nine in
  the quantization kernels -- now goes through the same `checked_u32` /
  `checked_i32` conversions the rest of the backend already used, so an
  out-of-range extent is a typed `ArithmeticOverflow` rather than a quiet
  miscomputation. The six casts left are provably safe: five re-cast a value
  `checked_i32` already validated, and one derives from a literal block size.
- **The large-file inventory covers the two catalog files that crossed the
  threshold.** `exec/catalog/inference.rs` (1210 lines) and
  `exec/catalog/attributes.rs` (1204) grew past 1200 as routing primitives were
  added, and `tools/check-large-files.sh` had been failing on them. Both now
  carry a named reason and a stated extraction target in `docs/HANDOFF.md`,
  which is what that gate asks for.

### Removed

- **The in-place mutation operations.** `add_`, `sub_`, `mul_`, `div_`,
  `zero_` and `fill_`, their six catalog rows, and the `Mutation` semantic
  profile are gone. None of them mutated in place: each dispatched the ordinary
  out-of-place operation and rebound `self.inner`, so `zero_` was
  `mul_scalar(0.0)` and `fill_(v)` was `mul_scalar(0.0).add_scalar(v)` -- two
  allocations behind a name promising none. The catalog rows were advertised by
  no backend at all, and the profile's own documentation said why: *"Not
  executable: `ExecutionRequest::inputs` is a slice of shared borrows, so an
  executor cannot reach a mutable operand at all."* One test used them; nothing
  else in the workspace did. `docs/FROZEN_FOUNDATIONS.md` already listed true
  in-place mutation as deferred pending a design for ownership, views and
  autograd versioning, and that is now the whole story rather than half of it.
- **`InPlaceShapeMismatch` and `parse_inplace_shape_mismatch`**, which
  humanised a diagnostic no code path can emit any more.

### Added

- **`Tensor::one_hot`, the fourth routing primitive.** `index.one_hot::<E>()`
  encodes an integer index tensor as boolean rows with one `true` at the named
  slot, so a `[T]` routing decision over `E` experts becomes the `[T, E]`
  dispatch matrix the router multiplies by. The depth is a const parameter, so
  it is both the attribute the descriptor validates and the extent the type
  appends: the two cannot disagree because they are the same number. Declared
  on CPU only, with a conformance fixture; out-of-range indices encode as
  all-`false` rows per ONNX `OneHot` rather than erroring. Catalog grows to
  174 operations, 164 backend-executable.

- **A custom operation with seven operands across three element types**, at
  `crates/incin-backends/examples/calibration_update.rs`. Per-channel
  quantization calibration needs the activations (`f32`), the channel each
  element belongs to (`u32`), and five running statistics (`f64`, because
  summing squares over a batch in `f32` loses the precision the pass exists to
  establish). It implements `Operation`, implements `Execute` for the real CPU
  backend, and runs on real storage.

  It is a worked answer to whether that arrangement is expressible. As a
  *custom* operation it is: the custom path calls `Operation::infer_outputs`
  and nothing else, so the per-operand contract there is the whole contract. As
  a catalog operation it is not: `verify_outputs` requires built-in operands to
  share a dtype, and the only heterogeneity admitted is one designated integer
  index operand, named in a hardcoded match over six operations. What a custom
  operation still cannot do is *advertise* the arrangement, because
  `CapabilityQuery` carries one dtype with no operand index -- the same split
  the built-in `cross_entropy_loss` lives with.

  The example asserts its own arithmetic and that its contract refuses swapped,
  narrowed, short and ragged operand lists, and CI runs it rather than only
  building it.

- **A differentiable custom operation with two inputs, two outputs, and its own
  backward**, at `crates/incin-backends/examples/polar_cartesian.rs`.
  Polar-to-Cartesian takes radius and angle and returns `x` and `y` -- the
  multi-output inference a single-output catalog row cannot express. Its two
  outputs travel the typed path with one `ShapeValue` each (see the entry
  below), so the descriptor cross-checks both geometries. A squared-error
  readout closes the graph, and the three backward recipes (one per polar
  output, one for the loss) are assembled into core `TapeNode`s and walked by the same
  `incin_core::exec::tape::backward` the CPU backend calls, so a tensor
  consumed by both outputs receives the sum of both contributions. The example
  checks the forward values and hand-derived gradients against textbook
  answers, sweeps every input element against central finite differences
  (worst relative error `1e-11`), asserts the contract refusals, and fits
  `(r, theta)` to a target point by gradient descent through nothing but its
  own backward. The one seam it does not cross is the backend's thread-local
  tape push, which stays `pub(crate)` by design; an in-tree backend would move
  the same node construction into its `Execute` impl.

- **Typed dispatch takes N outputs with per-output dtypes.** `execute_shaped`
  held exactly one `&ShapeValue<S>`, so a multi-output operation -- TopK's
  values and indices, or any custom op inferring more than one -- could not
  travel the typed path and re-derived its geometry frontend-side, unchecked.
  A sealed `ExpectedShapes` trait (one borrowed buffer per output, in order,
  plus a combined proof and evidence) is implemented for `ShapeValue<S>` and
  for 2/3/4-tuples; `infer_typed` and `infer_custom_typed` compare
  element-wise, and new `execute_shaped_n` carries the tuples while
  `execute_shaped` stays the arity-1 alias, so the 100+ tensor-surface call
  sites compile unchanged. Combined evidence takes the weakest proof with no
  statics -- one output's geometry says nothing about another's -- which is
  byte-identical to the old evidence for arity 1, and the comparison still
  borrows on both sides, so the hot path allocates nothing new. Rejections
  name the output: count via `OutputArity`, wrong or missing shape via
  `MetadataMismatch` with the index. The `polar_cartesian` example now runs
  its two-output op through the typed path with identical results.
  `Tensor::topk` follows it: values and indices travel one `execute_shaped_n`
  call with a proof each instead of an untyped dispatch plus unchecked
  `from_parts`, so a tampered second geometry is refused naming output 1.

### Changed

- **CUDA movement rows claim the dense storage set.** Transpose, broadcast,
  narrow and concat move bytes by element width with no arithmetic, and a
  byte-exact hardware matrix (`cuda_shape_dtypes.rs`, six dtypes times four
  kernels times leading/trailing-axis geometries, plus a `q8_0` refusal)
  proves it -- so `BroadcastAs`, the coarse `Broadcast` pair, and new exact
  `TransposeExact`/`TransposeView`/`Narrow`/`ConcatExact` rows advertise
  `i64`, `bf16`, `f16`, `f32`, `f64` and `bool` instead of `f32` alone. A
  public-dispatch test proves the rows admit what the kernels do (`i64`
  transpose and broadcast through admission, values checked). The remaining
  narrow rows (comparisons, creation, selection, scalars) keep their shape
  for #90's audit.

### Fixed

- **CUDA `layer_norm` runs its backward.** The fused Welford forward had no
  gradient path, so its capability row stayed inference-only. The kernel now
  also writes per-row mean and inverse-std when the grad mode records (a flag
  plus scratch stand-ins keep inference launches allocation-free), and a new
  fused backward kernel replays exactly those statistics: input gradients
  from `rstd * (gw - mean(gw) - y * mean(gw * y))` with the weight folded in
  *before* the means, weight/bias gradients by atomic accumulation. The
  `Execute` impl delegates to a tape-tracked backend method, and the
  capability row is training-capable. Seven hardware tests prove it: forward
  parity (guarding the template edit), backward parity against the composed
  CPU reference, a uniform-gradient analytic identity, the no-bias two-output
  case, a white-box proof that perturbed statistics move the gradients (so
  the kernel replays rather than recomputes), and launch refusals. Writing
  the tests caught a real defect first: the kernel averaged the upstream
  gradient before weighting it, which passes every uniform-weight check and
  fails parity -- the comment in the template names the trap.

- **CUDA losses train: `unbroadcast` completes scalar gradients, and the IR
  differentiator agrees on `sign(0)`.** Verifying the premise that no CUDA
  loss trains turned it around: `mse_loss` already trained through composed
  tape entries, and `l1`, `bce_with_logits_loss` and cross-entropy do too,
  with two defects fixed along the way. First, `cuda::tape::unbroadcast`
  reduced leading and size-1 axes but handed anything smaller straight on,
  so a mean-seeded scalar gradient reached a binary launch that refuses it;
  it now materializes the broadcast after a compatibility check. Second, the
  symbolic `diff` answered -1 for `d|x|/dx` at 0 while CPU `Sign`, PyTorch
  and the hand-written CUDA expression answer 0, so the fused abs path
  diverged from the unfused one exactly where l1 losses evaluate; the IR
  nests the select now. Third, the cross-entropy executor called the raw
  gather launch, which records no tape entry: routing through the tracked
  gather closes the walk and the logits gradient matches `(softmax -
  onehot)/batch` by hand. Each fix carries a regression test proven
  red-then-green (stash round-trips for the first two). What remains of the
  GPU-training gap is `dropout`, `group_norm` and `instance_norm`.

- **Three more training-capable CUDA rows that recorded nothing: `softmax`,
  `rms_norm`, `transpose_view` -- plus attention end to end.** Each
  advertised training while its backward silently dropped, the same defect
  class as cross-entropy's gather link. `softmax` now composes from tracked
  primitives (`log_softmax` plus `exp`), which also makes its docstring true;
  `rms_norm` saves one norm factor per row and replays a hand recipe built
  only from tracked primitives; `transpose_view`'s recipe materializes
  rather than viewing again, because a strided gradient would read back
  flat. Scaled dot-product attention trains through all of them. Four
  hardware tests prove it against CPU references and hand definitions, each
  inverted red-then-green. Writing the rms test caught a real formula slip
  first: the weight gradient summed `g*w*z` instead of `g*z`, understating
  every lane by its own weight -- CPU parity does not negotiate.

- **The reference page taught testing instead of usage.** Every section ended
  in a dump of worked examples while the items themselves showed either
  nothing or one-line occurrence matches, and the pool behind both included
  `#[test]` suites whose scaffolding and assertion plumbing confuse a reader
  trying to learn the call. The pool is now the book chapters plus the
  runnable example programs, both compiled by CI; each operation, type,
  dtype, layout, shape and target method opens onto the guide blocks that
  name it, with an honest empty state and a book link where none does. The
  section-end dumps are gone. The shapes section opens with the chapter's
  own static/mixed/dynamic blocks as a gallery, and the data-flow tab walks
  a custom operation from declaration to dispatch through the runnable polar
  example -- contract, kernel, readout, backward -- resolved from that file
  at build time. The browser suite was rewritten to the new contract: inline
  examples, gallery and authoring steps, same scale/tick/scroll/docs-link
  checks as before.

- **A third vacuous suite, and what it found.** `cuda_shape_dtypes.rs` asserted
  that `DTypeDescriptor::size_bytes` returns 1, 2, 4 or 8 for eight dtypes. It
  launched nothing, opened no device, and would have passed unchanged with the
  whole CUDA backend deleted -- while standing in as coverage for the movement
  operations it is named for. It now launches transpose, broadcast and narrow
  and checks that elements land where they belong, including the two cases a
  naive kernel gets wrong while passing the obvious one: broadcasting along a
  trailing rather than a leading axis, and narrowing an inner axis, where the
  window is not one contiguous run. Eight tests, green on a GTX 1650 SUPER.
- **The capability page kept its examples at arm's length.** Every section ended
  in a dump of worked book examples, while opening an operation showed only the
  registry rows and opening an element type showed nothing at all -- so a reader
  asking "what does using this look like" scrolled past the item to a pile of
  long-form snippets indexed by chapter rather than by item. Expanding an
  operation now lists up to three one-line use sites inline (the first line
  naming it, each linked to the book chapter or test file it came from), and
  each element-type card has a use-sites disclosure doing the same. The lines
  come from the same literal-name usage index the type, layout, shape and
  target rows already open onto, so an item with no snippet says so rather than
  being given an unrelated one, and the section-end worked examples stay where
  they are. The browser suite opens `matmul` and `f32` and requires the lines
  to name what was opened.
- **`unresolved link to `Grad`` broke two CI jobs.** The `TargetExt` doc
  comment added with the parameter-allocation work linked a marker that is not
  in that scope, which failed the CPU Test Suite and the Documentation Build on
  the same root cause.

- **The browser test suites never ran an assertion.** `tools/test-book-site.py`
  drove a real headless browser through ~20 assertions and then decided the
  outcome with `"BOOK_TEST=PASS" not in output`, where `output` is
  `--dump-dom`. The dump includes the harness's own `<script>` source, and that
  source contains the sentinel as a literal, so the substring was always
  present: the suite reported success with `check(false)` as the first
  statement of `run()`. Both suites now read the verdict out of the result
  element, and assemble the sentinel from two pieces so the literal cannot
  reappear in the source. Turning the assertions on exposed the four defects
  below, all of them previously green.
- **A permalink reloaded the chapter it pointed into.** Clicking a heading
  anchor in the chapter already on screen refetched that chapter and replaced
  its body, so the heading just clicked was removed from the document while the
  request was in flight, the reader lost their place, and every in-page anchor
  cost a network round trip. A route into the mounted chapter is now a scroll.
  A stale response from a superseded navigation is also discarded rather than
  landing over the newer chapter.
- **Arrow-key chapter navigation never worked.** The global `keydown` handler
  opened with `event.target.matches(...)`; `matches` is defined on `Element`
  and not on `Document`, so a keydown delivered to the document threw, and the
  throw took the arrow-key branches below it with it.
- **The capability page scrolled sideways on a phone, and its meters were not a
  scale.** The page is a column flex item and resolved its cross size to
  max-content; the category table carries `min-width: 680px`, so max-content
  stayed 769px however narrow the viewport got, and every child stretched with
  it. The page now takes a definite width and the table scrolls inside its own
  container. Separately, each row's bar was drawn against the widest backend
  *on that row* -- a denominator of 1 on 26 rows, 4 on 81 and 8 on 54 -- so a
  full bar meant four different amounts of support and no two rows could be
  compared. Every bar is now drawn against all nine dtypes, and the best any
  backend reaches on that operation is marked as a tick instead of as the
  scale.

### Changed

- **The capability meters read as a compatibility scale.** Colour now encodes
  coverage rather than implementation: a stepped red-to-green ramp, one step per
  element type, drawn against the same nine on every row. Length and colour say
  the same thing, so the colour reinforces the reading instead of smuggling in a
  second variable -- which is what made the first version of this page
  unreadable, with hue meaning implementation and fill depth meaning coverage.
  The ramp is stepped rather than smooth because the quantity is a count of
  whole element types.
- **`composed` is explained where it is used, and marked once.** It means the
  backend runs the operation by combining other operations it supports rather
  than with a kernel written for it: it executes and is correct, and it costs
  more. It is now a dot beside the operation name -- 29 of 179 operations are
  composed on some backend -- instead of tinting the number and bar it shares
  with coverage.
- **The key moved above the table it explains**, so the encoding is met before
  the marks rather than after 179 rows of them, and each operation now carries
  its public entry point as a signature line (`Tensor::abs`) above its
  description, in the shape a reader will type it.

- **CUDA advertises less than its movement kernels deliver.** The capability
  registry lists `f32` alone for transpose, broadcast and narrow. Writing the
  boundary test above to assert that an unadvertised element type is refused
  turned the expectation around: all three kernels accept `i64` and move it to
  exactly the right places. They move bytes by element width and do no
  arithmetic, so nothing in them is specific to `f32`. This is the same shape
  as the elementwise rows that declared contiguous-only while a complete
  strided kernel sat behind them: a declaration that makes a working kernel
  unreachable through the public API. The measurement is recorded as a test;
  widening the rows is left as open work, because a declaration should be
  widened against evidence for every dtype it would then claim, not the one
  that happened to be measured.

### Added

- **Code on the reference page is syntax highlighted**, by the same
  highlighter the book uses. It lived inside `book.js`, which this page does
  not load; it is now one shared file both pages load, rather than a second
  copy to keep in step.
- **Every item opens to real usage.** Types, layouts, shape types, target
  methods and element types are clickable like operations, and opening one
  shows compiled code that uses it: 530 names are covered by 602 snippets drawn
  from the book chapters and from the repository's own test functions, all of
  which the compiler checks. The heading says *used in*, never *the example
  for*, because the match is a literal occurrence of the name in the code -- a
  fact about the snippet, not a claim about its meaning. An item with no
  snippet says so instead of showing an unrelated one.
- **Every proportional bar on the page uses the coverage ramp.** The summary
  and backend bars kept a fixed hue while their markup already asked for a step
  class, so they read on a different scale from every other bar on the page.
  One helper now maps a percentage to the same nine steps everywhere, and the
  page test checks each bar's colour against its own width in every section.
- **Element types state the storage they actually occupy**, which answers what
  a `bool` costs: one byte per element, eight times its information content. A
  dtype's encoding is a block -- logical values per block, bytes per block,
  alignment -- so packing is already expressible, and `q8_0` uses it for 32
  values in 34 bytes. A bit-packed boolean would be 8 values in 1 byte; what
  stands in the way is not the encoding but that strides are counted in
  elements and resolved to byte offsets, so a sub-byte element turns every
  stride computation, view and kernel index into a bit address.
- **Three further sections: layouts, shapes and the target API.** Layouts lists
  the markers and traits with what each claims, and what each backend accepts
  contiguous versus strided -- the gap being a capability question, since a
  contiguous-only row makes a working strided kernel unreachable. Shapes lists
  all 99 documented types across the fifteen shape modules. The target API
  lists every `TargetExt` method, all of which return a `NoGrad` tensor except
  `parameter`.
- **The reference covers the type surface, element types, backends and the
  dispatch pipeline, not only operations.** Five sections now: operations; every
  public struct, trait, enum and type alias across the ten shipped crates (3210
  of them, 1955 linked to docs.rs); the nine element types with the backends and
  operation counts that accept each; what each backend advertises across the
  catalog; and the ordered stages a call passes through before a kernel runs,
  with the failure class each one owns.

  All of it is read from sources the repository already gates on rather than
  restated by hand: the type surface from the reviewed `cargo public-api`
  baselines, the element types from the `DTypeId` enum that defines them, the
  backend and dtype counts from the capability payload, and the pipeline from
  the lowering chapter and `exec/dispatch.rs`. The type reference is fetched on
  demand -- it is three times the weight of everything else on the page, and
  most readers never open it.
- **Every operation links to its documentation on docs.rs.** 147 of the 179
  resolve to a checked item anchor; the other 32 carry a search of the same
  crate, because the published release either predates them or documents them
  under another name. Nothing is guessed: `tools/build-docsrs-links.py`
  proposes candidates for each operation -- the catalogued entry point, the
  bare name, and the structural forms (`cmp_eq` through `eq`, the `_dim` and
  `_keepdim` reductions through their base) -- and keeps only the ones that
  match an anchor docs.rs actually publishes. Near-misses are deliberately not
  mapped: `prod_dim` is not `prod_all` and `scatter_add` is not `scatter`, and
  associating them would repeat the plausible-but-wrong guess that produced a
  wrong test-coverage metric earlier in this project. The resolver is run by
  hand and its result committed, so the payload generator CI drift-checks stays
  offline and reproducible, and it refuses to build if any operation lacks a
  link.
- **The capability page test grew with the page.** It now also pins that every
  bar carries the coverage step its number states and is painted that step's
  colour, that the ramp shows nine distinct colours **in each of the five book
  themes**, that the composed dot appears only where an operation is composed,
  and that the key precedes the table. The theme check earned its place
  immediately: the ramp's light palette was first written behind a
  `data-theme="dark"` that no theme in this book sets, and then behind
  `html:not(.js)`, which matches every theme here because the shell overwrites
  `html.className` with the theme name -- so all five themes rendered the light
  ramp and nothing said so.
- **`tools/test-api-page.py`**, which holds the capability page to the contract
  its meters imply: one denominator for the whole page, every bar drawn at the
  ratio it states, a reference tick exactly where another backend reaches
  further, summary cards on the same absolute footing, and no horizontal
  overflow at 390px. Both defects above were reported by a reader looking at
  the rendered page, which is the wrong place to find them. Wired into CI, the
  Pages deploy and the release workflow.
- **A pointwise result proves it is dense.** Unary and binary pointwise
  operations, the scalar forms and the `core::ops` operators now state
  `RowMajor<S>` instead of carrying the operand's layout, so
  `t.relu()?.reshape_view::<s![12]>()?` needs no runtime stride scan even when
  `t` proved nothing. Backed by conformance tests on CPU and CUDA that feed a
  genuinely strided operand, rather than by what the backends happen to do.
  Carrying the operand's layout was also latently false: it would have handed a
  `ChannelsLast` claim to a row-major buffer.
- **Reductions, comparisons and the rest of the allocating surface prove they
  are dense too.** `sum`/`mean`/`max`/`min`/`logsumexp` and their `_keepdim`
  forms, `cumsum`, the six comparison operators, `logical_and`/`or`/`not`,
  `masked_fill`, `where_cond` and `lerp` now state `RowMajor` of their result
  shape. `reduce.rs`
  already *documented* that "the results are freshly allocated dense buffers";
  nothing checked it against an operand that was not already dense, and two
  signatures disagreed with the sentence in opposite directions -- the axis
  reductions claimed nothing, while `cumsum` returned `Self` and so claimed
  whatever its *operand* claimed. `cumsum` is the one that mattered: it is
  shape-preserving, so carrying the operand's layout typechecked, and would have
  been a false claim for any operand that was not already row-major.
- Impl blocks that pinned `L` to its default and so were unreachable from a
  tensor carrying a proof: the six comparison operators, `logical_and`/`or`/
  `not`, the scalar `Mul`/`Add`/`Sub` operators for all four scalar types, and
  `masked_fill`'s mask parameter. Found by widening the reductions -- once `sum`
  returned a proof, the next call in the chain stopped compiling.
  `scaled_dot_product_attention` now ties its `q` operand to the impl block's
  layout parameter, which both lets it accept a proven operand and keeps
  `Tensor::scaled_dot_product_attention(..)` inferrable through a type alias
  (a type alias's parameter defaults do not apply in expression position, so an
  unconstrained `L` would have made the call ambiguous).
- The `incin` facade's `Tensor` alias gained the layout parameter, and a
  matching `Dense` alias that defaults its backend the same way. The alias fixed
  `L` to the default, so a facade user could not name the type of anything that
  returned a proof.
- **`matmul`, `addmm` and `Linear` prove their results are dense.** They were
  the operations explicitly left claiming nothing, on the grounds that the
  conformance evidence covered the pointwise surface only. That evidence now
  exists on both backends, so they claim: `matmul` returns
  `Dense<S1::Output, ..>`, `addmm` returns `Dense<S1, ..>` (it previously
  returned `Self`, so like `cumsum` it handed the *bias operand's* layout to a
  GEMM result), and all four `Linear` `Module` impls return `Dense`. `addmm`'s
  `mat1`/`mat2` and `matmul`'s operands accept any layout.
- The CPU test feeds a real strided operand through a GEMM and checks the
  numbers, not only the strides -- a kernel reading the operand linearly would
  return a correctly shaped buffer of wrong values. The CUDA test checks the
  same product, so the backends are compared against one answer rather than
  each against itself.
- **`BatchNorm2d` proves its result is dense; `Dropout` earns the right to carry
  its operand's.** Both were shape-preserving `nn` layers returning the
  operand's layout. `BatchNorm2d` has no identity path -- every call dispatches
  and writes a fresh buffer -- so it now returns `Dense`. `Dropout` genuinely
  does: in eval mode, or at `p == 0`, it hands back the very tensor it was
  given, strides and all, so carrying is right there. Its bound moved from
  `Layout` to the sealed `FreshDense<S>`, which is what makes the *other* branch
  honest: that branch writes a dense buffer, so the layout carried across both
  has to be one a fresh dense allocation also satisfies. Bounding on `Layout`
  compiles today only because `Dyn` and `RowMajor` are the only layouts and both
  are dense.
- **`RestateFor<S2>`, and a layout proof that survives `into_shape`.** The rule
  that a layout cannot be carried across a shape change has one exception, and
  `into_shape`/`into_dyn`/`to_shape` are it: they change no dimension. They
  re-describe the *same* extents under a different shape type, over the same
  buffer, with the same strides -- `S2::try_from_dims` is what makes them
  fallible, and what rules out the case where the two shapes disagree. So
  `RowMajor<S1>` and `RowMajor<S2>` denote identical strides whenever the
  conversion succeeds, and dropping to `Dyn` there discarded a fact that was
  still true. `RestateFor` is the type-level half of that argument; it is not
  sealed, because unlike `FreshDense` nothing about it can be minted.
- `RmsNorm` returns `Dense`. Its chain ends in `broadcast_mul`, which allocates;
  the proof was being lost on the `into_shape` back to a static shape. That was
  the concrete thing `RestateFor` was written for.
- The four loss `forward` methods -- `MSELoss`, `L1Loss`, `CrossEntropyLoss`,
  `BCEWithLogitsLoss` -- accept operands carrying any layout and state `Dense`
  for the result they allocate. They pinned `L` on *both* arguments, so once
  `Linear` returned a proof, feeding its output straight into a loss stopped
  compiling.
- **The shape operations are sorted into views and materialisations, by
  measurement rather than by signature.** `unsqueeze`, `try_squeeze`, the three
  `flatten` forms, `unfold`, `scatter` and `scatter_add` materialise and now
  state `Dense`. `try_narrow`, `broadcast_to`, `expand` and `chunk` are genuine
  views and keep claiming nothing.

  This group is the one where the answer could not be read off the signature.
  `unsqueeze` and `try_squeeze` look like pure metadata edits, and they are --
  until the operand is strided, at which point they route through a reshape
  that materialises. `scatter` and `scatter_add` returned `Self`, which is the
  sixth appearance of the shape-preserving pattern. The split was established
  by feeding each one a `transpose_view` result and reading the strides back,
  and both halves are asserted: claiming the views are dense would be as wrong
  as leaving the rest at `Dyn`.

  `chunk` is why a stride check alone is not enough to call something a view --
  its second piece shares the strides and moves the *offset*, so the test pins
  that too.
- **The manipulation surface proves its results are dense.** `concat`, `stack`,
  `gather`, `index_select`, `repeat`, `pad`, `diag`, `pixel_shuffle`,
  `to_dtype`, `triu`, `tril`, `group_norm` and `instance_norm` all allocate and
  now say so.

  Four of them -- `triu`, `tril`, `group_norm`, `instance_norm` -- returned
  `Self`, which is the fifth appearance of the same pattern and again in the
  shape-preserving members of the group. Everything else here changes the shape
  and so was forced to state *something*, which is exactly why it stated `Dyn`
  and nobody had to look again.

  **Breaking**: an annotation written `Tensor<S, B, K, G>` for one of these
  results becomes `Dense<S, B, K, G>`. Five in the workspace's own tests and
  examples needed it.
- **The spatial and embedding layers prove their results are dense.** `Conv1d`,
  `Conv2d`, `AvgPool2d` and `Embedding` were the `nn` layers still stating
  nothing. Unlike the pointwise surface they have no strided-operand case to
  construct: CPU advertises `spatial_layouts = CONTIGUOUS`, so a strided input
  is refused before a kernel runs, and a `RowMajor` result cannot be wrong for
  an operand the backend will not accept -- the same argument that makes the
  CUDA reduction claim safe.
- **The order-statistic family proves its results are dense.** `argmax`,
  `argmin`, `argsort` and `topk` were the reductions left stating nothing; they
  write fresh index buffers -- `topk` and `argsort` write two -- so the same
  rule applies. **Breaking**: their return types change, and `topk` returns a
  pair of `Dense` rather than a pair of `Tensor`.
- `Tensor::into_layout::<L>()`, the general form of `into_row_major`. The layout
  names the strides it needs through `FreshLayout::strides`, those are compared
  against the tensor's actual metadata, and the claim is granted only on a
  match. Fallible for the same reason: strides are a runtime fact and the only
  honest route is to look. Still no unchecked counterpart.
- `tensor_in` and `zeros_in` take the layout as their **first** type parameter,
  so the common call needs no turbofish at all -- the annotation that names the
  proof chooses the layout, `let x: Dense<s![2, 2], _> = Cpu.tensor_in(data)?`.
  The ordering is deliberate rather than cosmetic: Rust's turbofish is
  all-or-nothing, and the two parameters differ in whether they can be inferred.
  The data type is the argument's own, so it is always fixable at the argument
  by binding the value or suffixing the literal; the layout appears nowhere in
  the call. Putting it first means the parameter that occasionally needs naming
  is the one that can be named alone.
- `TargetExt::zeros_in`, the layout-expressing counterpart to `zeros`, and the
  `restate_layout` hook behind it for the paths where the backend allocates
  rather than the host uploading.
- `crates/incin-core/tests/target_layout_examples.rs`, seven worked cases from
  the design note kept as tests rather than prose, so the examples in the book
  cannot drift from what compiles.
- **Parameters and state tensors carry a proof from their allocation.**
  `TargetExt::state_tensor` dispatches a creation kernel and every one writes
  dense, so it returns `Dense` -- routed through `restate_layout`, so the claim
  is checked against the metadata rather than resting on that sentence staying
  true of some future kernel. `parameter` carries it through to a
  gradient-tracking tensor via the new `TargetTensorInGrad` alias.
- `require_grad` and `detach` **carry** the operand's layout rather than
  restating it. They join `Dropout` as the only operations entitled to: both
  re-tag the autograd identity and hand back the same storage, so whatever was
  true of the buffer's strides is still true of them. `require_grad` was found
  by the proof from `state_tensor` flowing into it and failing to compile.
- Two `forget_layout` calls in `nn_target.rs` are gone. They existed because
  one branch allocated and the other did not; once `state_tensor` claimed, both
  branches agreed and the weakening deleted itself -- the second time the
  design has removed a weakening rather than needing one added, after `Linear`.
- **`ChannelsLast` is constructible.** `HostInterop::from_bytes_strided` lets a
  backend accept strides the caller chose; it is defaulted to a capability
  refusal, so the contract stays additive and a backend that has not opted in
  simply cannot be asked for a non-dense allocation. CPU opts in. `tensor_in`
  permutes host data into the layout's order before upload rather than
  refusing, using the new `shapes::scatter_positions`.

  This is the case the design note flagged as most likely to be got wrong: a
  permutation that walks the nesting incorrectly produces a buffer of exactly
  the right length, shape and strides, holding the wrong values. Its test
  asserts the round-trip *and* carries a negative control showing what an
  unpermuted upload reads back as, so the assertion is discriminating rather
  than merely plausible.
- `TensorData` covers rank three and four, so an NCHW literal can be written.
  Four is where it stops because that is where the shapes are -- the
  convolutions, the pooling layers, and the rank channels-last is defined
  against. Closes the remaining half of #116.
- **`ChannelsLast<S>`, the second layout -- which is what makes the first one's
  bounds mean anything.** NHWC memory under an NCHW shape: `dims()` still
  reports `[N, C, H, W]` and only the strides move, so channels is the
  fastest-varying axis. It deliberately does **not** implement `Contiguous`.

  The point is not convolution, though that is the eventual use. Until this
  existed, `Dyn` and `RowMajor` were the only layouts and *both* satisfied every
  bound that mentioned them, so `reshape_view`'s `Contiguous` requirement was
  vacuously true for the entire inhabited world and had never rejected
  anything. `tests/compile_fail/reshape_view_needs_contiguous.rs` is that bound
  becoming a check.

  Rank four only: channels-last is defined against NCHW, and a layout that
  silently accepted any rank would be claiming a geometry it had not checked.

  Not yet constructible through any public path -- `zeros` is bounded on the
  sealed `FreshDense`, and the target API refuses a layout whose strides no
  backend can allocate. Making it inhabitable needs backend allocation support
  and is the next step; see
  `docs/plan/research/0.2.0/layout-at-construction.md`.
- **`FreshLayout<S>`, and a target API that can express a layout.** The trait
  carries `fn strides(dims) -> StrideBuf`: the strides an allocation must use
  for the layout to describe it. A constructor bounded on it asks the layout,
  allocates with the answer, and only then names it in the type -- so the claim
  is honest by construction, the value that produced the strides being the one
  in the type. That is why it is unsealed where `FreshDense` had to be sealed:
  the hazard the seal closed was a constructor stamping a layout onto an
  allocation that had nothing to do with it, which cannot happen once the
  layout chose the allocation.
- `TargetExt::tensor_in` and the `allocate_in` hook beneath it, plus the
  `TargetTensorIn<T, S, K, L>` alias. `TargetTensor` names five of six
  parameters, so nothing the target API returned could carry a proof. The new
  methods are additive rather than a widening of the old ones because a generic
  parameter on a function cannot have a default -- widening `tensor` itself
  would make every existing call site ambiguous, the same trap
  `scaled_dot_product_attention` hit.
- A layout whose strides no backend can allocate is **refused** at construction,
  through the existing capability vocabulary, rather than satisfied with a dense
  buffer wearing the wrong type. Nothing can trigger it yet, because both
  layouts ask for dense strides; it is there because a creation API that cannot
  say no is a way to mint a proof.
- `incin_core::shapes::dense_strides`, the suffix-product helper both
  implementors share.
- `Tensor::forget_layout`, the weakening counterpart to `into_row_major`. Total
  where the promotion is fallible, since claiming less can never claim wrongly.
  It exists for the case where two branches must meet and only one allocates;
  using it on a value that stays proven discards exactly what the layout
  parameter carries.
- The `incin` facade prelude now re-exports `Dense`, `RowMajor`, `Layout`,
  `Contiguous` and `FreshDense`. They reached `incin-core`'s prelude when the
  parameter landed but not the facade, so a user of the `incin` crate could hold
  a layout-carrying tensor and had no way to name its type.
- `Debug`, `Display`, `backward`, `backward_with`, the loss family and all four
  `core::ops` operator impls accept layout-carrying tensors. Each previously
  bound `L` to its default, so `-t` and `a + b` did not compile for a proven
  tensor.
- **Typed layout.** `Tensor` gained a sixth parameter, `L: Layout`, describing
  where a tensor's elements live: strides, offset, alignment and contiguity.
  It defaults to `Dyn` -- the same runtime-selected marker the shape, dtype,
  device and placement slots use -- which claims nothing, so existing code is
  unchanged
  and every runtime path stays available. `RowMajor<S>` derives its strides from
  the shape; `Dense<S, B, ..>` is the ergonomic alias. Facts are traits --
  `Contiguous` -- and `LayoutOf<S>` states rank congruence. There is
  deliberately no `AlignedTo<N>`: alignment is a property of the allocation
  rather than of the shape, so no layout derived from `S` can honestly imply it.
  See the [Layout chapter](docs/book/src/layout.md). Every tensor module accepts
  a layout-carrying operand. Shape-preserving operations carry the operand's
  layout through, so a proof survives a chain; shape-changing ones state theirs
  as `Dyn`, since a layout describes one geometry and cannot be carried to
  another.
- **Constructors yield a layout proof.** `zeros`, `ones`, `randn`, `full` and
  their siblings are generic over `L`, so `let t: Dense<s![3, 4], B> =
  Tensor::zeros(())?` produces a real `RowMajor` from the allocation itself,
  with no runtime promotion. Asking for `Tensor<S, B>` still yields `Dyn`,
  so nothing that predates the parameter changed. Before this, `reshape_view`
  was the only API bounded on `Contiguous` and nothing could satisfy it without
  going through `into_row_major` -- a runtime stride scan -- which left the
  static layout system behaviourally equal to a runtime check.
- `FreshDense<S>`, the **sealed** bound that makes the above safe. A constructor
  generic over `L` is otherwise a minting press: name any layout and receive a
  tensor claiming it. That is harmless while a fresh allocation genuinely is
  both `Dyn` and `RowMajor`, and stops being harmless the moment a second
  real layout such as `ChannelsLast` exists. Only this crate decides what a
  fresh allocation may claim.
- `Tensor::into_row_major`, a *checked* promotion from runtime strides to a
  type-level layout. There is deliberately no unchecked counterpart.
- `Tensor::reshape_view`, bounded on `L: Contiguous`: reinterprets a buffer
  under a new shape without copying. Reshaping a non-contiguous tensor is a
  compile error rather than the runtime failure it is elsewhere.
- `transpose_view`, a transpose that permutes shape and strides over the same
  buffer instead of copying, and the `TransposeView` operation behind it. CPU
  and CUDA implement it; WGPU deliberately does not advertise it, because its
  pointwise shaders address linearly and would read a view's elements in the
  wrong order. Which of the two transposes is faster depends on how often the
  result is read, so the framework offers both rather than choosing (#113).
  Measured on a GTX 1650: the view is ~45% faster for a single pointwise
  consumer and ~23% slower by eight, crossing over at about four reads.
- `AnyTensor` and `TensorOf<T>`, so generic code names one type parameter
  instead of six. The parameters stay reachable as associated types, so a bound
  that genuinely needs one still writes `T::Layout: Contiguous`; only the ones a
  helper does not constrain stop having to be written down.
- `Shape::STATIC_EXTENTS`, per-axis extents settled by the shape type, carried
  to backends on `ShapeEvidence` alongside the existing proof level, rank and
  element count.
- Proof-directed CUDA kernel specialisation. `ShapeEvidence` had no backend
  readers; it now has three. A statically proven element count that divides the
  vector width proves a packed kernel's ragged tail unreachable, so it is not
  emitted; proven per-axis extents let a strided kernel use literal divisors,
  which the compiler lowers to multiply-and-shift; and the `shape` array and its
  per-launch upload are dropped entirely when the extents are baked in.
- CUDA pointwise kernels are now lowered from `codegen`'s expression IR rather
  than from hand-written CUDA C literals, via `codegen::fragment::lower_scalar`.
  Derivatives come from `IrExpr::diff` rather than a second hand-written string,
  so a forward and its backward can no longer disagree.
- Fused CUDA unary backward: `grad_out * f'(x)` in one kernel, removing a launch
  and a full-size intermediate per operation per backward pass.
- `log_softmax`, `logsumexp` and `scatter_add`, with `DuplicateIndexRule::Accumulate`.

- The Layout chapter is now part of the Cargo-backed doctest aggregation. It was
  added to `SUMMARY.md` when the parameter landed and never to
  `crates/incin/src/lib.rs`, so `tools/check-docs.py` -- which only runs in the
  Pages workflow, off `master` -- had been failing on `develop` unnoticed and
  the chapter's samples were compiled by nothing.

### Removed

- The pointwise operations' "output carries the operand's layout" contract.
  **Breaking**: roughly forty methods change return type. A signature written
  as `Tensor<S, B, K, G>` for a pointwise result becomes `Dense<S, B, K, G>`,
  or keeps its shape by calling `forget_layout`; which is right depends on
  whether the caller wants the proof.
- The "output carries the operand's layout" contract from the reduction,
  comparison, logical, `masked_fill` and `lerp` surfaces as well. **Breaking**:
  `sum`, `mean`, `max`, `min`, `logsumexp`, their `_keepdim` forms, `cumsum`,
  `eq`/`ne`/`lt`/`le`/`gt`/`ge`, `logical_and`/`or`/`not`, `masked_fill` and
  `lerp` and `where_cond` change return type. An annotation written
  `Tensor<S, B, K, G>` for one of their results becomes `Dense<S, B, K, G>`, or
  keeps its shape by calling `forget_layout`.
- `Unknown`, the layout marker, in favour of the existing `Dyn`. **Breaking**
  for anyone who named it: `incin_core::shapes::Unknown` and the prelude
  re-export are gone, and `Tensor`'s sixth parameter now defaults to `Dyn`.
  Code that never wrote the marker down is unaffected, since the default is
  what changed name rather than meaning.

  One marker now covers every "decided at runtime" slot instead of two spellings
  for one idea, so a `where` clause that wants "unproven anything" names a single
  type. The objection this overrides is that a fully spelled-out tensor can now
  say `Dyn` twice, once for a dynamic shape and once for an unproven layout; the
  humanizer resolves that by position rather than by name, which is the more
  precise test in any case.

  Measured cost: with the default renamed, rustc stopped abbreviating one
  `compile_fail` rendering, which now spells out `f32, NoGrad, Local, Dyn` where
  it previously printed neither. The humanizer still strips the layout argument
  from it. The reason rustc's default-elision changed behaviour was not pinned
  down -- a reduced standalone case with the same parameter structure, defaults
  and trait impls still elides correctly.

### Changed (editor integrations)

- Hover and `expected .. found ..` diagnostics no longer print a tensor's
  layout argument when it is the default `Dyn`, which asserts nothing. A layout
  that is anything else is a real claim -- the difference between a tensor that
  can call `reshape_view` and one that cannot -- and is always shown. A `Dyn`
  *shape* is never elided, even though the layout slot spells its default the
  same way: the two are told apart by **position**, since the layout is the
  sixth of six parameters. That test is stricter than the name test it
  replaced, which fired on any trailing argument regardless of arity.

### Fixed

- **The array constructors produce the same shape type `s![..]` does.** `TensorData`
  built its shapes from `ConstDim<N>`, a different type that does not implement
  `ConcreteStaticExtent` -- so nothing `Cpu.tensor(..)` returned could reach
  `ElementCount`, and `reshape`/`reshape_view` were unavailable on the most
  ergonomic constructor in the crate. Routed through `typenum`'s
  `Const<N>`/`ToUInt` bridge, which the doc comment above the macro had claimed
  was already in use. Closes #116.

  A side effect worth having: `incin-diagnostics` translates `UInt<..>` chains
  to plain integers and had no `ConstDim` handling at all, so shapes from these
  constructors now humanize in editor diagnostics where they previously did not.

  **Breaking** only for code naming `<[[f32; 2]; 2] as TensorData>::Shape`
  explicitly. Four trybuild snapshots re-blessed for the new spelling.

- `incin-diagnostics` failed to compile under `--no-default-features`: its test
  module used `String` without importing it from `alloc`, which the crate needs
  because it is `no_std` without `std`. Never caught because the Feature
  Powerset job runs only on the nightly schedule, not on push.

- **CUDA pointwise refused the only operand that made its strided kernel
  reachable.** `elementwise_layouts` declared `CONTIGUOUS`, so the descriptor
  path answered `layout strided is unsupported for neg` for any non-contiguous
  CUDA tensor -- while the strided elementwise kernel exists, is benchmarked,
  and beats materialising for a single consumer. The declaration was true when
  written, since every CUDA operation materialised and no strided CUDA tensor
  could be built; `transpose_view` ended that and the row was not revisited,
  which meant this changelog's own "~45% faster for a single pointwise
  consumer" described a path callers could not take. 54 CUDA elementwise
  operations now advertise `strided`. The other capability rows stay narrow
  until each has the same evidence.
- The `Module` doctest had not compiled since the layout conversion introduced
  it: it spelled `crate::tensor::grad::NoGrad`, and inside a doctest `crate` is
  the doctest's own crate. Missed because `cargo test --all-targets`, the CI
  invocation, is the one form that does not build doctests.

- **Every codegen module rendered CUDA that could not compile.** All 21 emitted
  `#include <math.h>`, which NVRTC rejects outright -- it compiles a translation
  unit with no host headers on the include path. That is the shared reason
  behind the "modules with no consumer" backlog (#111): they could not have had
  one. The working kernel templates never did this; they include only
  `cuda_fp16.h`/`cuda_bf16.h` and call `expf`, `sqrtf` and `tanhf` directly,
  which NVRTC provides as builtins. `codegen_nvrtc_smoke` now compiles all 20
  CUDA entry points against the real device so this cannot return.
- **Hardware tests reported `ok` when there was no hardware.** Every CUDA and
  WGPU suite opened with a predicate and an early `return`, so a missing device
  produced a green line indistinguishable from a test that ran. `require_cuda`
  and `require_wgpu` fail instead. Verified in both directions: the suites pass
  with a device and fail under `CUDA_VISIBLE_DEVICES=""`.
- **The trybuild suites ran on exactly one machine.** Twenty-three test files
  guarded their compile-fail cases behind
  `if std::fs::read("/home/<user>/.cargo/config.toml").is_err() { return; }`, an
  absolute path into one developer's home directory. Anywhere else -- every CI
  runner included -- the read fails and the suite returns early reporting `ok`.
  That silently disabled all 70 compile-fail and compile-pass cases, which are
  the proof surface this framework exists for. They now run; all 70 pass.
- **The hardware floor could not see a suite disappear.** The workflow required
  at least 60 ignored CUDA tests against 66 actually running, leaving room for a
  whole suite to evaporate inside the guard meant to detect exactly that.
  `cargo xtask hardware-tests` derives the expectation from the `#[ignore]`
  reasons in the tree, and fails on an unclassified reason rather than
  defaulting to either side.

- **The CUDA optimizers did nothing and returned zeros.** `launch_sgd_step`,
  `launch_adam_step` and `launch_adamw_step` each allocated a zeroed output,
  compiled the kernel module, discarded their gradient and attributes with
  `let _ = (grad, attrs)`, and returned the zeros. No optimizer kernel was ever
  launched -- there was not one `.launch(` call in the file. Any model trained on
  CUDA had its parameters zeroed on the first optimizer step. The kernels
  themselves were complete and correct in `kernels/optimizer.cu`; only the launch
  was missing. This survived because `tests/cuda_optimizer.rs` recomputed Adam,
  AdamW and SGD in its own body and asserted its own arithmetic, calling no incin
  code at all.
- **CUDA kernels were compiled for NVRTC's default architecture, not the
  device's.** No `--gpu-architecture` was passed, so every kernel targeted a
  pre-sm_60 virtual architecture. That is not merely a lost optimisation: an
  intrinsic introduced after the default does not exist, so a kernel using one
  fails to *compile*, taking its whole module with it. The embedding module did
  exactly that -- `embedding_backward` needs `atomicAdd(double*, double)` from
  sm_60 -- which took the forward lookup down alongside it, so CUDA embeddings
  were entirely non-functional. The dispatcher now queries its device's compute
  capability and names the newest virtual architecture it supports.
- **CUDA `argmax`/`argmin` computed only the first output row.** The kernel
  reads its output position from `blockIdx.x` and uses a whole block to stride
  the reduction axis, but the launch sized the grid as `out_numel / 256` -- one
  thread per output rather than one block. For any output up to 256 elements
  that is a single block: row zero correct, every other row left at its
  zero-initialised value. Silently wrong, not an error. Found by rewriting
  `tests/cuda_reduce_ops.rs`, which had asserted that `size_bytes` returns `Ok`
  for three dtypes. The neighbouring welford, cumsum and topk launches already
  sized their grids correctly.
- **The CUDA module cache could serve the wrong kernel.** `KernelKey::cache_id`
  was built from the operation name and never the source, so callers that format
  a runtime value into their expression under a fixed name collided with
  themselves: `powf(2, 3)` returned `4` after `powf(x, 2)` had been compiled.
  Also reachable through `clamp` and through `mean`'s backward, which renders
  `x * 1/axis_len` under the constant name `"mul_scalar"`. The cache key now
  includes a digest of the kernel source.
- `IrUnaryOp::Gelu`'s symbolic derivative dropped a product-rule term, silently
  understating the gradient everywhere.
- A CUDA product reduction over a contiguous last axis reached `unreachable!()`:
  `prod` was accepted by the renderer and present in its warp-shuffle arm but
  missing from the load arm.
- CUDA `conv2d` refused a bias, making biased convolution executable on CPU and
  unreachable on CUDA, though both the CPU binder and the CUDA kernel already
  supported it.
- Compiled fusion offered candidates it could not justify: `find_candidates`
  paired nodes by position and never counted consumers, so a value with two
  readers was a candidate. It now proves exclusive consumption from the graph's
  own edges.

### Changed

- `DuplicateIndexRule` is `#[non_exhaustive]`.

## [0.1.0] - 2026-08-25

The first release intended for crates.io. CPU is the complete, verified
backend; the GPU backends, distributed planning, compiled execution, and the
automatic trainer ship as previews. See `docs/MIGRATION.md` for what an
upgrade from a `0.0.0` snapshot has to act on.

The `compiled` feature is specifically a CPU reference evaluator and plan
inspection surface under `incin::experimental::compiled`. It is not a stable
compiler, deployment target, or portable artifact ABI. Its preview plan
snapshots require matching artifact format and caller-supplied compatibility
major/minor values.


### Breaking changes

- **incin-data typed errors:** Public `incin-data` APIs no longer return
  `anyhow::Result`. `Downloader`, `MnistDataset` construction, and the Hugging
  Face Hub client now return the framework error (`incin_core::error::Result`),
  classifying failures as `Error::Io`, `Error::MalformedArtifact`,
  `Error::ResourceLimit`, or `Error::ArithmeticOverflow`. Transform pipelines
  and collation return the crate's typed `DataError`, which gained an
  `InvalidInput` variant for rejected transform inputs. Callers matching on
  error text or constructing `anyhow` errors from these surfaces must switch to
  the typed variants.

### Removed

- **`incin_backends::iteration` from the public surface.** The module exposed a
  single item, `tile_2d`, a 2D loop-tiling helper that takes runtime
  dimensions and sits below the descriptor contract. It is now `pub(crate)`,
  matching the "Internal Modules" rule in `docs/API_DESIGN.md`. No call site
  changed: internal kernels already used the `crate::iteration::` path, and the
  helper was never referenced from user documentation. Its test coverage moved
  into the module rather than being dropped.

- **`protoc` as a build dependency.** `incin-core`'s build script ran
  `prost-build` unconditionally, making a system protobuf compiler mandatory
  for every crate that depended on the facade. The generated ONNX module is
  checked in and regenerated with `cargo xtask onnx`; `cargo xtask onnx
  --check` verifies it in CI, which is now the only job that installs protoc.
- **`onnx-pb`**, unreleased since 2020, and the second `prost` major it pinned
  into the dependency tree.
- **`DummyBackend`, the shape-only stand-in backend**, and
  `incin-core`'s `test-utils` feature that carried it. It implemented
  `Execute<O>` for *every* catalog operation and stored a shape instead of
  data, so a test written against it passed whether or not any backend could
  run the operation and whatever values the operation produced. Its tests were
  migrated to the real CPU backend, which strengthened several of them: the
  distribution tests now check the support of the distribution they name
  rather than only the shape of the tensor, and `ones` is now distinguishable
  from `zeros`. `incin::test_utils` survives with fault injection only, and a
  consumer fixture proves enabling `test-utils` does not bring the stand-in
  back.
- **`tokio` and a duplicate `rustls`** from the default dependency graph; the
  Hub client that pulled them is now behind the `data-hub` feature.
- **Deprecated `candle` feature alias (`REL-002`, `D-014`):** Removed `candle` feature alias from `incin` and `incin-backends` in favor of explicit `external-candle`.
- `Backend::backward_with_nan_check` and its four implementations. NaN checking
  is `NanPolicy` on `ExecutionPolicy`; wrap the ordinary `backward` in
  `incin_core::exec::check_gradients(|| ..)`, which returns an error where the
  old method panicked.

### Added

- **A published API tier classification.** `docs/public-api/API_TIERS.md`
  assigns every module in every shipped crate to one of four tiers: stable user
  API, expert/backend-authoring API, intentional macro ABI, or preview. It
  states what a 0.1.0 consumer may rely on, and records which surfaces were
  reviewed as privatization candidates and deliberately kept.


- **`clip_grad_norm`** - total-norm gradient clipping over a `ParameterGroup`,
  returning the norm before rescaling.
- **`clip_grad_value`** - per-element gradient clamping over a
  `ParameterGroup`, backed by the `ValueClippingBackend` trait. Together with
  `clip_grad_norm` these were the training primitives the framework was
  missing.
- **`AutogradBackend::set_grad`** - the backend primitive clipping needs.
  Required rather than defaulted; see `docs/MIGRATION.md`.
- **WGPU unary activations are reachable.** `relu`, `step`, `mish`, `elu`,
  `gelu`, `abs`, `exp`, `neg`, `sqrt`, `log`, `tanh`, `sigmoid`, and `swish`
  had working shaders and `Execute` implementations but were never advertised
  by the capability registry, so canonical dispatch refused them. They are
  advertised now, verified numerically against reference implementations on a
  software adapter.
- **A capability assertion in both directions.** The compile-time check that
  every advertised WGPU row has an executor is joined by one proving every
  written executor is advertised, so a kernel cannot become unreachable again.
- **Metal in the generated capability matrix.** `METAL_CAPABILITIES` existed
  and `metal` was a documented feature, but the document had no column for it.
- **Publish metadata** on all ten publishable crates: keywords, categories,
  documentation links, and `[package.metadata.docs.rs]` feature sets, checked
  by `tools/check-publish-metadata.py`.
- **`rust-version = "1.88"`**, verified by a CI job pinned to that toolchain.
- **Supply-chain gate** - `deny.toml` and a `cargo deny check` CI job covering
  advisories, licences, duplicates, and registry provenance.
- **`data-hub` facade feature** for the Hugging Face Hub client, and
  `download`/`hub` features on `incin-data`.
- **`STATE_FORMAT_VERSION`** - an explicit schema version on both state
  formats. Safetensors carries it as `incin.format.version` metadata; postcard
  carries it as the first field of its envelope, ahead of the payload, so a
  version mismatch is reported as one rather than as a decode failure partway
  through. A file newer than the reader is refused with both numbers named.
  `CHECKPOINT_MANIFEST_VERSION` does the same for the sharded-checkpoint
  manifest, whose `version` field existed but was never read back.
- **Core Stabilization & Migration Guide (`REL-001`):** Completed comprehensive core stabilization review and added `docs/MIGRATION.md` detailing API migration pathways across backend storage decoupling (`EXE-006`..`EXE-009`), unified autograd graph engine (`GRD-001`..`GRD-006`), proof-carrying shape safety (`SHP-001`..`SHP-008`), and distributed placement proofs (`DST-001`..`DST-005`). `docs/MIGRATION.md` section 7 added for the compiled-graph subsystem.
- **Preview compiled graph tooling (`CMP-001`..`CMP-006`):** The `compiled` feature provides captured-plan inspection, guards, liveness analysis, and a descriptor-backed CPU reference evaluator only through `experimental::compiled`. Folding, prepacking, tuning, and fusion remain fail-closed where no executable semantics exist; no optimization claim is made.
- **Preview compiled-plan snapshots (`CMP-006`, `incin-core::compiled::artifact`):** `CompiledArtifact` wraps a `CompiledPlan` with an `ArtifactHeader`, caller-supplied compatibility metadata, and an Adler-32 integrity checksum. It is a local preview snapshot, not a deployment format or portable ABI; loading compares the requested compatibility values rather than the running framework version.
- **Distributed placement proofs (`DST-003`, `incin-core`'s `distributed`
  feature):** `Replicated`, `Sharded`, `Partial`, and `PipelineStage` extend
  the existing `Local` placement typestate, with `PlacementKind` as their
  runtime projection. `ShardDivisible` proves an exact typenum quotient through
  a zero `Rem`; dynamic extents use the same rule through `validate_shard`.
  `LegalTransition` admits only identity, local shard, all-gather, all-reduce,
  and reduce-scatter, while `CompletePlacement` prevents an unreduced
  `Partial` from reaching an ordinary consumer. `PlacementTransitionRule`
  validates typed global shape, descriptor output, input placements, and every
  local shape against mesh-derived world, tensor, and pipeline degrees before
  minting the private-field, private-constructor
  `ValidatedDistributed`. Physical mesh identity remains supplied by
  `DeviceMesh`, and executable collective ordering remains `DST-007`.
- **Typed logical device meshes (`DST-001`, `incin-core`'s `distributed`
  feature):** `incin_core::dist::mesh` adds `MeshSpec<Data<DP>,
  TensorParallel<TP>, Pipeline<PP>>` and `ValidMesh`, the compile-time half of
  `PROPOSALS.md` §3.8. A mesh holds no `DeviceId` - the claim is logical device
  selection, never hardware existence - so `ValidMesh` proves only that the
  degrees are nonzero and that `DP × TP × PP` is countable, over the same
  `typenum` `Mul` the shape rules use. `World` is an associated type so a
  caller can write `M: ValidMesh<World = U3>`, which is how §3.8's "`DP=3`,
  `TP=3`, or `PP=3` are valid for three GPUs and a rectangular `2 × 2` is not"
  becomes a compile error. The axes are positional and each position accepts
  only its own marker, because swapping tensor and pipeline keeps the world
  size and changes the meaning. Omitted axes default to one. `DeviceMesh::bind`,
  the topology fingerprint, and the runtime guards are `DST-002`.
- **Automatic `Trainer` (`UX-001`, `train` feature):** `incin::train` builds
  `PROPOSALS.md` §2's level-1 workflow - pick devices, get a validated plan,
  train. The load-bearing property is a refusal: an unsatisfiable device request
  is an error, never a CPU fallback, and `NotCompiledIn` (fix your
  `Cargo.toml`) is a separate variant from `DeviceUnavailable` (fix your
  machine). `DeviceSet` and `DevicePreference` join
  `incin_core::tensor::device`; they are separate types so that "I asked for
  CUDA and got CPU" is something the API can refuse rather than express.
  `DevicePreference::Fastest` may resolve to the CPU - that is what asking for
  it means - and records every family it skipped. Availability is answered
  through a `Machine` trait, so a three-GPU plan is testable on a runner with
  none. Multi-device `fit` is an explicit `CollectivesUnavailable` naming
  `DST-005` rather than a quiet single-GPU run.
- **Generated capability and feature documentation (`UX-013`):**
  `docs/capabilities.md` is rendered from `CPU_CAPABILITIES`,
  `CUDA_CAPABILITIES` and `WGPU_CAPABILITIES` by
  `incin_backends::capability_docs`, and `README.md`'s two feature tables are
  rendered from the Cargo manifests by `cargo xtask docs` - including the
  `Purpose` column, which comes from the `#` comment above each feature in the
  manifest. Both have a check that runs in CI (`cargo xtask docs --check` and
  the `generated_docs` suite), because a generator nobody runs is a handwritten
  table with extra steps. `DTypeId::name`, `DeviceKind::name` and
  `ImplementationKind::name` give the enums one spelling each, so the tables,
  the conformance suite and `cargo incin doctor` cannot disagree about what to
  call `f32`.
- **External-backend SDK and conformance suite (`EXE-010`):**
  `incin_backends::external::conformance` is the backend-authoring surface from
  `PROPOSALS.md` §2.9 - a `Subject` trait carrying the three things only an
  author can supply, `Tolerance` profiles, and eight checks identical for every
  backend. Every check consults the capability registry first, so an operation
  a backend does not claim is *skipped*, not failed. `external` is no longer
  gated on `external-candle`: authoring a backend no longer requires enabling
  the Candle adapter. `crates/incin-backends/tests/conformance.rs` carries a
  complete minimal template backend to copy, four deliberately broken ones that
  each fail exactly one check, and the Candle adapter passing all eight.
- **`cargo incin doctor` (`UX-014`):** one command reporting toolchain and
  crate versions, enabled Cargo features, the CPU ISA extensions the kernels
  branch on, each backend family's compiled-in and available state, cache paths
  and writeability, and capability probes for eight representative operations
  on every device that answered. Stable `key: value` text by default, `--json`
  for CI and support reports with a `schema_version`. Findings carry stable
  codes - `no-backend-compiled`, `backend-unavailable`, `cache-not-writable`,
  `deprecated-feature`, `toolchain-unknown`, `isa-unavailable` - and only the
  first exits non-zero. The report is `incin::doctor`, a library module, so it
  is testable; every observation goes behind a `Host` trait, so a three-GPU
  machine can be put in front of it on a runner with none. The command is
  read-only: writeability is read from mode bits rather than probed by writing.
- **Macro test suite (`CI-005`):** `crates/incin-macros/tests/` now carries the
  compile-pass, compile-fail, hygiene, rename, and rustfmt cases the macro
  policy in `PROPOSALS.md` requires - twelve trybuild cases plus guards that
  fail when a case stops asserting what it claims or when one of the five
  categories disappears. `cargo test -p incin-macros` previously ran nothing.
- **Structured backward failures and `NanPolicy` (`GRD-005`):** backward
  recipes return `Result` - the 115 `.expect("unbroadcast lhs (add)")` and
  `.unwrap()` sites inside them propagate now - and a failure arrives as
  `BackwardError`, naming the tensor and whether the non-finite value came from
  a recipe or from summing two contributions. NaN checking is an
  `ExecutionPolicy` axis (`incin_core::exec::check_gradients(|| ..)`), read by
  every backend's walk, and defaults to off because the check reads every
  element of every gradient. The CUDA backend had no check at all before this.
- **Backend-neutral autograd tape (`GRD-003`):** `incin_core::exec::tape` now
  owns the graph `PROPOSALS.md` §1.2.5 puts in the core - one `TensorId`, a
  `TapeNode` holding a node's inputs and backward recipe, a `Tape` owning the
  nodes, and the reverse walk that consumes them. `TapeStorage` is the whole of
  what a backend still supplies: identity, a ones seed, a fallible accumulate,
  and a non-finite predicate. The CPU backend runs on it; `GRD-004` moves WGPU
  and CUDA. The walk takes its nodes by value, so a backward recipe that itself
  records - every convolution backward does - cannot re-enter the tape it is
  draining.

- **`GradMode` and `no_grad` (`GRD-002`):** the type-level `Grad`/`NoGrad`
  markers now reach the layer that records. `GradMode` joins the other axes on
  `ExecutionPolicy`, is derived from `RequiresGrad::requires_grad` rather than
  declared beside it, and travels to the backends through the ambient policy
  `GRD-001` already installs. Every frontend operation runs its kernel under
  the mode its *result*'s marker derives, and the CPU, WGPU, and CUDA tapes
  refuse a push when that mode does not record - so a `NoGrad` chain creates no
  autograd node and retains no saved tensor, as `PROPOSALS.md` §1.2.5 requires.
  `incin_core::exec::no_grad(|| ..)` is the inference form and applies to
  `Grad` tensors too; an operand can only tighten the ambient mode, never raise
  it. `cpu::tape_depth()` (likewise `wgpu`, `cuda`) is newly public so the
  guarantee can be counted rather than assumed.
- **Dtype/kernel specialization architecture:** new `dtype_policy.rs` (single
  storage/compute/accumulator/output dtype resolver for CPU/CUDA/WGPU),
  `iteration.rs` (backend-neutral broadcast/layout iteration plan), and
  `kernel.rs` (typed CUDA source generation shared across pointwise,
  reduction, and normalization operation families). See `PROPOSALS.md` §3 for the consolidated design and phased roadmap.
- **CUDA autotuning foundation** (new `autotune` feature, `tuning.rs`): typed
  canonical launch-candidate keys, CUDA-event warmup/sample measurement,
  compute-capability-scoped caching, a Condvar-coordinated in-flight
  suppression claim so concurrent callers tuning the same problem/device/
  workload key block on the in-progress measurement instead of redundantly
  benchmarking it, and Tier-2 occupancy pruning for pointwise candidates
  (`cuOccupancyMaxActiveBlocksPerMultiprocessor`, conservative - only drops a
  candidate the driver confirms has zero active blocks). CUDA reductions and
  layer/batch norm are now generated from the dtype policy (replacing the
  checked-in F32-only `norm.cu`/`reduce.cu`) with warp/block cooperation and
  Welford accumulation; CUDA pointwise dispatch adds scalar-ILP and aligned
  packed (`half2`/`bfloat162`/`float4`/`double2`) access candidates. All CUDA
  work is compile/clippy-verified only - no CUDA hardware available in CI or
  local development at time of writing.
- **WGPU autograd, essentially complete:** `layer_norm`, `batch_norm`,
  `adaptive_avg_pool2d`/`avg_pool2d`/`max_pool2d`, `max_dim`/`min_dim`/
  `max_all`/`min_all`/`max_keepdim`/`min_keepdim`, and `cross_entropy_loss`
  are now gradient-correct on the WGPU backend, verified against a real
  software WGPU adapter (not just compile-checked) via finite-difference
  gradcheck tests. Pooling and the max/min-family reductions needed genuine
  new backward code (host-readback + recomputed-argmax/window scatter,
  mirroring the CPU backend's proven `cpu/ops/pool.rs`/`cpu/ops/reduce.rs`
  algorithms); `layer_norm`/`batch_norm`/`softmax`/`cross_entropy_loss` turned
  out to already be gradient-correct by composition from already-wired
  primitives and only needed verification. WGPU autograd coverage now
  matches CPU's, except `quantize`/`dequantize`/`quantized_matmul` (not wired
  on CPU either - not a WGPU-specific gap).
- **Cross-backend gradient parity:** extended `tests/gradient_parity.rs` with
  `max_pool2d` and `cross_entropy_loss` (non-zero target class) CPU-vs-WGPU
  checks, the permanent regression class this file exists to catch.

### Changed

- **`maximum` and `minimum` now propagate NaN.** They were implemented on
  Rust's `f32::max` and `f32::min`, which return the non-NaN operand and so
  swallowed NaN entirely. The operation profile's `IeeePropagate` rule requires
  a NaN operand to produce NaN, which is also what other array frameworks do,
  so a comparison against one of them will now agree where it previously
  diverged. Code that relied on the old NaN-swallowing behaviour to clean data
  must filter explicitly.

- **`scatter` defines its gradient as last-write-wins.** When two indices write
  the same output position, only the surviving write receives a cotangent and
  the overwritten one receives nothing.

- **Group and instance norm use the two-pass deviation form** for their
  statistics, which is more numerically stable and removed the need for a
  clamp.


- **Loss constructors no longer need a reduction turbofish.** `MSELoss`,
  `CrossEntropyLoss`, `L1Loss`, and `BCEWithLogitsLoss` each declare
  `R: ReductionMode = Mean`, but a type-parameter default does not drive
  inference for an associated function, so `MSELoss::new()` failed with `E0283`
  and every call site had to spell out `MSELoss::<Mean>::new()`. `new()` is now
  defined on the `Mean` instantiation only, so it resolves on its own and reads
  like `torch.nn.MSELoss()`. A non-default reduction is constructed with
  `MSELoss::<Sum>::with_reduction()`, which replaces the generic `new()`.

- **The CPU AVX2 kernels are reachable.** They were gated on
  `simd_lanes::<f32>() >= 8`, which reads `cfg!(target_feature = "avx2")` - false
  in any stock `cargo build`, since the default `x86_64` target is the baseline
  ISA. Every default build therefore dead-code-eliminated the SIMD path and ran
  a scalar loop. The kernels already carried
  `#[target_feature(enable = "avx2")]`, so the fix is the runtime detection that
  attribute exists for: `simd::avx2_detected()`, cached in a relaxed atomic.
  On an AVX2 machine, `eager/add_f32/65536` goes from 60.7 µs to **6.43 µs**,
  back inside its recorded budget, and `add_f32/1024` from 1.66 µs to 776 ns.
  A test now fails if the gate is narrowed back to a compile-time condition.
- **Whole-tensor reductions read contiguous storage directly.** `sum_all`,
  `mean_all`, `prod_all`, `max_all`, and `min_all` walked a logical odometer and
  fetched every element through a stride dot product plus a dtype match - about
  twenty cycles to read one number. `sum_f32/1024` goes from 6.97 µs to 5.41 µs
  under the full benchmark suite and from 6.97 µs to 1.32 µs in isolation;
  `sum_f32/65536` from 399 µs to 306 µs. The `f64` accumulator and the traversal
  order are unchanged, so reduced values are bit-identical.
- **Fewer allocations per operation.** The descriptor path inferred each output
  shape twice, once to derive it and once to verify the derivation against
  itself; `broadcast_shape` round-tripped its accumulator through a `ShapeBuf`
  on every operand, and the CPU backend built autograd tape entries even when the
  effective `GradMode` was going to discard them. A rank-2 elementwise add went
  from 27 allocations to 5 and a unary from 20 to 5, and the ceilings in
  `hot_path_allocations.rs` were rebased onto the new counts. Shape inference
  also produces a `ShapeBuf` throughout rather than a `Vec<usize>` that was then
  copied into one; `ShapeBuf` stores rank 8 and below inline, so an ordinary
  rank now reaches the descriptor without touching the heap. No validation was
  removed. See `docs/benchmarks/runtime-2026-08-17.md`.
- **`ModelExt::load` no longer takes a device.** The argument was ignored, so
  the signature described a relocation the call never performed. `load`
  restores state in place; moving a model between devices stays `ToDevice`.
- **An optimizer step that reaches no parameter is an error.** `SGD`, `Adam`,
  and `AdamW` returned `Ok(())` when no parameter in the group received a
  gradient, so a training loop could run to completion with parameters that
  never moved - including when `backward` ran on a different thread from the
  forward pass, since a tape is thread-local. Skipping some parameters remains
  legal.
- **`docs/plan/roadmap.md` derives its completion table from
  `docs/plan/ledger.toml`.** The two disagreed about the same task IDs; a new
  check keeps them equal, and the roadmap now states what "complete" means and
  how many rows carry recorded deviations.
- **README and crate documentation** describe CPU as the complete backend and
  the GPU backends as previews covering documented subsets, matching the
  generated capability matrix.
- **Dependencies:** `rand` 0.8 → 0.10, `rand_distr` 0.4 → 0.6, `hashbrown`
  0.14 → 0.17, `spin` 0.9 → 0.12, `safetensors` 0.4 → 0.8, `pollster` 0.3 →
  1.0, `criterion` 0.5 → 0.8.
- **Advanced indexing facade:** Curated `incin::advanced` to export only the
  documented type-level indexing selectors and traits.
- **Core advanced indexing facade:** Applied the same explicit export boundary
  to `incin_core::advanced`, keeping hidden implementation traits out of the
  downstream namespace.
- **Public API guard:** Stable `incin` and `incin-core` facade files now fail
  validation if a wildcard re-export is reintroduced.
- **Shape root exports:** Replaced wildcard exports from private shape
  implementation modules with explicit scalar, storage, proof, and dimension
  items.
- **Telemetry graph test:** Updated the graph snapshot test to use the named
  `incin_core::graph` namespace after removing `Graph` from the ordinary
  prelude.
- **Backend-authoring tests:** Updated test consumers to import `Backend` and
  `VariableBackend` from the explicit authoring namespace.
- **Transformer CPU test:** Made the gradient assertion deterministic by
  allowing valid zero-valued individual components while requiring a nonzero
  gradient somewhere in the model.
- **Macro fixtures:** Updated compile-pass macro fixtures to use the explicit
  backend-authoring namespace after the stable root API was narrowed.
- **Advanced facade:** Removed internal reshape and slice specification traits
  from the public advanced namespace; the user-facing selector contracts remain.
- **Core prelude:** Moved backend-authoring, tracing, and storage-encoding
  contracts to named modules, and added `incin_core::onnx` for ONNX helpers.
- **Telemetry preludes:** Replaced event-module wildcard exports with explicit
  event contracts in telemetry and visualization plugin crates.
- **Core aggregations:** Replaced wildcard exports for indexing, schedulers,
  and precision policy markers with explicit owning-module contracts.
- **Neural-network exports:** Replaced the `incin_core::nn` wildcard
  aggregations with explicit layer, optimizer, state, and statistics contracts.
- **Shape storage boundary:** Kept the internal `InlineOrHeap` representation
  out of the public shape prelude and removed the unused public
  `fold_static_numel` helper.
- **CPU allocation imports:** Removed unused random and Rayon import
  suppressions from the CPU creation kernel.
- **Editor prose:** Replaced dash-heavy comments and user-facing text in the
  Neovim, VS Code, and RustRover integrations with ordinary punctuation.
- **Descriptor macro policy:** Removed an obsolete unused-macro suppression
  from the shared descriptor executor declarations.
- **Dispatch scaffolding:** Removed the unused multi-operand dispatch macro;
  all live routes use the module-specific helper or explicit routing path.
- **Unsupported-operation scaffolding:** Removed unused creation, reduction,
  and tensor-operation declaration macros; float-operation declarations remain
  because CUDA, Metal, WGPU, and Candle still use them.
- **Target feature gating:** Compiled the non-CPU target implementation macro
  only when one of its target backends is enabled, removing its unused-macro
  suppression in CPU-only builds.
- **Capability macro exports:** Feature-gated backend-specific capability macro
  re-exports so CPU-only builds no longer need unused-import suppressions.
- **Rust toolchain reproducibility:** Pinned the supported compiler and stable
  CI, hardware, and release jobs to Rust 1.97.1 to keep diagnostics and builds
  repeatable.
- **CPU test helper isolation:** The finite-difference gradient checker is now
  compiled only for CPU unit tests instead of shipping as dormant production
  code behind a module-wide dead-code allowance.
- **CPU operation test cleanup:** Removed unused backend aliases from pooling,
  convolution, and embedding tests, and removed an obsolete macro forwarding
  helper that had no callers.
- **Dispatch dead-code cleanup:** Removed four private variable-creation
  dispatch wrappers that were never called; the execution registry remains the
  active path for variable creation operations.
- **Hidden API inventory:** Refreshed the reviewed source locations for the
  descriptor transform, paranoid-validation, and macro-support hidden items so
  the mechanical inventory check matches the current source.
- **Dummy backend scope:** The shape-only dummy backend is now compiled only
  for unit tests or the explicit `test-utils` feature, matching its documented
  role and keeping its test-support suppressions out of normal core builds.
- **Backend documentation:** Repaired stale placeholder references in the
  unsupported-operation macro documentation so each explanation names the
  operation family it describes.
- **CI package gate:** The ledger job now validates locked Cargo metadata and
  every publishable package archive, catching omitted sources, binaries, and
  license metadata before release packaging.
- **Core rustdoc links:** Removed invalid `GradMode` scope links and clarified
  the no-`std` policy-scope wording so core rustdoc passes with warnings denied.
- **Dummy backend dead code:** Removed an unused family of private float
  operation shims and all stale dead-code allowances from the test backend.
- **Dynamic marker scope:** Restricted the private `Dyn::marker` test helper to
  unit-test builds, removing the last production dead-code allowance in core.
- **Compile-fail diagnostics:** Updated the `Dyn` privacy regression snapshot
  to reflect that the test-only marker helper is no longer suggested to users.
- **Metal tuning isolation:** Metal benchmark winner selection and cache-claim
  helpers are now test-only, with production builds retaining only the
  candidate conversion and fallback policy they use.
- **Facade API tiers:** Removed backend-authoring traits from the stable
  `incin` root and default prelude, and removed `Graph` from the core prelude.
  The data prelude now also uses an explicit allow-list. The supported
  migration paths are recorded in `docs/MIGRATION.md`; backend contracts
  remain under the explicit `backend-authoring` feature.
- **Editor documentation prose:** Replaced em-dash-heavy phrasing in the
  VS Code, Neovim, and RustRover integration READMEs with ordinary punctuation
  so current user-facing documentation follows the repository prose style.
- **Candle adapter cleanup:** Removed unused unsupported-operation stubs and
  quantization placeholders from the legacy inherent surface. Unsupported
  capabilities remain represented by the descriptor capability registry.
- **`must_use` signal:** Removed 40 redundant method and function annotations
  whose return types were already `Option` or `Result`, while retaining
  annotations on builders, constructors, and semantically important values.
- **CUDA lint and structure:** Grouped internal two-dimensional column-to-image
  parameters into `Col2Im2dSpec`, kept the shared transposed-convolution
  backend contract explicit, and moved CUDA backend trait implementations
  before the test module so the CUDA all-targets lint gate passes with
  warnings denied.
- **Dead-code audit:** Removed unused raw conversion and complement helpers
  from the private axis-mask implementation, and removed a redundant CUDA
  identity suppression while retaining feature-gated test and dummy-backend
  helpers.
- **Rustdoc coverage:** Documented the public plan-report exit status constants
  so trainer builds remain warning-free under the facade documentation lint.
- **Architecture and build hygiene:** The shape buffer helpers remain available
  through the documented `incin_core::shapes` facade while their implementation
  modules are private. Unreferenced WGPU dispatch paths and CUDA kernel sources
  were removed, and backend layout and quantized-storage modules are now gated
  by the features that use them. WGPU lifetime owners and CUDA tuning helpers
  no longer rely on broad dead-code allowances. The book CI job installs
  Chromium in the job that runs the browser checks.
- **Feature isolation:** Distributed context imports and protocol decoding are
  now gated with `std`, while compiled distributed plans retain their
  no-std-compatible ownership imports. The supported `compiled,distributed`
  and `distributed` feature contracts both compile cleanly.
- **ONNX export surface:** Removed the unreferenced captured-graph export
  helper from the private exporter module; the reviewed eager-graph exporter
  remains the supported ONNX path.
- **Release packaging:** The editor release job now uses pinned Node.js and
  VS Code packaging-tool versions, and names the IntelliJ-platform archive
  independently from the RustRover integration directory. Release assets
  include the book, editor integrations, `incin-lsp`, and `cargo-incin`; the
  VS Code manifest now identifies the repository for package consumers.
- **Rustdoc coverage:** The public `incin` facade now enables local
  `missing_docs` warnings, and CI runs a warning-free facade-only rustdoc gate
  in addition to the workspace link and warning check.
- **Tensor byte views:** `Tensor::from_slice` now uses the `bytemuck` checked
  byte-slice conversion already guaranteed by `TensorElement`, removing a raw
  pointer reinterpretation from the core tensor boundary.
- `incin_backends::{cpu,wgpu,cuda}::storage::TensorId` are re-exports of
  `incin_core::exec::TensorId`; three independent identity counters became one.

### Fixed

- **Operations that advertised a gradient but recorded none.** The operation
  catalog publishes a `GradientRule` per operation and the capability
  registrations publish a per-operation training flag. Twenty-two operations
  across six families carried a `Defined` or `Piecewise` rule with training
  set while their CPU kernels were forward-only walks that pushed no autograd
  tape entry, so a backward pass through any of them stopped silently: the
  input was treated as a leaf and optimizers saw no gradient. This affected the
  scalar and pointwise binary family, the selection and indexing family
  (`masked_fill`, `index_select`, `scatter`), the Shape family, `cumsum`,
  `prod_all` and `prod_dim`, group and instance norm, training-mode batch norm,
  and the unary float operations including `powf` and `clamp`. All of them now
  record backwards, and `docs/capabilities.md` is truthful for those rows as a
  result. Covered by exact-value backward tests and finite-difference
  gradchecks, including through non-contiguous operands.

- **`lerp` rejected broadcast operands** it should have accepted.

- **The CUDA and Metal arms of the addition benchmark had never compiled.**
  Both called `.unwrap()` on `&input + &input`, which returns a tensor rather
  than a result. They are lint-clean under `--all-features` now, along with
  four `unsafe` blocks in the `incin-data` hub tests that carried no `SAFETY:`
  comment.


- **Every gradcheck uses an f32-appropriate finite-difference step, and
  catches gradient errors 25x smaller.** All 50 gradcheck call sites in
  `incin-backends` passed `eps=1e-4`, about a hundredth of the step that
  minimizes total error for f32 storage (`(6 * f32::EPSILON).cbrt()`, or
  roughly 9e-3). Because the rounding term grows as `1/eps`, that
  inflated the finite-difference noise floor by the same factor:
  measured worst-case error on correct gradients was 1.3e-2, essentially
  equal to the 1e-2 relative ceiling the assertions used, so those checks
  could not separate a small real defect from a rounding artifact. This
  is also what made the aarch64 Hardware Matrix failure (0.0145 observed
  on Apple Silicon, against x86_64's 0.00064 on the same test) look like
  a gradient bug rather than the step-size artifact it was. The step is
  now a shared `F32_STEP`, which drops the measured worst case across the
  crate to 1.0e-4; the relative ceiling becomes a shared `GRAD_TOL` of
  1e-3, clearing real noise by 10x instead of sitting on top of it; and
  the absolute noise floor that guards true-zero gradients falls from
  1e-3 to 5e-5. Measured by injecting a uniform scaling error into every
  analytic gradient, the suite previously needed a 5% error before all 38
  affected tests failed and caught only 3 at 0.1%; it now saturates at
  0.2% and catches 23 at 0.1%. A new hand-computed batched backward test
  additionally pins the equal-batch analytic gradient by exact
  arithmetic, which the existing hand-computed test did only for the
  unbatched case.
- **The deep dive is served inside the book.** README's "where to go next"
  linked the deep-dive chapters and "What's not finished" as raw GitHub
  markdown files; they now route to the rendered site pages, the
  introduction points at the part from inside the book, and all five
  chapters gained theme-aware diagrams (layer stack, execution route,
  proof lattice, dispatcher stages with their error taxonomy, the five
  proof stages, and macro hygiene flow) that follow both mdBook's themes
  and the Pages site's.
- **Candle 0.9.2 element types.** New `I16`, `I32`, and float8 variants are
  refused by name at the bridge rather than breaking the build or being
  reinterpreted as a same-width type Incin does have.
- **Every public example is compiled (`UX-013`).** 70 of the workspace's 79 doc
  examples were fenced ```` ```rust,ignore ````, so `cargo test --workspace
  --doc` reported success having compiled nine - and CI never ran it at all,
  because `cargo test --all-targets` excludes doctests. Compiling them found
  the examples documenting an API that does not exist: `from_slice` shown with
  one argument where it takes two (fifteen examples), `Param<Tensor<S, B>>`
  where the type is `Param<S, B>`, a rank-1 reshape argument written `()` where
  it is `((),)`, `dims()` compared against a `Vec` where a static shape returns
  an array, `incin::symbolic_dim!` which does not resolve (`dim!` is the public
  macro; `symbolic_dim!` is a `#[doc(hidden)]` alias the facade does not
  re-export), and an `incin-data` front page built on a `DataLoader` builder API
  that was never written. A test now fails on any reintroduced `ignore` fence.
- **`IndexSpec`, `LSTM` and `LSTMCell` are reachable from the prelude
  (`UX-013`).** `Tensor::slice` takes `&[IndexSpec]` and `IndexSpec` was not
  exported, so the documented call could not be written by a user of the
  prelude; `RNN` and `RNNCell` were exported while `LSTM` and `LSTMCell` were
  not.
- **`DummyBackend`'s binary operations broadcast (`UX-013`).** They returned the
  left operand's shape unchanged, which disagrees with every real backend:
  `broadcast_add` and its siblings reach `Backend::add` with differently shaped
  operands and hand the result to `Tensor::from_parts` against the *broadcast*
  type. `incin_core::shapes::broadcast::broadcast_dim_slices` is the one
  right-aligned rule both paths now use.
- **`--features external-candle` failed `clippy -D warnings` (`EXE-010`):** the
  `bytes` module was gated on `external-candle` alongside `cuda` and `wgpu`,
  but the Candle adapter never allocates by byte length, so that feature set
  compiled a module whose only function was dead.
- **WGPU device detection crashed when probed more than once (`UX-014`):**
  `incin_backends::detect::probe` built a fresh `wgpu::Instance` per call and
  dropped it, and two threads each probing twice segfaulted inside adapter
  enumeration. The instance is shared for the process lifetime now, matching
  what the WGPU backend already did; detection is still performed per call.
- **Macro hygiene (`CI-005`):** `s!`, `idx!`, `#[module]`, `model!`, and
  `import_model!` expanded to a relative `incin::prelude::…`, so any caller
  item named `incin` captured the expansion. All five emit absolute `::incin`
  paths now; use `s![@ ..]` inside the workspace, which expands to
  `crate::prelude::…`. A package rename in a caller's `Cargo.toml` remains
  unsupported and is documented on each macro.
- **`#[module]` argument validation (`CI-005`):** struct-level arguments were
  matched as substrings, so `#[module(no_such_argument)]` was silently accepted
  as `#[module]` and `#[module(not_internal)]` as `#[module(internal)]`. The
  list is parsed against a closed vocabulary and unknown keys are rejected.
- **Feature isolation and naming:** a bare install now enables only `std` and `cpu`; CUDA, WGPU, Candle, autotuning, and telemetry are explicit opt-ins. The third-party Candle adapter moved from `legacy::candle` to `external::candle`, and accelerator-only builds no longer reference CPU-only dispatch variants. Candle dtype conversion now returns an error instead of panicking on unsupported types.
- **C-10:** `Tensor::to_scalar<E>`/`to_vec1<E>` could construct an invalid `bool`
  (Miri-confirmed undefined behavior) when reading non-0/1 byte values from
  storage. Fixed by special-casing `bool` `TypeId` checks and enforcing a
  safe non-zero element truthiness check without unsound transmutes.
- **C-9:** WGPU `embedding`'s backward and `cross_entropy_loss`'s one-hot
  construction bit-reinterpreted F32-stored index/target bytes as `u32`
  (`buffer.to_vec::<u32>()`) instead of converting the value, silently
  corrupting every gradient/loss contribution for any non-zero class or
  vocab index (only index `0.0` happened to survive, since its IEEE bit
  pattern is `0x00000000`). Existing tests never caught this - both only
  exercised index/class `0`. Fixed to read `to_vec::<f32>()` and convert,
  matching the WGSL forward kernel's own `u32(indices[i])` value conversion.
- Pre-existing (not introduced this cycle) `cargo clippy --features cuda,std`
  and `--features wgpu,std` failures on `main`, found while auditing the
  above: mismatched feature gates in `backend_kind.rs`'s test module,
  `cpu/creation.rs`'s `TransferTo<Cpu>` test, and `tests/ops.rs`/
  `tests/gradient_parity.rs` assuming `cpu` was always enabled alongside
  `cuda`/`wgpu`.

---

## Development snapshot - 2026-07-22

### Changed

- Tensor device metadata is derived exclusively from the backend; `Tensor` now has the
  four parameters `Tensor<S, B, K, G>`.
- Runtime metadata is named `DTypeId`, `DeviceId`, and `DeviceKind`; GPU families
  remain representable even when their feature is disabled.
- Tensor allocation uses one `zeros`, `ones`, `rand`, or `randn` entry point for
  static and dynamic metadata. Allocating layers expose `build`.
- `from_slice` accepts the element type associated with its static dtype.
- `IncinBackend<T, D>` is the only concrete backend spelling exported by the
  public prelude; the former CPU, WGPU, and CUDA backend type names were removed. Device changes now use `TransferTo`, rebuilding destination-native storage through checked, dtype-aware host staging.

### Fixed

- CUDA-only builds can access the shared layout and quantized-storage helpers.
- CPU dynamic dtype allocation preserves the physical buffer variant and floating random
  initialization supports F32, F64, F16, and BF16.
- WGPU creation rejects non-F32 dtypes, wrong device families, invalid ordinals, and
  malformed byte payloads with typed errors.
- Runtime dispatch preserves physical dtype/device metadata, delegates reductions,
  and performs dynamic device transfers through dtype-aware host staging.

### Added
- **WGPU Autograd:** Implemented backward passes for `gelu`, `elu`, and `mish`
  activations in `WgpuBackendImpl`, including WGSL gradient kernels (`gelu_grad`,
  `elu_grad`, `mish_grad`) and tape entries in the autograd system.
- **Cross-Backend Parity Tests:** New `crates/incin-backends/tests/gradient_parity.rs`
  test suite verifies numeric agreement (≤ 1e-4) between `CpuBackendImpl` and
  `WgpuBackendImpl` for elementwise add, matmul, layer_norm, softmax, and
  cross_entropy_loss forward+backward passes.
- **`DTypeId::element_size()`:** New method returns the byte width of each
  dtype, used by safety checks in `to_scalar`/`to_vec1`.
- **Activation `ToDevice` impls:** Stateless activation modules (`ReLU`, `GELU`,
  `Swish`, `Mish`, `ELU`, `Softmax`, `Sigmoid`, `Tanh`) now implement
  `ToDevice<B, NewD>`, enabling their use as fields in `#[module]`-derived
  structs that call `to_device`.
- **Docs:** All 2,541 filler doc comments (`/// Core abstraction for \`X\`…`)
  replaced with real one-line descriptions across the entire workspace
  (`incin-core`, `incin-backends`, `incin-data`, `incin-macros`,
  `incin-telemetry`, `incin-viz`, `incin-viz-plugin-api`, test and
  example crates).
- **Real Doctests:** `s![]`, `idx![]`, and `#[module]` macro doc examples in
  `incin-macros/src/lib.rs` are compiled doctests (not `ignore`) and pass
  `cargo test --doc -p incin-macros`.

### Fixed
- **Safety:** `to_scalar` and `to_vec1` now validate the raw byte slice length
  against `DTypeId::element_size()` before interpreting bytes, preventing
  potential undefined behaviour on malformed storage.
- **Error Handling:** Replaced `panic!`/`unimplemented!` calls in `serialize.rs`
  (Q8_0 quantization path), `onnx_exporter.rs` (Q8_0 ONNX export), and
  `shapes/idx.rs` (multiple inferred dims) with clean `Result::Err` returns.
- **Security:** `FileTransport::open` now sets Unix file permissions to `0o600`
  (owner read/write only) on newly created telemetry log files.
- **Test Isolation:** All integration tests in `crates/incin/tests/` now
  explicitly target `CpuBackendImpl<f32, Cpu>` rather than `DefaultBackend`,
  preventing failures when `--features cuda` is active on CPU-only CI hosts.
- **CPU Feature Gate (C-8):** `cpu::ops::elementwise` components were previously
  gated under the `cuda` feature flag rather than `cpu`; corrected.

### Changed
- **`DefaultBackend`:** Always resolves to `CpuBackendImpl<f32, Cpu>` regardless of
  active GPU feature flags, ensuring a safe default on non-GPU hosts.

---

## Development snapshot - Backend Refactoring Sprint

### Changed
- **Backend Crates:** Moved `native`, `wgpu`, and `cuda` backends into their own
  distinct crate (`incin-backends`), standardizing trait bounds
  (`NumericOps`, `ModuleOps`, `ReductionOps`, etc.) across devices.
- **WGPU Migration:** Transitioned core components, backends, and app libraries
  from Metal to WGPU for unified cross-platform execution.
- **External Adapter Cleanup:** Deleted obsolete, dead-code `ndarray` and `burn`
  compatibility wrappers.

### Added
- Complete WGPU convolution implementations and telemetry tracking features.
