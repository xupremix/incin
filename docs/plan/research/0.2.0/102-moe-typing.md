#102 MoE routing under static shapes — M (L for GPU execution)
Finding: routing makes per-expert counts data-dependent; only the TOTAL (T*k) is provable. Precedent: dist/placement.rs splits Const vs Dyn placement. Options: capacity-factor (proves nothing true), Dyn boundary (rank-only proof), sorted buffer [T*k,D]+offsets[E+1] (proves exactly what routing guarantees).
Recommendation: static outer shapes, load as E+1 offset array; E,k const; capacity-factor only as explicit bounded-compute config; quantize() returns NoGrad by type (ties to #93); top-k gate weights differentiable-only documented.
Risk: silent token-drop if capacity defaults; aux-loss placement.
Unblocks: MoE impl, growth D2/D3.
