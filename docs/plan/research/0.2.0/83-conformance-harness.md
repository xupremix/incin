#83 CPU-oracle conformance harness — M
Finding: external/conformance.rs (649 lines) already has Tolerance profiles, Outcome, Report — but covers only 2 ops, f32, contiguous, training hardcoded false. CapabilityRule/registry enumeration + typed UnsupportedReason exist.
Recommendation: extend it: enumerate registrations() x dtype x layout x boundary-rank; CPU (158/158) oracle; advertised-but-refused = FAIL; executed-but-unadvertised = FAIL; skips distinguishable; per-dtype tolerance table with recorded exceptions; gradient comparison when rule.training; JSON artifact feeding a verified-on column in capabilities.md.
Risk: needs #82; bf16 tolerance looseness; seeding random ops.
Unblocks: close-out of #84,#86-#92,#106; #96 verification; #95 CPU reference.
