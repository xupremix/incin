# Contributing & Soundness Gates

This guide outlines our quality assurance standards, verification gates, and contribution guidelines.

---

## 1. Development Prerequisites

* **Rust Toolchain**: `1.88.0` (MSRV) or latest stable for development; nightly for Miri/Sanitizers.
* **mdBook**: `0.4.52` for building documentation.
* **Node.js**: `20.x+` for editor extensions and browser tests.

---

## 2. Verification Gates & Soundness Harness

Incin contains ~170 `unsafe` blocks in its high-performance CPU SIMD kernels. Before submitting a PR, ensure all soundness gates pass:

```bash
# 1. AddressSanitizer & LeakSanitizer (+AVX2 kernels)
bash tools/soundness.sh asan

# 2. ThreadSanitizer (parallel memory initialization races)
bash tools/soundness.sh tsan

# 3. Miri (aliasing & undefined behavior under Tree Borrows)
bash tools/soundness.sh miri

# 4. Full local CI pipeline
bash tools/ci-local.sh
```

---

## 3. Code Conventions (`CONVENTIONS.md`)

* **Line Budgets**: Source files should remain under 1,000 lines. Split large modules into dedicated submodules.
* **No Unshielded Unsafe**: All `unsafe` blocks must have an explicit `// SAFETY:` invariant comment and be registered in `docs/security/unsafe-ledger.md`.
* **Public Visibility**: Adhere to `docs/API_DESIGN.md`. Do not leak internal traits into the crate root.
* **Zero AI Attribution**: Commit messages must follow conventional commits without any `Co-Authored-By` AI trailers or machine-generated artifacts.
