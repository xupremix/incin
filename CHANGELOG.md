# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased
### Added
- Core tensor operations and neural network layers.
- Native backend (CPU/GPU) with CUDA and Metal support.
- Wgpu backend for cross-platform WebGPU acceleration.
- Parity integration tests between Native and Wgpu backends.
- Autograd tape mechanism for automatic differentiation.

### Fixed
- GPUSync race conditions in `test_adamw_step`.
- DummyBackend missing accurate shape calculus in convolution and pooling operations.
- Backend FloatElem generic consistency issues for legacy backends.
- Leaked internal API items hidden behind `pub(crate)` properly.
