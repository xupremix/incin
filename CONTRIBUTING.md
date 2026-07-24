# Contributing to Incin

First off, thank you for considering contributing to Incin! It's people like you that make Incin such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check if there's already an open issue on our issue tracker. If not, go ahead and open a new one!

## Development Setup

1. Clone the repository
2. Run `cargo build`
3. Run `cargo test --workspace` to ensure everything works
4. (Optional) Run `cargo check -p incin-wgpu --all-features` to verify GPU features.

## Submitting a Pull Request

- Please ensure your code passes `cargo fmt` and `cargo clippy --workspace -- -D warnings`.
- Add tests for your changes.
- Document any new `pub` items.
- Update `CHANGELOG.md` if necessary.

## Code of Conduct
Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.
