# Errors

Every fallible constructor and operation in Incin returns `Result<T, Error>`
(aliased `Result<T>`) — nothing panics on ordinary bad input. `Error` is a
large, typed enum: a dimension mismatch, an out-of-range index, and an
unsupported backend operation are three different variants, not three
strings that happen to say different things.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

fn might_fail() -> Result<()> {
    let x = Tensor::<Dyn, B>::zeros(vec![2, 3])?;
    let y = x.to_shape::<s![4, 4]>()?; // wrong element count
    Ok(())
}

fn main() {
    match might_fail() {
        Ok(()) => println!("ok"),
        Err(e) => eprintln!("failed: {e}"),
    }
}
```

`?` is the idiom throughout this book and throughout real code — every
example that returns `Result<()>` is meant to be read that way, not as
boilerplate to strip out.

## Backend refusals are typed too

A backend that doesn't support an operation for a given dtype, layout, or
rank refuses it with a typed reason (`UnsupportedReason::DType`,
`::Layout`, `::Rank`, ...) rather than a generic failure. This is what makes
[the GPU coverage gaps](./backends.md) discoverable programmatically instead
of only by reading source: a refusal names exactly what was missing.

## What panics, and what doesn't

Ordinary bad input — a wrong shape, an out-of-range axis, an unsupported
dtype for an operation — is always a typed `Err`, never a panic. `unwrap()`
and `expect()` in this book's examples are there because the input is known
good at that point (a freshly allocated tensor, a literal), the same way
you'd unwrap a `Vec` index you just checked the length of. In your own code,
prefer `?` and propagate `Result` unless you have the same kind of local
certainty.
