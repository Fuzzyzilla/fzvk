# Fuzzy's Funny Zero-cost Excessively-Typestate Vulkan Abstraction
A thin wrapper around [`ash`](https://github.com/ash-rs/ash) providing
compile-time typestate validations.

## Typestate
This crate makes extensive (perhaps excessive) use of the [typestate
pattern](https://en.wikipedia.org/wiki/Typestate_analysis) to validate *some*
operations. Of course, the scope of what can be declared at compile time is
inherently limited by the need for runtime flexibility, so it is mostly used in
locations where it is a common usage pattern that the "shape" of the data will
not be totally dynamic.

For example, an `Image` contains the following compile-time information:
* Its dimensionality (1D, 2D, 3D)
* Its array-ness
* Its format and aspects.
* Its vkImageUsageFlags
* Whether multisampling is in use.

As such, a fully-named image type looks something like:
```rust
let image : Image<(Storage, TransferDst), D2Array, R8G8B8_SRGB, SingleSampled> = todo!();
```
Since each of these facts are almost always known at compile time and have large
implications for valid usage of the resource, they are tracked at compile time
too.

## Why
After [doing the same thing to OpenGL](https://github.com/Fuzzyzilla/glhf), the
thoughts of which aspects of the vulkan API could likewise be validated using
compile-time information kept coming. I threw together the start of the crate
expecting some massive unforseen roadblock to materialize, but alas - none has.

## Usage
This crate is experimental and not ready for production. The
[examples](fzvk/examples/) folder contains the fullest view of what this crate
looks like in use. The examples and the crate are developed similtaneously to
ensure the library remains pleasant to use and to inform which areas need
implementing first.