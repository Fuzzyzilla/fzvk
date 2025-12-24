# `fzvk-shader`
(placeholder name)

Macros for importing and generating SPIR-V with some type-level information
encoded using `fzvk` types and traits.

Provides `spirv!` and `glsl!` macros, which import or compile SPIR-V into a
`&'static [u32]` bytecode slice along with several structures describing the
interfaces of the module, such as:
* An implementation of `fzvk::pipeline::StaticSpirV` defining it's
specialization interface.
* Safe-to-use structures representing all entry points and their execution
model/shader stage.
* Structures describing the descriptor sets, including required image properties
  of descriptor bindings.

See the examples for usage and interactions with the `fzvk` crate.

## Currently outside of scope
* Dynamic or
  [static](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-module-validation-standalone)
  module validation.
    * If this is what you want, check out
      [vulkano](https://github.com/vulkano-rs/vulkano) which provides a similar
      macro API with extremely thorough runtime validation.