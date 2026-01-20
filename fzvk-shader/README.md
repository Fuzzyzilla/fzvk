# `fzvk-shader`
(placeholder name)

Macros for importing and generating SPIR-V with some type-level information
encoded using `fzvk` types and traits.

Provides `spirv!` and `glsl!` macros, which import or compile SPIR-V into a
`&'static [u32]` bytecode slice, as well as analyzing the resulting SPIR-V
binary to create several strongly-typed descriptions of the interfaces of the
module:
 - [x] An implementation of `fzvk::pipeline::StaticSpirV` defining the module's
specialization constants.
 - [x] Safe-to-use structures representing all entry points and their execution
model/shader stage (no more unsafe `c"main"` all over your pipelines!).
 - [x] Structures describing the descriptor sets, including required image types
  of bindings.
 - [ ] Structures defining vertex attribute layout.
 - [ ] Structures defining shader inputs and outputs.
 - [ ] Structures defining Push constant layout.

See the main `fzvk` crate examples for usage and interactions with the `fzvk`
crate.

## Currently outside of scope
* Dynamic or
  [static](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-module-validation-standalone)
  module validation.
    * If this is what you want, check out
      [vulkano](https://github.com/vulkano-rs/vulkano) which provides a similar
      shader macro with extremely thorough runtime validation.