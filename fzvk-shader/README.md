# `fzvk-shader`
(placeholder name)

Macros for importing and generating SPIR-V with some type-level information.

Provides `spirv!` and `glsl!` macros, which import or compile SPIR-V into
bytecode along with an implementation of `fzvk::pipeline::StaticSpirV` defining
it's specialization interface and entry points. See the examples for usage.