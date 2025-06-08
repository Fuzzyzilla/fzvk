use std::marker::PhantomData;

use proc_macro::{TokenStream, TokenTree};
use rspirv::{
    self,
    spirv::{self, BuiltIn, Decoration},
};
use rustc_hash::FxHashMap;

trait UnwrapDisplay {
    type Ok;
    fn unwrap_display(self) -> Self::Ok;
}
impl<Ok, Err: std::error::Error> UnwrapDisplay for Result<Ok, Err> {
    type Ok = Ok;
    fn unwrap_display(self) -> Ok {
        match self {
            Ok(ok) => ok,
            Err(err) => panic!("{err}"),
        }
    }
}
fn execution_model_upper_str(ty: spirv::ExecutionModel) -> &'static str {
    use spirv::ExecutionModel as Model;
    match ty {
        Model::GLCompute => "COMPUTE",
        Model::TaskEXT => "TASK",
        Model::MeshEXT => "MESH",
        Model::TessellationControl => "TESS_CONTROL",
        Model::TessellationEvaluation => "TESS_EVALUATION",
        Model::Vertex => "VERTEX",
        Model::Geometry => "GEOMETRY",
        Model::Fragment => "FRAGMENT",
        _ => unimplemented!(),
    }
}
fn execution_model_typestate(ty: spirv::ExecutionModel) -> &'static str {
    use spirv::ExecutionModel as Model;
    match ty {
        Model::GLCompute => "::fzvk::pipeline::Compute",
        Model::TaskEXT => "::fzvk::pipeline::Task",
        Model::MeshEXT => "::fzvk::pipeline::Mesh",
        Model::TessellationControl => "::fzvk::pipeline::TessControl",
        Model::TessellationEvaluation => "::fzvk::pipeline::TessEvaluation",
        Model::Vertex => "::fzvk::pipeline::Vertex",
        Model::Geometry => "::fzvk::pipeline::Geometry",
        Model::Fragment => "::fzvk::pipeline::Fragment",
        _ => unimplemented!(),
    }
}

#[derive(Clone, Copy, Debug)]
enum InvalidType {
    Integer { signed: bool, width: u32 },
    Float { width: u32 },
}

#[derive(Clone, Copy)]
enum Type {
    F32,
    U32,
    I32,
    /// Boolean with *no external representation*, oopsie.
    Bool,
    /// We parsed a type, but it's not one we care about. Keep it around in case
    /// a spec constant references it, then we throw an error.
    Invalid(InvalidType),
}
struct Vector {
    base: Type,
    count: u32,
}
#[derive(Debug)]
enum SpecDefaultValue {
    /// Float value, stored as le bits.
    F32(u32),
    /// stored as le bits.
    U32(u32),
    /// stored as le bits.
    I32(u32),
    /// Boolean with *no external representation*, oopsie.
    Bool(bool),
}
enum ConstantValue {
    F32(u32),
    U32(u32),
    I32(u32),
    Invalid(InvalidType),
}
struct EntryPoint {
    /// UTF-8, No internal nulls. Multiple entry points with the same name are
    /// allowable as long as they don't also have the same execution model.
    name: String,
    /// loosely, the type of shader this entry point implements.
    model: spirv::ExecutionModel,
}
#[derive(Default)]
struct Module {
    // OpEntryPoints.
    entry_points: Vec<EntryPoint>,
    // Some basic types, created by OpTypeBool/Float/Int
    types: FxHashMap<u32, Type>,
    // All ResultIDs of types created by OpTypeVector
    vectors: FxHashMap<u32, Vector>,
    // All ResultIDs decorated by `BuiltIn WorkgroupSize`
    workgroup_sizes: Vec<u32>,
    // ResultID -> OpDecorate SpecId value
    spec_ids: FxHashMap<u32, u32>,
    // ResultID -> Name
    names: FxHashMap<u32, String>,
    // ResultID -> Constant value.
    constants: FxHashMap<u32, ConstantValue>,
    // ResultID -> SpecConstant value.
    spec_constants: FxHashMap<u32, SpecDefaultValue>,
}
impl rspirv::binary::Consumer for Module {
    fn consume_header(&mut self, module: rspirv::dr::ModuleHeader) -> rspirv::binary::ParseAction {
        rspirv::binary::ParseAction::Continue
    }
    fn initialize(&mut self) -> rspirv::binary::ParseAction {
        rspirv::binary::ParseAction::Continue
    }
    fn consume_instruction(
        &mut self,
        inst: rspirv::dr::Instruction,
    ) -> rspirv::binary::ParseAction {
        use rspirv::spirv::Op;
        match inst.class.opcode {
            Op::EntryPoint => {
                let [model, _entry, name, interfaces @ ..] = inst.operands.as_slice() else {
                    panic!("invalid entry point")
                };
                let model = model.unwrap_execution_model();
                let name = name.unwrap_literal_string();
                // The list of `in` and `out` variables this entry may access.
                let _ = interfaces;

                self.entry_points.push(EntryPoint {
                    name: name.to_owned(),
                    model,
                });
            }
            // Declaration of a name representing an int of some signedness and
            // some bitwidth.
            Op::TypeInt => {
                let id = inst.result_id.unwrap();
                let width = inst.operands[0].unwrap_literal_bit32();
                let signed = match inst.operands[1].unwrap_literal_bit32() {
                    0 => false,
                    1 => true,
                    _ => panic!("invalid signedness literal"),
                };
                let ty = if width == 32 {
                    if signed { Type::I32 } else { Type::U32 }
                } else {
                    // It's okay to observe a type we can't support, the error
                    // only occurs if we need to work with it.
                    Type::Invalid(InvalidType::Integer { width, signed })
                };
                self.types.insert(id, ty);
            }
            // Declaration of a float with some bitwidth.
            Op::TypeFloat => {
                let id = inst.result_id.unwrap();
                let width = inst.operands[0].unwrap_literal_bit32();
                assert_eq!(
                    inst.operands.len(),
                    1,
                    "alternate float encodings not supported"
                );
                let ty = if width == 32 {
                    Type::F32
                } else {
                    Type::Invalid(InvalidType::Float { width })
                };
                self.types.insert(id, ty);
            }
            // Bool. There are no arguments.
            Op::TypeBool => {
                let id = inst.result_id.unwrap();
                self.types.insert(id, Type::Bool);
            }
            Op::TypeVector => {
                let id = inst.result_id.unwrap();
                let [component, count] = inst.operands.as_slice() else {
                    panic!("wawa");
                };
                let component_ty = component.unwrap_id_ref();
                let count = count.unwrap_literal_bit32();

                let component_ty = self.types.get(&component_ty).unwrap();
                self.vectors.insert(
                    id,
                    Vector {
                        base: *component_ty,
                        count,
                    },
                );
            }
            // Used for "SpecId" showing into what index the spec constant will
            // be placed. This can come *before* the declaration of the id they
            // target, annoyingly.
            Op::Decorate => {
                let target = inst.operands[0].unwrap_id_ref();
                let decoration = inst.operands[1].unwrap_decoration();
                match decoration {
                    Decoration::SpecId => {
                        let spec_id = inst.operands[2].unwrap_literal_bit32();
                        // It is not an error for this to be specified several
                        // times for the same target, weirdly.
                        self.spec_ids.insert(target, spec_id);
                    }
                    Decoration::BuiltIn => {
                        let builtin = inst.operands[2].unwrap_built_in();
                        if builtin == BuiltIn::WorkgroupSize {
                            self.workgroup_sizes.push(target);
                        }
                    }
                    _ => println!("ignoring decoration {decoration:?}"),
                }
            }
            // For tagging constants with a human-readable variable name, taken
            // from the source code. Optional. This can come *before* the
            // declaration of the id they target, annoyingly.
            Op::Name => {
                let target = inst.operands[0].unwrap_id_ref();
                let name = inst.operands[1].unwrap_literal_string();

                self.names.insert(target, name.to_owned());
            }
            // An integer or floating-point constant.
            Op::SpecConstant => {
                let ty = inst.result_type.unwrap();
                let ty = self.types.get(&ty).unwrap();
                // We only support 32 bit rn.
                let default_bits = inst.operands[0].unwrap_literal_bit32();
                let default = match ty {
                    Type::F32 => SpecDefaultValue::F32(default_bits),
                    Type::U32 => SpecDefaultValue::U32(default_bits),
                    Type::I32 => SpecDefaultValue::I32(default_bits),
                    // Different instructions are used for booleans.
                    Type::Bool => panic!("invalid spec type bool"),
                    // We parsed this type before, but  it's not one we support.
                    Type::Invalid(ty) => {
                        panic!("spec constant uses an unsupported type {ty:?}")
                    }
                };
                let name = inst.result_id.unwrap();
                self.spec_constants.insert(name, default);
            }
            // A boolean spec constant.
            Op::SpecConstantFalse | Op::SpecConstantTrue => {
                let value = inst.class.opcode == Op::SpecConstantTrue;
                let ty = inst.result_type.unwrap();
                // Must be the Bool type.
                assert!(
                    self.types
                        .get(&ty)
                        .is_some_and(|ty| matches!(ty, Type::Bool))
                );
                let name = inst.result_id.unwrap();
                self.spec_constants
                    .insert(name, SpecDefaultValue::Bool(value));
            }
            // This is usable by GLSL, from eg:
            // ```glsl
            // layout(constant_id = 0) int A = 1;
            // layout(constant_id = 1) int B = A * 2;
            // ```
            // But im not in the mood to write a SPIR-V interpreter :3c Maybe
            // I'm misinterpreting, but it seems that this is not a spec
            // constant provided by the host, and is merely a way to derive
            // another constant from one provided by the host:
            // > Note that the OpSpecConstantOp instruction is not one that can
            // > be updated with a specialization constant. Nonetheless, it
            // still gets a SpecId, for some reason.
            Op::SpecConstantOp => (),
            // A non-specialization constant. This is used by
            // OpSpecConstantComposite.
            Op::Constant => {
                let name = inst.result_id.unwrap();
                let ty = inst.result_type.unwrap();

                let ty = self.types.get(&ty).unwrap();
                // We only support 32 bit types.
                let bits = inst.operands[0].unwrap_literal_bit32();
                let val = match *ty {
                    Type::U32 => ConstantValue::U32(bits),
                    Type::I32 => ConstantValue::I32(bits),
                    Type::F32 => ConstantValue::F32(bits),
                    Type::Invalid(i) => ConstantValue::Invalid(i),
                    // Constants of bool type are specified using different
                    // instructions, which we ignore.
                    Type::Bool => panic!("invalid const type bool"),
                };
                self.constants.insert(name, val);
            }
            // Spec constant that is a structure, array, matrix, or vector.
            // FIXME: While i don't think GLSL supports this, it *does* emit
            // this for gl_WorkgroupSize. These constants lack names. So we
            // could work backwards and say if a spec constant is used here,
            // it's named "local_size_x" or such!
            Op::SpecConstantComposite => {
                // We only support a u32 x 3 being applied to a BuiltIn
                // WorkGroupSize.
                let ty = inst.result_type.unwrap();
                let name = inst.result_id.unwrap();

                if !self.workgroup_sizes.contains(&name) {
                    unimplemented!("only WorkgroupSize can have an SpecConstantComposite value");
                }
                let ty = self.vectors.get(&ty).unwrap();
                assert!(
                    matches!(
                        ty,
                        Vector {
                            base: Type::U32,
                            count: 3
                        }
                    ),
                    "WorkgroupSize should be a u32x3 vector"
                );
                let [x, y, z] = inst.operands.as_slice() else {
                    panic!("bad number of args")
                };
                let [x, y, z] = [x, y, z].map(|dim| dim.unwrap_id_ref());

                // Rename the variables, now that we know they're being used as
                // local_sizes.
                if self.spec_constants.contains_key(&x) {
                    self.names.insert(x, "local_size_x".to_owned());
                }
                if self.spec_constants.contains_key(&y) {
                    self.names.insert(y, "local_size_y".to_owned());
                }
                if self.spec_constants.contains_key(&z) {
                    self.names.insert(z, "local_size_z".to_owned());
                }
            }
            _ => (),
        }
        rspirv::binary::ParseAction::Continue
    }
    fn finalize(&mut self) -> rspirv::binary::ParseAction {
        rspirv::binary::ParseAction::Continue
    }
}
fn input_string(input: TokenStream) -> String {
    let mut iter = input.into_iter();
    let string = iter.next().unwrap();
    assert_eq!(iter.count(), 0);
    let string = match string {
        proc_macro::TokenTree::Literal(lit) => lit,
        _ => panic!("expected a literal"),
    };

    let litrs::Literal::String(string) = litrs::Literal::from(string) else {
        panic!("expected a string literal")
    };
    string.value().to_owned()
}
fn from_spirv(spirv: &[u32]) -> TokenStream {
    let mut module = Module::default();
    rspirv::binary::parse_words(spirv, &mut module).unwrap_display();
    // Ensure we have at least one entry point, modules are invalid without one.
    assert!(!module.entry_points.is_empty());
    // Create a struct definition for all of the specialization constants.
    let (specialization_struct_def, specialization_struct_name) = if module
        .spec_constants
        .is_empty()
    {
        // If none, definition is empty and the unit type is the constant type.
        (quote::quote! {}, quote::quote! {()})
    } else {
        // Name of our struct.
        let name = quote::format_ident!("Specialization");
        // Name -> (ty, default val) of our constants. TokenTrees/idents don't
        // impl Hash or Eq. Everything feels wrong!
        struct Field {
            constant_id: u32,
            name: proc_macro2::Ident,
            ty: proc_macro2::TokenStream,
            default_value: proc_macro2::TokenStream,
        }
        let mut fields = Vec::<Field>::default();
        for (binding_id, default_value) in &module.spec_constants {
            let name = module
                .names
                .get(binding_id)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("constant_{binding_id}"));
            let (ty, default_value) = match default_value {
                SpecDefaultValue::Bool(v) => (
                    quote::quote! {::fzvk::Bool32},
                    if *v {
                        quote::quote! {::fzvk::Bool32::True}
                    } else {
                        quote::quote! {::fzvk::Bool32::False}
                    },
                ),
                SpecDefaultValue::U32(v) => (quote::quote! {u32}, quote::quote! {#v}),
                SpecDefaultValue::I32(v) => {
                    let value = i32::from_ne_bytes(v.to_ne_bytes());
                    (quote::quote! {i32}, quote::quote! {#value})
                }
                SpecDefaultValue::F32(v) => {
                    let le_bytes = v.to_le_bytes();
                    (
                        quote::quote! {f32},
                        quote::quote! {f32::from_le_bytes([#(#le_bytes),*])},
                    )
                }
            };
            let constant_id = module.spec_ids.get(binding_id).unwrap();
            fields.push(Field {
                constant_id: *constant_id,
                name: quote::format_ident!("{name}"),
                ty,
                default_value,
            })
        }
        let mut specialization_entries = Vec::<proc_macro2::TokenStream>::new();
        let mut offset = 0u32;
        for field in &fields {
            let id = field.constant_id;
            let entry = quote::quote! {
                ::fzvk::vk::SpecializationMapEntry {
                    constant_id: #id,
                    offset: #offset,
                    size: 4,
                }
            };
            specialization_entries.push(entry);
            offset += 4;
        }
        let field_name_1 = fields.iter().map(|field| &field.name);
        let field_name_2 = field_name_1.clone();
        let field_ty = fields.iter().map(|field| &field.ty);
        let field_default_value = fields.iter().map(|field| &field.default_value);
        // Definition and Specialization trait implementation
        let def = quote::quote! {
            /// Specialization constants, generated from SPIR-V.
            ///
            /// The `Default::default` implementation takes the default values
            /// from the shader.
            #[repr(C)]
            pub struct #name {
                pub #(#field_name_1: #field_ty),*
            }
            impl ::core::default::Default for #name {
                fn default() -> Self {
                    Self {
                        #(#field_name_2: #field_default_value),*
                    }
                }
            }
            unsafe impl ::fzvk::pipeline::Specialization for #name {
                const ENTRIES : &[::fzvk::vk::SpecializationMapEntry] = &[#(#specialization_entries),*];
            }
        };
        // Is it clear I don't know what im doing >~<
        (def, quote::quote! {#name})
    };
    let entry_constants = {
        struct EntryConst {
            const_name: proc_macro2::Ident,
            entry_point: proc_macro2::Literal,
            ty_constant: proc_macro2::TokenStream,
        }
        let mut entries = Vec::<EntryConst>::new();
        for this_entry in &module.entry_points {
            let is_unique = module
                .entry_points
                .iter()
                .filter(|entry| entry.name == this_entry.name)
                .count()
                == 1;
            let entry_name = std::ffi::CString::new(this_entry.name.clone()).unwrap();
            // Todo: GLSL case to UPPER_SNAKE_CASE
            let const_name = if !is_unique {
                this_entry.name.to_ascii_uppercase()
                    + "_"
                    + execution_model_upper_str(this_entry.model)
            } else {
                this_entry.name.to_ascii_uppercase()
            };
            let model_typestate_marker = execution_model_typestate(this_entry.model);
            entries.push(EntryConst {
                const_name: quote::format_ident!("{const_name}"),
                entry_point: proc_macro2::Literal::c_string(&entry_name),
                ty_constant: model_typestate_marker.parse().unwrap(),
            });
        }
        let const_name = entries.iter().map(|entry| &entry.const_name);
        let entry_point = entries.iter().map(|entry| &entry.entry_point);
        let ty_constant = entries.iter().map(|entry| &entry.ty_constant);
        quote::quote! {
            #(
                pub const #const_name :
                    ::fzvk::pipeline::StaticEntryName::<'static, Module, #ty_constant>
                    = unsafe {
                        ::fzvk::pipeline::StaticEntryName::new(#entry_point)
                    };
            )*
        }
    };

    quote::quote! {
        #specialization_struct_def
        pub struct Module;
        unsafe impl ::fzvk::pipeline::StaticSpirV for Module {
            type Specialization = #specialization_struct_name;
            const SPIRV : &[u32] = &[#(#spirv),*];
        }
        #entry_constants
    }
    .into()
}
/// Accepts a string literal as a filepath, expands to a static slice of the
/// bytes of SPIR-V within the file.
#[proc_macro]
pub fn spirv(input: TokenStream) -> TokenStream {
    let path = input_string(input);
    let spirv = read_spirv_words(path);
    from_spirv(&spirv)
}
/// Accepts a string literal containing GLSL source code, and expands to a
/// static slice of the compiled SPIR-V. Use e.g. `#pragma shader_stage(vertex)`
/// to specify the execution model.
#[proc_macro]
pub fn glsl(input: TokenStream) -> TokenStream {
    let source = input_string(input);
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_source_language(shaderc::SourceLanguage::GLSL);
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_0 as _,
    );
    options.set_target_spirv(shaderc::SpirvVersion::V1_0);
    let artifact = compiler
        .compile_into_spirv(
            &source,
            shaderc::ShaderKind::InferFromSource,
            "<source>",
            "main",
            Some(&options),
        )
        .unwrap_display();
    // ShaderC does not specify in what endian the bytes are returned.
    #[cfg(target_endian = "big")]
    unimplemented!("this is undocumented lol");
    let spirv = artifact.as_binary();
    from_spirv(spirv)
}

/// Read a file as u32s, swapping endian if necessary based on the file magic.
fn read_spirv_words(path: impl AsRef<std::path::Path>) -> Vec<u32> {
    const MAGIC: u32 = rspirv::spirv::MAGIC_NUMBER;
    const SWAPPED_MAGIC: u32 = MAGIC.swap_bytes();

    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() % 4 == 0);
    let mut words = vec![0u32; bytes.len() / 4];

    unsafe {
        // Have to copy bytes to bytes, not words to words, as `bytes` might not
        // be aligned!
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast(), bytes.len())
    };

    match words.first().copied() {
        Some(MAGIC) => words,
        Some(SWAPPED_MAGIC) => {
            for word in &mut words {
                *word = word.swap_bytes();
            }
            words
        }
        _ => panic!("bad file magic"),
    }
}

fn main() -> () {
    let path = std::env::args()
        .nth(1)
        .expect("usage: \"<spirv-path>\" OR \"<shader-path>\"");
    let path = std::path::PathBuf::from(path);
    let extension = path.extension().unwrap().to_str().unwrap();
    let spirv = if extension == "spirv" {
        read_spirv_words(&path)
    } else {
        let shader_kind = match extension {
            "glsl" => shaderc::ShaderKind::InferFromSource,

            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            _ => unimplemented!("unrecognized file extension"),
        };
        let source = std::fs::read_to_string(&path).unwrap();
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_source_language(shaderc::SourceLanguage::GLSL);
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_0 as _,
        );
        options.set_target_spirv(shaderc::SpirvVersion::V1_0);
        let artifact = compiler
            .compile_into_spirv(
                &source,
                shader_kind,
                path.file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .unwrap_or("<unknown>"),
                "main",
                Some(&options),
            )
            .unwrap();
        // ShaderC does not specify in what endian the bytes are returned.
        #[cfg(target_endian = "big")]
        unimplemented!("this is undocumented lol");
        artifact.as_binary().to_vec()
    };
    let mut module = Module::default();
    // Reinterprets as bytes, so it has to be LE words regardless of platform
    // endian.
    #[cfg(target_endian = "big")]
    unimplemented!("wants little endian, eegh");
    rspirv::binary::parse_words(spirv, &mut module).unwrap();
    for entry in &module.entry_points {
        println!("Entry point {:?}: {:?}", entry.name, entry.model);
    }
    for (id, value) in module.spec_constants {
        let name = module.names.get(&id);
        if let Some(name) = name {
            println!("specialization constant %{name} (%{id})");
        } else {
            println!("specialization constant %{id}");
        }
        println!(" - default value: {value:?}");
        if let Some(location) = module.spec_ids.get(&id) {
            println!(" - location: {location}");
        }
    }
    todo!()
}
