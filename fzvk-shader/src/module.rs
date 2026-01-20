//! # Module parsing
//! Parses a subset of the SPIR-V, only the information we currently need, and
//! presents it in a nice structured way for codegen to interact with it.
use rspirv::spirv;
use rustc_hash::FxHashMap;
mod descriptor_set;

pub fn execution_model_upper_str(ty: spirv::ExecutionModel) -> &'static str {
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
pub fn execution_model_typestate(ty: spirv::ExecutionModel) -> &'static str {
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
pub enum InvalidType {
    Integer { signed: bool, width: u32 },
    Float { width: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    F32,
    U32,
    I32,
    /// Boolean with *no external representation*, oopsie.
    Bool,
    /// We parsed a type, but it's not one we care about. Keep it around in case
    /// a spec constant references it, then we throw an error.
    Invalid(InvalidType),
}
#[derive(Debug)]
pub struct Vector {
    pub base: Type,
    pub count: u32,
}
#[derive(Debug)]
pub struct Array {
    pub base_ty: u32,
    // Points to an OpConstant, the value of this is not the count!
    pub count_id: u32,
}
#[derive(Debug)]
pub enum SpecDefaultValue {
    /// Float value, stored as le bits.
    F32(u32),
    /// stored as le bits.
    U32(u32),
    /// stored as le bits.
    I32(u32),
    /// Boolean with *no external representation*, oopsie.
    Bool(bool),
}
#[derive(Debug)]
pub enum ConstantValue {
    F32(u32),
    U32(u32),
    I32(u32),
    Invalid(InvalidType),
}
#[derive(Debug)]
pub struct EntryPoint {
    /// UTF-8, No internal nulls. Multiple entry points with the same name are
    /// allowable as long as they don't also have the same execution model.
    pub name: String,
    /// loosely, the type of shader this entry point implements.
    pub model: spirv::ExecutionModel,
    /// Which global OpVariables this entry point interacts with. This does
    /// *not* necessarily mean that all these interfaces are statically-used --
    /// it may be a superset thereof.
    pub interfaces: Vec<u32>,
}
#[derive(Default, Debug)]
pub struct Decorations {
    pub spec_id: Option<u32>,
    pub builtin: Option<spirv::BuiltIn>,
    pub descriptor_set: Option<u32>,
    pub descriptor_binding: Option<u32>,
    pub is_shader_storage_buffer: bool,
    pub is_uniform_buffer: bool,
}
#[derive(Debug)]
struct Pointer {
    storage_class: spirv::StorageClass,
    pointee_ty: u32,
}
#[derive(Debug)]
struct Variable {
    pointer_ty: u32,
}
#[derive(Default, Debug)]
pub struct Module {
    pub version: (u8, u8),
    // OpEntryPoints.
    pub entry_points: Vec<EntryPoint>,
    // Some basic types, created by OpTypeBool/Float/Int
    pub types: FxHashMap<u32, Type>,
    // All ResultIDs of types created by OpTypeVector
    pub vectors: FxHashMap<u32, Vector>,
    // ResultID -> Array of another type + count denoted by an ID ref to an
    // OpConstant.
    pub array_types: FxHashMap<u32, Array>,
    // ResultID -> Decorations applied to it
    pub decorations: FxHashMap<u32, Decorations>,
    // ResultID -> Name
    pub names: FxHashMap<u32, String>,
    // ResultID -> Constant value.
    pub constants: FxHashMap<u32, ConstantValue>,
    // ResultID -> SpecConstant value.
    pub spec_constants: FxHashMap<u32, SpecDefaultValue>,
    // ResultID -> ImageTy, SamplerTy, etc.
    pub resource_types: FxHashMap<u32, descriptor_set::ResourceType>,
    // ResultID -> Pointer storage class and type.
    pub pointer_types: FxHashMap<u32, Pointer>,
    // ResultID -> OpVariable
    pub variables: FxHashMap<u32, Variable>,
}
impl rspirv::binary::Consumer for Module {
    fn consume_header(&mut self, module: rspirv::dr::ModuleHeader) -> rspirv::binary::ParseAction {
        self.version = module.version();
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

                self.entry_points.push(EntryPoint {
                    name: name.to_owned(),
                    model,
                    // The list of `in` and `out` variables this entry may
                    // access. UNFORTUNATELY - prior to SPIRV 1.4, this does
                    // *not* include all OpVariables, only those with Input and
                    // Output storage class. todo: Polyfill this?
                    interfaces: interfaces
                        .iter()
                        .map(|operand| operand.unwrap_id_ref())
                        .collect(),
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
            Op::TypeArray => {
                let id = inst.result_id.unwrap();
                let [element_type, length_const_id] = inst.operands.as_slice() else {
                    panic!("bad operands");
                };
                self.array_types.insert(
                    id,
                    Array {
                        base_ty: element_type.unwrap_id_ref(),
                        count_id: length_const_id.unwrap_id_ref(),
                    },
                );
            }
            Op::TypeImage => {
                let id = inst.result_id.unwrap();
                self.resource_types.insert(
                    id,
                    descriptor_set::ResourceType::Image(descriptor_set::ImageType::new(
                        &inst.operands,
                    )),
                );
            }
            Op::TypeSampler => {
                let id = inst.result_id.unwrap();
                self.resource_types.insert(
                    id,
                    descriptor_set::ResourceType::Sampler(descriptor_set::Sampler),
                );
            }
            Op::TypeSampledImage => {
                let id = inst.result_id.unwrap();
                let image_ty = inst.operands[0].unwrap_id_ref();
                self.resource_types.insert(
                    id,
                    descriptor_set::ResourceType::CombinedImageSampler(
                        descriptor_set::CombinedImageSampler { image_ty },
                    ),
                );
            }
            Op::TypePointer => {
                let id = inst.result_id.unwrap();
                let storage_class = inst.operands[0].unwrap_storage_class();
                let ty = inst.operands[1].unwrap_id_ref();
                self.pointer_types.insert(
                    id,
                    Pointer {
                        storage_class,
                        pointee_ty: ty,
                    },
                );
            }
            Op::Variable => {
                let id = inst.result_id.unwrap();
                let pointer_ty = inst.result_type.unwrap();
                let [_storage_class, _initializer @ ..] = inst.operands.as_slice() else {
                    panic!("bad parameters");
                };
                // _storage_classs is always equal to the pointer's storage
                // class.
                self.variables.insert(id, Variable { pointer_ty });
            }
            Op::Decorate => {
                use spirv::Decoration as Dec;
                let target = inst.operands[0].unwrap_id_ref();
                let decorations = self.decorations.entry(target).or_default();
                let decoration = inst.operands[1].unwrap_decoration();
                match decoration {
                    Dec::SpecId => {
                        let spec_id = inst.operands[2].unwrap_literal_bit32();
                        decorations.spec_id = Some(spec_id);
                    }
                    Dec::BuiltIn => {
                        let builtin = inst.operands[2].unwrap_built_in();
                        decorations.builtin = Some(builtin);
                    }
                    Dec::Binding => {
                        let binding = inst.operands[2].unwrap_literal_bit32();
                        decorations.descriptor_binding = Some(binding);
                    }
                    Dec::DescriptorSet => {
                        let set = inst.operands[2].unwrap_literal_bit32();
                        decorations.descriptor_set = Some(set);
                    }
                    Dec::BufferBlock => {
                        decorations.is_shader_storage_buffer = true;
                    }
                    Dec::Block => {
                        decorations.is_uniform_buffer = true;
                    }
                    _ => (), //println!("ignoring decoration {decoration:?}"),
                };
            }
            // For tagging constants with a human-readable variable name, taken
            // from the source code. Optional. This can come *before* the
            // declaration of the id they target, annoyingly.
            Op::Name => {
                let target = inst.operands[0].unwrap_id_ref();
                let name = inst.operands[1].unwrap_literal_string();

                self.names.insert(target, name.to_owned());
            }
            Op::MemberName => (),
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

                if self.decorations.get(&name).is_none_or(|decorations| {
                    decorations.builtin != Some(spirv::BuiltIn::WorkgroupSize)
                }) {
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
