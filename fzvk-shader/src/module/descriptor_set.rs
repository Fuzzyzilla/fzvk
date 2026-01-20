//! # Descriptor set analysis
//! References the parsed module to construct the statically-known constraints
//! on descriptor sets.
use crate::OperandExt;
use rspirv::spirv;
use rustc_hash::FxHashMap;
pub fn spirv_image_format_to_fzvk_format_path(
    format: spirv::ImageFormat,
    depth_mode: DepthMode,
) -> &'static str {
    match depth_mode {
        DepthMode::Unknown => unimplemented!(),
        DepthMode::Depth if matches!(format, spirv::ImageFormat::Unknown) => {
            "::fzvk::descriptor::AnyDepth"
        }
        DepthMode::NotDepth => {
            use spirv::ImageFormat as Format;
            match format {
                Format::Unknown => "::fzvk::descriptor::AnyColor",

                Format::Rgb10a2ui => "::fzvk::format::A2Bgr10i",
                Format::Rgb10A2 => "::fzvk::format::A2Rgb10",
                Format::R11fG11fB10f => "::fzvk::format::B10Gr11Uf",

                Format::Rgba8i => "::fzvk::format::Rgba8i",
                Format::Rgba8Snorm => "::fzvk::format::Rgba8Inorm",
                Format::Rgba8ui => "::fzvk::format::Rgba8u",
                Format::Rgba8 => "::fzvk::format::Rgba8",

                Format::Rg8i => "::fzvk::format::Rg8i",
                Format::Rg8Snorm => "::fzvk::format::Rg8Inorm",
                Format::Rg8ui => "::fzvk::format::Rg8u",
                Format::Rg8 => "::fzvk::format::Rg8",

                Format::R8i => "::fzvk::format::R8i",
                Format::R8Snorm => "::fzvk::format::R8Inorm",
                Format::R8ui => "::fzvk::format::R8u",
                Format::R8 => "::fzvk::format::R8",

                Format::Rgba16f => "::fzvk::format::Rgba16f",
                Format::Rgba16i => "::fzvk::format::Rgba16i",
                Format::Rgba16Snorm => "::fzvk::format::Rgba16Inorm",
                Format::Rgba16ui => "::fzvk::format::Rgba16u",
                Format::Rgba16 => "::fzvk::format::Rgba16",

                Format::Rg16f => "::fzvk::format::Rg16f",
                Format::Rg16i => "::fzvk::format::Rg16i",
                Format::Rg16Snorm => "::fzvk::format::Rg16Inorm",
                Format::Rg16ui => "::fzvk::format::Rg16u",
                Format::Rg16 => "::fzvk::format::Rg16",

                Format::R16f => "::fzvk::format::R16f",
                Format::R16i => "::fzvk::format::R16i",
                Format::R16Snorm => "::fzvk::format::R16Inorm",
                Format::R16ui => "::fzvk::format::R16u",
                Format::R16 => "::fzvk::format::R16",

                Format::Rgba32f => "::fzvk::format::Rgba32f",
                Format::Rgba32i => "::fzvk::format::Rgba32i",
                Format::Rgba32ui => "::fzvk::format::Rgba32u",

                Format::Rg32f => "::fzvk::format::Rg32f",
                Format::Rg32i => "::fzvk::format::Rg32i",
                Format::Rg32ui => "::fzvk::format::Rg32u",

                Format::R32f => "::fzvk::format::R32f",
                Format::R32i => "::fzvk::format::R32i",
                Format::R32ui => "::fzvk::format::R32u",
                Format::R64i => "::fzvk::format::R64i",
                Format::R64ui => "::fzvk::format::R64u",
            }
        }

        _ => unimplemented!(),
    }
}
pub fn spirv_image_dimenstion_to_fzvk_dimension_path(
    dimension: ImageDimensionality,
) -> &'static str {
    use ImageDimensionality as Dim;
    match dimension {
        Dim::D1 => "::fzvk::image::D1",
        Dim::D1Array => "::fzvk::image::D1Array",
        Dim::D2 => "::fzvk::image::D2",
        Dim::D2Array => "::fzvk::image::D2Array",
        Dim::D3 => "::fzvk::image::D3",
        Dim::D3Array | Dim::Cube | Dim::CubeArray => unimplemented!(),
    }
}
pub fn is_multisampled_to_fzvk_path(is_multisampled: bool) -> &'static str {
    if is_multisampled {
        "::fzvk::image::SingleSampled"
    } else {
        "::fzvk::image::MultiSampled"
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DepthMode {
    NotDepth = 0,
    Depth = 1,
    Unknown = 2,
}
#[derive(Debug)]
pub enum SamplingMode {
    Unknown = 0,
    Sampled = 1,
    // Storage OR subpass input.
    Storage = 2,
}
impl TryFrom<u32> for DepthMode {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::NotDepth,
            1 => Self::Depth,
            2 => Self::Unknown,
            _ => return Err(()),
        })
    }
}
impl TryFrom<u32> for SamplingMode {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::Unknown,
            1 => Self::Sampled,
            2 => Self::Storage,
            _ => return Err(()),
        })
    }
}
#[derive(Debug)]
pub struct ImageType {
    // The type ID that results from sampling/loading from the image. This isn't
    // relevant to us, since this isn't a property of the external image, but is
    // instead how the texture addresser should cast the fetched values.
    pub _sample_ty: u32,
    // "Buffer" dimensionality means a vulkan TexelBuffer.
    pub dim: spirv::Dim,
    pub depth_mode: DepthMode,
    pub is_arrayed: bool,
    pub is_multisampled: bool,
    pub sampling_mode: SamplingMode,
    pub format: spirv::ImageFormat,
    // Only used for Kernel execution modes, not shader. The access qualifiers
    // of this are expressed through the NonReadabe/Writable decorations.
    pub _access_qualifier: Option<spirv::AccessQualifier>,
}
impl ImageType {
    pub fn new(operands: &[rspirv::dr::Operand]) -> Self {
        let [
            sample_ty,
            dim,
            depth_mode,
            is_arrayed,
            is_multisampled,
            sampling_mode,
            format,
            // Access qualifier is for *kernel* modules, not shaders, so we dont
            // care~
            _access_qualifier @ ..,
        ] = operands
        else {
            panic!("bad number of args")
        };
        Self {
            _sample_ty: sample_ty.unwrap_id_ref(),
            dim: dim.unwrap_dim(),
            depth_mode: depth_mode.unwrap_literal_bit32().try_into().unwrap(),
            is_arrayed: is_arrayed.unwrap_literal_bool(),
            is_multisampled: is_multisampled.unwrap_literal_bool(),
            sampling_mode: sampling_mode.unwrap_literal_bit32().try_into().unwrap(),
            format: format.unwrap_image_format(),
            _access_qualifier: None,
        }
    }
    pub fn requirements(&self) -> ImageRequirements {
        ImageRequirements {
            dim: match (self.dim, self.is_arrayed) {
                (spirv::Dim::Dim1D, false) => ImageDimensionality::D1,
                (spirv::Dim::Dim1D, true) => ImageDimensionality::D1Array,
                (spirv::Dim::Dim2D, false) => ImageDimensionality::D2,
                (spirv::Dim::Dim2D, true) => ImageDimensionality::D2Array,
                (spirv::Dim::Dim3D, false) => ImageDimensionality::D3,
                (spirv::Dim::Dim3D, true) => ImageDimensionality::D3Array,
                (spirv::Dim::DimCube, false) => ImageDimensionality::Cube,
                (spirv::Dim::DimCube, true) => ImageDimensionality::CubeArray,
                _ => unimplemented!(),
            },
            format: self.format,
            multisampled: self.is_multisampled,
            depth: self.depth_mode,
        }
    }
}
#[derive(Debug)]
pub struct Sampler;
#[derive(Debug)]
pub struct CombinedImageSampler {
    pub image_ty: u32,
}
#[derive(Debug)]
pub enum ResourceType {
    Image(ImageType),
    Sampler(Sampler),
    CombinedImageSampler(CombinedImageSampler),
}

#[derive(Debug)]
pub struct DescriptorBinding {
    pub name: Option<String>,
    pub ty: DescriptorBindingType,
    // *may* be zero, in which case the type is irrelevant
    pub array_count: u32,
}
impl DescriptorBinding {
    pub const EMPTY: Self = Self {
        name: None,
        array_count: 0,
        ty: DescriptorBindingType::Empty,
    };
}
#[derive(Debug)]
pub enum ImageDimensionality {
    D1,
    D1Array,
    D2,
    D2Array,
    D3,
    D3Array,
    Cube,
    CubeArray,
}
#[derive(Debug)]
pub struct ImageRequirements {
    dim: ImageDimensionality,
    format: spirv::ImageFormat,
    multisampled: bool,
    depth: DepthMode,
}
#[derive(Debug)]
pub enum DescriptorBindingType {
    // This binding number is not included in the descriptor set. This is
    // defined in the vulkan spec as being equivalent to an arbitrary type of
    // descriptor with an array count of 0.
    Empty,
    StorageImage(ImageRequirements),
    SampledImage(ImageRequirements),
    Sampler,
    CombinedImageSampler(ImageRequirements),
    SubpassInput,
    StorageBuffer,
    UniformBuffer,
    UniformTexelBuffer(),
    StorageTexelBuffer(),
}
#[derive(Debug)]
pub struct DescriptorSet(pub Vec<DescriptorBinding>);

impl super::Module {
    fn get_descriptor_sets(&self) -> FxHashMap<u32, DescriptorSet> {
        // Set -> (binding -> variable_ty)
        let mut set_binding_variable = FxHashMap::<u32, FxHashMap<u32, u32>>::default();
        // Iterate through every decorated binding, and collect all of the ones
        // decorated by "DescriptorSet" and note their types and IDs.
        for (id, decorators) in &self.decorations {
            if let Some(descriptor_set) = decorators.descriptor_set {
                let has_duplicate = set_binding_variable
                    .entry(descriptor_set)
                    .or_default()
                    .insert(decorators.descriptor_binding.unwrap(), *id)
                    .is_some();
                if has_duplicate {
                    // This actually *is* valid SPIR-V, but we can't handle it
                    // :3
                    unimplemented!("Two resources cannot share the same (set,binding) pair");
                }
            }
        }

        let mut descriptor_sets = FxHashMap::<u32, DescriptorSet>::default();
        /// Iterate over the set numbers, and their list of (binding ->
        /// variable_ty)
        for (set, binding_to_variable_id) in &set_binding_variable {
            let max_binding_nr = *binding_to_variable_id.keys().max().unwrap();
            let descriptor_bindings = descriptor_sets
                .entry(*set)
                .or_insert(DescriptorSet(Vec::new()));
            // We loop over every index, even if they are empty. Since we repr
            // descriptor sets as tuples, missing IDs are instead treated as an
            // array of arbitrary type with zero elements.
            for binding_nr in 0..=max_binding_nr {
                let variable_id = binding_to_variable_id.get(&binding_nr).copied();
                let binding = if let Some(variable_id) = variable_id {
                    let variable = self.variables.get(&variable_id).unwrap();
                    let pointer_ty = self.pointer_types.get(&variable.pointer_ty).unwrap();
                    // MAY Point to OpType[Resource] OR OpTypeStructure
                    // decorated with BufferBlock OR an OpTypeArray thereof! :O
                    // The array types give us the descriptor array count.
                    let (pointee_id, array_count) = {
                        if let Some(array) = self.array_types.get(&pointer_ty.pointee_ty) {
                            let count = match self.constants.get(&array.count_id).unwrap() {
                                super::ConstantValue::U32(count) => *count,
                                _ => unimplemented!(
                                    "Descriptor set array length must be a simple compile-time constant."
                                ),
                            };
                            (array.base_ty, count)
                        } else {
                            // Not an array, one less layer of indirection for
                            // us~.
                            (pointer_ty.pointee_ty, 1)
                        }
                    };
                    let pointee_decorations = self.decorations.get(&pointee_id);
                    if pointee_decorations.is_some_and(|pointee| {
                        pointee.is_shader_storage_buffer || pointee.is_uniform_buffer
                    }) {
                        let pointee_decorations = pointee_decorations.unwrap();
                        // This is a uniform or storage buffer structure.
                        let binding_ty = if pointee_decorations.is_shader_storage_buffer {
                            DescriptorBindingType::StorageBuffer
                        } else {
                            DescriptorBindingType::UniformBuffer
                        };
                        DescriptorBinding {
                            name: self.names.get(&variable_id).cloned(),
                            ty: binding_ty,
                            array_count,
                        }
                    } else {
                        // This is something else...
                        let resource_ty =
                            self.resource_types.get(&pointee_id).unwrap_or_else(|| {
                                panic!("Set {set} binding {binding_nr} is of unknown type")
                            });
                        let binding_ty = match resource_ty {
                            ResourceType::Image(img) => {
                                if matches!(img.dim, spirv::Dim::DimSubpassData) {
                                    DescriptorBindingType::SubpassInput
                                } else {
                                    match img.sampling_mode {
                                        SamplingMode::Sampled => {
                                            DescriptorBindingType::SampledImage(img.requirements())
                                        }
                                        SamplingMode::Storage => {
                                            DescriptorBindingType::StorageImage(img.requirements())
                                        }
                                        SamplingMode::Unknown => unimplemented!(),
                                    }
                                }
                            }
                            ResourceType::Sampler(_sampler) => DescriptorBindingType::Sampler,
                            ResourceType::CombinedImageSampler(combined) => {
                                let ResourceType::Image(img) =
                                    self.resource_types.get(&combined.image_ty).unwrap()
                                else {
                                    unimplemented!();
                                };
                                DescriptorBindingType::CombinedImageSampler(img.requirements())
                            }
                        };

                        DescriptorBinding {
                            name: self.names.get(&variable_id).cloned(),
                            ty: binding_ty,
                            array_count,
                        }
                    }
                } else {
                    // Nothing was assigned to this binding.
                    DescriptorBinding::EMPTY
                };
                descriptor_bindings.0.push(binding);
            }
        }
        descriptor_sets
    }
}
