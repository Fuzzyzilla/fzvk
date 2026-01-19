//! # Descriptor Sets and Bindings
//!
//! A Desciptor set can be conceptualized as a (non-nesting) tuple of zero or
//! more (possibly arrayed) bindings, where the binding number refers to the
//! fields of the tuple:
//! ```ignore
//! struct MyDescriptorSet (
//!     StorageBuffer, // self.0 == binding 0, equivalent to an array of 1.
//!     [<arbitrary>; 0], // .1 == binding 1, see note below~
//!     [CombinedImageSampler; 2], // .2 == binding 2
//! )
//! ```
//! Note that, when specifying a descriptor set layout, bindings may be placed
//! with gaps between them. [In
//! reality](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#VkDescriptorSetLayoutBinding),
//! these are exactly equivalent to an array of zero elements of arbitrary type,
//! meaning this tuple interpretation is without loss of generality. Such empty
//! bidings are valid but do incur a memory cost *as if* there was a resource in
//! that binding, and should be avoided where possible.

use super::*;
pub trait DescriptorType {
    const TYPE: vk::DescriptorType;
}
pub struct StorageImage;
impl DescriptorType for StorageImage {
    const TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_IMAGE;
}
pub struct SampledImage;
impl DescriptorType for SampledImage {
    const TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLED_IMAGE;
}
pub struct InputAttachment;
impl DescriptorType for InputAttachment {
    const TYPE: vk::DescriptorType = vk::DescriptorType::INPUT_ATTACHMENT;
}
pub struct Sampler;
impl DescriptorType for Sampler {
    const TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLER;
}
pub struct StorageBuffer;
impl DescriptorType for StorageBuffer {
    const TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_BUFFER;
}
pub struct UniformBuffer;
impl DescriptorType for UniformBuffer {
    const TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_BUFFER;
}
pub struct StorageTexelBuffer;
impl DescriptorType for StorageTexelBuffer {
    const TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_TEXEL_BUFFER;
}
pub struct UniformTexelBuffer;
impl DescriptorType for UniformTexelBuffer {
    const TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_TEXEL_BUFFER;
}
pub struct DynamicStorageBuffer;
impl DescriptorType for DynamicStorageBuffer {
    const TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_BUFFER_DYNAMIC;
}
pub struct DynamicUniformBuffer;
impl DescriptorType for DynamicUniformBuffer {
    const TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC;
}

pub trait FormatRequirement {}
/// Format requirement allowing any color format.
pub struct AnyColor;
impl FormatRequirement for AnyColor {}
/// Format requirement allowing any depth format.
pub struct AnyDepth;
impl FormatRequirement for AnyDepth {}
pub trait SatisfiesFormatRequirement<Format: FormatRequirement>: format::Format {}
/// All formats with a color aspect satisfy [`AnyColor`].
impl<T: format::Format<AspectMask = format::aspect::Color>> SatisfiesFormatRequirement<AnyColor>
    for T
{
}
/// All formats with a depth aspect satisfy [`AnyDepth`].
impl<
    Aspect: format::aspect::AspectSupersetOf<format::aspect::Depth>,
    T: format::Format<AspectMask = Aspect>,
> SatisfiesFormatRequirement<AnyDepth> for T
{
}
/// Formats of course satisfy themselves.
impl<T: format::Format + FormatRequirement> SatisfiesFormatRequirement<T> for T {}

macro_rules! image_formats {
    ($($name:ident,)*) => {
        $(impl FormatRequirement for crate::format::$name {})*
    };
}
image_formats! {
    A2Bgr10i,
    A2Rgb10,
    B10Gr11Uf,

    Rgba8i,
    Rgba8Inorm,
    Rgba8u,
    Rgba8,

    Rg8i,
    Rg8Inorm,
    Rg8u,
    Rg8,

    R8i,
    R8Inorm,
    R8u,
    R8,

    Rgba16f,
    Rgba16i,
    Rgba16Inorm,
    Rgba16u,
    Rgba16,

    Rg16f,
    Rg16i,
    Rg16Inorm,
    Rg16u,
    Rg16,

    R16f,
    R16i,
    R16Inorm,
    R16u,
    R16,

    Rgba32f,
    Rgba32i,
    Rgba32u,

    Rg32f,
    Rg32i,
    Rg32u,

    R32f,
    R32i,
    R32u,
    R64i,
    R64u,
}

pub unsafe trait HasResource<Ty: DescriptorType> {}
unsafe impl<Usage, Dim, Format, Samples, Aspect> HasResource<StorageImage>
    for image::ImageView<Usage, Dim, Format, Samples, Aspect>
where
    Usage: usage::ImageSuperset<usage::Storage>,
    // FIXME: bounds on these?
    Dim: image::Dimensionality,
    Format: format::Format,
    Samples: image::ImageSamples,
    Aspect: format::aspect::AspectMask,
{
}
pub trait DescriptorBinding {
    const TYPE: vk::DescriptorType;
    const COUNT: u32;
    type Param<'a>;
    fn create_info<'a>(
        param: Self::Param<'a>,
        binding: u32,
        shaders: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'a>;
}
impl DescriptorBinding for () {
    /// Arbitrary type, since `COUNT` is zero this is ignored!
    const TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLER;
    const COUNT: u32 = 0;
    type Param<'a> = ();
    fn create_info<'a>(
        (): Self::Param<'a>,
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(Self::TYPE)
            .descriptor_count(<Self as DescriptorBinding>::COUNT)
            .binding(binding)
            .stage_flags(stage_flags)
    }
}
impl<const N: usize> DescriptorBinding for [(); N] {
    const TYPE: vk::DescriptorType = <() as DescriptorBinding>::TYPE;
    const COUNT: u32 = 0;
    type Param<'a> = ();
    fn create_info<'a>(
        param: Self::Param<'a>,
        binding: u32,
        shaders: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'a> {
        <() as DescriptorBinding>::create_info(param, binding, shaders)
    }
}
pub trait DescriptorSetLayout {
    type CreateInfoParameters<'a>;
    fn create_info<'a>(
        parameters: Self::CreateInfoParameters<'a>,
    ) -> impl HasDescriptorSetCreateInfo + 'a;
}
impl<A: DescriptorBinding> DescriptorSetLayout for (A,) {
    type CreateInfoParameters<'a> = (A::Param<'a>,);
    fn create_info<'a>(
        parameters: Self::CreateInfoParameters<'a>,
    ) -> impl HasDescriptorSetCreateInfo + 'a {
        DescriptorSetCreateInfo([A::create_info(
            parameters.0,
            0,
            vk::ShaderStageFlags::empty(),
        )])
    }
}
pub trait HasDescriptorSetCreateInfo {
    fn create_info(&'_ self) -> vk::DescriptorSetLayoutCreateInfo<'_>;
}
pub struct DescriptorSetCreateInfo<'a, const BINDINGS: usize>(
    [vk::DescriptorSetLayoutBinding<'a>; BINDINGS],
);
impl<'a, const BINDINGS: usize> HasDescriptorSetCreateInfo
    for DescriptorSetCreateInfo<'a, BINDINGS>
{
    fn create_info<'this>(&'this self) -> vk::DescriptorSetLayoutCreateInfo<'this> {
        vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.0)
            .flags(vk::DescriptorSetLayoutCreateFlags::empty())
    }
}
