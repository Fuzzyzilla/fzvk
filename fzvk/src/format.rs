use super::vk;
pub trait Format {
    const FORMAT: vk::Format;
    /// Size of an unpacked texel, in bytes.
    const TEXEL_SIZE: u32;
    /// Dimesions of a compressed block, in texels, WidthxHeightxDepth. If not a
    /// block format, the block size is 1x1x1.
    const BLOCK_DIMENSIONS: [u32; 3];
    /// Size of a compressed block, in bytes. If not a block format, the block
    /// size is equivalent to [`Format::TEXEL_SIZE`].
    const BLOCK_SIZE: u32;
}
/// The kind of data a format stores. Formats may have several aspects, notably
/// [`Depth`] and [`Stencil`] formats are often seen together.
pub trait Aspect {
    const ASPECT: vk::ImageAspectFlags;
}
pub struct Color;
impl Aspect for Color {
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;
}
pub struct Depth;
impl Aspect for Depth {
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::DEPTH;
}
pub struct Stencil;
impl Aspect for Stencil {
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::STENCIL;
}
/// Implemented for all types which have a given aspect flag. For example,
/// `D16Unorm : HasAspect<Depth>`
pub trait HasAspect<A: Aspect> {}
impl<Format: ColorFormat> HasAspect<Color> for Format {}
impl<Format: DepthFormat> HasAspect<Depth> for Format {}
impl<Format: StencilFormat> HasAspect<Stencil> for Format {}

/// Implemented between two formats if the formats are defined to be
/// [*compatible*](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#formats-compatibility-classes).
///
/// Images of format `A` can have views of format `B` iff `A:
/// CompatibleWith<B>`.
///
/// Color formats are compatible if they satisfy the following:
/// * They are both an uncompressed format, and
///     * They occupy the same number of [bits per texel](Format::TEXEL_SIZE).
///     * They are both an alpha-only format, or are both not an alpha-only
///       format (e.g. [`R8_UNORM`] and [`A8_UNORM`] are *not* compatible.)
/// * They are both an compressed format, and
///     * They have the same compression scheme.
///     * They have the same [block dimensions](Format::BLOCK_DIMENSIONS).
///     * They have the same [block bitdepth](Format::BLOCK_SIZE).
///     * (Effectively, they only differ in the unpacked representation of
///       texels)
///
/// Depth/Stencil formats are only compatible with themselves.
pub trait CompatibleWith<Other: Format> {}
pub trait ColorFormat: Format {}
pub trait DepthFormat: Format {}
pub trait StencilFormat: Format {}
pub trait DepthStencilFormat: DepthFormat + StencilFormat {}

macro_rules! color_nonblock_formats {
    [$($name:ident {FORMAT = $format:expr, SIZE = $size:expr}),+$(,)?] => {
        $(
            // Fixme
            #[allow(non_camel_case_types)]
            pub struct $name;
            impl $crate::format::Format for $name {
                const FORMAT : ::ash::vk::Format = $format;
                const TEXEL_SIZE : u32 = $size;
                const BLOCK_DIMENSIONS : [u32; 3] = [1;3];
                const BLOCK_SIZE : u32 = $size;
            }
            impl $crate::format::ColorFormat for $name {}
        )+
    };
}
color_nonblock_formats! {
    R4G4_UNORM_PACK8 {FORMAT = vk::Format::R4G4_UNORM_PACK8, SIZE = 1},
    R4G4B4A4_UNORM_PACK16 {FORMAT = vk::Format::R4G4B4A4_UNORM_PACK16, SIZE = 2},
    B4G4R4A4_UNORM_PACK16 {FORMAT = vk::Format::B4G4R4A4_UNORM_PACK16, SIZE = 2},
    R5G6B5_UNORM_PACK16 {FORMAT = vk::Format::R5G6B5_UNORM_PACK16, SIZE = 2},
    B5G6R5_UNORM_PACK16 {FORMAT = vk::Format::B5G6R5_UNORM_PACK16, SIZE = 2},
    R5G5B5A1_UNORM_PACK16 {FORMAT = vk::Format::R5G5B5A1_UNORM_PACK16, SIZE = 2},
    B5G5R5A1_UNORM_PACK16 {FORMAT = vk::Format::B5G5R5A1_UNORM_PACK16, SIZE = 2},
    A1R5G5B5_UNORM_PACK16 {FORMAT = vk::Format::A1R5G5B5_UNORM_PACK16, SIZE = 2},
    R8_UNORM {FORMAT = vk::Format::R8_UNORM, SIZE = 1},
    R8_SNORM {FORMAT = vk::Format::R8_SNORM, SIZE = 1},
    R8_USCALED {FORMAT = vk::Format::R8_USCALED, SIZE = 1},
    R8_SSCALED {FORMAT = vk::Format::R8_SSCALED, SIZE = 1},
    R8_UINT {FORMAT = vk::Format::R8_UINT, SIZE = 1},
    R8_SINT {FORMAT = vk::Format::R8_SINT, SIZE = 1},
    R8_SRGB {FORMAT = vk::Format::R8_SRGB, SIZE = 1},
    R8G8_UNORM {FORMAT = vk::Format::R8G8_UNORM, SIZE = 2},
    R8G8_SNORM {FORMAT = vk::Format::R8G8_SNORM, SIZE = 2},
    R8G8_USCALED {FORMAT = vk::Format::R8G8_USCALED, SIZE = 2},
    R8G8_SSCALED {FORMAT = vk::Format::R8G8_SSCALED, SIZE = 2},
    R8G8_UINT {FORMAT = vk::Format::R8G8_UINT, SIZE = 2},
    R8G8_SINT {FORMAT = vk::Format::R8G8_SINT, SIZE = 2},
    R8G8_SRGB {FORMAT = vk::Format::R8G8_SRGB, SIZE = 2},
    R8G8B8_UNORM {FORMAT = vk::Format::R8G8B8_UNORM, SIZE = 3},
    R8G8B8_SNORM {FORMAT = vk::Format::R8G8B8_SNORM, SIZE = 3},
    R8G8B8_USCALED {FORMAT = vk::Format::R8G8B8_USCALED, SIZE = 3},
    R8G8B8_SSCALED {FORMAT = vk::Format::R8G8B8_SSCALED, SIZE = 3},
    R8G8B8_UINT {FORMAT = vk::Format::R8G8B8_UINT, SIZE = 3},
    R8G8B8_SINT {FORMAT = vk::Format::R8G8B8_SINT, SIZE = 3},
    R8G8B8_SRGB {FORMAT = vk::Format::R8G8B8_SRGB, SIZE = 3},
    B8G8R8_UNORM {FORMAT = vk::Format::B8G8R8_UNORM, SIZE = 3},
    B8G8R8_SNORM {FORMAT = vk::Format::B8G8R8_SNORM, SIZE = 3},
    B8G8R8_USCALED {FORMAT = vk::Format::B8G8R8_USCALED, SIZE = 3},
    B8G8R8_SSCALED {FORMAT = vk::Format::B8G8R8_SSCALED, SIZE = 3},
    B8G8R8_UINT {FORMAT = vk::Format::B8G8R8_UINT, SIZE = 3},
    B8G8R8_SINT {FORMAT = vk::Format::B8G8R8_SINT, SIZE = 3},
    B8G8R8_SRGB {FORMAT = vk::Format::B8G8R8_SRGB, SIZE = 3},
    R8G8B8A8_UNORM {FORMAT = vk::Format::R8G8B8A8_UNORM, SIZE = 4},
    R8G8B8A8_SNORM {FORMAT = vk::Format::R8G8B8A8_SNORM, SIZE = 4},
    R8G8B8A8_USCALED {FORMAT = vk::Format::R8G8B8A8_USCALED, SIZE = 4},
    R8G8B8A8_SSCALED {FORMAT = vk::Format::R8G8B8A8_SSCALED, SIZE = 4},
    R8G8B8A8_UINT {FORMAT = vk::Format::R8G8B8A8_UINT, SIZE = 4},
    R8G8B8A8_SINT {FORMAT = vk::Format::R8G8B8A8_SINT, SIZE = 4},
    R8G8B8A8_SRGB {FORMAT = vk::Format::R8G8B8A8_SRGB, SIZE = 4},
    B8G8R8A8_UNORM {FORMAT = vk::Format::B8G8R8A8_UNORM, SIZE = 4},
    B8G8R8A8_SNORM {FORMAT = vk::Format::B8G8R8A8_SNORM, SIZE = 4},
    B8G8R8A8_USCALED {FORMAT = vk::Format::B8G8R8A8_USCALED, SIZE = 4},
    B8G8R8A8_SSCALED {FORMAT = vk::Format::B8G8R8A8_SSCALED, SIZE = 4},
    B8G8R8A8_UINT {FORMAT = vk::Format::B8G8R8A8_UINT, SIZE = 4},
    B8G8R8A8_SINT {FORMAT = vk::Format::B8G8R8A8_SINT, SIZE = 4},
    B8G8R8A8_SRGB {FORMAT = vk::Format::B8G8R8A8_SRGB, SIZE = 4},
    A8B8G8R8_UNORM_PACK32 {FORMAT = vk::Format::A8B8G8R8_UNORM_PACK32, SIZE = 4},
    A8B8G8R8_SNORM_PACK32 {FORMAT = vk::Format::A8B8G8R8_SNORM_PACK32, SIZE = 4},
    A8B8G8R8_USCALED_PACK32 {FORMAT = vk::Format::A8B8G8R8_USCALED_PACK32, SIZE = 4},
    A8B8G8R8_SSCALED_PACK32 {FORMAT = vk::Format::A8B8G8R8_SSCALED_PACK32, SIZE = 4},
    A8B8G8R8_UINT_PACK32 {FORMAT = vk::Format::A8B8G8R8_UINT_PACK32, SIZE = 4},
    A8B8G8R8_SINT_PACK32 {FORMAT = vk::Format::A8B8G8R8_SINT_PACK32, SIZE = 4},
    A8B8G8R8_SRGB_PACK32 {FORMAT = vk::Format::A8B8G8R8_SRGB_PACK32, SIZE = 4},
    A2R10G10B10_UNORM_PACK32 {FORMAT = vk::Format::A2R10G10B10_UNORM_PACK32, SIZE = 4},
    A2R10G10B10_SNORM_PACK32 {FORMAT = vk::Format::A2R10G10B10_SNORM_PACK32, SIZE = 4},
    A2R10G10B10_USCALED_PACK32 {FORMAT = vk::Format::A2R10G10B10_USCALED_PACK32, SIZE = 4},
    A2R10G10B10_SSCALED_PACK32 {FORMAT = vk::Format::A2R10G10B10_SSCALED_PACK32, SIZE = 4},
    A2R10G10B10_UINT_PACK32 {FORMAT = vk::Format::A2R10G10B10_UINT_PACK32, SIZE = 4},
    A2R10G10B10_SINT_PACK32 {FORMAT = vk::Format::A2R10G10B10_SINT_PACK32, SIZE = 4},
    A2B10G10R10_UNORM_PACK32 {FORMAT = vk::Format::A2B10G10R10_UNORM_PACK32, SIZE = 4},
    A2B10G10R10_SNORM_PACK32 {FORMAT = vk::Format::A2B10G10R10_SNORM_PACK32, SIZE = 4},
    A2B10G10R10_USCALED_PACK32 {FORMAT = vk::Format::A2B10G10R10_USCALED_PACK32, SIZE = 4},
    A2B10G10R10_SSCALED_PACK32 {FORMAT = vk::Format::A2B10G10R10_SSCALED_PACK32, SIZE = 4},
    A2B10G10R10_UINT_PACK32 {FORMAT = vk::Format::A2B10G10R10_UINT_PACK32, SIZE = 4},
    A2B10G10R10_SINT_PACK32 {FORMAT = vk::Format::A2B10G10R10_SINT_PACK32, SIZE = 4},
    R16_UNORM {FORMAT = vk::Format::R16_UNORM, SIZE = 2},
    R16_SNORM {FORMAT = vk::Format::R16_SNORM, SIZE = 2},
    R16_USCALED {FORMAT = vk::Format::R16_USCALED, SIZE = 2},
    R16_SSCALED {FORMAT = vk::Format::R16_SSCALED, SIZE = 2},
    R16_UINT {FORMAT = vk::Format::R16_UINT, SIZE = 2},
    R16_SINT {FORMAT = vk::Format::R16_SINT, SIZE = 2},
    R16_SFLOAT {FORMAT = vk::Format::R16_SFLOAT, SIZE = 2},
    R16G16_UNORM {FORMAT = vk::Format::R16G16_UNORM, SIZE = 4},
    R16G16_SNORM {FORMAT = vk::Format::R16G16_SNORM, SIZE = 4},
    R16G16_USCALED {FORMAT = vk::Format::R16G16_USCALED, SIZE = 4},
    R16G16_SSCALED {FORMAT = vk::Format::R16G16_SSCALED, SIZE = 4},
    R16G16_UINT {FORMAT = vk::Format::R16G16_UINT, SIZE = 4},
    R16G16_SINT {FORMAT = vk::Format::R16G16_SINT, SIZE = 4},
    R16G16_SFLOAT {FORMAT = vk::Format::R16G16_SFLOAT, SIZE = 4},
    R16G16B16_UNORM {FORMAT = vk::Format::R16G16B16_UNORM, SIZE = 6},
    R16G16B16_SNORM {FORMAT = vk::Format::R16G16B16_SNORM, SIZE = 6},
    R16G16B16_USCALED {FORMAT = vk::Format::R16G16B16_USCALED, SIZE = 6},
    R16G16B16_SSCALED {FORMAT = vk::Format::R16G16B16_SSCALED, SIZE = 6},
    R16G16B16_UINT {FORMAT = vk::Format::R16G16B16_UINT, SIZE = 6},
    R16G16B16_SINT {FORMAT = vk::Format::R16G16B16_SINT, SIZE = 6},
    R16G16B16_SFLOAT {FORMAT = vk::Format::R16G16B16_SFLOAT, SIZE = 6},
    R16G16B16A16_UNORM {FORMAT = vk::Format::R16G16B16A16_UNORM, SIZE = 8},
    R16G16B16A16_SNORM {FORMAT = vk::Format::R16G16B16A16_SNORM, SIZE = 8},
    R16G16B16A16_USCALED {FORMAT = vk::Format::R16G16B16A16_USCALED, SIZE = 8},
    R16G16B16A16_SSCALED {FORMAT = vk::Format::R16G16B16A16_SSCALED, SIZE = 8},
    R16G16B16A16_UINT {FORMAT = vk::Format::R16G16B16A16_UINT, SIZE = 8},
    R16G16B16A16_SINT {FORMAT = vk::Format::R16G16B16A16_SINT, SIZE = 8},
    R16G16B16A16_SFLOAT {FORMAT = vk::Format::R16G16B16A16_SFLOAT, SIZE = 8},
    R32_UINT {FORMAT = vk::Format::R32_UINT, SIZE = 4},
    R32_SINT {FORMAT = vk::Format::R32_SINT, SIZE = 4},
    R32_SFLOAT {FORMAT = vk::Format::R32_SFLOAT, SIZE = 4},
    R32G32_UINT {FORMAT = vk::Format::R32G32_UINT, SIZE = 8},
    R32G32_SINT {FORMAT = vk::Format::R32G32_SINT, SIZE = 8},
    R32G32_SFLOAT {FORMAT = vk::Format::R32G32_SFLOAT, SIZE = 8},
    R32G32B32_UINT {FORMAT = vk::Format::R32G32B32_UINT, SIZE = 12},
    R32G32B32_SINT {FORMAT = vk::Format::R32G32B32_SINT, SIZE = 12},
    R32G32B32_SFLOAT {FORMAT = vk::Format::R32G32B32_SFLOAT, SIZE = 12},
    R32G32B32A32_UINT {FORMAT = vk::Format::R32G32B32A32_UINT, SIZE = 16},
    R32G32B32A32_SINT {FORMAT = vk::Format::R32G32B32A32_SINT, SIZE = 16},
    R32G32B32A32_SFLOAT {FORMAT = vk::Format::R32G32B32A32_SFLOAT, SIZE = 16},
    R64_UINT {FORMAT = vk::Format::R64_UINT, SIZE = 8},
    R64_SINT {FORMAT = vk::Format::R64_SINT, SIZE = 8},
    R64_SFLOAT {FORMAT = vk::Format::R64_SFLOAT, SIZE = 8},
    R64G64_UINT {FORMAT = vk::Format::R64G64_UINT, SIZE = 16},
    R64G64_SINT {FORMAT = vk::Format::R64G64_SINT, SIZE = 16},
    R64G64_SFLOAT {FORMAT = vk::Format::R64G64_SFLOAT, SIZE = 16},
    R64G64B64_UINT {FORMAT = vk::Format::R64G64B64_UINT, SIZE = 24},
    R64G64B64_SINT {FORMAT = vk::Format::R64G64B64_SINT, SIZE = 24},
    R64G64B64_SFLOAT {FORMAT = vk::Format::R64G64B64_SFLOAT, SIZE = 24},
    R64G64B64A64_UINT {FORMAT = vk::Format::R64G64B64A64_UINT, SIZE = 32},
    R64G64B64A64_SINT {FORMAT = vk::Format::R64G64B64A64_SINT, SIZE = 32},
    R64G64B64A64_SFLOAT {FORMAT = vk::Format::R64G64B64A64_SFLOAT, SIZE = 32},
    B10G11R11_UFLOAT_PACK32 {FORMAT = vk::Format::B10G11R11_UFLOAT_PACK32, SIZE = 4},
    E5B9G9R9_UFLOAT_PACK32 {FORMAT = vk::Format::E5B9G9R9_UFLOAT_PACK32, SIZE = 4},
}
/*
macro_rules! compatibility_class {
    ($first:ident, $rest:tt?) => {
        compatibility_class_for_each! {$first, $rest}
        compatibility_class! {$rest}
    };
    ($first:ident$(,)?) => {};
}*/
/// Takes a comma separated set of formats, {a, b, c, d, ..., z}, and implements
/// CompatibleWith for pairs (a,b), (a,c), (a,d), ... (a,z).
macro_rules! compatibility_class_for_each {
    {$first:ident$(,)?} => {};
    ($first:ident, $next:ident $(, $rest:ident)* $(,)?) => {
        impl $crate::format::CompatibleWith<$next> for $first {}
        impl $crate::format::CompatibleWith<$first> for $next {}
        // Call again, skipping `next`
        compatibility_class_for_each!{$first, $($rest),*}
    };
}
/// Takes a comma separated set of formats, and implements CompatibleWith<A> for
/// B for each A and B in the set, excluding the Identity (A for A).
macro_rules! compatibility_class {
    () => {};
    ($first:ident$(,)?) => {};
    ($first:ident $(, $more:ident)* $(,)?) => {
        // Implement the first as compatible with the rest
        compatibility_class_for_each!{$first, $($more),*}
        // Do it again, discarding the first, so now the second is compatible
        // with the rest...
        compatibility_class!{$($more),+}
    };
}
impl<T: Format> CompatibleWith<T> for T {}
// https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#formats-compatibility-classes
compatibility_class! {
    //R8_BOOL_ARM,
    R4G4_UNORM_PACK8,
    R8_UNORM,
    R8_SNORM,
    R8_USCALED,
    R8_SSCALED,
    R8_UINT,
    R8_SINT,
    R8_SRGB
}
compatibility_class! {
    //A1B5G5R5_UNORM_PACK16, R10X6_UNORM_PACK16, R12X4_UNORM_PACK16,
    //A4R4G4B4_UNORM_PACK16, A4B4G4R4_UNORM_PACK16, R10X6_UINT_PACK16_ARM,
    //R12X4_UINT_PACK16_ARM, R14X2_UINT_PACK16_ARM, R14X2_UNORM_PACK16_ARM,
    R4G4B4A4_UNORM_PACK16,
    B4G4R4A4_UNORM_PACK16,
    R5G6B5_UNORM_PACK16,
    B5G6R5_UNORM_PACK16,
    R5G5B5A1_UNORM_PACK16,
    B5G5R5A1_UNORM_PACK16,
    A1R5G5B5_UNORM_PACK16,
    R8G8_UNORM,
    R8G8_SNORM,
    R8G8_USCALED,
    R8G8_SSCALED,
    R8G8_UINT,
    R8G8_SINT,
    R8G8_SRGB,
    R16_UNORM,
    R16_SNORM,
    R16_USCALED,
    R16_SSCALED,
    R16_UINT,
    R16_SINT,
    R16_SFLOAT
}
compatibility_class! {
    //A8_UNORM
}
compatibility_class! {
    R8G8B8_UNORM,
    R8G8B8_SNORM,
    R8G8B8_USCALED,
    R8G8B8_SSCALED,
    R8G8B8_UINT,
    R8G8B8_SINT,
    R8G8B8_SRGB,
    B8G8R8_UNORM,
    B8G8R8_SNORM,
    B8G8R8_USCALED,
    B8G8R8_SSCALED,
    B8G8R8_UINT,
    B8G8R8_SINT,
    B8G8R8_SRGB
}
compatibility_class! {
    //R10X6G10X6_UNORM_2PACK16, R12X4G12X4_UNORM_2PACK16, R16G16_SFIXED5_NV,
    //R10X6G10X6_UINT_2PACK16_ARM, R12X4G12X4_UINT_2PACK16_ARM,
    //R14X2G14X2_UINT_2PACK16_ARM, R14X2G14X2_UNORM_2PACK16_ARM,
    R8G8B8A8_UNORM,
    R8G8B8A8_SNORM,
    R8G8B8A8_USCALED,
    R8G8B8A8_SSCALED,
    R8G8B8A8_UINT,
    R8G8B8A8_SINT,
    R8G8B8A8_SRGB,
    B8G8R8A8_UNORM,
    B8G8R8A8_SNORM,
    B8G8R8A8_USCALED,
    B8G8R8A8_SSCALED,
    B8G8R8A8_UINT,
    B8G8R8A8_SINT,
    B8G8R8A8_SRGB,
    A8B8G8R8_UNORM_PACK32,
    A8B8G8R8_SNORM_PACK32,
    A8B8G8R8_USCALED_PACK32,
    A8B8G8R8_SSCALED_PACK32,
    A8B8G8R8_UINT_PACK32,
    A8B8G8R8_SINT_PACK32,
    A8B8G8R8_SRGB_PACK32,
    A2R10G10B10_UNORM_PACK32,
    A2R10G10B10_SNORM_PACK32,
    A2R10G10B10_USCALED_PACK32,
    A2R10G10B10_SSCALED_PACK32,
    A2R10G10B10_UINT_PACK32,
    A2R10G10B10_SINT_PACK32,
    A2B10G10R10_UNORM_PACK32,
    A2B10G10R10_SNORM_PACK32,
    A2B10G10R10_USCALED_PACK32,
    A2B10G10R10_SSCALED_PACK32,
    A2B10G10R10_UINT_PACK32,
    A2B10G10R10_SINT_PACK32,
    R16G16_UNORM,
    R16G16_SNORM,
    R16G16_USCALED,
    R16G16_SSCALED,
    R16G16_UINT,
    R16G16_SINT,
    R16G16_SFLOAT,
    R32_UINT,
    R32_SINT,
    R32_SFLOAT,
    B10G11R11_UFLOAT_PACK32,
    E5B9G9R9_UFLOAT_PACK32
}
compatibility_class! {
    R16G16B16_UNORM,
    R16G16B16_SNORM,
    R16G16B16_USCALED,
    R16G16B16_SSCALED,
    R16G16B16_UINT,
    R16G16B16_SINT,
    R16G16B16_SFLOAT
}
compatibility_class! {
    R16G16B16A16_UNORM,
    R16G16B16A16_SNORM,
    R16G16B16A16_USCALED,
    R16G16B16A16_SSCALED,
    R16G16B16A16_UINT,
    R16G16B16A16_SINT,
    R16G16B16A16_SFLOAT,
    R32G32_UINT,
    R32G32_SINT,
    R32G32_SFLOAT,
    R64_UINT,
    R64_SINT,
    R64_SFLOAT
}
compatibility_class! {
    R32G32B32_UINT,
    R32G32B32_SINT,
    R32G32B32_SFLOAT
}
compatibility_class! {
    R32G32B32A32_UINT,
    R32G32B32A32_SINT,
    R32G32B32A32_SFLOAT,
    R64G64_UINT,
    R64G64_SINT,
    R64G64_SFLOAT
}
compatibility_class! {
    R64G64B64_UINT,
    R64G64B64_SINT,
    R64G64B64_SFLOAT
}
compatibility_class! {
    R64G64B64A64_UINT,
    R64G64B64A64_SINT,
    R64G64B64A64_SFLOAT
}
compatibility_class! {
    D16_UNORM
}
compatibility_class! {
    X8_D24_UNORM_PACK32
}
compatibility_class! {
    D32_SFLOAT
}
compatibility_class! {
    S8_UINT
}
compatibility_class! {
    D16_UNORM_S8_UINT
}
compatibility_class! {
    D24_UNORM_S8_UINT
}
compatibility_class! {
    D32_SFLOAT_S8_UINT
}
/*
compatibility_class! {
    BC1_RGB_UNORM_BLOCK,
    BC1_RGB_SRGB_BLOCK
}
compatibility_class! {
    BC1_RGBA_UNORM_BLOCK,
    BC1_RGBA_SRGB_BLOCK
}
compatibility_class! {
    BC2_UNORM_BLOCK,
    BC2_SRGB_BLOCK
}
compatibility_class! {
    BC3_UNORM_BLOCK,
    BC3_SRGB_BLOCK
}
compatibility_class! {
    BC4_UNORM_BLOCK,
    BC4_SNORM_BLOCK
}
compatibility_class! {
    BC5_UNORM_BLOCK,
    BC5_SNORM_BLOCK
}
compatibility_class! {
    BC6H_UFLOAT_BLOCK,
    BC6H_SFLOAT_BLOCK
}
compatibility_class! {
    BC7_UNORM_BLOCK,
    BC7_SRGB_BLOCK
}
compatibility_class! {
    ETC2_R8G8B8_UNORM_BLOCK,
    ETC2_R8G8B8_SRGB_BLOCK
}
compatibility_class! {
    ETC2_R8G8B8A1_UNORM_BLOCK,
    ETC2_R8G8B8A1_SRGB_BLOCK
}
compatibility_class! {
    ETC2_R8G8B8A8_UNORM_BLOCK,
    ETC2_R8G8B8A8_SRGB_BLOCK
}
compatibility_class! {
    EAC_R11_UNORM_BLOCK,
    EAC_R11_SNORM_BLOCK
}
compatibility_class! {
    EAC_R11G11_UNORM_BLOCK,
    EAC_R11G11_SNORM_BLOCK
}
compatibility_class! {
    ASTC_4x4_SFLOAT_BLOCK,
    ASTC_4x4_UNORM_BLOCK,
    ASTC_4x4_SRGB_BLOCK
}
compatibility_class! {
    ASTC_5x4_SFLOAT_BLOCK,
    ASTC_5x4_UNORM_BLOCK,
    ASTC_5x4_SRGB_BLOCK
}
compatibility_class! {
    ASTC_5x5_SFLOAT_BLOCK,
    ASTC_5x5_UNORM_BLOCK,
    ASTC_5x5_SRGB_BLOCK
}
compatibility_class! {
    ASTC_6x5_SFLOAT_BLOCK,
    ASTC_6x5_UNORM_BLOCK,
    ASTC_6x5_SRGB_BLOCK
}
compatibility_class! {
    ASTC_6x6_SFLOAT_BLOCK,
    ASTC_6x6_UNORM_BLOCK,
    ASTC_6x6_SRGB_BLOCK
}
compatibility_class! {
    ASTC_8x5_SFLOAT_BLOCK,
    ASTC_8x5_UNORM_BLOCK,
    ASTC_8x5_SRGB_BLOCK
}
compatibility_class! {
    ASTC_8x6_SFLOAT_BLOCK,
    ASTC_8x6_UNORM_BLOCK,
    ASTC_8x6_SRGB_BLOCK
}
compatibility_class! {
    ASTC_8x8_SFLOAT_BLOCK,
    ASTC_8x8_UNORM_BLOCK,
    ASTC_8x8_SRGB_BLOCK
}
compatibility_class! {
    ASTC_10x5_SFLOAT_BLOCK,
    ASTC_10x5_UNORM_BLOCK,
    ASTC_10x5_SRGB_BLOCK
}
compatibility_class! {
    ASTC_10x6_SFLOAT_BLOCK,
    ASTC_10x6_UNORM_BLOCK,
    ASTC_10x6_SRGB_BLOCK
}
compatibility_class! {
    ASTC_10x8_SFLOAT_BLOCK,
    ASTC_10x8_UNORM_BLOCK,
    ASTC_10x8_SRGB_BLOCK
}
compatibility_class! {
    ASTC_10x10_SFLOAT_BLOCK,
    ASTC_10x10_UNORM_BLOCK,
    ASTC_10x10_SRGB_BLOCK
}
compatibility_class! {
    ASTC_12x10_SFLOAT_BLOCK,
    ASTC_12x10_UNORM_BLOCK,
    ASTC_12x10_SRGB_BLOCK
}
compatibility_class! {
    ASTC_12x12_SFLOAT_BLOCK,
    ASTC_12x12_UNORM_BLOCK,
    ASTC_12x12_SRGB_BLOCK
}
compatibility_class! {
    G8B8G8R8_422_UNORM
}
compatibility_class! {
    B8G8R8G8_422_UNORM
}
compatibility_class! {
    G8_B8_R8_3PLANE_420_UNORM
}
compatibility_class! {
    G8_B8R8_2PLANE_420_UNORM
}
compatibility_class! {
    G8_B8_R8_3PLANE_422_UNORM
}
compatibility_class! {
    G8_B8R8_2PLANE_422_UNORM
}
compatibility_class! {
    G8_B8_R8_3PLANE_444_UNORM
}
compatibility_class! {
    R10X6G10X6B10X6A10X6_UNORM_4PACK16,
    R10X6G10X6B10X6A10X6_UINT_4PACK16_ARM
}
compatibility_class! {
    G10X6B10X6G10X6R10X6_422_UNORM_4PACK16
}
compatibility_class! {
    B10X6G10X6R10X6G10X6_422_UNORM_4PACK16
}
compatibility_class! {
    G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16
}
compatibility_class! {
    G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
}
compatibility_class! {
    G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16
}
compatibility_class! {
    G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16
}
compatibility_class! {
    G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16
}
compatibility_class! {
    R12X4G12X4B12X4A12X4_UNORM_4PACK16,
    R12X4G12X4B12X4A12X4_UINT_4PACK16_ARM
}
compatibility_class! {
    G12X4B12X4G12X4R12X4_422_UNORM_4PACK16
}
compatibility_class! {
    B12X4G12X4R12X4G12X4_422_UNORM_4PACK16
}
compatibility_class! {
    G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16
}
compatibility_class! {
    G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16
}
compatibility_class! {
    G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16
}
compatibility_class! {
    G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16
}
compatibility_class! {
    G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16
}
compatibility_class! {
    G16B16G16R16_422_UNORM
}
compatibility_class! {
    B16G16R16G16_422_UNORM
}
compatibility_class! {
    G16_B16_R16_3PLANE_420_UNORM
}
compatibility_class! {
    G16_B16R16_2PLANE_420_UNORM
}
compatibility_class! {
    G16_B16_R16_3PLANE_422_UNORM
}
compatibility_class! {
    G16_B16R16_2PLANE_422_UNORM
}
compatibility_class! {
    G16_B16_R16_3PLANE_444_UNORM
}
compatibility_class! {
    PVRTC1_2BPP_UNORM_BLOCK_IMG,
    PVRTC1_2BPP_SRGB_BLOCK_IMG
}
compatibility_class! {
    PVRTC1_4BPP_UNORM_BLOCK_IMG,
    PVRTC1_4BPP_SRGB_BLOCK_IMG
}
compatibility_class! {
    PVRTC2_2BPP_UNORM_BLOCK_IMG,
    PVRTC2_2BPP_SRGB_BLOCK_IMG
}
compatibility_class! {
    PVRTC2_4BPP_UNORM_BLOCK_IMG,
    PVRTC2_4BPP_SRGB_BLOCK_IMG
}
compatibility_class! {
    G8_B8R8_2PLANE_444_UNORM
}
compatibility_class! {
    G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
}
compatibility_class! {
    G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16
}
compatibility_class! {
    G16_B16R16_2PLANE_444_UNORM
}
compatibility_class! {
    R14X2G14X2B14X2A14X2_UINT_4PACK16_ARM,
    R14X2G14X2B14X2A14X2_UNORM_4PACK16_ARM
}
compatibility_class! {
    G14X2_B14X2R14X2_2PLANE_420_UNORM_3PACK16_ARM
}
compatibility_class! {
    G14X2_B14X2R14X2_2PLANE_422_UNORM_3PACK16_ARM
}
*/
