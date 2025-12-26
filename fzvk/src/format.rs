use super::vk;
/// A vulkan vertex or texel format.
pub unsafe trait Format: 'static {
    type Aspect: Aspect;
    /// An opaque type representing the compatibility of this format. Images
    /// views with different formats of the same compatibility can safely alias
    /// each other.
    type CompatibilityClass;
    /// The vulkan format enum value this type represents.
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
    /// The `ImageAspectFlags` this type represents.
    const ASPECT: vk::ImageAspectFlags;
}
/// The `Color` aspect of a format.
pub struct Color;
impl Aspect for Color {
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;
}
/// The `Depth` aspect of a format.
pub struct Depth;
impl Aspect for Depth {
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::DEPTH;
}
/// The `Stencil` aspect of a format.
pub struct Stencil;
impl Aspect for Stencil {
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::STENCIL;
}
/// Implemented for all types which have a given aspect flag. For example,
/// `R8_UNORM: HasAspect<Color>`
pub unsafe trait HasAspect<A: Aspect>: Format {}

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
/// Each Depth/Stencil format is not compatible with any format other than
/// itself.
///
/// Note that portability subset devices may have the further restriction that
/// the number and bitsize of the channels must be the same, which is not
/// reflected in this trait.
pub trait CompatibleWith<Other: Format>: Format {}
/// Implemented for pairs when their [compatibility
/// classes](Format::CompatibilityClass) are identical. This is also implements
/// the identity compatibility, `A: CompatibleWith<A>`
impl<A: Format, B: Format> CompatibleWith<B> for A where
    A::CompatibilityClass: IsSame<B::CompatibilityClass>
{
}
/// Detail of [`CompatibleWith`]. `X: IsSame<Y>` is only implemented if X and Y
/// are the same type. [`CompatibleWith`] Can be implemented without this trait,
/// but this makes the error messages much more readable.
trait IsSame<Other> {}
impl<T> IsSame<T> for T {}

/// Defines many formats and their compatibility classes.
/// ```ignore
/// {
///     class CompatibilityClassName {
///         FormatName = {FORMAT = <vk format constant>, SIZE = <size in bytes>},
///         ...more formats in this class
///     },
///     ...more classes
/// }
/// ```
macro_rules! color_nonblock_formats {
    [
        $(

            $(
                #[$class_meta:meta]
            )*
            class $class:ident {
                $(
                    $(
                        #[$format_meta:meta]
                    )*
                    $name:ident {FORMAT = $format:expr, SIZE = $size:expr}
                ),*
                $(,)?
            }
        ),*
        $(,)?
    ] => {
        $(
            $(#[$class_meta])*
            pub struct $class;
            $(
                // Fixme
                #[allow(non_camel_case_types)]
                $(#[$format_meta])*
                pub struct $name;
                unsafe impl $crate::format::Format for $name {
                    type Aspect = Color;
                    type CompatibilityClass = $class;
                    const FORMAT : ::ash::vk::Format = $format;
                    const TEXEL_SIZE : u32 = $size;
                    const BLOCK_DIMENSIONS : [u32; 3] = [1;3];
                    const BLOCK_SIZE : u32 = $size;
                }
                unsafe impl $crate::format::HasAspect<$crate::format::Color> for $name {}
            )*
        )*
    };
}

color_nonblock_formats! {
    /// Marker for the Compatibility class of 8-bit alpha-only formats.
    class Alpha8 {
        /// Requires `KHR_maintainance5` or `VK1.4`
        A8_UNORM {FORMAT = vk::Format::A8_UNORM_KHR, SIZE = 1},
    },
    /// Marker for the Compatibility class of 8-bit uncompressed color formats.
    class Color8 {
        R4G4_UNORM_PACK8 {FORMAT = vk::Format::R4G4_UNORM_PACK8, SIZE = 1},
        R8_UNORM {FORMAT = vk::Format::R8_UNORM, SIZE = 1},
        R8_SNORM {FORMAT = vk::Format::R8_SNORM, SIZE = 1},
        R8_USCALED {FORMAT = vk::Format::R8_USCALED, SIZE = 1},
        R8_SSCALED {FORMAT = vk::Format::R8_SSCALED, SIZE = 1},
        R8_UINT {FORMAT = vk::Format::R8_UINT, SIZE = 1},
        R8_SINT {FORMAT = vk::Format::R8_SINT, SIZE = 1},
        R8_SRGB {FORMAT = vk::Format::R8_SRGB, SIZE = 1},
    },
    /// Marker for the Compatibility class of 16-bit uncompressed color formats.
    class Color16 {
        R4G4B4A4_UNORM_PACK16 {FORMAT = vk::Format::R4G4B4A4_UNORM_PACK16, SIZE = 2},
        B4G4R4A4_UNORM_PACK16 {FORMAT = vk::Format::B4G4R4A4_UNORM_PACK16, SIZE = 2},
        R5G6B5_UNORM_PACK16 {FORMAT = vk::Format::R5G6B5_UNORM_PACK16, SIZE = 2},
        B5G6R5_UNORM_PACK16 {FORMAT = vk::Format::B5G6R5_UNORM_PACK16, SIZE = 2},
        R5G5B5A1_UNORM_PACK16 {FORMAT = vk::Format::R5G5B5A1_UNORM_PACK16, SIZE = 2},
        B5G5R5A1_UNORM_PACK16 {FORMAT = vk::Format::B5G5R5A1_UNORM_PACK16, SIZE = 2},
        A1R5G5B5_UNORM_PACK16 {FORMAT = vk::Format::A1R5G5B5_UNORM_PACK16, SIZE = 2},
        R8G8_UNORM {FORMAT = vk::Format::R8G8_UNORM, SIZE = 2},
        R8G8_SNORM {FORMAT = vk::Format::R8G8_SNORM, SIZE = 2},
        R8G8_USCALED {FORMAT = vk::Format::R8G8_USCALED, SIZE = 2},
        R8G8_SSCALED {FORMAT = vk::Format::R8G8_SSCALED, SIZE = 2},
        R8G8_UINT {FORMAT = vk::Format::R8G8_UINT, SIZE = 2},
        R8G8_SINT {FORMAT = vk::Format::R8G8_SINT, SIZE = 2},
        R8G8_SRGB {FORMAT = vk::Format::R8G8_SRGB, SIZE = 2},
        R16_UNORM {FORMAT = vk::Format::R16_UNORM, SIZE = 2},
        R16_SNORM {FORMAT = vk::Format::R16_SNORM, SIZE = 2},
        R16_USCALED {FORMAT = vk::Format::R16_USCALED, SIZE = 2},
        R16_SSCALED {FORMAT = vk::Format::R16_SSCALED, SIZE = 2},
        R16_UINT {FORMAT = vk::Format::R16_UINT, SIZE = 2},
        R16_SINT {FORMAT = vk::Format::R16_SINT, SIZE = 2},
        R16_SFLOAT {FORMAT = vk::Format::R16_SFLOAT, SIZE = 2},
    },
    /// Marker for the Compatibility class of 24-bit uncompressed color formats.
    class Color24 {
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
    },
    /// Marker for the Compatibility class of 32-bit uncompressed color formats.
    class Color32 {
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

        R16G16_UNORM {FORMAT = vk::Format::R16G16_UNORM, SIZE = 4},
        R16G16_SNORM {FORMAT = vk::Format::R16G16_SNORM, SIZE = 4},
        R16G16_USCALED {FORMAT = vk::Format::R16G16_USCALED, SIZE = 4},
        R16G16_SSCALED {FORMAT = vk::Format::R16G16_SSCALED, SIZE = 4},
        R16G16_UINT {FORMAT = vk::Format::R16G16_UINT, SIZE = 4},
        R16G16_SINT {FORMAT = vk::Format::R16G16_SINT, SIZE = 4},
        R16G16_SFLOAT {FORMAT = vk::Format::R16G16_SFLOAT, SIZE = 4},

        R32_UINT {FORMAT = vk::Format::R32_UINT, SIZE = 4},
        R32_SINT {FORMAT = vk::Format::R32_SINT, SIZE = 4},
        R32_SFLOAT {FORMAT = vk::Format::R32_SFLOAT, SIZE = 4},

        B10G11R11_UFLOAT_PACK32 {FORMAT = vk::Format::B10G11R11_UFLOAT_PACK32, SIZE = 4},
        E5B9G9R9_UFLOAT_PACK32 {FORMAT = vk::Format::E5B9G9R9_UFLOAT_PACK32, SIZE = 4},
    },
    /// Marker for the Compatibility class of 48-bit uncompressed color formats.
    class Color48 {
        R16G16B16_UNORM {FORMAT = vk::Format::R16G16B16_UNORM, SIZE = 6},
        R16G16B16_SNORM {FORMAT = vk::Format::R16G16B16_SNORM, SIZE = 6},
        R16G16B16_USCALED {FORMAT = vk::Format::R16G16B16_USCALED, SIZE = 6},
        R16G16B16_SSCALED {FORMAT = vk::Format::R16G16B16_SSCALED, SIZE = 6},
        R16G16B16_UINT {FORMAT = vk::Format::R16G16B16_UINT, SIZE = 6},
        R16G16B16_SINT {FORMAT = vk::Format::R16G16B16_SINT, SIZE = 6},
        R16G16B16_SFLOAT {FORMAT = vk::Format::R16G16B16_SFLOAT, SIZE = 6},
    },
    /// Marker for the Compatibility class of 64-bit uncompressed color formats.
    class Color64 {
        R16G16B16A16_UNORM {FORMAT = vk::Format::R16G16B16A16_UNORM, SIZE = 8},
        R16G16B16A16_SNORM {FORMAT = vk::Format::R16G16B16A16_SNORM, SIZE = 8},
        R16G16B16A16_USCALED {FORMAT = vk::Format::R16G16B16A16_USCALED, SIZE = 8},
        R16G16B16A16_SSCALED {FORMAT = vk::Format::R16G16B16A16_SSCALED, SIZE = 8},
        R16G16B16A16_UINT {FORMAT = vk::Format::R16G16B16A16_UINT, SIZE = 8},
        R16G16B16A16_SINT {FORMAT = vk::Format::R16G16B16A16_SINT, SIZE = 8},
        R16G16B16A16_SFLOAT {FORMAT = vk::Format::R16G16B16A16_SFLOAT, SIZE = 8},

        R32G32_UINT {FORMAT = vk::Format::R32G32_UINT, SIZE = 8},
        R32G32_SINT {FORMAT = vk::Format::R32G32_SINT, SIZE = 8},
        R32G32_SFLOAT {FORMAT = vk::Format::R32G32_SFLOAT, SIZE = 8},

        R64_UINT {FORMAT = vk::Format::R64_UINT, SIZE = 8},
        R64_SINT {FORMAT = vk::Format::R64_SINT, SIZE = 8},
        R64_SFLOAT {FORMAT = vk::Format::R64_SFLOAT, SIZE = 8},
    },
    /// Marker for the Compatibility class of 96-bit uncompressed color formats.
    class Color96 {
        R32G32B32_UINT {FORMAT = vk::Format::R32G32B32_UINT, SIZE = 12},
        R32G32B32_SINT {FORMAT = vk::Format::R32G32B32_SINT, SIZE = 12},
        R32G32B32_SFLOAT {FORMAT = vk::Format::R32G32B32_SFLOAT, SIZE = 12},
    },
    /// Marker for the Compatibility class of 128-bit uncompressed color
    /// formats.
    class Color128 {
        R32G32B32A32_UINT {FORMAT = vk::Format::R32G32B32A32_UINT, SIZE = 16},
        R32G32B32A32_SINT {FORMAT = vk::Format::R32G32B32A32_SINT, SIZE = 16},
        R32G32B32A32_SFLOAT {FORMAT = vk::Format::R32G32B32A32_SFLOAT, SIZE = 16},

        R64G64_UINT {FORMAT = vk::Format::R64G64_UINT, SIZE = 16},
        R64G64_SINT {FORMAT = vk::Format::R64G64_SINT, SIZE = 16},
        R64G64_SFLOAT {FORMAT = vk::Format::R64G64_SFLOAT, SIZE = 16},
    },
    /// Marker for the Compatibility class of 192-bit uncompressed color
    /// formats.
    class Color192 {
        R64G64B64_UINT {FORMAT = vk::Format::R64G64B64_UINT, SIZE = 24},
        R64G64B64_SINT {FORMAT = vk::Format::R64G64B64_SINT, SIZE = 24},
        R64G64B64_SFLOAT {FORMAT = vk::Format::R64G64B64_SFLOAT, SIZE = 24},
    },
    /// Marker for the Compatibility class of 256-bit uncompressed color
    /// formats.
    class Color256 {
        R64G64B64A64_UINT {FORMAT = vk::Format::R64G64B64A64_UINT, SIZE = 32},
        R64G64B64A64_SINT {FORMAT = vk::Format::R64G64B64A64_SINT, SIZE = 32},
        R64G64B64A64_SFLOAT {FORMAT = vk::Format::R64G64B64A64_SFLOAT, SIZE = 32},
    }
}
