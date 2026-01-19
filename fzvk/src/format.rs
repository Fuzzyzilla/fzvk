//! # Formats
use super::vk;
use core::num::NonZero;
/// A vulkan vertex or texel format.
pub unsafe trait Format: 'static + Default {
    /// Performs dual purpose: serves as a marker type for the
    /// available aspects of the image via the [`AspectSuperset`] trait.
    /// Additionally, values of this type specify one-or-more of the aspects of
    /// this image.
    type AspectMask: aspect::AspectMask;
    /// An opaque type representing the compatibility of this format. Images
    /// views with different formats of the same compatibility can safely alias
    /// each other. See the [`CompatibleWith`] trait.
    type CompatibilityClass;
    /// The vulkan format enum value this type represents.
    const FORMAT: vk::Format;
    /// Size of an unpacked texel, in bytes.
    const TEXEL_SIZE: NonZero<u32>;
    /// Dimesions of a compressed block, in texels, WidthxHeightxDepth. If not a
    /// block format, the block size is 1x1x1.
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3];
    /// Size of a compressed block, in bytes. If not a block format, the block
    /// size is equivalent to [`Format::TEXEL_SIZE`].
    const BLOCK_SIZE: NonZero<u32>;
}
pub mod aspect {
    use super::vk;
    /// The kind of data a format stores. Formats may have several aspects,
    /// notably [`Depth`] and [`Stencil`] formats are often seen together.
    pub trait AspectMask {
        const ALL: Self;
        const ALL_FLAGS: vk::ImageAspectFlags;
        fn aspect_mask(&self) -> vk::ImageAspectFlags;
    }
    /// The `Color` aspect of a format.
    #[derive(Default)]
    pub struct Color;
    impl AspectMask for Color {
        const ALL: Self = Self;
        const ALL_FLAGS: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;
        fn aspect_mask(&self) -> vk::ImageAspectFlags {
            vk::ImageAspectFlags::COLOR
        }
    }
    /// The `Depth` aspect of a format.
    #[derive(Default)]
    pub struct Depth;
    impl AspectMask for Depth {
        const ALL: Self = Self;
        const ALL_FLAGS: vk::ImageAspectFlags = vk::ImageAspectFlags::DEPTH;
        fn aspect_mask(&self) -> vk::ImageAspectFlags {
            vk::ImageAspectFlags::DEPTH
        }
    }
    /// The `Stencil` aspect of a format.
    #[derive(Default)]
    pub struct Stencil;
    impl AspectMask for Stencil {
        const ALL: Self = Self;
        const ALL_FLAGS: vk::ImageAspectFlags = vk::ImageAspectFlags::STENCIL;
        fn aspect_mask(&self) -> vk::ImageAspectFlags {
            vk::ImageAspectFlags::STENCIL
        }
    }
    /// Either or both of the `Depth` or `Stencil` aspects of a format.
    #[derive(Copy, Clone)]
    #[repr(u32)]
    pub enum DepthStencil {
        Depth = vk::ImageAspectFlags::DEPTH.as_raw(),
        Stencil = vk::ImageAspectFlags::STENCIL.as_raw(),
        DepthStencil =
            vk::ImageAspectFlags::DEPTH.as_raw() | vk::ImageAspectFlags::STENCIL.as_raw(),
    }
    impl AspectMask for DepthStencil {
        const ALL: Self = Self::DepthStencil;
        const ALL_FLAGS: vk::ImageAspectFlags = vk::ImageAspectFlags::from_raw(
            vk::ImageAspectFlags::DEPTH.as_raw() | vk::ImageAspectFlags::STENCIL.as_raw(),
        );
        fn aspect_mask(&self) -> vk::ImageAspectFlags {
            vk::ImageAspectFlags::from_raw(*self as u32)
        }
    }
    /// Implemented for all Aspect marker traits which have a given aspect flag.
    /// For example, `DepthStencil: AspectSuperset<Depth>`
    pub unsafe trait AspectSupersetOf<A: AspectMask>: AspectMask {}
    unsafe impl<T: AspectMask> AspectSupersetOf<T> for T {}
    unsafe impl AspectSupersetOf<Depth> for DepthStencil {}
    unsafe impl AspectSupersetOf<Stencil> for DepthStencil {}

    pub struct ViewAspects<A: AspectMask>(core::marker::PhantomData<A>);
    impl<A: AspectMask> ViewAspects<A> {
        const FLAGS: vk::ImageAspectFlags = A::ALL_FLAGS;
    }
    impl<A: AspectMask> Default for ViewAspects<A> {
        fn default() -> Self {
            Self(core::marker::PhantomData)
        }
    }
}

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
///     class CompatibilityClassName{SIZE = size_bytes} {
///         FormatName = <vk format constant>,
///         ...more formats in this class
///     },
///     ...more classes
/// }
/// ```
macro_rules! color_nonblock_formats {
    [
        $(
            $(#[$class_meta:meta])*
            class $class:ident{SIZE = $size:expr} {
                $(
                    $(
                        #[$format_meta:meta]
                    )*
                    $name:ident = $format:ident
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
                //#[doc(alias = stringify!($format))]
                $(#[$format_meta])*
                #[derive(Copy, Clone, Default)]
                pub struct $name;
                unsafe impl $crate::format::Format for $name {
                    type AspectMask = $crate::format::aspect::Color;
                    type CompatibilityClass = $class;
                    const FORMAT : ::ash::vk::Format = <::ash::vk::Format>::$format;
                    const TEXEL_SIZE : ::core::num::NonZero<u32> = ::core::num::NonZero::new($size).unwrap();
                    const BLOCK_DIMENSIONS : [::core::num::NonZero<u32>; 3] = [::core::num::NonZero::new(1u32).unwrap();3];
                    const BLOCK_SIZE : ::core::num::NonZero<u32> = ::core::num::NonZero::new($size).unwrap();
                }
            )*
        )*
    };
}
/*
macro_rules! color_nonblock_format_impl {
    ($(#[$meta:meta])*
    $name:ident $class:ident $format:ident $size:literal $alias:literal) => {
        $(#[$meta])*
        #[doc(alias = $alias)]
        pub struct $name;
        unsafe impl $crate::format::Format for $name {
            type AspectMask = Color;
            type CompatibilityClass = $class;
            const FORMAT : ::ash::vk::Format = <::ash::vk::Format>::$format;
            const TEXEL_SIZE : u32 = $size;
            const BLOCK_DIMENSIONS : [u32; 3] = [1;3];
            const BLOCK_SIZE : u32 = $size;
        }
    };
}*/

color_nonblock_formats! {
    /// Marker for the Compatibility class of 8-bit alpha-only formats.
    class Alpha8{SIZE = 1} {
        /// Requires `KHR_maintainance5` or `VK1.4`
        A8Unorm = A8_UNORM_KHR,
    },
    /// Marker for the Compatibility class of 8-bit uncompressed color formats.
    class Color8{SIZE = 1} {
        Rg4 = R4G4_UNORM_PACK8,
        R8 = R8_UNORM,
        R8Inorm = R8_SNORM,
        R8Uscaled = R8_USCALED,
        R8Iscaled = R8_SSCALED,
        R8u = R8_UINT,
        R8i = R8_SINT,
        R8Srgb = R8_SRGB,
    },
    /// Marker for the Compatibility class of 16-bit uncompressed color formats.
    class Color16{SIZE = 2} {
        Rgba4 = R4G4B4A4_UNORM_PACK16,
        Bgra4 = B4G4R4A4_UNORM_PACK16,
        R5G6B5 = R5G6B5_UNORM_PACK16,
        B5G6R5 = B5G6R5_UNORM_PACK16,
        Rgb5A1 = R5G5B5A1_UNORM_PACK16,
        Bgr5A1 = B5G5R5A1_UNORM_PACK16,
        A1Rgb5 = A1R5G5B5_UNORM_PACK16,
        Rg8 = R8G8_UNORM,
        Rg8Inorm = R8G8_SNORM,
        Rg8Uscaled = R8G8_USCALED,
        Rg8Iscaled = R8G8_SSCALED,
        Rg8u = R8G8_UINT,
        Rg8i = R8G8_SINT,
        Rg8Srgb = R8G8_SRGB,
        R16 = R16_UNORM,
        R16Inorm = R16_SNORM,
        R16Uscaled = R16_USCALED,
        R16Iscaled = R16_SSCALED,
        R16u = R16_UINT,
        R16i = R16_SINT,
        R16f = R16_SFLOAT,
    },
    /// Marker for the Compatibility class of 24-bit uncompressed color formats.
    class Color24{SIZE = 3} {
        Rgb8 = R8G8B8_UNORM,
        Rgb8Inorm = R8G8B8_SNORM,
        Rgb8Uscaled = R8G8B8_USCALED,
        Rgb8Iscaled = R8G8B8_SSCALED,
        Rgb8u = R8G8B8_UINT,
        Rgb8i = R8G8B8_SINT,
        Rgb8Srgb = R8G8B8_SRGB,
        Bgr8 = B8G8R8_UNORM,
        Bgr8Inorm = B8G8R8_SNORM,
        Bgr8Uscaled = B8G8R8_USCALED,
        Bgr8Iscaled = B8G8R8_SSCALED,
        Bgr8u = B8G8R8_UINT,
        Bgr8i = B8G8R8_SINT,
        Bgr8Srgb = B8G8R8_SRGB,
    },
    /// Marker for the Compatibility class of 32-bit uncompressed color formats.
    class Color32{SIZE = 4} {
        Rgba8 = R8G8B8A8_UNORM,
        Rgba8Inorm = R8G8B8A8_SNORM,
        Rgba8Uscaled = R8G8B8A8_USCALED,
        Rgba8Iscaled = R8G8B8A8_SSCALED,
        Rgba8u = R8G8B8A8_UINT,
        Rgba8i = R8G8B8A8_SINT,
        Rgba8Srgb = R8G8B8A8_SRGB,
        Bgra8 = B8G8R8A8_UNORM,
        Bgra8Inorm = B8G8R8A8_SNORM,
        Bgra8Uscaled = B8G8R8A8_USCALED,
        Bgra8Iscaled = B8G8R8A8_SSCALED,
        Bgra8u = B8G8R8A8_UINT,
        Bgra8i = B8G8R8A8_SINT,
        Bgra8Srgb = B8G8R8A8_SRGB,
        Abgr8 = A8B8G8R8_UNORM_PACK32,
        Abgr8Inorm = A8B8G8R8_SNORM_PACK32,
        Abgr8Uscaled = A8B8G8R8_USCALED_PACK32,
        Abgr8Iscaled = A8B8G8R8_SSCALED_PACK32,
        Abgr8u = A8B8G8R8_UINT_PACK32,
        Abgr8i = A8B8G8R8_SINT_PACK32,
        Abgr8Srgb = A8B8G8R8_SRGB_PACK32,

        A2Rgb10 = A2R10G10B10_UNORM_PACK32,
        A2Rgb10Inorm = A2R10G10B10_SNORM_PACK32,
        A2Rgb10Uscaled = A2R10G10B10_USCALED_PACK32,
        A2Rgb10Iscaled = A2R10G10B10_SSCALED_PACK32,
        A2Rgb10u = A2R10G10B10_UINT_PACK32,
        A2Rgb10i = A2R10G10B10_SINT_PACK32,
        A2Bgr10 = A2B10G10R10_UNORM_PACK32,
        A2Bgr10Inorm = A2B10G10R10_SNORM_PACK32,
        A2Bgr10Uscaled = A2B10G10R10_USCALED_PACK32,
        A2Bgr10Iscaled = A2B10G10R10_SSCALED_PACK32,
        A2Bgr10u = A2B10G10R10_UINT_PACK32,
        A2Bgr10i = A2B10G10R10_SINT_PACK32,

        Rg16 = R16G16_UNORM,
        Rg16Inorm = R16G16_SNORM,
        Rg16Uscaled = R16G16_USCALED,
        Rg16Iscaled = R16G16_SSCALED,
        Rg16u = R16G16_UINT,
        Rg16i = R16G16_SINT,
        Rg16f = R16G16_SFLOAT,

        R32u = R32_UINT,
        R32i = R32_SINT,
        R32f = R32_SFLOAT,

        B10Gr11Uf = B10G11R11_UFLOAT_PACK32,
        E5Bgr9Uf = E5B9G9R9_UFLOAT_PACK32,
    },
    /// Marker for the Compatibility class of 48-bit uncompressed color formats.
    class Color48{SIZE = 6} {
        Rgb16 = R16G16B16_UNORM,
        Rgb16Inorm = R16G16B16_SNORM,
        Rgb16Uscaled = R16G16B16_USCALED,
        Rgb16Iscaled = R16G16B16_SSCALED,
        Rgb16u = R16G16B16_UINT,
        Rgb16i = R16G16B16_SINT,
        Rgb16f = R16G16B16_SFLOAT,
    },
    /// Marker for the Compatibility class of 64-bit uncompressed color formats.
    class Color64{SIZE = 8} {
        Rgba16 = R16G16B16A16_UNORM,
        Rgba16Inorm = R16G16B16A16_SNORM,
        Rgba16Uscaled = R16G16B16A16_USCALED,
        Rgba16Iscaled = R16G16B16A16_SSCALED,
        Rgba16u = R16G16B16A16_UINT,
        Rgba16i = R16G16B16A16_SINT,
        Rgba16f = R16G16B16A16_SFLOAT,

        Rg32u = R32G32_UINT,
        Rg32i = R32G32_SINT,
        Rg32f = R32G32_SFLOAT,

        R64u = R64_UINT,
        R64i = R64_SINT,
        R64f = R64_SFLOAT,
    },
    /// Marker for the Compatibility class of 96-bit uncompressed color formats.
    class Color96{SIZE = 12} {
        Rgb32u = R32G32B32_UINT,
        Rgb32i = R32G32B32_SINT,
        Rgb32f = R32G32B32_SFLOAT,
    },
    /// Marker for the Compatibility class of 128-bit uncompressed color
    /// formats.
    class Color128{SIZE = 16} {
        Rgba32u = R32G32B32A32_UINT,
        Rgba32i = R32G32B32A32_SINT,
        Rgba32f = R32G32B32A32_SFLOAT,

        Rg64u = R64G64_UINT,
        Rg64i = R64G64_SINT,
        Rg64f = R64G64_SFLOAT,
    },
    /// Marker for the Compatibility class of 192-bit uncompressed color
    /// formats.
    class Color192{SIZE = 24} {
        Rgb64u = R64G64B64_UINT,
        Rgb64i = R64G64B64_SINT,
        Rgb64f = R64G64B64_SFLOAT,
    },
    /// Marker for the Compatibility class of 256-bit uncompressed color
    /// formats.
    class Color256{SIZE = 32} {
        Rgba64u = R64G64B64A64_UINT,
        Rgba64i = R64G64B64A64_SINT,
        Rgba64f = R64G64B64A64_SFLOAT,
    }
}
// ================== Depth and/or Stencil formats =============================
// Didn't bother writing a macro for these lol.
#[derive(Copy, Clone, Default)]
pub struct D16;
unsafe impl Format for D16 {
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3] = [NonZero::new(1).unwrap(); 3];
    const BLOCK_SIZE: NonZero<u32> = Self::TEXEL_SIZE;
    const TEXEL_SIZE: NonZero<u32> = NonZero::new(2).unwrap();
    const FORMAT: vk::Format = vk::Format::D16_UNORM;
    type AspectMask = aspect::Depth;
    type CompatibilityClass = Self;
}
#[derive(Copy, Clone, Default)]
pub struct D16S8u;
unsafe impl Format for D16S8u {
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3] = [NonZero::new(1).unwrap(); 3];
    const BLOCK_SIZE: NonZero<u32> = Self::TEXEL_SIZE;
    const TEXEL_SIZE: NonZero<u32> = NonZero::new(3).unwrap();
    const FORMAT: vk::Format = vk::Format::D16_UNORM_S8_UINT;
    type AspectMask = aspect::DepthStencil;
    type CompatibilityClass = Self;
}
#[derive(Copy, Clone, Default)]
pub struct X8D24;
unsafe impl Format for X8D24 {
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3] = [NonZero::new(1).unwrap(); 3];
    const BLOCK_SIZE: NonZero<u32> = Self::TEXEL_SIZE;
    const TEXEL_SIZE: NonZero<u32> = NonZero::new(4).unwrap();
    const FORMAT: vk::Format = vk::Format::X8_D24_UNORM_PACK32;
    type AspectMask = aspect::Depth;
    type CompatibilityClass = Self;
}
#[derive(Copy, Clone, Default)]
pub struct D24S8u;
unsafe impl Format for D24S8u {
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3] = [NonZero::new(1).unwrap(); 3];
    const BLOCK_SIZE: NonZero<u32> = Self::TEXEL_SIZE;
    const TEXEL_SIZE: NonZero<u32> = NonZero::new(4).unwrap();
    const FORMAT: vk::Format = vk::Format::D24_UNORM_S8_UINT;
    type AspectMask = aspect::DepthStencil;
    type CompatibilityClass = Self;
}
#[derive(Copy, Clone, Default)]
pub struct D32f;
unsafe impl Format for D32f {
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3] = [NonZero::new(1).unwrap(); 3];
    const BLOCK_SIZE: NonZero<u32> = Self::TEXEL_SIZE;
    const TEXEL_SIZE: NonZero<u32> = NonZero::new(4).unwrap();
    const FORMAT: vk::Format = vk::Format::D32_SFLOAT;
    type AspectMask = aspect::Depth;
    type CompatibilityClass = Self;
}
#[derive(Copy, Clone, Default)]
pub struct D32fS8u;
unsafe impl Format for D32fS8u {
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3] = [NonZero::new(1).unwrap(); 3];
    const BLOCK_SIZE: NonZero<u32> = Self::TEXEL_SIZE;
    const TEXEL_SIZE: NonZero<u32> = NonZero::new(5).unwrap();
    const FORMAT: vk::Format = vk::Format::D32_SFLOAT_S8_UINT;
    type AspectMask = aspect::DepthStencil;
    type CompatibilityClass = Self;
}
#[derive(Copy, Clone, Default)]
pub struct S8u;
unsafe impl Format for S8u {
    const BLOCK_DIMENSIONS: [NonZero<u32>; 3] = [NonZero::new(1).unwrap(); 3];
    const BLOCK_SIZE: NonZero<u32> = Self::TEXEL_SIZE;
    const TEXEL_SIZE: NonZero<u32> = NonZero::new(1).unwrap();
    const FORMAT: vk::Format = vk::Format::S8_UINT;
    // The lone user of the `Stencil` type! *yippee sfx*
    type AspectMask = aspect::Stencil;
    type CompatibilityClass = Self;
}
