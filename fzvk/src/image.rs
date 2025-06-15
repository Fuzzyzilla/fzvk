//! `vkImage[View]`
use crate::{
    buffer::{BufferPitch, RowPitch, RowSlicePitch},
    format,
    usage::ImageUsage,
    vk,
};
use core::num::NonZero;

/// Specifies how an image view should source it's channels from the underlying
/// image.
///
/// For example, if the `r` field is set to the `B` swizzle, texel fetches from
/// the image view will pull the red color component from the blue channel of
/// the underlying image.
#[derive(Clone, Copy)]
pub struct ComponentMapping {
    pub r: vk::ComponentSwizzle,
    pub g: vk::ComponentSwizzle,
    pub b: vk::ComponentSwizzle,
    pub a: vk::ComponentSwizzle,
}
impl ComponentMapping {
    /// Identity swizzle, each component is mapped to itself.
    pub const IDENTITY: Self = Self::splat(vk::ComponentSwizzle::IDENTITY);
    /// Make a swizzle with every component mapped to the same `channel`.
    pub const fn splat(channel: vk::ComponentSwizzle) -> Self {
        Self {
            r: channel,
            g: channel,
            b: channel,
            a: channel,
        }
    }
}
impl From<ComponentMapping> for vk::ComponentMapping {
    fn from(value: ComponentMapping) -> Self {
        let ComponentMapping { r, g, b, a } = value;
        vk::ComponentMapping { r, g, b, a }
    }
}
/// The identity swizzle.
impl Default for ComponentMapping {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A trait representing the extent of a image of various dimensionalities.
pub unsafe trait Extent {
    /// The dimensionality of the extent represented by this type. For example,
    /// 2D Array has dimensionality 2.
    type Dim: Dimensionality;
    /// The extrapolated extent3D. All dimensions must be non-zero. Axes beyond
    /// the type's dimensionality should be `1`.
    fn extent(&self) -> vk::Extent3D;
    /// The number of layers, or 1 if not a layered type.
    fn layers(&self) -> NonZero<u32>;
}
/// A trait representing an offset of an image of various dimensionalities.
pub trait Offset {
    /// The value represeting the `(0, 0, 0)` point.
    const ORIGIN: Self;
    /// The dimensionality of the extent represented by this type. For example,
    /// 2D Array has dimensionality 2.
    type Dim: Dimensionality;
    /// The extrapolated extent3D. All dimensions must be non-zero. Axes beyond
    /// the type's dimensionality should be `1`.
    fn offset(&self) -> vk::Offset3D;
}
pub struct Offset1D(pub i32);
impl Offset for Offset1D {
    const ORIGIN: Self = Self(0);
    type Dim = D1;
    fn offset(&self) -> vk::Offset3D {
        vk::Offset3D {
            x: self.0,
            y: 0,
            z: 0,
        }
    }
}
/// The width of a 1-dimensional image.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Extent1D(pub NonZero<u32>);
unsafe impl Extent for Extent1D {
    type Dim = D1;
    fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.0.get(),
            height: 1,
            depth: 1,
        }
    }
    fn layers(&self) -> NonZero<u32> {
        NonZero::new(1).unwrap()
    }
}
/// The width and layers of a 1-dimensional array image.
pub struct Extent1DArray {
    pub width: NonZero<u32>,
    pub layers: ArrayCount,
}
unsafe impl Extent for Extent1DArray {
    type Dim = D1Array;
    fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width.get(),
            height: 1,
            depth: 1,
        }
    }
    fn layers(&self) -> NonZero<u32> {
        self.layers.as_nonzero()
    }
}
/// The width and height of a 2-dimensional array image.
pub struct Extent2DArray {
    pub width: NonZero<u32>,
    pub height: NonZero<u32>,
    pub layers: ArrayCount,
}
impl Extent2D {
    pub fn with_layers(self, layers: ArrayCount) -> Extent2DArray {
        Extent2DArray {
            width: self.width,
            height: self.height,
            layers,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Offset2D {
    x: i32,
    y: i32,
}
impl Offset for Offset2D {
    const ORIGIN: Self = Self { x: 0, y: 0 };
    type Dim = D2;
    fn offset(&self) -> vk::Offset3D {
        vk::Offset3D {
            x: self.x,
            y: self.y,
            z: 0,
        }
    }
}
unsafe impl Extent for Extent2DArray {
    type Dim = D2Array;
    fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width.get(),
            height: self.height.get(),
            depth: 1,
        }
    }
    fn layers(&self) -> NonZero<u32> {
        self.layers.as_nonzero()
    }
}
/// The width and height of a 2-dimensional image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent2D {
    pub width: NonZero<u32>,
    pub height: NonZero<u32>,
}
unsafe impl Extent for Extent2D {
    type Dim = D2;
    fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width.get(),
            height: self.height.get(),
            depth: 1,
        }
    }
    fn layers(&self) -> NonZero<u32> {
        NonZero::new(1).unwrap()
    }
}
/// The width, height, and depth of a 3-dimensional image.
pub struct Extent3D {
    pub width: NonZero<u32>,
    pub height: NonZero<u32>,
    pub depth: NonZero<u32>,
}
unsafe impl Extent for Extent3D {
    type Dim = D3;
    fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width.get(),
            height: self.height.get(),
            depth: self.depth.get(),
        }
    }
    fn layers(&self) -> NonZero<u32> {
        NonZero::new(1).unwrap()
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Offset3D {
    x: i32,
    y: i32,
    z: i32,
}
impl Offset for Offset3D {
    const ORIGIN: Self = Self { x: 0, y: 0, z: 0 };
    type Dim = D3;
    fn offset(&self) -> vk::Offset3D {
        vk::Offset3D {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}
#[derive(Clone, Copy)]
#[repr(transparent)]
/// The number of images in an array, two or greater.
///
/// In vulkan, the only difference between an image and an array image is
/// whether it is created with an array count > 1. Note that this differs for
/// views, where an array view with 1 layer and a regular view are distict
/// concepts.
pub struct ArrayCount(NonZero<u32>);
impl ArrayCount {
    /// The minimum array count.
    pub const TWO: Self = Self::new(2).unwrap();
    /// Create an array count, >= 2 None if an invalid count.
    pub const fn new(layers: u32) -> Option<Self> {
        if layers >= 2 {
            // Safety - just checked.
            Some(unsafe { Self::new_unchecked(layers) })
        } else {
            None
        }
    }
    /// Create an array count without bounds checking.
    /// # Safety
    /// `layers` must be `>= 2`
    pub const unsafe fn new_unchecked(layers: u32) -> Self {
        debug_assert!(layers >= 2);
        Self(NonZero::new_unchecked(layers))
    }
    /// Get the number of layers with guarantee that it is nonzero.
    pub const fn as_nonzero(self) -> NonZero<u32> {
        self.0
    }
    /// Get the number of layers as an integer.
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}
#[repr(transparent)]
pub struct MipCount(pub NonZero<u32>);
impl MipCount {
    pub const ONE: Self = Self(NonZero::<u32>::MIN);
}

/// The number of dimensions in an image, including whether it's an array.
pub trait Dimensionality {
    /// The size in each dimension, plus an array layer count if applicable.
    type Extent: Extent;
    type SubresourceLayers: SubresourceLayers;
    /// An offset into an image of this dimensionality.
    type Offset: Offset;
    /// How an N-dimensional image's addresses should be translated into
    /// 1-dimensional buffer addresses.
    type Pitch: BufferPitch;
    /// Vulkan representation of this dimensionality.
    const IMAGE_TYPE: vk::ImageType;
    const VIEW_TYPE: vk::ImageViewType;
}
pub struct BufferPitchD3;
/// Typestate for a 1-dimensional image.
pub struct D1;
/// Typestate for a 1-dimensional array image.
pub struct D1Array;
/// Typestate for a 2-dimensional image.
pub struct D2;
/// Typestate for a 2-dimensional array image.
pub struct D2Array;
/// Typestate for a 3-dimensional image.
pub struct D3;
impl Dimensionality for D1 {
    type Extent = Extent1D;
    type Offset = Offset1D;
    type Pitch = ();
    type SubresourceLayers = SubresourceMip;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_1D;
}
impl Dimensionality for D1Array {
    type Extent = Extent1DArray;
    type Offset = Offset1D;
    type Pitch = RowPitch;
    type SubresourceLayers = SubresourceMipArray;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_1D_ARRAY;
}
impl Dimensionality for D2 {
    type Extent = Extent2D;
    type Offset = Offset2D;
    type Pitch = RowPitch;
    type SubresourceLayers = SubresourceMip;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_2D;
}
impl Dimensionality for D2Array {
    type Extent = Extent2DArray;
    type Offset = Offset2D;
    type Pitch = RowSlicePitch;
    type SubresourceLayers = SubresourceMipArray;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_2D_ARRAY;
}
impl Dimensionality for D3 {
    type Extent = Extent3D;
    type Offset = Offset3D;
    type Pitch = RowSlicePitch;
    type SubresourceLayers = SubresourceMip;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_3D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_3D;
}
pub trait SubresourceLayers {
    fn subresource_layers(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers;
}
/// A single mip level of a non-array image.
pub struct SubresourceMip(pub u32);
impl SubresourceMip {
    pub const ZERO: Self = Self(0);
}
impl SubresourceLayers for SubresourceMip {
    fn subresource_layers(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: self.0,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}
/// A single mip level of a range of image layers.
pub struct SubresourceMipArray {
    mip: u32,
    start_layer: u32,
    layer_count: NonZero<u32>,
}
impl SubresourceMipArray {
    /// Reference a span of layers at a given mip level.
    ///
    /// If a range is specified without an end, e.g. `0..`,
    /// `VK_REMAINING_LAYERS` is specified. # Panics If layers is an inverted or
    /// empty range.
    pub fn new(mip: u32, layers: impl core::ops::RangeBounds<u32>) -> Self {
        let start_layer = match layers.start_bound() {
            core::ops::Bound::Excluded(&a) => a.checked_add(1).unwrap(),
            core::ops::Bound::Included(&a) => a,
            core::ops::Bound::Unbounded => 0,
        };
        let layer_count = match layers.end_bound() {
            core::ops::Bound::Excluded(&end) => {
                assert!(start_layer < end);
                // Safety - just checked that end is strictly larger, therefore
                // a difference >= 1.
                unsafe { NonZero::new_unchecked(end - start_layer) }
            }
            core::ops::Bound::Included(&end) => {
                assert!(start_layer <= end);
                let count = (end - start_layer)
                    .checked_add(1)
                    .expect("overflow in SubresourceMipArray::new");
                // Unconditional +1, always nonzero.
                unsafe { NonZero::new_unchecked(count) }
            }
            core::ops::Bound::Unbounded => NonZero::new(vk::REMAINING_ARRAY_LAYERS).unwrap(),
        };
        Self {
            mip,
            start_layer,
            layer_count,
        }
    }
}
impl SubresourceLayers for SubresourceMipArray {
    fn subresource_layers(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: self.mip,
            base_array_layer: self.start_layer,
            layer_count: self.layer_count.get(),
        }
    }
}
/*
pub trait ResourceRange {
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
}
pub struct SubresourceRange(pub core::ops::Range<u32>);
pub struct LayeredSubresourceRange{pub mips: core::ops::Range<u32>, pub layers: core::ops::Range<u32>}*/

pub unsafe trait ImageSamples {
    fn flag(self) -> vk::SampleCountFlags;
}
#[derive(Clone, Copy)]
/// Typestate for an image which has one sample per pixel. See also
/// [`MultiSampled`].
pub struct SingleSampled;
unsafe impl ImageSamples for SingleSampled {
    fn flag(self) -> vk::SampleCountFlags {
        vk::SampleCountFlags::TYPE_1
    }
}
/// Typestate for an image which has more than one sample. See also
/// [`SingleSampled`].
#[repr(u32)]
#[derive(Clone, Copy)]
pub enum MultiSampled {
    Samples2 = vk::SampleCountFlags::TYPE_2.as_raw(),
    Samples4 = vk::SampleCountFlags::TYPE_4.as_raw(),
    Samples8 = vk::SampleCountFlags::TYPE_8.as_raw(),
    Samples16 = vk::SampleCountFlags::TYPE_16.as_raw(),
    Samples32 = vk::SampleCountFlags::TYPE_32.as_raw(),
    Samples64 = vk::SampleCountFlags::TYPE_64.as_raw(),
}
unsafe impl ImageSamples for MultiSampled {
    fn flag(self) -> vk::SampleCountFlags {
        vk::SampleCountFlags::from_raw(self as _)
    }
}

crate::thin_handle! {
    /// An owned image handle.
    /// # Typestates
    /// * `Usage`: The `VkImageUsageFlags` this image is statically known to
    ///   possess (e.g. `(Storage, TransferSrc)`)
    /// * `Dim`: The dimensionality of an image, including whether it is an
    ///   array of images (e.g. [`D2`])
    /// * `Format`: The texel format of the image.
    /// * `Samples`: Whether the image is [single-](SingleSampled) or
    ///   [multi-](MultiSampled)sampled.
    #[must_use = "dropping the handle will not destroy the image and may leak resources"]
    pub struct Image<Usage: ImageUsage, Dim: Dimensionality, Format: format::Format, Samples: ImageSamples>(
        vk::Image
    );
}

crate::thin_handle! {
    /// An owned image view handle.
    /// # Typestates
    /// * `Usage`: The `VkImageUsageFlags` this image is statically known to
    ///   possess (e.g. `(Storage, TransferSrc)`)
    /// * `Dim`: The dimensionality of an image, including whether it is an
    ///   array of images (e.g. [`D2`])
    /// * `Format`: The texel format of the image.
    /// * `Samples`: Whether the image is [single-](SingleSampled) or
    ///   [multi-](MultiSampled)sampled.
    #[must_use = "dropping the handle will not destroy the view and may leak resources"]
    pub struct ImageView<Usage: ImageUsage, Dim: Dimensionality, Format: format::Format, Samples: ImageSamples>(
        vk::ImageView
    );
}
