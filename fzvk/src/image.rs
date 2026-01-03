//! `vkImage[View]`
use crate::{
    ThinHandle,
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
    /// The extrapolated Offset3D. Axes beyond the type's dimensionality should
    /// be `0`.
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
    pub layers: CreateArrayCount,
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
    pub layers: CreateArrayCount,
}
impl Extent2D {
    pub fn with_layers(self, layers: CreateArrayCount) -> Extent2DArray {
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
pub struct CreateArrayCount(NonZero<u32>);
impl CreateArrayCount {
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
pub trait Dimensionality: 'static {
    /// The size in each dimension, plus an array layer count if applicable,
    /// used when creating the image.
    type CreateExtent: Extent;
    /// The size in each dimension, used when referring to a subregion of the
    /// image. Equivalent to the [`Dimensionality::CreateExtent`] of the
    /// equivalent non-arrayed [`Dimensionality`].
    type Extent: Extent;
    /// Indexes a single mipmap level, possibly across a range of layers if
    /// applicable.
    type SubresourceLayers: HasSubresourceLayers;
    /// Indexes a range of mipmap levels and possibly a range of layers if
    /// applicable.
    type SubresourceRange: HasSubresourceRange;
    /// An offset into an image of this dimensionality.
    type Offset: Offset;
    /// How an N-dimensional image's addresses should be translated into
    /// 1-dimensional buffer addresses.
    type Pitch: BufferPitch;
    /// Vulkan representation of this dimensionality.
    const IMAGE_TYPE: vk::ImageType;
    const VIEW_TYPE: vk::ImageViewType;
}
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
    type CreateExtent = Extent1D;
    type Extent = Extent1D;
    type Offset = Offset1D;
    type Pitch = ();
    type SubresourceLayers = SubresourceMip;
    type SubresourceRange = SubresourceMips;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_1D;
}
impl Dimensionality for D1Array {
    type CreateExtent = Extent1DArray;
    type Extent = Extent1D;
    type Offset = Offset1D;
    type Pitch = RowPitch;
    type SubresourceLayers = SubresourceLayers;
    type SubresourceRange = SubresourceRange;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_1D_ARRAY;
}
impl Dimensionality for D2 {
    type CreateExtent = Extent2D;
    type Extent = Extent2D;
    type Offset = Offset2D;
    type Pitch = RowPitch;
    type SubresourceLayers = SubresourceMip;
    type SubresourceRange = SubresourceMips;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_2D;
}
impl Dimensionality for D2Array {
    type CreateExtent = Extent2DArray;
    type Extent = Extent2D;
    type Offset = Offset2D;
    type Pitch = RowSlicePitch;
    type SubresourceLayers = SubresourceLayers;
    type SubresourceRange = SubresourceRange;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_2D_ARRAY;
}
impl Dimensionality for D3 {
    type CreateExtent = Extent3D;
    type Extent = Extent3D;
    type Offset = Offset3D;
    type Pitch = RowSlicePitch;
    type SubresourceLayers = SubresourceMip;
    type SubresourceRange = SubresourceMips;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_3D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_3D;
}

/// Implemented for dimensionalities which can be constructed from views of
/// `Other` dimensionality.
pub trait CanView<Other: Dimensionality>: Dimensionality {}
// Can, of course, view every dimensionality as itself.
impl<Dim: Dimensionality> CanView<Dim> for Dim {}

// Can view a single element of an array as a non-array.
impl CanView<D1Array> for D1 {}
impl CanView<D2Array> for D2 {}

// Can view non-array as an array of 1 layer.
impl CanView<D1> for D1Array {}
impl CanView<D2> for D2Array {}

pub unsafe trait HasSubresourceRange {
    /// The entire image, all mip levels and all array layers.
    const ALL: Self;
    /// Get the vulkan subresource range structure associated with this value.
    ///
    /// Neither [`vk::ImageSubresourceRange::layer_count`] nor
    /// [`vk::ImageSubresourceRange::level_count`] can be zero.
    fn subresource_range(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange;
}
pub unsafe trait HasSubresourceLayers {
    /// Get the vulkan subresource layers structure associated with this value.
    ///
    /// The [`vk::ImageSubresourceLayers::layer_count`] must not be 0 or
    /// `VK_REMAINING_LAYERS`.
    fn subresource_layers(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers;
}
/// Implemented for ranges that are closed on the right: `a..b`, `a..=b`, `..b`,
/// and `..=b` (but not, for example, `a..`)
///
/// # Safety
/// Types implementing this trait *must not* return
/// [`core::ops::Bound::Unbounded`] for [`core::ops::RangeBounds::end_bound`]
pub unsafe trait RangeClosedRight<T>: core::ops::RangeBounds<T> {}
unsafe impl<T> RangeClosedRight<T> for core::ops::Range<T> {}
unsafe impl<T> RangeClosedRight<T> for core::ops::RangeInclusive<T> {}
unsafe impl<T> RangeClosedRight<T> for core::ops::RangeTo<T> {}
unsafe impl<T> RangeClosedRight<T> for core::ops::RangeToInclusive<T> {}

/// A single mip level of a non-array image.
pub struct SubresourceMip(pub u32);
impl SubresourceMip {
    pub const BASE_LEVEL: Self = Self(0);
}
unsafe impl HasSubresourceLayers for SubresourceMip {
    fn subresource_layers(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: self.0,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}
pub struct SubresourceMips {
    start: u32,
    count: NonZero<u32>,
}
unsafe impl HasSubresourceRange for SubresourceMips {
    const ALL: Self = Self {
        start: 0,
        count: NonZero::new(vk::REMAINING_MIP_LEVELS).unwrap(),
    };
    fn subresource_range(&self, aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: self.start,
            level_count: self.count.get(),
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
}
/// A single mip level of a range of image layers.
pub struct SubresourceLayers {
    mip: u32,
    base_array_layer: u32,
    /// Must not be VK_REMAINING_LAYERS.
    layer_count: NonZero<u32>,
}
impl SubresourceLayers {
    /// Reference a range of layers at a given mip level.
    ///
    /// Due to [a
    /// mistake](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_maintenance5.html#_vk_remaining_array_layers_for_vkimagesubresourcelayers_layercount)
    /// in the vulkan specification, the `layers` parameter does not accept
    /// right-unbounded ranges, and the `VK_REMAINING_LAYERS` special constant
    /// is not usable.
    /// # Panics
    /// If `layers` refers to an empty or inverted range.
    pub fn range(mip: u32, layers: impl RangeClosedRight<u32>) -> Self {
        let base_array_layer = match layers.start_bound() {
            core::ops::Bound::Excluded(&a) => a.strict_add(1),
            core::ops::Bound::Included(&a) => a,
            core::ops::Bound::Unbounded => 0,
        };
        let layer_count = match layers.end_bound() {
            core::ops::Bound::Excluded(&end) => {
                assert!(base_array_layer < end);
                // Safety - just checked that end is strictly larger, therefore
                // a difference >= 1.
                unsafe { NonZero::new_unchecked(end - base_array_layer) }
            }
            core::ops::Bound::Included(&end) => {
                assert!(base_array_layer <= end);
                let count = (end - base_array_layer).strict_add(1);
                // Unconditional +1, always nonzero.
                unsafe { NonZero::new_unchecked(count) }
            }
            // Guarded by unsafe contract of `ClosedRight` trait.
            core::ops::Bound::Unbounded => unsafe { core::hint::unreachable_unchecked() },
        };
        // This constant is not allowed in this API.
        assert_ne!(layer_count.get(), vk::REMAINING_ARRAY_LAYERS);
        Self {
            mip,
            base_array_layer,
            layer_count,
        }
    }
    pub fn new(mip: u32, base_array_layer: u32, layer_count: NonZero<u32>) -> Self {
        Self {
            mip,
            base_array_layer,
            layer_count,
        }
    }
}
unsafe impl HasSubresourceLayers for SubresourceLayers {
    fn subresource_layers(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: self.mip,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count.get(),
        }
    }
}
pub struct SubresourceRange {
    base_mip_level: u32,
    /// May be `VK_REMAINING_LEVELS`
    level_count: NonZero<u32>,
    base_array_layer: u32,
    /// May be `VK_REMAINING_LAYERS`
    layer_count: NonZero<u32>,
}
unsafe impl HasSubresourceRange for SubresourceRange {
    const ALL: Self = Self {
        base_mip_level: 0,
        level_count: NonZero::new(vk::REMAINING_MIP_LEVELS).unwrap(),
        base_array_layer: 0,
        layer_count: NonZero::new(vk::REMAINING_ARRAY_LAYERS).unwrap(),
    };
    fn subresource_range(&self, aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: self.base_mip_level,
            level_count: self.level_count.get(),
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count.get(),
        }
    }
}
impl SubresourceRange {
    const REMAINING_MIP_LEVELS: NonZero<u32> = NonZero::new(vk::REMAINING_MIP_LEVELS).unwrap();
    const REMAINING_ARRAY_LAYERS: NonZero<u32> = NonZero::new(vk::REMAINING_ARRAY_LAYERS).unwrap();
    /// Construct a range referring to several mip levels across several layers
    /// of an image.
    ///
    /// `VK_REMAINING_*` constants should *not* be used in the ranges. Instead,
    /// leave the right side of the range unbounded.
    /// # Panics
    /// If either range refers to an empty or inverted range.
    pub fn range(
        mip_levels: impl core::ops::RangeBounds<u32>,
        array_layers: impl core::ops::RangeBounds<u32>,
    ) -> Self {
        use core::ops::Bound;
        let (base_mip_level, level_count) = {
            let start_inclusive = match mip_levels.start_bound().cloned() {
                Bound::Unbounded => 0,
                Bound::Included(x) => x,
                Bound::Excluded(x) => x.strict_add(1),
            };
            let count = match mip_levels.end_bound().cloned() {
                Bound::Unbounded => Self::REMAINING_MIP_LEVELS,
                // Safety: +1 means never zero.
                Bound::Included(x) => unsafe {
                    NonZero::new_unchecked(x.strict_sub(start_inclusive).strict_add(1))
                },
                Bound::Excluded(x) => NonZero::new(x.strict_sub(start_inclusive)).unwrap(),
            };
            (start_inclusive, count)
        };
        let (base_array_layer, layer_count) = {
            let start_inclusive = match array_layers.start_bound().cloned() {
                Bound::Unbounded => 0,
                Bound::Included(x) => x,
                Bound::Excluded(x) => x.strict_add(1),
            };
            let count = match array_layers.end_bound().cloned() {
                Bound::Unbounded => Self::REMAINING_ARRAY_LAYERS,
                // Safety: +1 means never zero.
                Bound::Included(x) => unsafe {
                    NonZero::new_unchecked(x.strict_sub(start_inclusive).strict_add(1))
                },
                Bound::Excluded(x) => NonZero::new(x.strict_sub(start_inclusive)).unwrap(),
            };
            (start_inclusive, count)
        };
        SubresourceRange {
            base_mip_level,
            level_count,
            base_array_layer,
            layer_count,
        }
    }
    pub fn new(
        base_mip_level: u32,
        level_count: NonZero<u32>,
        base_array_layer: u32,
        layer_count: NonZero<u32>,
    ) -> Self {
        Self {
            base_mip_level,
            level_count,
            base_array_layer,
            layer_count,
        }
    }
}
/*
pub trait ResourceRange {
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
}
pub struct SubresourceRange(pub core::ops::Range<u32>);
pub struct LayeredSubresourceRange{pub mips: core::ops::Range<u32>, pub layers: core::ops::Range<u32>}*/

pub unsafe trait ImageSamples: 'static {
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
    ///
    /// Notably lacking in this state is the Image's current layout. This is
    /// because the layout of the image can change without CPU interaction. In
    /// the scope of a recording command buffer however, this *can* be tracked.
    /// See [`Self::discard_layout`] and [`Self::assume_layout`].
    #[must_use = "dropping the handle will not destroy the image and may leak resources"]
    pub struct Image<Usage: ImageUsage, Dim: Dimensionality, Format: format::Format, Samples: ImageSamples>(
        vk::Image
    );
}

impl<Usage: ImageUsage, Dim: Dimensionality, Format: format::Format, Samples: ImageSamples>
    Image<Usage, Dim, Format, Samples>
{
    /// Reference this image with Undefined layout.
    ///
    /// It is always safe to assume an image is in undefined layout.
    pub unsafe fn discard_layout(
        &'_ self,
    ) -> ImageReference<'_, Usage, Dim, Format, Samples, layout::Undefined> {
        ImageReference(self.0, core::marker::PhantomData)
    }
    /// Reference this image as the given layout.
    pub unsafe fn assume_layout<Layout>(
        &'_ self,
        _layout: Layout,
    ) -> ImageReference<'_, Usage, Dim, Format, Samples, Layout>
    where
        Layout: layout::Layout,
        Self: layout::CanTransitionIntoLayout<Layout>,
    {
        ImageReference(self.0, core::marker::PhantomData)
    }
}

/// A reference to an image in a given [Layout](layout::Layout), for use during
/// the recording of command buffers. See [`Image`] documentation for rationale.
pub struct ImageReference<
    'a,
    Usage: ImageUsage,
    Dim: Dimensionality,
    Format: format::Format,
    Samples: ImageSamples,
    Layout: layout::Layout,
>(
    crate::handle::NonNull<vk::Image>,
    core::marker::PhantomData<(&'a Image<Usage, Dim, Format, Samples>, Layout)>,
);
unsafe impl<
    'a,
    Usage: ImageUsage,
    Dim: Dimensionality,
    Format: format::Format,
    Samples: ImageSamples,
    Layout: layout::Layout,
> ThinHandle for ImageReference<'a, Usage, Dim, Format, Samples, Layout>
{
    type Handle = vk::Image;
}

impl<
    'a,
    Usage: ImageUsage,
    Dim: Dimensionality,
    Format: format::Format,
    Samples: ImageSamples,
    Layout: layout::Layout,
> ImageReference<'a, Usage, Dim, Format, Samples, Layout>
{
    /// Transition into a new layout.
    ///
    /// If the old contents need not be preserved during this transition, or the
    /// image hasn't yet been initialized into a format at all,
    /// [`Self::reinitialize_as`] should be used.
    #[doc(alias = "hrt")]
    pub fn transition<NewLayout>(
        self,
        _into_layout: NewLayout,
    ) -> Transition<'a, Usage, Dim, Format, Samples, NewLayout>
    where
        NewLayout: layout::Layout,
        Image<Usage, Dim, Format, Samples>: layout::CanTransitionIntoLayout<NewLayout>,
    {
        Transition {
            from: Layout::LAYOUT,
            image: self.0,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: <Format::Aspect as format::Aspect>::ASPECT,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            _phantom: core::marker::PhantomData,
        }
    }
    /// Transition into a new layout, discarding the old contents of the image.
    /// This is potentially *much* more efficient than doing a normal
    /// [`Self::transition`], and should be used wherever the contents of the
    /// image need not be preserved.
    ///
    /// This is also the only valid way to initialize an image into a layout on
    /// it's first use.
    pub fn reinitialize_as<NewLayout>(
        self,
        _into_layout: NewLayout,
    ) -> Transition<'a, Usage, Dim, Format, Samples, NewLayout>
    where
        NewLayout: layout::Layout,
        Image<Usage, Dim, Format, Samples>: layout::CanTransitionIntoLayout<NewLayout>,
    {
        Transition {
            from: vk::ImageLayout::UNDEFINED,
            image: self.0,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: <Format::Aspect as format::Aspect>::ASPECT,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            _phantom: core::marker::PhantomData,
        }
    }
    /// Create a no-op transition. This method is unavailable on `Undefined`
    /// layout images.
    pub fn identity(self) -> Transition<'a, Usage, Dim, Format, Samples, Layout>
    where
        // This might seem redundant, but it prevents an undefined -> undefined
        // barrier. `CanTransitionIntoLayout` (Dissallows Undefined) =/= "can be
        // in layout" (Allows undefined) :3
        Image<Usage, Dim, Format, Samples>: layout::CanTransitionIntoLayout<Layout>,
    {
        Transition {
            from: Layout::LAYOUT,
            image: self.0,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: <Format::Aspect as format::Aspect>::ASPECT,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            _phantom: core::marker::PhantomData,
        }
    }
}

/// An opaque struct representing a pending image layout transition. Pass this
/// into [`crate::Device::barrier`] to get back an [`ImageReference`] in the new
/// layout.
#[must_use = "must be submitted to a `barrier` command to have any effect."]
pub struct Transition<
    'a,
    Usage: ImageUsage,
    Dim: Dimensionality,
    Format: format::Format,
    Samples: ImageSamples,
    IntoLayout: layout::Layout,
> {
    pub(crate) from: vk::ImageLayout,
    pub(crate) image: crate::handle::NonNull<vk::Image>,
    pub(crate) subresource_range: vk::ImageSubresourceRange,
    _phantom:
        core::marker::PhantomData<ImageReference<'a, Usage, Dim, Format, Samples, IntoLayout>>,
}
impl<
    'a,
    Usage: ImageUsage,
    Dim: Dimensionality,
    Format: format::Format,
    Samples: ImageSamples,
    IntoLayout: layout::Layout,
> Transition<'a, Usage, Dim, Format, Samples, IntoLayout>
{
    fn into_after_transition(self) -> ImageReference<'a, Usage, Dim, Format, Samples, IntoLayout> {
        ImageReference(self.image, core::marker::PhantomData)
    }
}

pub trait ImageTransitions<'a> {
    type AfterTransition<'b>;
    /// Internal use. See [`Transition`] for usage.
    fn as_barriers<'this>(
        &'this self,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    ) -> impl AsRef<[vk::ImageMemoryBarrier<'this>]>;
    /// Internal use. See [`Transition`] for usage.
    fn into_after_transition(self) -> Self::AfterTransition<'a>;
}
impl ImageTransitions<'_> for () {
    type AfterTransition<'b> = ();
    fn as_barriers<'this>(
        &'this self,
        _src_access_mask: vk::AccessFlags,
        _dst_access_mask: vk::AccessFlags,
    ) -> impl AsRef<[vk::ImageMemoryBarrier<'this>]> {
        []
    }
    fn into_after_transition(self) -> Self::AfterTransition<'static> {}
}
impl<
    'a,
    Usage: ImageUsage,
    Dim: Dimensionality,
    Format: format::Format,
    Samples: ImageSamples,
    IntoLayout: layout::Layout,
> ImageTransitions<'a> for Transition<'a, Usage, Dim, Format, Samples, IntoLayout>
{
    type AfterTransition<'b> = ImageReference<'b, Usage, Dim, Format, Samples, IntoLayout>;
    fn into_after_transition(self) -> Self::AfterTransition<'a> {
        self.into_after_transition()
    }
    fn as_barriers<'this>(
        &'this self,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    ) -> impl AsRef<[vk::ImageMemoryBarrier<'this>]> {
        [vk::ImageMemoryBarrier::default()
            .image(self.image.get())
            .subresource_range(self.subresource_range)
            .old_layout(self.from)
            .new_layout(IntoLayout::LAYOUT)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)]
    }
}
impl<'a, A: ImageTransitions<'a>> ImageTransitions<'a> for (A,) {
    type AfterTransition<'b> = (A::AfterTransition<'b>,);
    fn as_barriers<'this>(
        &'this self,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    ) -> impl AsRef<[vk::ImageMemoryBarrier<'this>]> {
        self.0.as_barriers(src_access_mask, dst_access_mask)
    }
    fn into_after_transition(self) -> Self::AfterTransition<'a> {
        (self.0.into_after_transition(),)
    }
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

pub mod layout {
    use super::*;
    pub trait Layout {
        const LAYOUT: vk::ImageLayout;
    }
    /// Implemented by types for which a transition into the layout denoted by
    /// `Which` is allowable.
    ///
    /// Note that this is not the same as "can be in layout" - this trait
    /// disallows transitions into [`Undefined`], although all images can *be*
    /// in [`Undefined`] layout.
    pub trait CanTransitionIntoLayout<Which: Layout> {}

    /// Image is usable as the given layout. For every given layout, this is
    /// implemented by the layout itself and the General layout which is usable
    /// for all tasks.
    pub trait UsableAs<Which: Layout>: Layout {}
    /// General layout can be used for any layout's task.
    impl<Other: Layout> UsableAs<Other> for General {}
    // Cannot `impl<T> UsableAs<T> for T` because that interferes with the
    // above.

    /// This layout cannot be transitioned into. Transitions *from* this layout
    /// mean "discard contents", which may yield better performance if the
    /// contents need not be preserved.
    pub struct Undefined;
    impl Layout for Undefined {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::UNDEFINED;
    }

    pub struct General;
    impl Layout for General {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::GENERAL;
    }
    // Anything can have General layout!
    impl<T> CanTransitionIntoLayout<General> for T {}

    pub struct ColorAttachment;
    impl Layout for ColorAttachment {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }
    impl UsableAs<Self> for ColorAttachment {}
    // Only color aspect images with usage including `COLOR_ATTACHMENT_BIT` can
    // have this layout.
    impl<
        Usage: crate::usage::ImageSuperset<crate::usage::ColorAttachment>,
        Dim: Dimensionality,
        Format: format::HasAspect<format::Color>,
        Samples: ImageSamples,
    > CanTransitionIntoLayout<ColorAttachment> for Image<Usage, Dim, Format, Samples>
    {
    }

    pub struct DepthStencilAttachment;
    impl Layout for DepthStencilAttachment {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    impl UsableAs<Self> for DepthStencilAttachment {}
    /* FIXME: Format needs to be bound by `HasAspect<Depth AND/OR Stencil>`
    impl<
        Usage: crate::usage::ImageSuperset<crate::usage::DepthStencilAttachment>,
        Dim: Dimensionality,
        Format: format::HasAspect<format::Depth>,
        Samples: ImageSamples,
    > CanHaveLayout<ColorAttachment> for Image<Usage, Dim, Format, Samples>
    {
    }
    pub struct DepthStencilReadOnly;
    impl Layout for DepthStencilReadOnly {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    }*/

    pub struct ShaderReadOnly;
    impl Layout for ShaderReadOnly {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    }
    impl UsableAs<Self> for ShaderReadOnly {}

    pub use crate::usage::TransferDst;
    impl Layout for TransferDst {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    }
    impl<
        Usage: crate::usage::ImageSuperset<crate::usage::TransferDst>,
        Dim: Dimensionality,
        Format: format::Format,
        Samples: ImageSamples,
    > CanTransitionIntoLayout<TransferDst> for Image<Usage, Dim, Format, Samples>
    {
    }
    impl UsableAs<Self> for TransferDst {}

    pub use crate::usage::TransferSrc;
    impl Layout for TransferSrc {
        const LAYOUT: vk::ImageLayout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    }
    impl<
        Usage: crate::usage::ImageSuperset<crate::usage::TransferSrc>,
        Dim: Dimensionality,
        Format: format::Format,
        Samples: ImageSamples,
    > CanTransitionIntoLayout<TransferSrc> for Image<Usage, Dim, Format, Samples>
    {
    }
    impl UsableAs<Self> for TransferSrc {}
}
