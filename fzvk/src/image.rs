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
pub trait Dimensionality: 'static {
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
    // Cannot `impl<T> UsableAs<T> for T` because that interferes with the above.

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
