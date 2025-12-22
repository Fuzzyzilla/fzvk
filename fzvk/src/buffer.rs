//! `vkBuffer`
use crate::{NonNull, ThinHandle, usage::*, vk};
use core::{marker::PhantomData, num::NonZero};

crate::thin_handle! {
    /// An owned image view handle.
    /// # Typestates
    /// * `Usage`: The `vkBufferUsageFlags` this image is statically known to
    ///   possess (e.g. `(Storage, TransferSrc)`)
    #[must_use = "dropping the handle will not destroy the buffer and may leak resources"]
    pub struct Buffer<Usage: BufferUsage>(vk::Buffer);
}
impl<SubUsage: BufferUsage, SuperUsage: BufferSuperset<SubUsage> + BufferUsage>
    AsMut<Buffer<SubUsage>> for Buffer<SuperUsage>
{
    fn as_mut(&mut self) -> &mut Buffer<SubUsage> {
        self.as_subusage_mut()
    }
}
impl<SubUsage: BufferUsage, SuperUsage: BufferSuperset<SubUsage> + BufferUsage>
    AsRef<Buffer<SubUsage>> for Buffer<SuperUsage>
{
    fn as_ref(&self) -> &Buffer<SubUsage> {
        self.as_subusage()
    }
}

impl<AcUsageess: BufferUsage> Buffer<AcUsageess> {
    /// Forget some of the usage types. To bring back the forgotten types, the
    /// *unsafe* function [`Self::into_usage_unchecked`] can be used.
    ///
    /// Use [`Self::as_subusage`] or [`Self::as_subusage_mut`] to get temporary
    /// subusage without forgetting the original flags.
    /// # Compiler Errors
    /// If you see `expected false, found true` as an error, that means the
    /// `SubUsage` flags is not statically known to be a subtype of this
    /// buffer's Usage flags. I do not know how to make a better error :3
    pub fn into_subusage<SubUsage>(self) -> Buffer<SubUsage>
    where
        SubUsage: BufferUsage,
        AcUsageess: BufferSuperset<SubUsage>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperUsage> *is a*
        // Buffer<SubUsage>.
        unsafe { self.with_state() }
    }
    pub fn reference<SubUsage: BufferUsage>(&self) -> BufferReference<SubUsage>
    where
        AcUsageess: BufferSuperset<SubUsage>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperUsage> *is a*
        // Buffer<SubUsage>.

        // Known non-null since self is non-null.
        unsafe { BufferReference::from_handle_unchecked(self.handle()) }
    }
    pub fn as_subusage<SubUsage>(&self) -> &Buffer<SubUsage>
    where
        SubUsage: BufferUsage,
        AcUsageess: BufferSuperset<SubUsage>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperUsage> *is a*
        // Buffer<SubUsage>.
        unsafe { self.with_state_ref() }
    }
    pub fn as_subusage_mut<SubUsage>(&mut self) -> &mut Buffer<SubUsage>
    where
        SubUsage: BufferUsage,
        AcUsageess: BufferSuperset<SubUsage>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperUsage> *is a*
        // Buffer<SubUsage>.
        unsafe { self.with_state_mut() }
    }
    /// Convert the type into a different kind of usage.
    /// # Safety
    /// The buffer cannot be used for any kind of usage that wasn't specified at
    /// creation time.
    pub unsafe fn into_usage_unchecked<NewUsage: BufferUsage>(self) -> Buffer<NewUsage> {
        // Safety: Forwarded to caller.
        unsafe { self.with_state() }
    }
    pub fn barrier(&'_ self) -> BufferBarrier<'_> {
        BufferBarrier {
            buffer: unsafe { self.handle() },
            offset: 0,
            len: vk::WHOLE_SIZE,
            _phantom: PhantomData,
        }
    }
}
pub struct BufferBarrier<'a> {
    pub(crate) buffer: vk::Buffer,
    pub(crate) offset: u64,
    pub(crate) len: u64,
    _phantom: PhantomData<&'a ()>,
}
/// A thin, shared reference to a [`Buffer`] with some subset of usages.
/// Acquired using [`Buffer::reference`].
///
/// This is used anywhere where vulkan expects a slice of buffers, where
/// `&[&Buffer]` is one layer of indirection too deep to be directly handed off
/// to the implementation.
#[repr(transparent)]
pub struct BufferReference<'a, Usage: BufferUsage>(
    NonNull<vk::Buffer>,
    PhantomData<(Usage, &'a Buffer<Usage>)>,
);

unsafe impl<Usage: BufferUsage> ThinHandle for BufferReference<'_, Usage> {
    type Handle = vk::Buffer;
}

/// The data needed for addressing calculations in buffer-image copies.
pub trait BufferPitch {
    /// The value representing that the vulkan implementation should assume
    /// tight packing between rows and slices, i.e. where
    /// [`BufferPitch::row_pitch`] and [`BufferPitch::slice_pitch`] are both
    /// `None`.
    const PACKED: Self;
    /// The distance in *texels* (not bytes!) between the start of one row and
    /// the start of the next, or `None` for tight packing (i.e. the `width` of
    /// the image).
    ///
    /// If not applicable for this type, `None` should be returned.
    fn row_pitch(&self) -> Option<NonZero<u32>>;
    /// The distance in *texels* (not bytes!) between the start of one 2D slice
    /// of a 3D image or 2D array image and the start of the next, or `None` for
    /// tight packing (i.e. the `width * height` of the image)
    ///
    /// If not applicable for this type, `None` should be returned.
    fn slice_pitch(&self) -> Option<NonZero<u32>>;
}
/// Controls buffer addressing for image operations where only one 2D image
/// slice is involved.
///
/// The distance in *texels* (not bytes!) between the start of one row and the
/// start of the next, or `None` for tight packing (i.e. the `width` of the
/// image)
///
/// If not None, must be >= the tight packed pitch (rows must not alias, even
/// for a read-only operation)
pub struct RowPitch(pub Option<NonZero<u32>>);
/// Controls buffer addressing for image operations where several 2D image
/// slices are involved.
pub struct RowSlicePitch {
    /// The distance in *texels* (not bytes!) between the start of one row and
    /// the start of the next, or `None` for tight packing (i.e. the `width` of
    /// the image)
    ///
    /// If not None, must be >= the tight packed pitch (rows must not alias,
    /// even for a read-only operation)
    pub row_pitch: Option<NonZero<u32>>,
    /// The distance in *texels* (not bytes!) between the start of one 2D slice
    /// of a 3D image or 2D array image and the start of the next, or `None` for
    /// tight packing (i.e. the `width * height` of the image)
    ///
    /// If not None, must be >= the tight packed pitch (slices must not alias,
    /// even for a read-only operation)
    pub slice_pitch: Option<NonZero<u32>>,
}

/// Buffer addressing for D1 images cannot be controlled.
impl BufferPitch for () {
    const PACKED: Self = ();
    fn row_pitch(&self) -> Option<NonZero<u32>> {
        None
    }
    fn slice_pitch(&self) -> Option<NonZero<u32>> {
        None
    }
}
impl BufferPitch for RowPitch {
    const PACKED: Self = Self(None);
    fn row_pitch(&self) -> Option<NonZero<u32>> {
        self.0
    }
    fn slice_pitch(&self) -> Option<NonZero<u32>> {
        None
    }
}
impl BufferPitch for RowSlicePitch {
    const PACKED: Self = Self {
        row_pitch: None,
        slice_pitch: None,
    };
    fn row_pitch(&self) -> Option<NonZero<u32>> {
        self.row_pitch
    }
    fn slice_pitch(&self) -> Option<NonZero<u32>> {
        self.slice_pitch
    }
}

/// Trait for integers which can be used as values in an index buffer.
pub trait IndexTy {
    /// The index enum for this type.
    const TYPE: vk::IndexType;
}
impl IndexTy for u16 {
    const TYPE: vk::IndexType = vk::IndexType::UINT16;
}
impl IndexTy for u32 {
    const TYPE: vk::IndexType = vk::IndexType::UINT32;
}
