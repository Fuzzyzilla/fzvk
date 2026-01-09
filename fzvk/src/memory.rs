use super::{Device, ThinHandle, buffer, format, image, usage, vk};

/// Represents a resource with no backing memory assigned.
///
/// The uses of resources in this state are extremely limited, effectively
/// limited to memory management APIs, and memory must be bound before they can
/// be used for anything else. This is a temporary state, and is thus
/// represented as an external wrapper instead of an internal typestate on the
/// underlying handle.
#[repr(transparent)]
pub struct Virtual<V: HasMemoryRequirements>(V);
impl<V: HasMemoryRequirements> Virtual<V> {
    /// Assume the virtual resource has had a complete, contiguous, and
    /// satisfactory range of memory bound to it through external APIs.
    pub unsafe fn assume_backed(self) -> V {
        self.0
    }
}

//pub struct BindInfo<T: ThinHandle> {}

// Safety: repr(transparent) over repr(transparent) over Handle. X3
unsafe impl<V: HasMemoryRequirements> ThinHandle for Virtual<V> {
    type Handle = V::Handle;
}
// Yes this technically allows for `Virtual<Virtual<Virtual<....>>>` but it
// works as intended so w/e. You can't construct that anyways.
impl<V: HasMemoryRequirements> HasMemoryRequirements for Virtual<V> {
    type Bound = V::Bound;
    unsafe fn memory_requirements(&self, device: &Device) -> vk::MemoryRequirements {
        self.0.memory_requirements(device)
    }
}
/// Trait for handles that require vulkan memory to be bound to them, `Images`
/// and `Buffers`.
pub trait HasMemoryRequirements: ThinHandle {
    /// The type that results from successfully binding memory to `self`.
    type Bound;
    unsafe fn memory_requirements(&self, device: &Device) -> vk::MemoryRequirements;
}
/// Trait for objects which are sound to have memory bound to them in their
/// current state.
pub trait BindMemory<Access: MemoryAccess>: HasMemoryRequirements {
    unsafe fn bind_memory(
        self,
        device: &Device,
        memory: &mut Memory<Access>,
        offset: u64,
    ) -> Result<Self::Bound, vk::Result>;
}
impl<Usage: usage::BufferUsage> HasMemoryRequirements for buffer::Buffer<Usage> {
    type Bound = Self;
    unsafe fn memory_requirements(&self, device: &Device) -> vk::MemoryRequirements {
        device.0.get_buffer_memory_requirements(self.handle())
    }
}
impl<
    Usage: usage::ImageUsage,
    Dim: image::Dimensionality,
    Format: format::Format,
    Samples: image::ImageSamples,
> HasMemoryRequirements for image::Image<Usage, Dim, Format, Samples>
{
    type Bound = Self;
    unsafe fn memory_requirements(&self, device: &Device) -> vk::MemoryRequirements {
        device.0.get_image_memory_requirements(self.handle())
    }
}
impl<Usage: usage::BufferUsage, Access: NonTransientAccess> BindMemory<Access>
    for Virtual<buffer::Buffer<Usage>>
{
    unsafe fn bind_memory(
        self,
        device: &Device,
        memory: &mut Memory<Access>,
        offset: u64,
    ) -> Result<Self::Bound, vk::Result> {
        device
            .0
            .bind_buffer_memory(self.0.handle(), memory.handle(), offset)
            .map(|()| self.0)
    }
}
impl<
    Usage: usage::ImageUsage,
    Dim: image::Dimensionality,
    Format: format::Format,
    Samples: image::ImageSamples,
    Access: NonTransientAccess,
> BindMemory<Access> for Virtual<image::Image<Usage, Dim, Format, Samples>>
{
    unsafe fn bind_memory(
        self,
        device: &Device,
        memory: &mut Memory<Access>,
        offset: u64,
    ) -> Result<Self::Bound, vk::Result> {
        device
            .0
            .bind_image_memory(self.0.handle(), memory.handle(), offset)
            .map(|()| self.0)
    }
}
impl<
    Usage: usage::ImageSuperset<usage::TransientAttachment>,
    Dim: image::Dimensionality,
    Format: format::Format,
    Samples: image::ImageSamples,
> BindMemory<LazilyAllocated> for Virtual<image::Image<Usage, Dim, Format, Samples>>
{
    unsafe fn bind_memory(
        self,
        device: &Device,
        memory: &mut Memory<LazilyAllocated>,
        offset: u64,
    ) -> Result<Self::Bound, vk::Result> {
        device
            .0
            .bind_image_memory(self.0.handle(), memory.handle(), offset)
            .map(|()| self.0)
    }
}

crate::typestate_enum! {
    /// Typestates for host-visibility of memory.
    ///
    /// Device-locality is not expressed on a typestate level as it has no
    /// implications for correct usage.
    pub enum trait MemoryAccess {
        /// Memory types with the `HOST_VISIBLE` flag.
        pub struct HostVisible,
        /// Memory types with the `HOST_VISIBLE` flag that have an outstanding
        /// CPU mapping.
        pub struct HostMapped,
        /// Memory types without the `HOST_VISIBLE` flag.
        pub struct DeviceOnly,
        /// Memory types with the `LAZILY_ALLOCATED` flag, which can only be
        /// bound to `Transient` usage images.
        pub struct LazilyAllocated,
    }
}
/// [`MemoryAccess`] types that are not [`LazilyAllocated`].
pub trait NonTransientAccess: MemoryAccess {}
impl NonTransientAccess for HostVisible {}
impl NonTransientAccess for HostMapped {}
impl NonTransientAccess for DeviceOnly {}

crate::thin_handle!(
    /// An allocation of continuous memory, accessible by the device.
    ///
    /// # Typestates
    /// * [`Access`](MemoryAccess): Whether the memory is host-visible or
    ///   currently mapped.
    pub struct Memory<Access: MemoryAccess>(vk::DeviceMemory);
);
/// Convert a Range type to a `(offset, size)` tuple, where the resulting `size`
/// may be the special `WHOLE_SIZE` constant.
pub(crate) fn range_to_offset_size(range: impl core::ops::RangeBounds<u64>) -> (u64, u64) {
    let offset = match range.start_bound().cloned() {
        core::ops::Bound::Unbounded => 0,
        core::ops::Bound::Included(x) => x,
        core::ops::Bound::Excluded(x) => x.strict_add(1),
    };
    let size = match range.end_bound().cloned() {
        core::ops::Bound::Unbounded => vk::WHOLE_SIZE,
        core::ops::Bound::Included(x) => x.strict_sub(offset).strict_add(1),
        core::ops::Bound::Excluded(x) => x.strict_sub(offset),
    };
    (offset, size)
}
