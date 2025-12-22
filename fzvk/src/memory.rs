use super::{Device, ThinHandle, buffer, format, image, usage, vk};

/// Represents a resource with no backing memory assigned.
///
/// The uses of resources in this state are extremely limited, effectively
/// limited to memory management APIs, and memory must be bound before they can
/// be used for anything else. This is a temporary state, and is thus
/// represented as an external wrapper instead of an internal typestate on the
/// underlying handle.
#[repr(transparent)]
pub struct Virtual<V: NeedsMemory>(V);
impl<V: NeedsMemory> Virtual<V> {
    /// Assume the virtual resource has had a complete, contiguous, and
    /// satisfactory range of memory bound to it through external APIs.
    pub unsafe fn assume_backed(self) -> V {
        self.0
    }
}

//pub struct BindInfo<T: ThinHandle> {}

// Safety: repr(transparent) over repr(transparent) over Handle. X3
unsafe impl<V: NeedsMemory> ThinHandle for Virtual<V> {
    type Handle = V::Handle;
}
// Yes this technically allows for `Virtual<Virtual<Virtual<....>>>` but it
// works as intended so w/e. You can't construct that anyways.
impl<V: NeedsMemory> NeedsMemory for Virtual<V> {
    type BindInfo<'a> = V::BindInfo<'a>;
    type Bound = V::Bound;
    unsafe fn memory_requirements(&self, device: &Device) -> vk::MemoryRequirements {
        self.0.memory_requirements(device)
    }
}
/// Trait for handles that require vulkan memory to be bound to them, `Images`
/// and `Buffers`.
pub trait NeedsMemory: ThinHandle {
    /// The information needed to bind memory to `self`.
    type BindInfo<'a>;
    /// The type that results from successfully binding memory to `self`.
    type Bound;
    unsafe fn memory_requirements(&self, device: &Device) -> vk::MemoryRequirements;
}
/// Trait for objects which are sound to have memory bound to them in their
/// current state.
pub trait BindMemory: NeedsMemory {
    unsafe fn bind_memory(
        self,
        device: &Device,
        info: <Self as NeedsMemory>::BindInfo<'_>,
    ) -> Result<Self::Bound, ()>;
}
impl<Usage: usage::BufferUsage> NeedsMemory for buffer::Buffer<Usage> {
    type BindInfo<'a> = vk::BindBufferMemoryInfo<'a>;
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
> NeedsMemory for image::Image<Usage, Dim, Format, Samples>
{
    type BindInfo<'a> = vk::BindImageMemoryInfo<'a>;
    type Bound = Self;
    unsafe fn memory_requirements(&self, device: &Device) -> vk::MemoryRequirements {
        device.0.get_image_memory_requirements(self.handle())
    }
}
impl<T: NeedsMemory> BindMemory for Virtual<T> {
    unsafe fn bind_memory(
        self,
        device: &Device,
        info: <Self as NeedsMemory>::BindInfo<'_>,
    ) -> Result<Self::Bound, ()> {
        todo!()
    }
}

crate::typestate_enum! {
    /// Typestates for host-visibility of memory.
    ///
    /// Device-locality is not expressed on a typestate level as it has no
    /// implications for correct usage.
    pub enum trait MemoryAccess {
        /// Typestate for memory which is host-visible and coherent.
        ///
        /// Coherent memory does not need explicit flushing or invalidation.
        pub struct HostCoherent,
        /// Typestate for memory which is host-visible and cached.
        ///
        /// Cached accesses are generally faster than uncached accesses, though
        /// they are not always [coherent](HostCoherent).
        pub struct HostCached,
        /// Typestate for memory which is host visible and both
        /// [cached](HostCached) and [coherent](HostCoherent).
        pub struct HostCachedCoherent,
        /// Typestate for memory which is not host-visible.
        pub struct Inaccessable,
    }
}
crate::thin_handle!(
    /// An allocation of continuous memory, accessible by the device.
    ///
    /// # Typestates
    /// * [`Access`](MemoryAccess): Whether the memory is host-visible.
    pub struct Memory<Access: MemoryAccess>(vk::DeviceMemory);
);
