use super::{
    NonNull, ThinHandle,
    image::Dimensionality,
    pipeline::{SubpassCount, ValidSubpassCount},
    vk,
};
use core::marker::PhantomData;

crate::thin_handle! {
    /// A pool from which many [`CommandBuffer`]s may be allocated. Any
    /// operation on any command buffer allocated from this pool requires
    /// synchronous access to the pool as well.
    #[must_use = "dropping the handle will not destroy the command pool and may leak resources"]
    pub struct CommandPool(vk::CommandPool);
}

/// A type representing whether a command buffer is Primary or Secondary.
pub trait CommandBufferLevel {
    const LEVEL: vk::CommandBufferLevel;
}
/// A primary command buffer, executed directly by the queue.
pub struct Primary;
impl CommandBufferLevel for Primary {
    const LEVEL: vk::CommandBufferLevel = vk::CommandBufferLevel::PRIMARY;
}
/// A second command buffer, exectuted indirectly by being "called" by
/// [`Primary`] command buffers.
pub struct Secondary;
impl CommandBufferLevel for Secondary {
    const LEVEL: vk::CommandBufferLevel = vk::CommandBufferLevel::SECONDARY;
}

crate::thin_handle! {
    /// A command buffer, allocated from a [`CommandPool`].
    ///
    /// All operations require synchronous access to this buffer, as well as the
    /// pool it was allocated from.
    /// # Typestates
    /// * `Level` - [`Primary`] or [`Secondary`], describing the allocation
    ///   kind, which restricts the operations that can be recorded into the
    ///   buffer.
    #[must_use = "dropping the handle will not deallocate the buffer and may leak resources"]
    pub struct CommandBuffer<Level: CommandBufferLevel>(vk::CommandBuffer);
}
impl<Level: CommandBufferLevel> CommandBuffer<Level> {
    pub fn reference(&self) -> CommandBufferReference<Level> {
        // Safety - known non null since it came from self. Immutable reference
        // won't ever change the typestate.
        unsafe { ThinHandle::from_handle_unchecked(self.handle()) }
    }
}

#[repr(transparent)]
pub struct CommandBufferReference<'a, Level: CommandBufferLevel>(
    NonNull<vk::CommandBuffer>,
    PhantomData<&'a CommandBuffer<Level>>,
);
unsafe impl<Level: CommandBufferLevel> ThinHandle for CommandBufferReference<'_, Level> {
    type Handle = vk::CommandBuffer;
}
impl<'a, Level: CommandBufferLevel> From<&'a CommandBuffer<Level>>
    for CommandBufferReference<'a, Level>
{
    fn from(value: &'a CommandBuffer<Level>) -> Self {
        value.reference()
    }
}

pub struct BufferImageCopy<Dim: Dimensionality> {
    /// The minima of the rectangular bound to copy.
    pub image_offset: Dim::Offset,
    /// The size of the rectangular bound to copy.
    pub image_extent: Dim::Extent,
    /// The start offset, in bytes, of the first texel to begin transferring.
    pub buffer_offset: u64,
    /// Parameters controlling how texel addresses are calculated within buffer.
    pub pitch: Dim::Pitch,
    /// Which mip level, and optionally which layers of an array image, to copy.
    pub layers: Dim::SubresourceLayers,
}

pub trait CommandBufferState {}
/// Typestate for a recording command buffer that is not currently rendering.
pub struct OutsideRender;
/// Typestate for a recording command that is inside a renderpass.
/// # Typestates
/// * `REMAINING_PASSES`: how many subpasses left in the renderpass. A
///   renderpass can only be ended once this reaches the final subpass.
pub struct RemainingSubpasses<const REMAINING_PASSES: usize>
where
    SubpassCount<REMAINING_PASSES>: ValidSubpassCount;
impl CommandBufferState for OutsideRender {}
impl<const REMAINING_PASSES: usize> CommandBufferState for RemainingSubpasses<REMAINING_PASSES> where
    SubpassCount<REMAINING_PASSES>: ValidSubpassCount
{
}

#[must_use = "dropping the handle will result in a command buffer orphaned in an incomplete state"]
/// A temporary handle to a command buffer in the `recording` state.
/// # Typestates
/// * `Level`: The allocated level of the command buffer being recorded, primary
///   or secondary. The level of the buffer affects the valid operations that
///   can be recorded into it.
/// * `State`: The current scope of the command buffer. For example, whether the
///   command buffer is currently "inside" a render pass, and can thus issue
///   `draw` commands.
pub struct RecordingBuffer<'a, Level: CommandBufferLevel, State: CommandBufferState> {
    #[allow(unused)]
    // It's not actually unused. ThinHandle trait reads it!
    buffer: NonNull<vk::CommandBuffer>,
    _typed_buffer: PhantomData<&'a mut CommandBuffer<Level>>,
    _state: PhantomData<State>,
    _pool: PhantomData<&'a mut CommandPool>,
}
unsafe impl<'a, Level: CommandBufferLevel, State: CommandBufferState> ThinHandle
    for RecordingBuffer<'a, Level, State>
{
    type Handle = vk::CommandBuffer;
}
