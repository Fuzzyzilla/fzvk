//! `vk*UsageFlags`.
use super::{Pred, Satisified, vk};

macro_rules! access_combinations {
    {impl $trait_name:ident for [$(($($name:ident),*$(,)?)),+$(,)?] {
        const $const_name:ident: $const_ty:ty;
    }} => {
        // Trailing comma to force it to be a tuple type, even for single
        // fields.
        $(
            // Hide, as we use a fake variadic when generating docs.
            #[doc(hidden)]
            impl<$($name : $trait_name),+> $trait_name for ($($name),*,) {
                const $const_name: $const_ty = <$const_ty>::from_raw($(<$name as $trait_name>::$const_name.as_raw())|*);
            }
        )+
    };
}

/// Compile-time representation of `vkBufferUsageFlags`.
///
/// Flags can be a single aspect, [`Storage`], or can be placed in a tuple to
/// combine them: `(Storage, Vertex, Indirect)`
pub trait BufferUsage: 'static {
    /// The bitflags for this buffer usage combination.
    const FLAGS: vk::BufferUsageFlags;
}
/// A buffer that can be bound to a command buffer as an index buffer.
pub struct Index;
impl BufferUsage for Index {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::INDEX_BUFFER;
}
/// A buffer that can be bound to a command buffer as a vertex buffer.
#[derive(Clone, Copy)]
pub struct Vertex;
impl BufferUsage for Vertex {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
}
/// A buffer that can be bound to a descriptor set as shader storage.
pub struct Storage;
impl BufferUsage for Storage {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::STORAGE_BUFFER;
}
/// A buffer or image that can used as the source of a copy command.
pub struct TransferSrc;
impl BufferUsage for TransferSrc {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;
}
/// A buffer or image that can used as the destination of a copy command.
pub struct TransferDst;
impl BufferUsage for TransferDst {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_DST;
}
/// A buffer that can be used as the source for dispatch or draw indirection
/// commands.
pub struct Indirect;
impl BufferUsage for Indirect {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::INDIRECT_BUFFER;
}
/// A buffer that can be bound to a descriptor set as uniform storage.
pub struct Uniform;
impl BufferUsage for Uniform {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
}
#[cfg_attr(doc, doc(fake_variadic))]
/// Implemented for tuples of the form of, for example, `(Storage, TransferDst,
/// TransferSrc)`. The resulting usage flags is the union of all members.
impl<A: BufferUsage> BufferUsage for (A,) {
    const FLAGS: vk::BufferUsageFlags = A::FLAGS;
}
access_combinations! {
    impl BufferUsage for [
        (A,B),
        (A,B,C),
        (A,B,C,D),
        (A,B,C,D,E),
        (A,B,C,D,E,F),
        (A,B,C,D,E,F,G),
    ] {
        const FLAGS : vk::BufferUsageFlags;
    }
}

/// Compile-time representation of `vkImageUsageFlags`.
///
/// Flags can be a single usage, [`Storage`], or can be placed in a tuple to
/// combine them: `(Storage, Sampled, ColorAttachment)`
pub trait ImageUsage: 'static {
    /// The bitflags for this image usage combination.
    const FLAGS: vk::ImageUsageFlags;
}
impl ImageUsage for TransferSrc {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::TRANSFER_SRC;
}
impl ImageUsage for TransferDst {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::TRANSFER_DST;
}
impl ImageUsage for Storage {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::STORAGE;
}
/// An image that can be bound to a descriptor set to be used with a sampler.
pub struct Sampled;
impl ImageUsage for Sampled {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
}
/// An image that can be attached to a Framebuffer as a color attachment.
pub struct ColorAttachment;
impl ImageUsage for ColorAttachment {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
}
/// An image that can be attached to a Framebuffer as a depth and/or stencil
/// attachment.
pub struct DepthStencilAttachment;
impl ImageUsage for Uniform {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
}
/// An image that can be attached to a Framebuffer for transient render
/// operations.
pub struct TransientAttachment;
impl ImageUsage for TransientAttachment {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
}
/// An image that can be attached to a Framebuffer as an input attachment.
pub struct InputAttachment;
impl ImageUsage for InputAttachment {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::INPUT_ATTACHMENT;
}
/// Implemented for tuples of the form of, for example, `(Storage, TransferDst,
/// TransferSrc)`. The resulting usage flags is the union of all members.
#[cfg_attr(doc, doc(fake_variadic))]
impl<A: ImageUsage> ImageUsage for (A,) {
    const FLAGS: vk::ImageUsageFlags = A::FLAGS;
}
access_combinations! {
    impl ImageUsage for [
        (A,B),
        (A,B,C),
        (A,B,C,D),
        (A,B,C,D,E),
        (A,B,C,D,E,F),
        (A,B,C,D,E,F,G),
        (A,B,C,D,E,F,G,H),
    ] {
        const FLAGS : vk::ImageUsageFlags;
    }
}

/// A trait representing whether a given buffer usage is a subset of another.
///
/// Buffers can be safely reinterpreted as a buffer of a subset of usages.
///
/// For example, `(Index, Vertex, Uniform)` is `BufferSuperset<(Uniform,
/// Vertex)>`, but is not `BufferSuperset<(Indirect, Vertex)>`
pub unsafe trait BufferSuperset<Sub: BufferUsage>: BufferUsage {}

unsafe impl<Super, Sub> BufferSuperset<Sub> for Super
where
    Super: BufferUsage,
    Sub: BufferUsage,
    Pred<{ Super::FLAGS.contains(Sub::FLAGS) }>: Satisified,
{
}
/// A trait representing whether a given image usage is a subset of another.
///
/// Images can be safely reinterpreted as a image of a subset of usages.
///
/// For example, `(Storage, TransferSrc)` is `BufferSuperset<(Storage,)>`, but
/// is not `BufferSuperset<(TransferDst, TransferSrc)>`
pub unsafe trait ImageSuperset<Sub: ImageUsage>: ImageUsage {}
unsafe impl<Super, Sub> ImageSuperset<Sub> for Super
where
    Super: ImageUsage,
    Sub: ImageUsage,
    Pred<{ Super::FLAGS.contains(Sub::FLAGS) }>: Satisified,
{
}
