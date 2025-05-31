//! # Fuzzy's Funny Zero-cost Vulkan abstraction.
//! Typestate :3
//!
//! To get started, use [`ash`] to fetch a [`ash::Device`] and pass it to [`Device::from_ash`].
//!
//! ## External Synchronization
//! This crate maps objects to *ownership* of the underlying handle. With this abstraction, a function
//! taking mutable reference to a handle or accepting a handle by value can assume it is the only thread
//! with access to the handle.
//!
//! ## Safety
//! Functions that take multiple objects must have the correct ownership hierarchies
//! (e.g. the parameters passed to [`Device::begin_command_buffer`] should be a command buffer,
//! the command pool from which it is allocated, and the device they both belong to.)
//!
//! ## Todo:
//! * [ ] Renderpasses
//! * [ ] Swapchains
//! * [ ] Pipelines
//! * [ ] Framebuffers
//! * [ ] Modules
//! * [ ] Descriptors
//! * [ ] Memory

// #![warn(missing_docs)]
#![allow(unsafe_op_in_unsafe_fn)]
#![feature(generic_const_exprs)]
#![allow(clippy::missing_safety_doc)]
use ash::vk;
use std::{marker::PhantomData, num::NonZero};

#[derive(Clone, Copy)]
#[repr(transparent)]
/// The unique set of family indices sharing a resource.
/// Always contains 2 or more indices.
pub struct SharingFamilies<'a>(&'a [u32]);
impl<'a> SharingFamilies<'a> {
    /// Wrap a set of families.
    /// # Safety
    /// * The slice must be length 2 or more.
    /// * Every element of the slice must be unique.
    pub const unsafe fn new_unchecked(families: &'a [u32]) -> Self {
        debug_assert!(families.len() >= 2);
        #[cfg(debug_assertions)]
        {
            // const fns are nightmarish :3
            let mut i = 0;
            while i < families.len() - 1 {
                let mut j = i + 1;
                while j < families.len() {
                    if families[i] == families[j] {
                        panic!("SharingFamilies requires all elements be unique");
                    }
                    j += 1;
                }
                i += 1;
            }
        }

        Self(families)
    }
    /// Create from an array.
    /// # Panics
    /// If not every element is unique.
    pub const fn new_array<const N: usize>(families: &'a [u32; N]) -> Self
    where
        Pred<{ N >= 2 }>: Satisified,
    {
        assert!(families.len() >= 2);
        // const fns are nightmarish :3
        let mut i = 0;
        while i < families.len() - 1 {
            let mut j = i + 1;
            while j < families.len() {
                if families[i] == families[j] {
                    panic!("SharingFamilies requires all elements be unique");
                }
                j += 1;
            }
            i += 1;
        }

        Self(families)
    }
    /// Get a slice of the queue families.
    /// Always two or more in length, and all values are unique.
    pub fn families(self) -> &'a [u32] {
        self.0
    }
}
#[derive(Clone, Copy)]
#[repr(i32)]
/// Whether a resource can be shared between queues or is exclusive to one queue.
pub enum SharingMode<'a> {
    /// The resource is owned by one queue, and changes of queue are mediated
    /// by memory barriers.
    Exclusive = vk::SharingMode::EXCLUSIVE.as_raw(),
    /// The resources is shared between several queues, which may concurrently access it
    /// (synchronization allowing).
    ///
    /// The slice *must* contain 2 or more values.
    Concurrent(SharingFamilies<'a>) = vk::SharingMode::CONCURRENT.as_raw(),
}
impl<'a> SharingMode<'a> {
    /*
    /// Create a sharing mode from a slice of families.
    /// If the number of families is zero or one, [`Self::Exclusive`].
    /// Otherwise, [`Self::Concurrent`]
    pub fn from_familes(values: &'a [u32]) -> Self {
        match values.len() {
            0 | 1 => Self::Exclusive,
            2.. => Self::Concurrent(unsafe { SharingFamilies::new_unchecked(values) }),
        }
    }*/
    /// Get the mode enum.
    pub fn mode(&self) -> vk::SharingMode {
        match self {
            Self::Exclusive => vk::SharingMode::EXCLUSIVE,
            Self::Concurrent(_) => vk::SharingMode::CONCURRENT,
        }
    }
    /// Get the sharing families if concurrent, or None.
    pub fn families(&self) -> Option<SharingFamilies> {
        match *self {
            Self::Exclusive => None,
            Self::Concurrent(values) => Some(values),
        }
    }
    /// Get the slice of family indices, or an empty slice if Exclusive.
    pub fn family_slice(&self) -> &[u32] {
        self.families()
            .map(SharingFamilies::families)
            .unwrap_or(&[])
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

/// An error that occurs when the vulkan implementation must allocate memory,
/// e.g. when creating a new handle (image, buffer, etc.) or begining a command buffer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum AllocationError {
    /// The implementation could not allocate enough host memory.
    OutOfHostMemory = vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw(),
    /// The implementation could not allocate enough device memory.
    OutOfDeviceMemory = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw(),
}
impl AllocationError {
    /// Convert from a vulkan result code.
    /// Must be one of `ERROR_OUT_OF_HOST_MEMORY` or `ERROR_OUT_OF_DEVICE_MEMORY` otherwise
    /// the result is unspecified uwu.
    fn from_vk(value: vk::Result) -> Self {
        // Communicates intent better.
        #[allow(clippy::wildcard_in_or_patterns)]
        match value {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfHostMemory,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | _ => Self::OutOfDeviceMemory,
        }
    }
}

/// Create a trait, and create several unit structs implementing it.
///
/// For enumerating a number of Type-State parameters.
///
/// ```
/// state_set!(pub enum trait Soup { pub struct Stew, pub struct Chili, pub struct Oatmeal });
/// ```
macro_rules! typestate_enum {
    {$(#[$outer_attr:meta])* pub enum trait $trait:ident {$($(#[$attr:meta])*pub struct $state:ident),*$(,)?}} => {
        $(#[$outer_attr])*
        pub trait $trait: 'static {}
        $(
            $(#[$attr])*
            pub struct $state;
            impl $trait for $state {}
        )*
    };
}
/// Create a struct as a thin wrapper around another type, with potentially
/// several type-state bounds placed in a [`PhantomData`], and implements [`ThinHandle`] on it.
///
/// ```
/// thin_handle!(pub struct SoupHandle<Kind: Soup>(u8));
/// ```
macro_rules! thin_handle {
    {$(#[$attr:meta])* pub struct $name:ident$(<$($param_name:ident: $bound:path),+$(,)?>)?($underlying:ty$(,)?);} => {
        $(#[$attr])*
        #[repr(transparent)]
        pub struct $name$(<$($param_name: $bound)+>)?($underlying, $(::core::marker::PhantomData<$($param_name,)+>)?);
        /// # Safety
        /// Definition generated by macro.
        unsafe impl $(<$($param_name : $bound,)+>)? crate::ThinHandle for $name$(<$($param_name,)+>)? {
            type Handle = $underlying;
        }
    };
    /*{$(#[$attr:meta])* pub struct $name:ident<$lifetime:lifetime, $($param_name:ident: $bound:path),*$(,)?>($underlying:ty, $(&mut $borrow_ty:ty),+);} => {
        $(#[$attr])*
        #[repr(transparent)]
        pub struct $name<$lifetime, $($param_name: $bound),*>($underlying, ::core::marker::PhantomData<$($param_name),*, $(&$lifetime mut $borrow_ty),+>);
        /// # Safety
        /// Definition generated by macro.
        unsafe impl <$lifetime, $($param_name : $bound),*> crate::ThinHandle for $name<$lifetime, $($param_name),*> {
            type Handle = $underlying;
        }
    };*/
}
/// Trait for types which are a thin wrapper over a Vulkan handle.
/// # Safety
/// `Self` must be `repr(transparent)` over `Self::Handle`
pub unsafe trait ThinHandle: Sized {
    type Handle: Copy;
    // Doesn't work? Always succeeds even when it shouldn't. Weird.
    // const SOUP: () = assert!(std::mem::size_of::<Self>() == std::mem::size_of::<Self::Handle>(),);

    /// Get a copy of the underlying handle.
    /// # Safety
    /// Must not be used to change the underyling object in such a way that its
    /// state no longer matches the existing value of `Self`.
    unsafe fn handle(&self) -> Self::Handle {
        debug_assert_eq!(
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Self::Handle>()
        );
        unsafe { std::mem::transmute_copy(self) }
    }
    /// Discard the typestate and access the underlying handle.
    ///
    /// Note that this is still unsafe, as this handle may refer to the same object
    /// as some other handle, who's typestate must be respected by the caller.
    #[must_use = "dropping the handle may leak resources"]
    unsafe fn into_handle(self) -> Self::Handle {
        unsafe { self.handle() }
    }
    /// Get references to the underlying handles.
    /// # Safety
    /// Must not be used to change the underyling object in such a way that its
    /// state no longer matches the existing value of `Self`
    unsafe fn handles_of(values: &[Self]) -> &[Self::Handle] {
        debug_assert_eq!(
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Self::Handle>()
        );
        unsafe { std::mem::transmute::<&[Self], &[Self::Handle]>(values) }
    }
    /// Create an object from the relavant handle.
    /// # Safety
    /// The handle must be in a state consistent with the typestate of `Self`
    unsafe fn from_handle(handle: Self::Handle) -> Self {
        // Manual transmute since the sizes aren't known.
        debug_assert_eq!(
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Self::Handle>()
        );
        unsafe { (&raw const handle).cast::<Self>().read() }
    }
    /// Change the typestate of the thin handle. This is a jackhammer to drive a nail, so
    /// implementors may provide a safe subset of this operation or otherwise constrain it.
    ///
    /// Usage: ([`Buffer`] implements it's own functions for this, such as [`Buffer::as_subaccess`] just as an example:)
    /// ```no_run
    /// let buffer : Buffer<Storage> = todo!();
    /// let vertex_buffer = unsafe { buffer.with_state::<Buffer<Vertex>>() };
    /// ```
    /// # Safety
    /// The handle must be in a state consistent with the typestate of `Other`
    #[must_use = "dropping the handle may leak resources"]
    unsafe fn with_state<Other: ThinHandle<Handle = Self::Handle>>(self) -> Other {
        Other::from_handle(self.handle())
    }
    /// Change the typestate of the thin handle. This is a jackhammer to drive a nail, so
    /// implementors may provide a safe subset of this operation or otherwise constrain it.
    /// # Safety
    /// The handle must be in a state consistent with the typestate of `Other`
    unsafe fn with_state_mut<Other: ThinHandle<Handle = Self::Handle>>(&mut self) -> &mut Other {
        debug_assert_eq!(
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Self::Handle>()
        );
        std::mem::transmute(self)
    }
    /// Change the typestate of the thin handle. This is a jackhammer to drive a nail, so
    /// implementors may provide a safe subset of this operation or otherwise constrain it.
    /// # Safety
    /// The handle must be in a state consistent with the typestate of `Other`
    unsafe fn with_state_ref<Other: ThinHandle<Handle = Self::Handle>>(&self) -> &Other {
        debug_assert_eq!(
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Self::Handle>()
        );
        std::mem::transmute(self)
    }
}

macro_rules! access_combinations {
    {impl $trait_name:ident for [$(($($name:ident),*$(,)?)),+$(,)?] {
        const $const_name:ident: $const_ty:ty;
    }} => {
        // Trailing comma to force it to be a tuple type, even for single fields.
        $(impl<$($name : $trait_name),+> $trait_name for ($($name),*,) {
            const $const_name: $const_ty = <$const_ty>::from_raw($(<$name as $trait_name>::$const_name.as_raw())|*);
        })+
    };
}

/// Compile-time representation of `vkBufferUsageFlags`.
///
/// Flags can be a single aspect, [`Storage`], or can be placed in a tuple to combine them: `(Storage, Vertex, Indirect)`
pub trait BufferAccess: 'static {
    /// The bitflags for this buffer usage combination.
    const FLAGS: vk::BufferUsageFlags;
}
/// A buffer that can be bound to a command buffer as an index buffer.
pub struct Index;
impl BufferAccess for Index {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::INDEX_BUFFER;
}
/// A buffer that can be bound to a command buffer as a vertex buffer.
pub struct Vertex;
impl BufferAccess for Vertex {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
}
/// A buffer that can be bound to a descriptor set as shader storage.
pub struct Storage;
impl BufferAccess for Storage {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::STORAGE_BUFFER;
}
/// A buffer or image that can used as the source of a copy command.
pub struct TransferSrc;
impl BufferAccess for TransferSrc {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;
}
/// A buffer or image that can used as the destination of a copy command.
pub struct TransferDst;
impl BufferAccess for TransferDst {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_DST;
}
/// A buffer that can be used as the source for dispatch or draw indirection commands.
pub struct Indirect;
impl BufferAccess for Indirect {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::INDIRECT_BUFFER;
}
/// A buffer that can be bound to a descriptor set as uniform storage.
pub struct Uniform;
impl BufferAccess for Uniform {
    const FLAGS: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
}
access_combinations! {
    impl BufferAccess for [
        (A,),
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
/// Flags can be a single aspect, [`Storage`], or can be placed in a tuple to combine them: `(Storage, Sampled, ColorAttachment)`
pub trait ImageAccess: 'static {
    /// The bitflags for this image usage combination.
    const FLAGS: vk::ImageUsageFlags;
}
impl ImageAccess for TransferSrc {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::TRANSFER_SRC;
}
impl ImageAccess for TransferDst {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::TRANSFER_DST;
}
impl ImageAccess for Storage {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::STORAGE;
}
/// An image that can be bound to a descriptor set to be used with a sampler.
pub struct Sampled;
impl ImageAccess for Sampled {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
}
/// An image that can be attached to a Framebuffer as a color attachment.
pub struct ColorAttachment;
impl ImageAccess for ColorAttachment {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
}
/// An image that can be attached to a Framebuffer as a depth and/or stencil attachment.
pub struct DepthStencilAttachment;
impl ImageAccess for Uniform {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
}
/// An image that can be attached to a Framebuffer for transient render operations.
pub struct TransientAttachment;
impl ImageAccess for TransientAttachment {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
}
/// An image that can be attached to a Framebuffer as an input attachment.
pub struct InputAttachment;
impl ImageAccess for InputAttachment {
    const FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::INPUT_ATTACHMENT;
}
access_combinations! {
    impl ImageAccess for [
        (A,),
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

pub struct Pred<const B: bool>;
pub trait Satisified {}
impl Satisified for Pred<true> {}

/// A trait representing whether a given buffer usage is a subset of another.
///
/// Buffers can be safely reinterpreted as a buffer of a subset of usages.
///
/// For example, `(Index, Vertex, Uniform)` is `BufferSuperset<(Uniform, Vertex)>`,
/// but is not `BufferSuperset<(Indirect, Vertex)>`
pub unsafe trait BufferSuperset<Sub: BufferAccess>: BufferAccess {}

unsafe impl<Super, Sub> BufferSuperset<Sub> for Super
where
    Super: BufferAccess,
    Sub: BufferAccess,
    Pred<{ Super::FLAGS.contains(Sub::FLAGS) }>: Satisified,
{
}
/// A trait representing whether a given image usage is a subset of another.
///
/// Images can be safely reinterpreted as a image of a subset of usages.
///
/// For example, `(Storage, TransferSrc)` is `BufferSuperset<(Storage,)>`,
/// but is not `BufferSuperset<(TransferDst, TransferSrc)>`
pub unsafe trait ImageSuperset<Sub: ImageAccess>: ImageAccess {}
unsafe impl<Super, Sub> ImageSuperset<Sub> for Super
where
    Super: ImageAccess,
    Sub: ImageAccess,
    Pred<{ Super::FLAGS.contains(Sub::FLAGS) }>: Satisified,
{
}

/// A trait representing the extent of a image of various dimensionalities.
pub unsafe trait Extent {
    /// The dimensionality of the extent represented by this type.
    /// For example, 2D Array has dimensionality 2.
    type Dim: Dimensionality;
    /// The extrapolated extent3D.
    /// All dimensions must be non-zero. Axes beyond the type's dimensionality should be `1`.
    fn extent(&self) -> vk::Extent3D;
    /// The number of layers, or 1 if not a layered type.
    fn layers(&self) -> NonZero<u32>;
}
/// The width of a 1-dimensional image.
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
    pub layers: NonZero<u32>,
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
        self.layers
    }
}
/// The width and height of a 2-dimensional array image.
pub struct Extent2DArray {
    pub width: NonZero<u32>,
    pub height: NonZero<u32>,
    pub layers: NonZero<u32>,
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
        self.layers
    }
}
/// The width and height of a 2-dimensional image.
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
/// The number of dimensions in an image, including whether it's an array.
pub trait Dimensionality {
    type Extent: Extent;
    const TYPE: vk::ImageType;
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
    type Extent = Extent1D;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
}
impl Dimensionality for D1Array {
    type Extent = Extent1DArray;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
}
impl Dimensionality for D2 {
    type Extent = Extent2D;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
}
impl Dimensionality for D2Array {
    type Extent = Extent2DArray;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
}
impl Dimensionality for D3 {
    type Extent = Extent3D;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_3D;
}

/// A value representing an image's pixel format.
pub trait ImageFormat {
    fn format(&self) -> vk::Format;
}
/// Formats with only a color aspect.
#[derive(Clone, Copy)]
#[repr(i32)]
pub enum ColorFormat {
    Rgba8Unorm = vk::Format::R8G8B8A8_UNORM.as_raw(),
    Rgba8Srgb = vk::Format::R8G8B8A8_SRGB.as_raw(),
}
impl ImageFormat for ColorFormat {
    fn format(&self) -> vk::Format {
        vk::Format::from_raw(*self as _)
    }
}
#[derive(Clone, Copy)]
/// Formats with a depth and/or stencil aspect.
pub enum DepthStencilFormat {}
impl ImageFormat for DepthStencilFormat {
    fn format(&self) -> vk::Format {
        vk::Format::from_raw(*self as _)
    }
}

/// An owned image handle.
/// # Typestates
/// * `Access`: The `VkImageUsageFlags` this image is statically known to possess (e.g. `(Storage, TransferSrc)`)
/// * `Dim`: The dimensionality of an image, including whether it is an array of images (e.g. [`D2`])
/// * `Format`: The Image aspects this image's format is known to possess (e.g. [`ColorFormat`]).
#[repr(transparent)]
#[must_use = "dropping the handle will not destroy the image and may leak resources"]
pub struct Image<Access: ImageAccess, Dim: Dimensionality, Format: ImageFormat>(
    vk::Image,
    PhantomData<(Access, Dim, Format)>,
);
unsafe impl<Access: ImageAccess, Dim: Dimensionality, Format: ImageFormat> ThinHandle
    for Image<Access, Dim, Format>
{
    type Handle = vk::Image;
}

/// An owned image view handle.
/// # Typestates
/// * `Access`: The `VkImageUsageFlags` this image is statically known to possess (e.g. `(Storage, TransferSrc)`),
///   which is a subset of the usages of the parent [`Image`].
/// * `Dim`: The dimensionality of an image, including whether it is an array of images (e.g. [`D2`])
/// * `Format`: The Image aspects this image's format is known to possess (e.g. [`ColorFormat`]).
#[repr(transparent)]
#[must_use = "dropping the handle will not destroy the view and may leak resources"]
pub struct ImageView<Access: ImageAccess, Dim: Dimensionality, Format: ImageFormat>(
    vk::ImageView,
    PhantomData<(Access, Dim, Format)>,
);
unsafe impl<Access: ImageAccess, Dim: Dimensionality, Format: ImageFormat> ThinHandle
    for ImageView<Access, Dim, Format>
{
    type Handle = vk::ImageView;
}

/// An owned image view handle.
/// # Typestates
/// * `Access`: The `vkBufferUsageFlags` this image is statically known to possess (e.g. `(Storage, TransferSrc)`)
#[repr(transparent)]
#[must_use = "dropping the handle will not destroy the buffer and may leak resources"]
pub struct Buffer<Access: BufferAccess>(vk::Buffer, PhantomData<Access>);
unsafe impl<Access: BufferAccess> ThinHandle for Buffer<Access> {
    type Handle = vk::Buffer;
}
impl<SubAccess: BufferAccess, SuperAccess: BufferSuperset<SubAccess> + BufferAccess>
    AsMut<Buffer<SubAccess>> for Buffer<SuperAccess>
{
    fn as_mut(&mut self) -> &mut Buffer<SubAccess> {
        self.as_subaccess_mut()
    }
}
impl<SubAccess: BufferAccess, SuperAccess: BufferSuperset<SubAccess> + BufferAccess>
    AsRef<Buffer<SubAccess>> for Buffer<SuperAccess>
{
    fn as_ref(&self) -> &Buffer<SubAccess> {
        self.as_subaccess()
    }
}

impl<Access: BufferAccess> Buffer<Access> {
    /// Forget some of the access types. To bring back the forgotten types, the *unsafe*
    /// function [`Self::into_access_unchecked`] can be used.
    ///
    /// Use [`Self::as_subaccess`] or [`Self::as_subaccess_mut`] to get temporary subaccess without
    /// forgetting the original flags.
    /// # Compiler Errors
    /// If you see `expected false, found true` as an error, that means
    /// the `SubAccess` flags is not statically known to be a subtype of this buffer's Access flags.
    /// I do not know how to make a better error :3
    pub fn into_subaccess<SubAccess>(self) -> Buffer<SubAccess>
    where
        SubAccess: BufferAccess,
        Access: BufferSuperset<SubAccess>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperAccess> *is a* Buffer<SubAccess>.
        unsafe { self.with_state() }
    }
    pub fn reference<SubAccess: BufferAccess>(&self) -> BufferReference<SubAccess>
    where
        Access: BufferSuperset<SubAccess>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperAccess> *is a* Buffer<SubAccess>.
        unsafe { BufferReference::from_handle(self.handle()) }
    }
    pub fn as_subaccess<SubAccess>(&self) -> &Buffer<SubAccess>
    where
        SubAccess: BufferAccess,
        Access: BufferSuperset<SubAccess>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperAccess> *is a* Buffer<SubAccess>.
        unsafe { self.with_state_ref() }
    }
    pub fn as_subaccess_mut<SubAccess>(&mut self) -> &mut Buffer<SubAccess>
    where
        SubAccess: BufferAccess,
        Access: BufferSuperset<SubAccess>,
    {
        // Safety: Subset<> bound guarantees Buffer<SuperAccess> *is a* Buffer<SubAccess>.
        unsafe { self.with_state_mut() }
    }
    /// Convert the type into a different kind of access.
    /// # Safety
    /// The buffer cannot be used for any kind of access that wasn't specified at creation time.
    pub unsafe fn into_access_unchecked<NewAccess: BufferAccess>(self) -> Buffer<NewAccess> {
        // Safety: Forwarded to caller.
        unsafe { self.with_state() }
    }
}
/// A thin, shared reference to a [`Buffer`] with some subset of usages. Acquired using [`Buffer::reference`].
///
/// This is used anywhere where vulkan expects a slice of buffers, where `&[&Buffer]` is one layer of
/// indirection too deep to be directly handed off to the implementation.
#[repr(transparent)]
pub struct BufferReference<'a, Access: BufferAccess>(
    vk::Buffer,
    PhantomData<(Access, &'a Buffer<Access>)>,
);

unsafe impl<Access: BufferAccess> ThinHandle for BufferReference<'_, Access> {
    type Handle = vk::Buffer;
}

/// A vulkan device.
///
/// This contains all the function pointers needed to operate. All device-scope operations go through this object.
pub struct Device(ash::Device);

thin_handle! {
    pub struct Queue(vk::Queue);
}
pub trait FenceState {}
pub trait KnownFenceState: FenceState {
    const CREATION_FLAGS: vk::FenceCreateFlags;
}
/// Typestate for a `Fence` which is currently signaled.
pub struct Signaled;
impl FenceState for Signaled {}
impl KnownFenceState for Signaled {
    const CREATION_FLAGS: vk::FenceCreateFlags = vk::FenceCreateFlags::SIGNALED;
}
/// Typestate for a `Fence` which is currently unsignaled and not in the process of becoming
/// signaled.
pub struct Unsignaled;
impl FenceState for Unsignaled {}
impl KnownFenceState for Unsignaled {
    const CREATION_FLAGS: vk::FenceCreateFlags = vk::FenceCreateFlags::empty();
}
/// Typestate for a `Fence` which is eventually going to become signaled due to a previous
/// queue submission operation.
///
/// This is a *hint* that the Fence is in a state where it *may* be accessed from any time
/// by an external source.
pub struct Pending;
impl FenceState for Pending {}
/// A synchronization primitive for GPU->CPU communication.
/// # Typestate
/// * `State`: Whether the fence is signaled, unsignaled, or in the process of becoming signaled.
#[repr(transparent)]
pub struct Fence<State: FenceState>(vk::Fence, PhantomData<State>);

unsafe impl<State: FenceState> ThinHandle for Fence<State> {
    type Handle = vk::Fence;
}
thin_handle! {
    /// A synchronization primitive for CPU->GPU communication and fine-grained *intra*-queue GPU->GPU dependencies.
    pub struct Event(vk::Event);
}

thin_handle! {
    /// A synchronization primitive for GPU->GPU communication and coarse-grained inter-queue dependencies.
    pub struct Semaphore(vk::Semaphore);
}

thin_handle! {
    /// A pool from which many [`CommandBuffer`]s may be allocated.
    /// Any operation on any command buffer allocated from this pool requires
    /// synchronous access to the pool as well.
    #[must_use = "dropping the handle will not destroy the command pool and may leak resources"]
    pub struct CommandPool(vk::CommandPool);
}
typestate_enum! {
    /// A type representing whether a command buffer is Primary or Secondary.
    pub enum trait CommandBufferLevel {
        /// A primary command buffer, executed directly by the queue.
        pub struct Primary,
        /// A second command buffer, exectuted indirectly by being "called" by [`Primary`] command buffers.
        pub struct Secondary,
    }
}
thin_handle! {
    /// A command buffer, allocated from a [`CommandPool`].
    ///
    /// All operations require synchronous access to this buffer, as well as the pool it was allocated from.
    /// # Typestates
    /// * `Level` - [`Primary`] or [`Secondary`], describing the allocation kind, which restricts
    ///   the operations that can be recorded into the buffer.
    #[must_use = "dropping the handle will not deallocate the buffer and may leak resources"]
    pub struct CommandBuffer<Level: CommandBufferLevel>(vk::CommandBuffer);
}

/// A render pass, describing the flow of rendering operations on a Framebuffer.
///
/// # Typestates
/// * `SUBPASSES` - A non-zero integer representing the number of subpasses this renderpass consists of.
#[repr(transparent)]
#[must_use = "dropping the handle will not destroy the renderpass and may leak resources"]
pub struct RenderPass<const SUBPASSES: usize>(vk::RenderPass)
where
    SubpassCount<SUBPASSES>: ValidSubpassCount;
unsafe impl<const N: usize> ThinHandle for RenderPass<N>
where
    SubpassCount<N>: ValidSubpassCount,
{
    type Handle = vk::RenderPass;
}

/// An error that occurs when waiting on a fence.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum SubmitError {
    /// The implementation could not allocate enough host memory.
    OutOfHostMemory = vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw(),
    /// The implementation could not allocate enough device memory.
    OutOfDeviceMemory = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw(),
    /// The device context is irreparably destroyed. Oops!
    DeviceLost = vk::Result::ERROR_DEVICE_LOST.as_raw(),
}
impl SubmitError {
    /// Convert from a vulkan result code.
    /// Must be one of `ERROR_OUT_OF_HOST_MEMORY`, `ERROR_OUT_OF_DEVICE_MEMORY`, or `ERROR_DEVICE_LOST`, otherwise
    /// the result is unspecified uwu.
    fn from_vk(value: vk::Result) -> Self {
        // Communicates intent better.
        #[allow(clippy::wildcard_in_or_patterns)]
        match value {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfHostMemory,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Self::OutOfDeviceMemory,
            vk::Result::ERROR_DEVICE_LOST | _ => Self::DeviceLost,
        }
    }
}
#[repr(i32)]
pub enum FencePoll<Success, Pending> {
    Signaled(Success) = vk::Result::SUCCESS.as_raw(),
    Unsignaled(Pending) = vk::Result::NOT_READY.as_raw(),
}
pub enum Timeout {
    /// Don't wait, check the status and immediately return.
    Poll,
    /// Wait the specified number of nanoseconds. The resolution is unspecified,
    /// and the implementation may wait longer or shorter than the given time.
    Nanos(NonZero<u64>),
    /// Wait forever, until signaled or an error occurs.
    /// If you're using this statically, consider using the non-timeout variant of the call
    /// which provides better type-level information.
    Infinite,
}
impl Timeout {
    fn get(self) -> u64 {
        match self {
            Self::Poll => 0,
            Self::Nanos(n) => n.get(),
            Self::Infinite => u64::MAX,
        }
    }
}

impl Device {
    /// Wrap an [`ash`] device.
    ///
    /// This is the main entry point for this crate.
    pub fn from_ash(device: ash::Device) -> Self {
        Device(device)
    }
    /// Submit work to a queue.
    /// An optional fence will be signaled upon completion.
    pub unsafe fn submit_with_fence(
        &self,
        queue: &mut Queue,
        fence: Fence<Unsignaled>,
    ) -> Result<Fence<Pending>, SubmitError> {
        self.0
            .queue_submit(queue.handle(), &[], fence.handle())
            .map_err(SubmitError::from_vk)?;
        Ok(fence.with_state())
    }
    /// Create a fence in the given state, [`Signaled`] or [`Unsignaled`].
    /// ```no_run
    /// # let device : Device = todo!();
    /// let signaled = device.create_fence::<Unsignaled>().unwrap();
    /// ```
    pub unsafe fn create_fence<State: KnownFenceState>(
        &self,
    ) -> Result<Fence<State>, AllocationError> {
        self.0
            .create_fence(
                &vk::FenceCreateInfo::default().flags(State::CREATION_FLAGS),
                None,
            )
            .map(|handle| Fence::from_handle(handle))
            .map_err(AllocationError::from_vk)
    }
    /// Set each fence to the "Unsignaled" state. It is valid to reset an already unsignaled
    /// fence.
    pub unsafe fn reset_fences<const N: usize, State: KnownFenceState>(
        &self,
        fences: [Fence<State>; N],
    ) -> Result<[Fence<Unsignaled>; N], AllocationError> {
        self.0
            .reset_fences(ThinHandle::handles_of(&fences))
            .map_err(AllocationError::from_vk)?;
        Ok(fences.map(|f| Fence::with_state(f)))
    }
    /// Poll the fence. If the result is [`FencePoll::Unsignaled`], the result *may* be immediately
    /// out-of-date.
    pub unsafe fn fence_status(
        &self,
        fence: Fence<Pending>,
    ) -> Result<FencePoll<Fence<Signaled>, Fence<Pending>>, SubmitError> {
        let handle = fence.into_handle();
        let status = self
            .0
            .get_fence_status(handle)
            .map_err(SubmitError::from_vk)?;
        Ok(if status {
            FencePoll::Signaled(Fence::from_handle(handle))
        } else {
            FencePoll::Unsignaled(Fence::from_handle(handle))
        })
    }
    /// Convinience wrapper around [`Self::wait_all_fences_timeout`] for once fence.
    pub unsafe fn wait_fence_timeout(
        &self,
        fence: Fence<Pending>,
        timeout: Timeout,
    ) -> Result<FencePoll<Fence<Signaled>, Fence<Pending>>, SubmitError> {
        match self.wait_all_fences_timeout([fence], timeout) {
            Ok(FencePoll::Signaled([s])) => Ok(FencePoll::Signaled(s)),
            Ok(FencePoll::Unsignaled([u])) => Ok(FencePoll::Unsignaled(u)),
            Err(e) => Err(e),
        }
    }
    /// Convinience wrapper around [`Self::wait_all_fences`] for once fence.
    pub unsafe fn wait_fence(&self, fence: Fence<Pending>) -> Result<Fence<Signaled>, SubmitError> {
        match self.wait_fence_timeout(fence, Timeout::Infinite) {
            Ok(FencePoll::Signaled(s)) => Ok(s),
            Ok(FencePoll::Unsignaled(_)) => unreachable!(),
            Err(e) => Err(e),
        }
    }
    /// Wait for every fence in the array to become signaled, or fail after some timeout.
    /// If timeout is not [`Timeout::Infinite`], the result may be immediately out-of-date.
    pub unsafe fn wait_all_fences_timeout<const N: usize>(
        &self,
        fences: [Fence<Pending>; N],
        timeout: Timeout,
    ) -> Result<FencePoll<[Fence<Signaled>; N], [Fence<Pending>; N]>, SubmitError> {
        if N == 0 {
            // UB to wait on an empty slice, but since we know at compile time, we can just... do nothing :D
            // Can't use a literal because generics. Return some arbitrary ZST:
            let empty = std::mem::zeroed();
            assert_eq!(std::mem::size_of_val(&empty), 0);
            return Ok(FencePoll::Signaled(empty));
        }
        let res = self
            .0
            .wait_for_fences(ThinHandle::handles_of(&fences), true, timeout.get());
        match res {
            // Reinterpret with Signaled typestate
            Ok(()) => Ok(FencePoll::Signaled(fences.map(|f| f.with_state()))),
            // Unsignalled, all are still pending
            Err(vk::Result::TIMEOUT) => Ok(FencePoll::Unsignaled(fences)),
            Err(e) => Err(SubmitError::from_vk(e)),
        }
    }
    /// Wait for every fence in the array to become signaled.
    /// Convinience wrapper around [`Self::wait_all_fences_timeout`] with simpler types.
    pub unsafe fn wait_all_fences<const N: usize>(
        &self,
        fences: [Fence<Pending>; N],
    ) -> Result<[Fence<Signaled>; N], SubmitError> {
        match self.wait_all_fences_timeout(fences, Timeout::Infinite) {
            Ok(FencePoll::Signaled(s)) => Ok(s),
            Ok(FencePoll::Unsignaled(_)) => unreachable!(),
            Err(e) => Err(e),
        }
    }
    /// Wait for any fence in the slice to become signaled, or until a timeout.
    /// Returns `Ok(true)` if one or more fences were signaled, `Ok(false)` if not.
    pub unsafe fn wait_any_fence_timeout(
        &self,
        // To my surprise, this is internally synchronized, so no need for &mut.
        fences: &[Fence<Pending>],
        timeout: Timeout,
    ) -> Result<bool, SubmitError> {
        // Cost?? In *my* zero cost abstraction???
        if fences.is_empty() {
            return Ok(true);
        }
        let res = self
            .0
            .wait_for_fences(ThinHandle::handles_of(fences), false, timeout.get());
        match res {
            Ok(()) => Ok(true),
            // Unsignalled, all are still pending
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(SubmitError::from_vk(e)),
        }
    }
    pub unsafe fn wait_any_fence(&self, fences: &[Fence<Pending>]) -> Result<bool, SubmitError> {
        // For parity with the wait_all* set of functions. Unfortunately,
        // there's no extra type-information we can communicate here, unlike the other family.
        self.wait_any_fence_timeout(fences, Timeout::Infinite)
    }
    /// Destroy a fence.
    /// The bound [`KnownFenceState`] is used to ensure that the fence is not in use within
    /// a currently-executing submission.
    pub unsafe fn destroy_fence<State: KnownFenceState>(&self, fences: Fence<State>) -> &Self {
        self.0.destroy_fence(fences.into_handle(), None);
        self
    }
    /// Create a [`CommandPool`] from which [`CommandBuffer`]s may be allocated.
    pub unsafe fn create_command_pool(
        &self,
        info: &vk::CommandPoolCreateInfo,
    ) -> Result<CommandPool, AllocationError> {
        self.0
            .create_command_pool(info, None)
            .map(CommandPool)
            .map_err(AllocationError::from_vk)
    }
    /// Destroy a [`CommandPool`].
    /// # Safety
    /// Any [`CommandBuffer`]s allocated from this pool do not need to be [freed](Self::free_command_buffers) prior
    /// to this call, however they become invalid to access and should be dropped.
    pub unsafe fn destroy_command_pool(&self, pool: CommandPool) -> &Self {
        self.0.destroy_command_pool(pool.0, None);
        self
    }
    /// Allocate [`CommandBuffer`]s from the given pool.
    ///
    /// The constant `N` is the number of command buffers to allocate. Use
    /// this along with a destructuring to create several buffers at once:
    /// ```no_run
    /// # let device : Device = todo!();
    /// let mut pool = device.create_command_pool(&todo!());
    /// let [buffer_a, buffer_b] = device.allocate_primary_buffers(&mut pool).unwrap();
    /// ```
    pub unsafe fn allocate_primary_buffers<const N: usize>(
        &self,
        pool: &mut CommandPool,
    ) -> Result<[CommandBuffer<Primary>; N], AllocationError> {
        let vec = self
            .0
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(N as _)
                    .command_pool(pool.handle())
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
            .map_err(AllocationError::from_vk)?;
        let array = <[vk::CommandBuffer; N]>::try_from(vec).unwrap();
        Ok(array.map(|h| CommandBuffer::from_handle(h)))
    }
    /// Free [`CommandBuffer`]s back to the given pool.
    pub unsafe fn free_command_buffers<const N: usize, Level: CommandBufferLevel>(
        &self,
        pool: &mut CommandPool,
        buffers: [CommandBuffer<Level>; N],
    ) -> &Self {
        self.0
            .free_command_buffers(pool.handle(), ThinHandle::handles_of(&buffers));
        self
    }
    /// Create a buffer. The `BufferUsageFlags` are passed at compile time:
    ///
    /// ```no_run
    /// # let device: Device = todo!();
    /// let buffer = device.create_buffer::<(Storage, TransferSrc, TransferDst)>(
    ///         1024,
    ///         SharingMode::Concurrent(SharingFamilies::from_array(&[0, 1])),
    ///     );
    /// ```
    pub unsafe fn create_buffer<Access: BufferAccess>(
        &self,
        size: u64,
        sharing: SharingMode,
    ) -> Result<Buffer<Access>, AllocationError> {
        self.0
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .usage(Access::FLAGS)
                    .size(size)
                    .sharing_mode(sharing.mode())
                    .queue_family_indices(sharing.family_slice()),
                None,
            )
            .map(|handle| Buffer::from_handle(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_buffer<Access: BufferAccess>(&self, buffer: Buffer<Access>) -> &Self {
        self.0.destroy_buffer(buffer.handle(), None);
        self
    }
    /// Create an image. The `vkImageUsageFlags`, Dimensionality, and Aspects are passed at compile time:
    ///
    /// ```no_run
    /// # let device: Device = todo!();
    /// let image = device.create_image::<(TransferSrc, ColorAttachment), _, _>(
    ///         // Extent2D makes a `D2` image.
    ///         // You can also use Extent1DArray, Extent3D, etc.
    ///         Extent2D {
    ///             width: NonZero::new(128).unwrap(),
    ///             height: NonZero::new(128).unwrap()
    ///         },
    ///         NonZero::new(1).unwrap(),
    ///         // ColorFormat makes a COLOR_ASPECT image.
    ///         // You can also use DepthStencilFormat
    ///         ColorFormat::Rgba8Unorm,
    ///         vk::ImageTiling::Optimal,
    ///         SharingMode::Exclusive
    ///     ).unwrap();
    /// ```
    ///
    /// See: [`ImageAccess`], [`Extent`], [`ColorFormat`], [`DepthStencilFormat`].
    pub unsafe fn create_image<Access: ImageAccess, Ext: Extent, Format: ImageFormat>(
        &self,
        extent: Ext,
        mip_levels: NonZero<u32>,
        format: Format,
        samples: vk::SampleCountFlags,
        tiling: vk::ImageTiling,
        sharing: SharingMode,
    ) -> Result<Image<Access, Ext::Dim, Format>, AllocationError> {
        self.0
            .create_image(
                &vk::ImageCreateInfo::default()
                    .extent(extent.extent())
                    .array_layers(extent.layers().get())
                    .queue_family_indices(sharing.family_slice())
                    .sharing_mode(sharing.mode())
                    .usage(Access::FLAGS)
                    .image_type(Ext::Dim::TYPE)
                    .format(format.format())
                    .mip_levels(mip_levels.get())
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .samples(samples)
                    .tiling(tiling),
                None,
            )
            .map(|handle| Image::from_handle(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn create_render_pass<const N: usize>(
        &self,
        subpasses: &[vk::SubpassDescription; N],
    ) -> Result<RenderPass<N>, AllocationError>
    where
        SubpassCount<N>: ValidSubpassCount,
    {
        self.0
            .create_render_pass(
                &vk::RenderPassCreateInfo::default().subpasses(subpasses),
                None,
            )
            .map(|handle| RenderPass::from_handle(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_image<Access: ImageAccess, Dim: Dimensionality, Format: ImageFormat>(
        &self,
        image: Image<Access, Dim, Format>,
    ) -> &Self {
        self.0.destroy_image(image.handle(), None);
        self
    }
    pub unsafe fn begin_command_buffer<'a, Level: CommandBufferLevel>(
        &'_ self,
        _pool: &'a mut CommandPool,
        buffer: &'a mut CommandBuffer<Level>,
    ) -> Result<RecordingBuffer<'a, Level, OutsideRender>, AllocationError> {
        self.0
            .begin_command_buffer(buffer.handle(), &vk::CommandBufferBeginInfo::default())
            .map_err(AllocationError::from_vk)?;
        Ok(RecordingBuffer {
            buffer: buffer.handle(),
            _typed_buffer: PhantomData,
            _state: PhantomData,
            _pool: PhantomData,
        })
    }
    pub unsafe fn bind_vertex_buffers<
        'a,
        'b,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
    >(
        &'a self,
        buffer: &'b mut RecordingBuffer<Level, MaybeInsideRender>,
        first: u32,
        vertex_buffers: &'b [BufferReference<'b, Vertex>],
        offsets: &'b [u64],
    ) -> &'a Self {
        self.0.cmd_bind_vertex_buffers(
            buffer.buffer,
            first,
            ThinHandle::handles_of(vertex_buffers),
            offsets,
        );
        self
    }
    pub unsafe fn bind_index_buffer<
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
    >(
        &self,
        buffer: &mut RecordingBuffer<Level, MaybeInsideRender>,
        index_buffer: impl AsRef<Buffer<Index>>,
        offset: u64,
        ty: vk::IndexType,
    ) -> &Self {
        self.0
            .cmd_bind_index_buffer(buffer.handle(), index_buffer.as_ref().handle(), offset, ty);
        self
    }
    pub unsafe fn end_command_buffer<Level: CommandBufferLevel>(
        &self,
        buffer: RecordingBuffer<Level, OutsideRender>,
    ) -> Result<&Self, AllocationError> {
        self.0
            .end_command_buffer(buffer.buffer)
            .map_err(AllocationError::from_vk)?;
        Ok(self)
    }
    pub unsafe fn begin_render_pass<const N: usize, Level: CommandBufferLevel>(
        &self,
        buffer: RecordingBuffer<Level, OutsideRender>,
        renderpass: &RenderPass<N>,
    ) -> RecordingBuffer<Level, RemainingSubpasses<N>>
    where
        SubpassCount<N>: ValidSubpassCount,
    {
        self.0
            .cmd_begin_render_pass(buffer.buffer, todo!(), vk::SubpassContents::INLINE);
        unsafe { buffer.with_state() }
    }
    pub unsafe fn draw<const N: usize, Level: CommandBufferLevel>(
        &self,
        buffer: &mut RecordingBuffer<Level, RemainingSubpasses<N>>,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> &Self
    where
        SubpassCount<N>: ValidSubpassCount,
    {
        self.0.cmd_draw(
            buffer.buffer,
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        );
        self
    }
    pub unsafe fn draw_indexed<const N: usize, Level: CommandBufferLevel>(
        &self,
        buffer: &mut RecordingBuffer<Level, RemainingSubpasses<N>>,
        vertex_offset: i32,
        indices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> &Self
    where
        SubpassCount<N>: ValidSubpassCount,
    {
        self.0.cmd_draw_indexed(
            buffer.buffer,
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            vertex_offset,
            instances.start,
        );
        self
    }
    pub unsafe fn next_subpass<const N: usize, Level: CommandBufferLevel>(
        &self,
        buffer: RecordingBuffer<Level, RemainingSubpasses<N>>,
    ) -> RecordingBuffer<Level, RemainingSubpasses<{ N - 1 }>>
    where
        SubpassCount<N>: HasNextSubpass,
        // Shouldn't HasNextSubpass already imply this?
        // I guess there's a reason why generic const arithmetic is unstable :3c
        SubpassCount<{ N - 1 }>: ValidSubpassCount,
    {
        self.0
            .cmd_next_subpass(buffer.buffer, vk::SubpassContents::INLINE);
        unsafe { buffer.with_state() }
    }
    pub unsafe fn end_render_pass<Level: CommandBufferLevel>(
        &self,
        buffer: RecordingBuffer<Level, RemainingSubpasses<1>>,
    ) -> RecordingBuffer<Level, OutsideRender> {
        self.0.cmd_end_render_pass(buffer.buffer);
        unsafe { buffer.with_state() }
    }
}
unsafe fn waawa() {
    let device: Device = todo!();
    let mut queue: Queue = todo!();
    let mut pool = device
        .create_command_pool(&vk::CommandPoolCreateInfo::default())
        .unwrap();
    let buffer = device
        .create_buffer::<(Vertex, Index)>(1024, SharingMode::Exclusive)
        .unwrap();
    let image = device
        .create_image::<(TransferDst, TransferSrc), _, _>(
            Extent2D {
                width: NonZero::new(10).unwrap(),
                height: NonZero::new(10).unwrap(),
            },
            NonZero::new(1).unwrap(),
            ColorFormat::Rgba8Unorm,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )
        .unwrap();
    let fence = device.create_fence::<Unsignaled>().unwrap();
    let renderpass = device.create_render_pass(&[todo!(), todo!()]).unwrap();
    let [mut cb1, mut cb2] = device.allocate_primary_buffers(&mut pool).unwrap();

    let recording = device.begin_command_buffer(&mut pool, &mut cb1).unwrap();
    let mut render_pass = device.begin_render_pass(recording, &renderpass);
    device
        .bind_vertex_buffers(&mut render_pass, 0, &[buffer.reference()], &[0])
        .bind_index_buffer(&mut render_pass, buffer, 512, vk::IndexType::UINT16)
        .draw_indexed(&mut render_pass, 0, 0..10, 0..1);
    let mut render_pass = device.next_subpass(render_pass);
    let recording = device.end_render_pass(render_pass);
    device
        .end_command_buffer(recording)
        .unwrap()
        .free_command_buffers(&mut pool, [cb1, cb2])
        .destroy_command_pool(pool)
        .destroy_buffer(buffer)
        .destroy_image(image);
    let fence: Fence<Unsignaled> = device.create_fence::<Unsignaled>().unwrap();
    let fence: Fence<Pending> = device.submit_with_fence(&mut queue, fence).unwrap();
    let fence: Fence<Signaled> = device.wait_fence(fence).unwrap();
    device.destroy_fence(fence);
}

pub trait CommandBufferState {}
/// Typestate for a recording command buffer that is not currently rendering.
pub struct OutsideRender;
/// Typestate for a recording command that is inside a renderpass.
/// # Typestates
/// * `REMAINING_PASSES`: how many subpasses left in the renderpass. A renderpass can only
///   be ended once this reaches the final subpass.
pub struct RemainingSubpasses<const REMAINING_PASSES: usize>
where
    SubpassCount<REMAINING_PASSES>: ValidSubpassCount;
impl CommandBufferState for OutsideRender {}
impl<const REMAINING_PASSES: usize> CommandBufferState for RemainingSubpasses<REMAINING_PASSES> where
    SubpassCount<REMAINING_PASSES>: ValidSubpassCount
{
}

pub struct SubpassCount<const N: usize>;
/// Implemented for all non-zero subpass counts.
pub trait ValidSubpassCount {}
impl<const N: usize> ValidSubpassCount for SubpassCount<N> where Pred<{ N > 0 }>: Satisified {}
/// Implemented for all valid subpass counts where `{Count - 1}` is also a valid subpass count.
pub trait HasNextSubpass: ValidSubpassCount {
    type NextSubpass: ValidSubpassCount;
}
impl<const N: usize> HasNextSubpass for SubpassCount<N>
where
    SubpassCount<N>: ValidSubpassCount,
    SubpassCount<{ N - 1 }>: ValidSubpassCount,
{
    type NextSubpass = SubpassCount<{ N - 1 }>;
}

#[must_use = "dropping the handle will result in a command buffer orphaned in an incomplete state"]
/// A temporary handle to a command buffer in the `recording` state.
/// # Typestates
/// * `Level`: The allocated level of the command buffer being recorded, primary or secondary.
///   The level of the buffer affects the valid operations that can be recorded into it.
/// * `State`: The current scope of the command buffer. For example, whether the command buffer
///   is currently "inside" a render pass, and can thus issue `draw` commands.
pub struct RecordingBuffer<'a, Level: CommandBufferLevel, State: CommandBufferState> {
    buffer: vk::CommandBuffer,
    _typed_buffer: PhantomData<&'a mut CommandBuffer<Level>>,
    _state: PhantomData<State>,
    _pool: PhantomData<&'a mut CommandPool>,
}
unsafe impl<'a, Level: CommandBufferLevel, State: CommandBufferState> ThinHandle
    for RecordingBuffer<'a, Level, State>
{
    type Handle = vk::CommandBuffer;
}
