//! # Fuzzy's Funny Zero-cost Excessively-Typestate Vulkan Abstraction
//! A thin wrapper around [`ash`]. This crate is not associated with the ash project!
//!
//! To get started, fetch an [`ash::Device`] and pass it to [`Device::from_ash`].
//!
//! ## Typestate
//! This crate makes extensive (perhaps excessive) use of the
//! [typestate pattern](https://en.wikipedia.org/wiki/Typestate_analysis) to validate *some* operations.
//! Of course, the scope of what can be declared at compile time is inherently limited by the need for
//! runtime flexibility, so it is mostly used in locations where it is a common usage pattern that the
//! "shape" of the data will not be totally dynamic.
//!
//! For example, an [`Image`] contains the following compile-time information:
//! * Whether it is a Color or Depth/Stencil Aspect image.
//! * Its dimensionality (1D, 2D, 3D)
//! * Its array-ness
//! * Its vkImageUsageFlags
//! * Whether multisampling is in use.
//!
//! A fully-named image type thus looks something like:
//! ```no_run
//! # use fzvk::*;
//! let image : Image<(Storage, TransferDst), D2Array, ColorFormat, SingleSampled> = todo!();
//! ```
//! Since each of these facts are almost always known at compile time and have large implications
//! for valid usage of the resource, they are tracked at compile time too.
//!
//! This allows for functions to validate, at compile time, whether certain aspects of the operation
//! are valid and produce a compiler error if the state of an object is not statically known to be correct.
//! See [`Device::bind_vertex_buffers`] for what that looks like.
//!
//! ### What if it *needs* to be dynamic?
//! If there is a finite enumeration of dynamic states, `enum`s or even `union`s around the possible
//! typestates are good choices.
//!
//! For states too dynamic even for that strategy, raw vulkan may be used.
//! Since this crate is a thin wrapper around [`ash`], it is perfectly valid to mix and match API
//! calls from both. See the [`ThinHandle`] trait for how to move data in and out of the typestate
//! ecosystem and the safety considerations of this.
//!
//! ## Zero cost
//! * Types do not contain any runtime state beyond their vulkan handles.
//!   * Typestate "data" is zero-sized and compiles down to nothing at all
//!   * Handles do not contain the necessary function pointers, thus all operations must go through
//!     the central [`Device`] type.
//!   * The big exception to this is [`Device`], which contains several kilobytes of function pointers
//!     on the stack. We can't all be winners :3
//! * Allocations are avoided like the plague.
//!   * Wherever possible, const-generic arrays are used instead to make variable-length operations
//!     occur exclusively on the stack.
//! * Vulkan functions are only called when explicity issued by the user, or when
//!   unambiguously implied at compile time.
//!   * *Handles do not implement Drop*
//!
//! ## Errors
//! All vulkan runtime errors (Device lost, out-of-memory, etc.), with the exception of "expected" errors
//! like `ERROR_TIMEOUT` or s`ERROR_NOT_READY`, are assumed to be fatal to the library.
//! While this will not result in panics or undefined behavior, it may result in resource leaks. Fixme!
//!
//! ## Safety
//! While this crate attempts to validate some subset of the vulkan correct-usage rules, it does not
//! (and *can not*) validate everything. Vulkan is still a deeply unsafe API with many opportunities for
//! incorrect usage to result in undefined behavior.
//!
//! Functions that take multiple objects must have the correct ownership hierarchies
//! (e.g. the parameters passed to [`Device::begin_command_buffer`] should be a command buffer,
//! the command pool from which it is allocated, and the device they both belong to.) This crate
//! takes advantange of this fact for additional compile-time checks, such as *external synchronization*.
//!
//! ### External Synchronization
//! This crate maps objects (Like [`Buffer`], [`Fence`], etc.) to *ownership* of the underlying handle.
//! With this abstraction, a function taking mutable reference to a handle or accepting a handle by
//! value can assume it is the only thread with access to the handle.
//!
//! ## Todo:
//! * [ ] Renderpasses
//!   * [ ] Dynamic Rendering
//!   * [ ] Framebuffers
//! * [ ] Swapchains
//! * [ ] Pipelines
//!   * [ ] Layout
//!   * [ ] Graphics
//!   * [ ] Compute
//!   * [X] Modules
//!     * [X] Specialization
//!       * [ ] Okay but do it better
//!   * [ ] Cache
//! * [ ] Sync primitives
//!   * [X] Fences
//!   * [ ] Semaphores
//!   * [ ] Events
//! * [ ] Descriptor
//!   * [ ] Layout
//!   * [ ] Pool
//!   * [ ] Set
//! * [ ] Memory
//!   * [ ] Host accessible
//! * [ ] Use NonNull NON_DISPATCHABLE_HANDLEs

// #![warn(missing_docs)]
#![allow(unsafe_op_in_unsafe_fn)]
#![feature(generic_const_exprs)]
#![allow(clippy::missing_safety_doc)]
use ash::vk::{self, Handle};
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
impl std::error::Error for AllocationError {}
impl std::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfHostMemory => write!(f, "out of host memory"),
            Self::OutOfDeviceMemory => write!(f, "out of device memory"),
        }
    }
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
    /// * The handle must be in a state consistent with the typestate of `Self`
    /// * The handle must not be `VK_NULL_HANDLE`
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
/// A trait representing an offset of an image of various dimensionalities.
pub trait Offset {
    /// The value represeting the `(0, 0, 0)` point.
    const ORIGIN: Self;
    /// The dimensionality of the extent represented by this type.
    /// For example, 2D Array has dimensionality 2.
    type Dim: Dimensionality;
    /// The extrapolated extent3D.
    /// All dimensions must be non-zero. Axes beyond the type's dimensionality should be `1`.
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
/// whether it is created with an array count > 1.
pub struct ArrayCount(NonZero<u32>);
impl ArrayCount {
    /// Create an array count, >= 2 None if an invalid count.
    pub fn new(layers: u32) -> Option<Self> {
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
    pub unsafe fn new_unchecked(layers: u32) -> Self {
        debug_assert!(layers >= 2);
        Self(NonZero::new_unchecked(layers))
    }
    /// Get the number of layers with guarantee that it is nonzero.
    pub fn as_nonzero(self) -> NonZero<u32> {
        self.0
    }
    /// Get the number of layers as an integer.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[repr(u32)]
pub enum Aspect {
    Color = vk::ImageAspectFlags::COLOR.as_raw(),
    Depth = vk::ImageAspectFlags::DEPTH.as_raw(),
    Stencil = vk::ImageAspectFlags::STENCIL.as_raw(),
}
/// The number of dimensions in an image, including whether it's an array.
pub trait Dimensionality {
    type Extent: Extent;
    type SubresourceLayers: SubresourceLayers;
    type Offset: Offset;
    type Pitch: BufferPitch;
    //type Offset: Offset;
    //type SubresourceRange: Range,
    const TYPE: vk::ImageType;
}
/// The data needed for addressing calculations in buffer-image copies.
pub trait BufferPitch {
    /// The value representing that the vulkan implementation should assume tight
    /// packing between rows and slices, i.e. where [`BufferPitch::row_pitch`] and
    /// [`BufferPitch::slice_pitch`] are both `None`.
    const PACKED: Self;
    fn row_pitch(&self) -> Option<NonZero<u32>>;
    fn slice_pitch(&self) -> Option<NonZero<u32>>;
}
/// Controls buffer addressing for image operations where only one 2D image slice is involved.
///
/// The distance in *texels* (not bytes!) between the start of one row and the start
/// of the next, or `None` for tight packing (i.e. the `width` of the image)
///
/// If not None, must be >= the tight packed pitch (rows must not alias, even for
/// a read-only operation)
pub struct RowPitch(pub Option<NonZero<u32>>);
/// Controls buffer addressing for image operations where several 2D image slices are involved.
pub struct RowSlicePitch {
    /// The distance in *texels* (not bytes!) between the start of one row and the start
    /// of the next, or `None` for tight packing (i.e. the `width` of the image)
    ///
    /// If not None, must be >= the tight packed pitch (rows must not alias, even for
    /// a read-only operation)
    pub row_pitch: Option<NonZero<u32>>,
    /// The distance in *texels* (not bytes!) between the start of one 2D slice of a 3D image or 2D array image
    /// and the start of the next, or `None` for tight packing (i.e. the `width * height` of the image)
    ///
    /// If not None, must be >= the tight packed pitch (slices must not alias, even for
    /// a read-only operation)
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
    const TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
}
impl Dimensionality for D1Array {
    type Extent = Extent1DArray;
    type Offset = Offset1D;
    type Pitch = RowPitch;
    type SubresourceLayers = SubresourceMipArray;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_1D;
}
impl Dimensionality for D2 {
    type Extent = Extent2D;
    type Offset = Offset2D;
    type Pitch = RowPitch;
    type SubresourceLayers = SubresourceMip;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
}
impl Dimensionality for D2Array {
    type Extent = Extent2DArray;
    type Offset = Offset2D;
    type Pitch = RowSlicePitch;
    type SubresourceLayers = SubresourceMipArray;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
}
impl Dimensionality for D3 {
    type Extent = Extent3D;
    type Offset = Offset3D;
    type Pitch = RowSlicePitch;
    type SubresourceLayers = SubresourceMip;
    const TYPE: vk::ImageType = vk::ImageType::TYPE_3D;
}
pub trait SubresourceLayers {
    fn subresource_layers(&self, aspect: Aspect) -> vk::ImageSubresourceLayers;
}
/// A single mip level of a non-array image.
pub struct SubresourceMip(pub u32);
impl SubresourceMip {
    pub const ZERO: Self = Self(0);
}
impl SubresourceLayers for SubresourceMip {
    fn subresource_layers(&self, aspect: Aspect) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::from_raw(aspect as _),
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
    /// If a range is specified without an end, e.g. `0..`, `VK_REMAINING_LAYERS`
    /// is specified.
    /// # Panics
    /// If layers is an inverted or empty range.
    pub fn new(mip: u32, layers: impl std::ops::RangeBounds<u32>) -> Self {
        let start_layer = match layers.start_bound() {
            std::ops::Bound::Excluded(&a) => a.checked_add(1).unwrap(),
            std::ops::Bound::Included(&a) => a,
            std::ops::Bound::Unbounded => 0,
        };
        let layer_count = match layers.end_bound() {
            std::ops::Bound::Excluded(&end) => {
                assert!(start_layer < end);
                // Safety - just checked that end is strictly larger, therefore a difference >= 1.
                unsafe { NonZero::new_unchecked(end - start_layer) }
            }
            std::ops::Bound::Included(&end) => {
                assert!(start_layer <= end);
                let count = (end - start_layer)
                    .checked_add(1)
                    .expect("overflow in SubresourceMipArray::new");
                // Unconditional +1, always nonzero.
                unsafe { NonZero::new_unchecked(count) }
            }
            std::ops::Bound::Unbounded => NonZero::new(vk::REMAINING_ARRAY_LAYERS).unwrap(),
        };
        Self {
            mip,
            start_layer,
            layer_count,
        }
    }
}
impl SubresourceLayers for SubresourceMipArray {
    fn subresource_layers(&self, aspect: Aspect) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::from_raw(aspect as _),
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
pub struct SubresourceRange(pub std::ops::Range<u32>);
pub struct LayeredSubresourceRange{pub mips: std::ops::Range<u32>, pub layers: std::ops::Range<u32>}*/

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
    R8Unorm = vk::Format::R8_UNORM.as_raw(),
    Rg8Uint = vk::Format::R8G8_UINT.as_raw(),
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

pub unsafe trait ImageSamples {
    fn flag(self) -> vk::SampleCountFlags;
}
#[derive(Clone, Copy)]
/// Typestate for an image which has one sample per pixel.
/// See also [`MultiSampled`].
pub struct SingleSampled;
unsafe impl ImageSamples for SingleSampled {
    fn flag(self) -> vk::SampleCountFlags {
        vk::SampleCountFlags::TYPE_1
    }
}
/// Typestate for an image which has more than one sample.
/// See also [`SingleSampled`].
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

/// An owned image handle.
/// # Typestates
/// * `Access`: The `VkImageUsageFlags` this image is statically known to possess (e.g. `(Storage, TransferSrc)`)
/// * `Dim`: The dimensionality of an image, including whether it is an array of images (e.g. [`D2`])
/// * `Format`: The Image aspects this image's format is known to possess (e.g. [`ColorFormat`]).
/// * `Samples`: Whether the image is [single-](SingleSampled) or [multi-](MultiSampled)sampled.
#[repr(transparent)]
#[must_use = "dropping the handle will not destroy the image and may leak resources"]
pub struct Image<
    Access: ImageAccess,
    Dim: Dimensionality,
    Format: ImageFormat,
    Samples: ImageSamples,
>(vk::Image, PhantomData<(Access, Dim, Format, Samples)>);
unsafe impl<Access: ImageAccess, Dim: Dimensionality, Format: ImageFormat, Samples: ImageSamples>
    ThinHandle for Image<Access, Dim, Format, Samples>
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

/// A Vulkan "Logical Device," and all associated function pointers.
///
/// This contains all the function pointers needed to operate. All device-scope operations go through this object.
///
/// To create one, acquire an [`ash::Device`] as documented by `ash` and pass it to [`Device::from_ash`].
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
#[must_use = "dropping the handle will not destroy the fence and may leak resources"]
pub struct Fence<State: FenceState>(vk::Fence, PhantomData<State>);

unsafe impl<State: FenceState> ThinHandle for Fence<State> {
    type Handle = vk::Fence;
}
thin_handle! {
    /// A synchronization primitive for CPU->GPU communication and fine-grained *intra*-queue GPU->GPU dependencies.
#[must_use = "dropping the handle will not destroy the event and may leak resources"]
    pub struct Event(vk::Event);
}

thin_handle! {
    /// A synchronization primitive for GPU->GPU communication and coarse-grained inter-queue dependencies.
#[must_use = "dropping the handle will not destroy the semaphore and may leak resources"]
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
thin_handle! {
    /// A block of shader code that has been forwarded to the implementation.
    /// Includes one or more "entry points" of possibly heterogeneous shader stages.
    ///
    /// Use [`Self::entry`] to reference a specific entry point for use in a pipeline.
    #[must_use = "dropping the handle will not deallocate the module and may leak resources"]
    pub struct ShaderModule(vk::ShaderModule);
}

/// The stage of a pipeline a shader is for.
pub trait ShaderStage {
    const FLAG: vk::ShaderStageFlags;
}
impl ShaderStage for Vertex {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::VERTEX;
}
pub struct Geometry;
impl ShaderStage for Geometry {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::GEOMETRY;
}
pub struct TessControl;
impl ShaderStage for TessControl {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::TESSELLATION_CONTROL;
}
pub struct TessEvaluation;
impl ShaderStage for TessEvaluation {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::TESSELLATION_EVALUATION;
}
pub struct TaskEXT;
impl ShaderStage for TaskEXT {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::TASK_EXT;
}
pub struct MeshEXT;
impl ShaderStage for MeshEXT {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::MESH_EXT;
}
pub struct Fragment;
impl ShaderStage for Fragment {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::FRAGMENT;
}
pub struct Compute;
impl ShaderStage for Compute {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::COMPUTE;
}

impl ShaderModule {
    /// Reference a specific entry point within this module.
    /// # Safety
    /// * The module must contain an entry point called `name`
    /// * The entry point must be of a type consistent with `Stage`
    /// * The device must be capable of using a shader of type `Stage`
    pub unsafe fn entry<'a, Stage: ShaderStage>(
        &'a self,
        name: &'a std::ffi::CStr,
        _stage: Stage,
    ) -> Entry<'a, Stage> {
        Entry {
            module: self.handle(),
            entry: name,
            _marker: PhantomData,
        }
    }
    /// Helper function for GLSL modules, where there is always exactly one
    /// entry point and it is always called "main".
    ///  # Safety
    /// * The module must contain an entry point called `main`
    /// * The entry point must be of a type consistent with `Stage`
    /// * The device must be capable of using a shader of type `Stage`
    pub unsafe fn main<Stage: ShaderStage>(&self, _stage: Stage) -> Entry<Stage> {
        self.entry(c"main", _stage)
    }
}

/// A reference to a shader module, bundled with an entry point name and the stage of said entry point.
///
/// Created by [`ShaderModule::entry`].
/// # Typestates
/// * `Stage`: The type of the shader, e.g. [`Vertex`], [`Compute`]
#[derive(Copy, Clone)]
pub struct Entry<'a, Stage: ShaderStage> {
    module: vk::ShaderModule,
    entry: &'a std::ffi::CStr,
    _marker: PhantomData<(&'a ShaderModule, Stage)>,
}
impl<'a, Stage: ShaderStage> Entry<'a, Stage> {
    /// Provide constants to the shader compiler.
    ///
    /// If your shader does not use specialization, you can specialize on the unit type:
    /// ```no_run
    /// # let module : ShaderModule = todo!();
    /// let specialized_entry = module.main().specialize(());
    /// ```
    /// # Safety
    /// * Every constant necessary for a complete module entry point must be populated by the [entries](Specialization::ENTRIES) of `S`.
    /// * Every types required by the shader must match the types provided by `S`
    pub unsafe fn specialize<S: Specialization>(
        self,
        specialization: &'a S,
    ) -> SpecializedEntry<'a, Stage> {
        SpecializedEntry {
            module: self.module,
            entry: self.entry,
            specialization_info: vk::SpecializationInfo {
                data_size: std::mem::size_of::<S>(),
                p_data: (specialization as *const S).cast(),
                ..vk::SpecializationInfo::default().map_entries(S::ENTRIES)
            },
            _marker: PhantomData,
        }
    }
}

/// A reference to a shader module, bundled with an entry point name, the stage of said entry point, and
/// the specialization constants needed to make it concrete.
///
/// Created by [`Entry::specialize`].
/// # Typestates
/// * `Stage`: The type of the shader, e.g. [`Vertex`], [`Compute`]
pub struct SpecializedEntry<'a, Stage: ShaderStage> {
    module: vk::ShaderModule,
    entry: &'a std::ffi::CStr,
    // &dyn but funny
    specialization_info: vk::SpecializationInfo<'a>,
    // specialization_map: &'a [vk::SpecializationMapEntry],
    // specialization_data: *const u8,
    // specialization_size: usize,
    _marker: PhantomData<(&'a ShaderModule, Stage)>,
}
impl<'a, Stage: ShaderStage> SpecializedEntry<'a, Stage> {
    pub fn specialization_info<'this>(&'this self) -> &'this vk::SpecializationInfo<'this>
    where
        'a: 'this,
    {
        &self.specialization_info
    }
    pub fn create_info(&'_ self) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .module(self.module)
            .name(self.entry)
            .stage(Stage::FLAG)
            .specialization_info(self.specialization_info())
    }
}

/// A type which can be used as a specialization constant.
///
/// # Safety
/// * Each entry in `ENTRIES` must refer to a range inside the bounds of `Self`.
/// * Each range in `Self` referred to by `ENTRIES` must be a valid value of the
///   type the shader expects.
///   * This req sucks uwu. Basically the layout of this structure and the shader *must* agree.
pub unsafe trait Specialization {
    const ENTRIES: &'static [vk::SpecializationMapEntry];
}
unsafe impl Specialization for () {
    const ENTRIES: &'static [vk::SpecializationMapEntry] = &[];
}

/// Shaders controlling generation of primitives
pub enum PreRasterShaders<'a> {
    /// VertexInput -> Primitive Assembly -> Vertex -> (Tess?) -> (Geometry?)
    Vertex {
        vertex: SpecializedEntry<'a, Vertex>,
        tess: Option<(
            SpecializedEntry<'a, TessControl>,
            SpecializedEntry<'a, TessEvaluation>,
        )>,
        geometry: Option<SpecializedEntry<'a, Geometry>>,
    },
    /// (Task?) -> Mesh
    Mesh(
        Option<SpecializedEntry<'a, TaskEXT>>,
        SpecializedEntry<'a, MeshEXT>,
    ),
}
/// The set of shaders that forms a complete graphics pipeline.
pub struct GraphicsShaders<'a> {
    /// Shaders that generate primitives to be rasterized
    pre_raster: PreRasterShaders<'a>,
    /// Shader after primitives are rasterized.
    fragment: Option<SpecializedEntry<'a, Fragment>>,
}
impl<'a> GraphicsShaders<'a> {
    /// Maximum number of shader stages possible in a single pipeline
    const MAX_SHADER_STAGES: usize = 5;
    pub fn create_infos<'this>(
        &'this self,
    ) -> tinyvec::ArrayVec<[vk::PipelineShaderStageCreateInfo<'this>; Self::MAX_SHADER_STAGES]>
    where
        'a: 'this,
    {
        let Self {
            pre_raster,
            fragment,
        } = self;
        // This default()s the array. Whyy?
        let mut vec = tinyvec::ArrayVec::new();
        if let PreRasterShaders::Vertex {
            vertex,
            tess,
            geometry,
        } = &pre_raster
        {
            vec.push(vertex.create_info());
            if let Some((tessc, tesse)) = tess {
                vec.push(tessc.create_info());
                vec.push(tesse.create_info());
            }
            if let Some(geometry) = geometry {
                vec.push(geometry.create_info());
            }
        }
        if let Some(fragment) = fragment {
            vec.push(fragment.create_info());
        }

        vec
    }
}

pub trait BindPoint {
    const BIND_POINT: vk::PipelineBindPoint;
}
pub struct Graphics;
impl BindPoint for Graphics {
    const BIND_POINT: vk::PipelineBindPoint = vk::PipelineBindPoint::GRAPHICS;
}
impl BindPoint for Compute {
    const BIND_POINT: vk::PipelineBindPoint = vk::PipelineBindPoint::COMPUTE;
}

thin_handle! {
    pub struct Pipeline<Kind: BindPoint>(vk::Pipeline);

}

pub struct ComputePipelineCreateInfo<'a> {
    pub shader: SpecializedEntry<'a, Compute>,
    pub layout: vk::PipelineLayout,
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
impl std::error::Error for SubmitError {}
impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfHostMemory => write!(f, "out of host memory"),
            Self::OutOfDeviceMemory => write!(f, "out of device memory"),
            Self::DeviceLost => write!(f, "device lost"),
        }
    }
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
/// The result of querying or waiting on the state of a [`Fence`].
pub enum FencePoll<Success, Pending> {
    Signaled(Success) = vk::Result::SUCCESS.as_raw(),
    Unsignaled(Pending) = vk::Result::NOT_READY.as_raw(),
}
/// The timeout for a host-side wait.
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

/// A thin handle (an image, a fence, etc.) that has been associated with a device and
/// can thus have its own associated functions.
///
/// Use [`Device::bind`] to acquire.
pub struct Bound<'device, T: ThinHandle>(T, &'device Device);
impl<'device, T: ThinHandle> Bound<'device, T> {
    /// Extract the typed handle.
    pub fn into_inner(self) -> T {
        self.0
    }
    /// Access the untyped handle.
    /// # Safety
    /// See [`ThinHandle::handle`]
    pub unsafe fn handle(&self) -> T::Handle {
        self.0.handle()
    }
    /// Assign a new typstate the bound handle.
    /// # Safety
    /// See [`ThinHandle::with_state`]
    pub unsafe fn with_state<U: ThinHandle<Handle = T::Handle>>(self) -> Bound<'device, U> {
        Bound(self.0.with_state(), self.1)
    }
}

impl<'a> Bound<'a, Fence<Pending>> {
    pub unsafe fn wait(self) -> Result<Bound<'a, Fence<Signaled>>, SubmitError> {
        self.1.wait_fence(self.0).map(|fence| self.1.bind(fence))
    }
}
impl<'a, State: KnownFenceState> Bound<'a, Fence<State>> {
    pub unsafe fn reset(self) -> Result<Bound<'a, Fence<Unsignaled>>, AllocationError> {
        self.1.reset_fence(self.0).map(|fence| self.1.bind(fence))
    }
    pub unsafe fn destroy(self) {
        unsafe {
            self.1.destroy_fence(self.0);
        }
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
    /*
    /// The distance in *texels* (not bytes!) between the start of one row and the start
    /// of the next, or `None` for tight packing (i.e. the `width` of the image)
    ///
    /// If not None, must be >= the tight packed pitch (rows must not alias, even for
    /// a read-only operation)
    pub row_pitch: Option<NonZero<u32>>,
    /// The distance in *texels* (not bytes!) between the start of one 2D slice of a 3D image or 2D array image
    /// and the start of the next, or `None` for tight packing (i.e. the `width * height` of the image)
    ///
    /// If not None, must be >= the tight packed pitch (slices must not alias, even for
    /// a read-only operation)
    pub slice_pitch: Option<NonZero<u32>>,
    */
    /// Which mip level, and optionally which layers of an array image, to copy.
    pub layers: Dim::SubresourceLayers,
}

impl Device {
    /// Bundle a handle with a device pointer, allowing the handle to
    /// have it's own associated functions and making method chaining possible even
    /// through several operations which change the type of it's operand.
    ///
    /// Generally, if an operation takes only the device and the handle and returns either,
    /// it can be executed using a [`Bound`] handle for cleaner syntax.
    ///
    /// This is a zero-cost operation.
    ///
    /// ```no_run
    /// # let device: Device = todo!();
    /// let pending_fence: Fence<Pending> = todo!();
    /// let completed_fence = device.wait(pending_fence).unwrap();
    /// device.destroy_fence(completed_fence);
    /// ```
    /// becomes
    /// ```no_run
    /// # let device: Device = todo!();
    /// let pending_fence: Fence<Pending> = todo!();
    /// device.bind(pending_fence)
    ///     .wait().unwrap()
    ///     .destroy();
    /// ```
    ///
    /// # Safety
    /// The handle must be acquired from this device.
    pub unsafe fn bind<Handle: ThinHandle>(&self, handle: Handle) -> Bound<Handle> {
        Bound(handle, self)
    }
    /// Wrap an [`ash`] device.
    ///
    /// This is the main entry point for this crate.
    pub fn from_ash(device: ash::Device) -> Self {
        Device(device)
    }
    /// Submit work to a queue.
    /// The fence will be signaled upon completion.
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
    /// Wait for a queue to complete all prior submissions, *as if* a fence on all prior
    /// submissions had been waited on.
    pub unsafe fn queue_wait_idle(&self, queue: &mut Queue) -> Result<(), SubmitError> {
        self.0
            .queue_wait_idle(queue.handle())
            .map_err(SubmitError::from_vk)
    }
    /// *As if* all queues owned by the device had `queue_wait_idle` called on them.
    /// # Safety
    /// * Requires *unique* (mutable) access to all device queues for the duration of the call.
    ///
    /// This cannot currently be proven at compile time. Passing mutable references to *all*
    /// queues allows it to be checked, but it cannot be proven that all queues are present.
    pub unsafe fn wait_idle(&self, _queues: &mut [&mut Queue]) -> Result<(), SubmitError> {
        self.0.device_wait_idle().map_err(SubmitError::from_vk)
    }
    pub unsafe fn create_module(&self, spirv: &[u32]) -> Result<ShaderModule, AllocationError> {
        debug_assert!(!spirv.is_empty());
        self.0
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
            .map(|handle| ShaderModule::from_handle(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_module(&self, module: ShaderModule) -> &Self {
        self.0.destroy_shader_module(module.into_handle(), None);
        self
    }
    /// Create a fence in the given state, [`Signaled`] or [`Unsignaled`].
    /// ```no_run
    /// # let device : Device = todo!();
    /// let signaled = device.create_fence::<Unsignaled>().unwrap();
    /// ```
    pub unsafe fn create_fence<State: KnownFenceState>(
        &self,
    ) -> Result<Bound<Fence<State>>, AllocationError> {
        self.0
            .create_fence(
                &vk::FenceCreateInfo::default().flags(State::CREATION_FLAGS),
                None,
            )
            .map(|handle| Fence::from_handle(handle))
            .map(|handle| self.bind(handle))
            .map_err(AllocationError::from_vk)
    }
    /// Set each fence to the "Unsignaled" state. It is valid to reset an already unsignaled
    /// fence.
    pub unsafe fn reset_fence<State: KnownFenceState>(
        &self,
        fence: Fence<State>,
    ) -> Result<Fence<Unsignaled>, AllocationError> {
        self.reset_fences([fence]).map(|[fence]| fence)
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
    /// If the result is [`FencePoll::Unsignaled`], the result may become immediately out-of-date.
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
    /// let buffer = device.create_buffer(
    ///         (Storage, TransferSrc, TransferDst),
    ///         1024,
    ///         SharingMode::Concurrent(SharingFamilies::from_array(&[0, 1])),
    ///     );
    /// ```
    pub unsafe fn create_buffer<Access: BufferAccess>(
        &self,
        _usage: Access,
        size: NonZero<u64>,
        sharing: SharingMode,
    ) -> Result<Buffer<Access>, AllocationError> {
        self.0
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .usage(Access::FLAGS)
                    .size(size.get())
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
    /// Create an image. The `vkImageUsageFlags`, Dimensionality, Aspect mask, and Multisampled-ness
    /// are passed at compile time:
    ///
    /// ```no_run
    /// # let device: Device = todo!();
    /// let image = device.create_image(
    ///         // The usages the image will be created with.
    ///         // This becomes part of the image's type.
    ///         // A tuple of usages or a single usage may be provided.
    ///         (TransferSrc, ColorAttachment),
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
    ///         // Makes a non-multisampled image.
    ///         // You can also use the MultiSampled enum.
    ///         SingleSampled,
    ///         vk::ImageTiling::Optimal,
    ///         SharingMode::Exclusive
    ///     ).unwrap();
    /// ```
    ///
    /// See: [`ImageAccess`], [`Extent`], [`ColorFormat`], [`DepthStencilFormat`], [`MultiSampled`].
    pub unsafe fn create_image<
        Access: ImageAccess,
        Ext: Extent,
        Format: ImageFormat,
        Samples: ImageSamples,
    >(
        &self,
        _usage: Access,
        extent: Ext,
        mip_levels: NonZero<u32>,
        format: Format,
        samples: Samples,
        tiling: vk::ImageTiling,
        sharing: SharingMode,
    ) -> Result<Image<Access, Ext::Dim, Format, Samples>, AllocationError> {
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
                    .samples(samples.flag())
                    .tiling(tiling),
                None,
            )
            .map(|handle| Image::from_handle(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_image<
        Access: ImageAccess,
        Dim: Dimensionality,
        Format: ImageFormat,
        Samples: ImageSamples,
    >(
        &self,
        image: Image<Access, Dim, Format, Samples>,
    ) -> &Self {
        self.0.destroy_image(image.handle(), None);
        self
    }
    /// Create (possibly) several compute pipelines.
    /// If *any* pipeline fails to be created, the others are freed
    /// and the error is returned.
    pub unsafe fn create_compute_pipelines<const N: usize>(
        &self,
        _cache: Option<&mut ()>,
        infos: [ComputePipelineCreateInfo; N],
    ) -> Result<[Pipeline<Compute>; N], AllocationError> {
        let infos = infos.each_ref().map(|info| {
            vk::ComputePipelineCreateInfo::default()
                .layout(info.layout)
                .stage(info.shader.create_info())
        });
        let mut output = std::mem::MaybeUninit::<[vk::Pipeline; N]>::uninit();
        // We call the raw form (not ash's wrapper) since it works on slices and vecs
        // while we work on arrays and can thus skip an allocation
        let res = (self.0.fp_v1_0().create_compute_pipelines)(
            self.0.handle(),
            vk::PipelineCache::null(),
            N.try_into().unwrap(),
            infos.as_ptr(),
            std::ptr::null(),
            output.as_mut_ptr().cast(),
        );
        // # Safety
        // Always populates the whole array, failures are populated with NULL.
        // Complete failure (i.e. no successes) even results in all NULLs. How polite!
        let output = output.assume_init();

        match res {
            vk::Result::SUCCESS => Ok(output.map(|handle| {
                debug_assert!(!handle.is_null());
                Pipeline::from_handle(handle)
            })),
            _ => {
                for handle in output {
                    if !handle.is_null() {
                        self.0.destroy_pipeline(handle, None);
                    }
                }
                Err(AllocationError::from_vk(res))
            }
        }
    }
    pub unsafe fn destroy_pipeline<Kind: BindPoint>(&self, pipeline: Pipeline<Kind>) -> &Self {
        self.0.destroy_pipeline(pipeline.into_handle(), None);
        self
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
    pub unsafe fn bind_pipeline<
        'a,
        'b,
        Kind: BindPoint,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
    >(
        &'a self,
        buffer: &'b mut RecordingBuffer<Level, MaybeInsideRender>,
        pipeline: &'b Pipeline<Kind>,
    ) -> &'a Self {
        self.0
            .cmd_bind_pipeline(buffer.handle(), Kind::BIND_POINT, pipeline.handle());
        self
    }
    pub unsafe fn dispatch<'a, Level: CommandBufferLevel, MaybeInsideRender: CommandBufferState>(
        &'a self,
        buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        [x, y, z]: [NonZero<u32>; 3],
    ) -> &'a Self {
        self.0
            .cmd_dispatch(buffer.handle(), x.get(), y.get(), z.get());
        self
    }
    pub unsafe fn copy_buffer_to_image<
        'a,
        const N: usize,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
        SuperAccess: ImageSuperset<TransferDst>,
        Dim: Dimensionality,
    >(
        &'a self,
        command_buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        buffer: impl AsRef<Buffer<TransferSrc>>,
        // FIXME: DepthStencil needs a way to specify *which* of the two aspects.
        image: Image<SuperAccess, Dim, ColorFormat, SingleSampled>,
        image_layout: vk::ImageLayout,
        regions: [BufferImageCopy<Dim>; N],
    ) -> &'a Self {
        self.0.cmd_copy_buffer_to_image(
            command_buffer.handle(),
            buffer.as_ref().handle(),
            image.handle(),
            image_layout,
            &regions.map(|region| vk::BufferImageCopy {
                image_offset: region.image_offset.offset(),
                image_extent: region.image_extent.extent(),
                buffer_offset: region.buffer_offset,
                // None for automatic pitch calculation is mapped to a value of 0, convenient!
                buffer_row_length: region.pitch.row_pitch().map(NonZero::get).unwrap_or(0),
                buffer_image_height: region.pitch.slice_pitch().map(NonZero::get).unwrap_or(0),
                // FIXME: Aspect should be dynamic!
                image_subresource: region.layers.subresource_layers(Aspect::Color),
            }),
        );
        todo!()
    }
    pub unsafe fn copy_image_to_buffer<
        'a,
        const N: usize,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
        SuperAccess: ImageSuperset<TransferSrc>,
        Dim: Dimensionality,
    >(
        &'a self,
        command_buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        // FIXME: DepthStencil needs a way to specify *which* of the two aspects.
        image: Image<SuperAccess, Dim, ColorFormat, SingleSampled>,
        image_layout: vk::ImageLayout,
        buffer: impl AsRef<Buffer<TransferDst>>,
        regions: [BufferImageCopy<Dim>; N],
    ) -> &'a Self {
        self.0.cmd_copy_image_to_buffer(
            command_buffer.handle(),
            image.handle(),
            image_layout,
            buffer.as_ref().handle(),
            &regions.map(|region| vk::BufferImageCopy {
                image_offset: region.image_offset.offset(),
                image_extent: region.image_extent.extent(),
                buffer_offset: region.buffer_offset,
                // None for automatic pitch calculation is mapped to a value of 0, convenient!
                buffer_row_length: region.pitch.row_pitch().map(NonZero::get).unwrap_or(0),
                buffer_image_height: region.pitch.slice_pitch().map(NonZero::get).unwrap_or(0),
                // FIXME: Aspect should be dynamic!
                image_subresource: region.layers.subresource_layers(Aspect::Color),
            }),
        );
        todo!()
    }
    pub unsafe fn dispatch_base<
        'a,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
    >(
        &'a self,
        buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        base: [u32; 3],
        count: [NonZero<u32>; 3],
    ) -> &'a Self {
        self.0.cmd_dispatch_base(
            buffer.handle(),
            base[0],
            base[1],
            base[2],
            count[0].get(),
            count[1].get(),
            count[2].get(),
        );
        self
    }
    pub unsafe fn dispatch_indirect<
        'a,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
    >(
        &'a self,
        buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        indirect: impl AsRef<Buffer<Indirect>>,
        offset: u64,
    ) -> &'a Self {
        self.0
            .cmd_dispatch_indirect(buffer.handle(), indirect.as_ref().handle(), offset);
        self
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

    let image_extent = Extent2D {
        width: NonZero::new(128).unwrap(),
        height: NonZero::new(128).unwrap(),
    };

    let mut pool = device
        .create_command_pool(&vk::CommandPoolCreateInfo::default())
        .unwrap();
    let buffer = device
        .create_buffer(
            (Vertex, Index, TransferSrc),
            NonZero::new(
                u64::from(image_extent.width.get()) * u64::from(image_extent.height.get()) * 4,
            )
            .unwrap(),
            SharingMode::Exclusive,
        )
        .unwrap();
    let image = device
        .create_image(
            (TransferDst, TransferSrc),
            image_extent,
            NonZero::new(1).unwrap(),
            ColorFormat::Rgba8Unorm,
            SingleSampled,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )
        .unwrap();
    let fence = device.create_fence::<Unsignaled>().unwrap();
    let renderpass = device.create_render_pass(&[todo!(), todo!()]).unwrap();
    let [mut cb1, mut cb2] = device.allocate_primary_buffers(&mut pool).unwrap();

    let mut recording = device.begin_command_buffer(&mut pool, &mut cb1).unwrap();
    device.copy_buffer_to_image(
        &mut recording,
        buffer,
        image,
        vk::ImageLayout::UNDEFINED,
        [BufferImageCopy {
            buffer_offset: 0,
            image_extent,
            layers: SubresourceMip(0),
            image_offset: Offset::ORIGIN,
            pitch: BufferPitch::PACKED,
        }],
    );
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
    let fence = device.create_fence::<Unsignaled>().unwrap().into_inner();
    let fence = device.submit_with_fence(&mut queue, fence).unwrap();
    let fence = device.wait_fence(fence).unwrap();
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
