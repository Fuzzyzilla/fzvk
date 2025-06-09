//! # Object Handles
//!
//! Objects in vulkan are referred to by opaque 64-bit "handles." For example,
//! on some implementations, this may be a pointer to an internal datastructure.
//! On others, it may be a direct encoding of the object itself. Or it could be
//! neither!
//!
//! ## Comparison
//! Vulkan does not require that distinct objects must have distinct handles due
//! to the aforementioned possibility that handles may be a direct encoding of
//! the object. As such, the value of handles being equal does not necessarily
//! mean that two handles refer to the same object, and these types can not
//! meaninfully implement `Eq` or `Hash`.
//!
//! If you still wish to compare handles in this way, they can be compared as
//! integers using the [`Handle::as_raw`](ash::vk::Handle::as_raw) function.
//!
//! It should be noted that this crate assumes owned handle types (e.g.
//! [`Buffer`](crate::Buffer)) are *unique*, so multiple owned objects will
//! never refer to the same underlying vulkan object (but may, due to the avove,
//! have the same handle). Thus, by a more rigid definition of equality, they
//! would never compare equal to each other anyway.
use ash::vk::Handle as RawHandle;
use core::{marker::PhantomData, num::NonZero};

#[derive(Clone, Copy)]
#[repr(transparent)]
/// A [raw Vulkan handle](ash::vk::Handle) which is known to be non-null,
/// allowing for niche optimizations.
///
/// Contrary to other objects of this crate, this does not represent any kind of
/// ownership over the underlying object, and is thus `Copy`.
pub struct NonNull<Inner: RawHandle>(NonZero<u64>, PhantomData<Inner>);
impl<Inner: RawHandle> NonNull<Inner> {
    /// Wrap the handle, or None if the handle is null.
    pub fn new(handle: Inner) -> Option<Self> {
        Some(Self(NonZero::new(handle.as_raw())?, PhantomData))
    }
    /// Wrap the handle, assuming it's not null.
    /// # Safety
    /// `handle.is_null()` must be false.
    pub unsafe fn new_unchecked(handle: Inner) -> Self {
        // Safety - contract forwarded to caller
        let nonzero = unsafe { NonZero::new_unchecked(handle.as_raw()) };
        Self(nonzero, PhantomData)
    }
    /// Access the underlying handle.
    pub fn get(self) -> Inner {
        Inner::from_raw(self.0.get())
    }
}

/// Trait for types which are a thin wrapper over a Non-Null Vulkan handle.
/// # Safety
/// `Self` must be `repr(transparent)` over a [`NonNull`]`<Self::Handle>`
pub unsafe trait ThinHandle: Sized {
    type Handle: Copy + RawHandle;
    // Doesn't work? Always succeeds even when it shouldn't. Weird. const SOUP:
    // () = assert!(core::mem::size_of::<Self>() ==
    // core::mem::size_of::<Self::Handle>(),);

    /// Get a copy of the underlying handle.
    /// # Safety
    /// Must not be used to change the underyling object in such a way that its
    /// state no longer matches the existing value of `Self`.
    unsafe fn handle(&self) -> Self::Handle {
        debug_assert_eq!(
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Handle>()
        );
        unsafe { core::mem::transmute_copy(self) }
    }
    /// Discard the typestate and access the underlying handle.
    ///
    /// Note that this is still unsafe, as this handle may refer to the same
    /// object as some other handle, who's typestate must be respected by the
    /// caller.
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
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Handle>()
        );
        unsafe { core::mem::transmute::<&[Self], &[Self::Handle]>(values) }
    }
    /// Create an object from the relavant handle.
    /// # Safety
    /// * The handle must be in a state consistent with the typestate of `Self`
    /// * The handle must not be `VK_NULL_HANDLE`
    unsafe fn from_handle_unchecked(handle: Self::Handle) -> Self {
        // Manual transmute since the sizes aren't known.
        debug_assert_eq!(
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Handle>()
        );
        debug_assert!(!handle.is_null());
        unsafe { (&raw const handle).cast::<Self>().read() }
    }
    /// Create an object from the relavant handle.
    /// # Safety
    /// * The handle must be in a state consistent with the typestate of `Self`
    unsafe fn from_handle(handle: Self::Handle) -> Option<Self> {
        if handle.is_null() {
            return None;
        }
        // Manual transmute since the sizes aren't known.
        debug_assert_eq!(
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Handle>()
        );
        debug_assert!(!handle.is_null());
        Some(unsafe { (&raw const handle).cast::<Self>().read() })
    }
    /// Change the typestate of the thin handle. This is a jackhammer to drive a
    /// nail, so implementors may provide a safe subset of this operation or
    /// otherwise constrain it.
    ///
    /// Usage: ([`Buffer`](crate::buffer::Buffer) implements it's own functions
    /// for this, such as
    /// [`Buffer::as_subusage`](crate::buffer::Buffer::as_subusage) just as an
    /// example:)
    /// ```no_run
    /// # use fzvk::*;
    /// let buffer : Buffer<Storage> = todo!();
    /// let vertex_buffer = unsafe { buffer.with_state::<Buffer<Vertex>>() };
    /// ```
    /// # Safety
    /// The handle must be in a state consistent with the typestate of `Other`
    #[must_use = "dropping the handle may leak resources"]
    unsafe fn with_state<Other: ThinHandle<Handle = Self::Handle>>(self) -> Other {
        // Safety - not null since we just got it from a ThinHandle which is non
        // null.
        //
        // The state safety condition is handled by the caller.
        Other::from_handle_unchecked(self.handle())
    }
    /// Change the typestate of the thin handle. This is a jackhammer to drive a
    /// nail, so implementors may provide a safe subset of this operation or
    /// otherwise constrain it. # Safety The handle must be in a state
    /// consistent with the typestate of `Other`
    unsafe fn with_state_mut<Other: ThinHandle<Handle = Self::Handle>>(&mut self) -> &mut Other {
        debug_assert_eq!(
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Handle>()
        );
        core::mem::transmute(self)
    }
    /// Change the typestate of the thin handle. This is a jackhammer to drive a
    /// nail, so implementors may provide a safe subset of this operation or
    /// otherwise constrain it. # Safety The handle must be in a state
    /// consistent with the typestate of `Other`
    unsafe fn with_state_ref<Other: ThinHandle<Handle = Self::Handle>>(&self) -> &Other {
        debug_assert_eq!(
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Handle>()
        );
        core::mem::transmute(self)
    }
}
