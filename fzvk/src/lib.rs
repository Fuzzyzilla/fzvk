//! # Fuzzy's Funny Zero-cost Excessively-Typestate Vulkan Abstraction
//! A thin wrapper around [`ash`]. This crate is not associated with the ash
//! project!
//!
//! To get started, fetch an [`ash::Device`] and pass it to
//! [`Device::from_ash`].
//!
//! ## Typestate
//! This crate makes extensive (perhaps excessive) use of the [typestate
//! pattern](https://en.wikipedia.org/wiki/Typestate_analysis) to validate
//! *some* operations. Of course, the scope of what can be declared at compile
//! time is inherently limited by the need for runtime flexibility, so it is
//! mostly used in locations where it is a common usage pattern that the "shape"
//! of the data will not be totally dynamic.
//!
//! For example, an [`Image`] contains the following compile-time information:
//! * Its dimensionality (1D, 2D, 3D)
//! * Its array-ness
//! * Its format and aspects.
//! * Its vkImageUsageFlags
//! * Whether multisampling is in use.
//!
//! A fully-named image type thus looks something like:
//! ```no_run
//! # use fzvk::*;
//! let image : Image<(Storage, TransferDst), D2Array, format::Rgba8Srgb, SingleSampled> = todo!();
//! ```
//! Since each of these facts are almost always known at compile time and have
//! large implications for valid usage of the resource, they are tracked at
//! compile time too.
//!
//! This allows for functions to validate, at compile time, whether certain
//! aspects of the operation are valid and produce a compiler error if the state
//! of an object is not statically known to be correct. See
//! [`Device::bind_vertex_buffers`] for what that looks like.
//!
//! ### What if it *needs* to be dynamic?
//! If there is a finite enumeration of dynamic states, `enum`s or even `union`s
//! around the possible typestates are good choices.
//!
//! For states too dynamic even for that strategy, raw vulkan may be used. Since
//! this crate is a thin wrapper around [`ash`], it is perfectly valid to mix
//! and match API calls from both. See the [`ThinHandle`] trait for how to move
//! data in and out of the typestate ecosystem and the safety considerations of
//! this.
//!
//! ## Zero cost
//! * Types do not contain any runtime state beyond their vulkan handles.
//!   * Typestate "data" is zero-sized and compiles down to nothing at all
//!   * Handles do not contain the necessary function pointers, thus all
//!     operations must go through the central [`Device`] type.
//!   * The big exception to this is [`Device`], which contains several
//!     kilobytes of function pointers on the stack. We can't all be winners :3
//! * Allocations are avoided like the plague.
//!   * Wherever possible, const-generic arrays are used instead to make
//!     variable-length operations occur exclusively on the stack.
//! * Even `if` statements are used with some trepedation. If the condition can
//!   be moved to the type level, it will be.
//! * Vulkan functions are only called when explicity issued by the user, or
//!   when unambiguously implied at compile time.
//!   * *Handles do not implement Drop*
//!
//! ## Feature Flags
//! * `alloc` (default) - Some helper functions for reading dynamically-sized
//!   data from the implementation. Otherwise, the user will need to do the
//!   get-length-get-data dance manually.
//!
//! ## Errors
//! All vulkan runtime errors (Device lost, out-of-memory, etc.), with the
//! exception of "expected" errors like `ERROR_TIMEOUT` or s`ERROR_NOT_READY`,
//! are assumed to be fatal to the library. While this will not result in panics
//! or undefined behavior, it may result in resource leaks. Fixme!
//!
//! ## Safety
//! While this crate attempts to validate some subset of the vulkan
//! correct-usage rules, it does not (and *can not*) validate everything. Vulkan
//! is still a deeply unsafe API with many opportunities for incorrect usage to
//! result in undefined behavior.
//!
//! Functions that take multiple objects must have the correct ownership
//! hierarchies (e.g. the parameters passed to [`Device::begin_command_buffer`]
//! should be a command buffer, the command pool from which it is allocated, and
//! the device they both belong to.) This crate takes advantange of this fact
//! for additional compile-time checks, such as *external synchronization*.
//!
//! ### External Synchronization
//! This crate maps objects (Like [`Buffer`], [`Fence`], etc.) to *ownership* of
//! the underlying handle. With this abstraction, a function taking mutable
//! reference to a handle or accepting a handle by value can assume it is the
//! only thread with access to the handle.
//!
//! [`PipelineCache`] objects are always worked with by mutable reference, and
//! can thus soundly be created with or without `EXTERNALLY_SYNCHRONIZED_BIT`
//!
//! ## Coverage
//! This crate does not aim to cover the entire Vulkan API, let alone all its
//! extensions. Primarily, this will support the subset of the Vulkan API that I
//! find useful and have a good intuition for.
//!
//! However, since this crate allows for intermixing with raw `ash` calls, all
//! missing features and extensions can be used, just without the syntactic
//! sugar.
//!
//! Notable exclusions:
//! * Currenlty only VK1.0 is supported, this will change in the future.
//! * Cubemap imageviews
//! * Compressed/block-based/multiplanar image formats.
//! * Sparse objects.
//!
//! Probable Extensions:
//! * [ ] `KHR_swapchain` (and associated platform extensions)
//! * [ ] `KHR_dynamic_rendering`
//!   * [ ] `KHR_dynamic_rendering_local_read`
//! * [ ] `KHR_portability_subset`
//! * [ ] `KHR_dedicated_allocation`
//! * [ ] `KHR_timeline_semaphore` (some learning is needed on my part uwu)
//! * [ ] `EXT_mesh_shader` (i just like them a lot)
//! * [ ] (incomplete list)
//!
//! ### Todo:
//! * [ ] Swapchains
//! * [ ] Images
//!   * [X] Color
//!   * [ ] Depth/Stencil
//! * [X] Buffers
//! * [ ] Pipelines
//!   * [X] Cache
//!   * [ ] Layout
//!     * [ ] Push constants
//!     * [ ] Descriptors
//!   * [ ] Graphics
//!     * [ ] Dynamic Rendering
//!     * [ ] Renderpasses
//!       * [ ] Framebuffers
//!   * [X] Compute
//!   * [X] Modules
//!     * [X] Specialization
//!       * [X] Okay but do it better (proc_macro)
//! * [ ] Sync primitives
//!   * [X] Fences
//!   * [X] Semaphores
//!   * [ ] Events
//! * [ ] Descriptor
//!   * [ ] Layout
//!   * [ ] Pool
//!   * [ ] Set
//!   * [ ] Samplers
//! * [ ] Memory
//!   * [ ] Host accessible
//! * [ ] Use NonNull NON_DISPATCHABLE_HANDLEs

// #![warn(missing_docs)]
#![allow(unsafe_op_in_unsafe_fn)]
#![feature(generic_const_exprs)]
#![allow(clippy::missing_safety_doc)]
// "fake variadics"
#![cfg_attr(doc, feature(rustdoc_internals))]
#![no_std]
#[cfg(feature = "alloc")]
extern crate alloc;

pub mod buffer;
pub use buffer::*;
pub mod command_buffer;
pub use command_buffer::*;
pub mod descriptor;
pub mod format;
pub mod handle;
pub use handle::*;
pub mod image;
pub use image::*;
pub mod memory;
pub mod pipeline;
pub use pipeline::*;
pub mod sampler;
pub mod usage;
pub use usage::*;
pub mod sync;
pub use ash::{self, vk};
use core::num::NonZero;
pub use sync::*;
use vk::Handle;

/// A boolean value in 32-bits.
///
/// It is immediate undefined behavior to observe a value that isn't 0 or 1 as
/// this type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum Bool32 {
    False = 0,
    True = 1,
}
impl Bool32 {
    pub const fn new(v: bool) -> Self {
        if v { Self::True } else { Self::False }
    }
    pub const fn get(self) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
        }
    }
}
impl From<Bool32> for u32 {
    fn from(value: Bool32) -> Self {
        value as Self
    }
}
impl From<bool> for Bool32 {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}
impl From<Bool32> for bool {
    fn from(value: Bool32) -> Self {
        value.get()
    }
}
/// An enum type that transparently represents a vulkan enum.
pub unsafe trait VkEnum: Sized {
    type Enum;
    fn into_vk(self) -> Self::Enum {
        // We have to use
        unsafe { core::mem::transmute_copy(&self) }
    }
}
/// An error that occurs when the vulkan implementation must allocate memory,
/// e.g. when creating a new handle (image, buffer, etc.) or begining a command
/// buffer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum AllocationError {
    /// The implementation could not allocate enough host memory.
    OutOfHostMemory = vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw(),
    /// The implementation could not allocate enough device memory.
    OutOfDeviceMemory = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw(),
}
impl core::error::Error for AllocationError {}
impl core::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfHostMemory => write!(f, "out of host memory"),
            Self::OutOfDeviceMemory => write!(f, "out of device memory"),
        }
    }
}
impl AllocationError {
    /// Convert from a vulkan result code. Must be one of
    /// `ERROR_OUT_OF_HOST_MEMORY` or `ERROR_OUT_OF_DEVICE_MEMORY` otherwise the
    /// result is unspecified uwu.
    fn from_vk(value: vk::Result) -> Self {
        // Communicates intent better.
        #[allow(clippy::wildcard_in_or_patterns)]
        match value {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfHostMemory,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | _ => Self::OutOfDeviceMemory,
        }
    }
}
#[macro_use]
mod macro_use {
    /// Create a struct as a thin wrapper around another type, with potentially
    /// several type-state bounds placed in a
    /// [`PhantomData`](core::marker::PhantomData), and implements
    /// [`ThinHandle`](super::ThinHandle) on it.
    ///
    /// ```ignore
    /// thin_handle!(pub struct SoupHandle<Kind: Soup>(u8));
    /// ```
    #[macro_export]
    macro_rules! thin_handle {
        {$(#[$attr:meta])* pub struct $name:ident$(<$($param_name:ident: $bound:path),+$(,)?>)?($underlying:ty$(,)?);} => {
            $(#[$attr])*
            #[repr(transparent)]
            pub struct $name$(<$($param_name: $bound),+>)?($crate::handle::NonNull<$underlying>, $(::core::marker::PhantomData<($($param_name),+,)>)?);
            /// # Safety
            /// Definition generated by macro.
            unsafe impl $(<$($param_name : $bound),+>)? $crate::handle::ThinHandle for $name$(<$($param_name,)+>)? {
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

    /// Create a trait, and create several unit structs implementing it.
    ///
    /// For enumerating a number of Type-State parameters.
    ///
    /// ```ignore
    /// typestate_enum!(pub enum trait Soup { pub struct Stew, pub struct Chili, pub struct Oatmeal });
    /// ```
    #[macro_export]
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
    /// Implement a flags trait for tuples of single flags, where the tuple
    /// implements the OR of the flags.
    /// ```ignore
    /// #[cfg_attr(doc, doc(fake_variadic))]
    /// impl<A: BufferUsage> BufferUsage for (A,) {
    ///     const FLAGS: vk::BufferUsageFlags = A::FLAGS;
    /// }
    /// flag_combinations! {
    ///    impl BufferUsage for [
    ///        (A,B),
    ///        (A,B,C),
    ///    ] {
    ///        const FLAGS : vk::BufferUsageFlags;
    ///    }
    /// }
    /// ```
    #[macro_export]
    macro_rules! flag_combinations {
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

    /// Create an enum that trivially converts to a vulkan equivalent.
    /// ```ignore
    /// vk_enum!(pub enum Filter: vk::Filter {
    ///     Nearest = NEAREST,
    ///     Linear = LINEAR,
    /// });
    /// ```
    #[macro_export]
    macro_rules! vk_enum {
        ($(#[$meta:meta])*
            pub enum $name:ident: $vk_name:ty {
            $($key:ident = $value:ident,)*
        }) => {
            $(#[$meta])*
            #[derive(Clone, Copy)]
            #[repr(i32)]
            pub enum $name {
                $($key = <$vk_name>::$value.as_raw(),)*
            }
            unsafe impl $crate::VkEnum for $name {
                type Enum = $vk_name;
            }
        };
    }
}

pub struct Pred<const B: bool>;
pub trait Satisified {}
impl Satisified for Pred<true> {}

/// A Vulkan "Logical Device," and all associated function pointers.
///
/// This contains all the function pointers needed to operate. All device-scope
/// operations go through this object.
///
/// To create one, acquire an [`ash::Device`] as documented by `ash` and pass it
/// to [`Device::from_ash`].
pub struct Device<'a>(&'a ash::Device);
thin_handle! {
    pub struct Queue(vk::Queue);
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
impl core::error::Error for SubmitError {}
impl core::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfHostMemory => write!(f, "out of host memory"),
            Self::OutOfDeviceMemory => write!(f, "out of device memory"),
            Self::DeviceLost => write!(f, "device lost"),
        }
    }
}
impl SubmitError {
    /// Convert from a vulkan result code. Must be one of
    /// `ERROR_OUT_OF_HOST_MEMORY`, `ERROR_OUT_OF_DEVICE_MEMORY`, or
    /// `ERROR_DEVICE_LOST`, otherwise the result is unspecified uwu.
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
    /// Wait forever, until signaled or an error occurs. If you're using this
    /// statically, consider using the non-timeout variant of the call which
    /// provides better type-level information.
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

/// A thin handle (an image, a fence, etc.) that has been associated with a
/// device and can thus have its own associated functions.
///
/// Use [`Device::bind`] to acquire.
pub struct Bound<'device, T: ThinHandle>(T, &'device Device<'device>);
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

#[must_use = "the handles contained will be leaked"]
pub struct SubmitWithFence<const WAITS: usize, const SIGNALS: usize> {
    pub waited_semaphores: [Semaphore<Waiting>; WAITS],
    pub signaled_semaphores: [Semaphore<Pending>; SIGNALS],
    pub pending_fence: Fence<Pending>,
}

/// An error that can occur while reading a pipeline cache. Contains the
/// sub-slice that the data takes up.
///
/// `Incomplete` is *not* a fatal error, just that the implementation had more
/// data than the array had room for.
pub enum TryGetPipelineCache<'a> {
    /// Fetched the whole data.
    Ok(&'a mut [u8]),
    /// Fetched as much data as we could, but there was too much.
    Incomplete(&'a mut [u8]),
}

impl<'device> Device<'device> {
    /// Bundle a handle with a device pointer, allowing the handle to have it's
    /// own associated functions and making method chaining possible even
    /// through several operations which change the type of it's operand.
    ///
    /// Generally, if an operation takes only the device and the handle and
    /// returns either, it can be executed using a [`Bound`] handle for cleaner
    /// syntax.
    ///
    /// This is a zero-cost operation.
    ///
    /// ```no_run
    /// # use fzvk::*;
    /// # let device: Device = todo!();
    /// let pending_fence: Fence<Pending> = todo!();
    /// # unsafe {
    /// let completed_fence = device.wait_fence(pending_fence).unwrap();
    /// device.destroy_fence(completed_fence);
    /// # }
    /// ```
    /// becomes
    /// ```no_run
    /// # use fzvk::*;
    /// # let device: Device = todo!();
    /// let pending_fence: Fence<Pending> = todo!();
    /// # unsafe {
    /// device.bind(pending_fence)
    ///     .wait().unwrap()
    ///     .destroy();
    /// # }
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
    pub fn from_ash(device: &'device ash::Device) -> Self {
        Device(device)
    }
    /// Submit work to a queue. The fence will be signaled upon completion.
    ///
    /// FIXME: this only allows for one batch at a time. While I believe this is
    /// at no loss of generality, it is at a *major* loss in performance.
    /// # Panics
    /// If `buffers` is empty.
    pub unsafe fn submit_with_fence<const WAITS: usize, const SIGNALS: usize>(
        &self,
        queue: &mut Queue,
        wait_semaphores: [WaitSemaphore; WAITS],
        buffers: &[CommandBufferReference<Primary>],
        signal_semaphores: [Semaphore<Unsignaled>; SIGNALS],
        signal_fence: Fence<Unsignaled>,
    ) -> Result<SubmitWithFence<WAITS, SIGNALS>, SubmitError> {
        // Usually we can just silently ignore it and return Ok(()), however we
        // have no way to issue adummy wait/signal operations to all the
        // necessary fences and semaphores.
        assert_ne!(buffers.len(), 0);
        self.0
            .queue_submit(
                queue.handle(),
                &[vk::SubmitInfo::default()
                    .wait_semaphores(
                        &wait_semaphores
                            .each_ref()
                            .map(|wait| wait.semaphore.handle()),
                    )
                    .wait_dst_stage_mask(
                        &wait_semaphores.each_ref().map(|wait| wait.wait_stage.get()),
                    )
                    .command_buffers(ThinHandle::handles_of(buffers))
                    .signal_semaphores(ThinHandle::handles_of(&signal_semaphores))],
                signal_fence.handle(),
            )
            .map_err(SubmitError::from_vk)?;
        Ok(SubmitWithFence {
            waited_semaphores: wait_semaphores.map(|wait| wait.semaphore.with_state()),
            signaled_semaphores: signal_semaphores.map(|signal| signal.with_state()),
            pending_fence: signal_fence.with_state(),
        })
    }
    /// Wait for a queue to complete all prior submissions, *as if* a fence on
    /// all prior submissions had been waited on.
    pub unsafe fn queue_wait_idle(&self, queue: &mut Queue) -> Result<(), SubmitError> {
        self.0
            .queue_wait_idle(queue.handle())
            .map_err(SubmitError::from_vk)
    }
    /// *As if* all queues owned by the device had `queue_wait_idle` called on
    /// them. # Safety
    /// * Requires *unique* (mutable) access to all device queues for the
    ///   duration of the call.
    ///
    /// This cannot currently be proven at compile time. Passing mutable
    /// references to *all* queues allows it to be checked, but it cannot be
    /// proven that all queues are present.
    pub unsafe fn wait_idle(&self, _queues: &mut [&mut Queue]) -> Result<(), SubmitError> {
        self.0.device_wait_idle().map_err(SubmitError::from_vk)
    }
    pub unsafe fn create_module<Module: StaticSpirV>(
        &self,
    ) -> Result<ShaderModule<Module>, AllocationError> {
        debug_assert!(!Module::SPIRV.is_empty());
        self.0
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(Module::SPIRV),
                None,
            )
            .map(|handle| ThinHandle::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn create_dynamic_module(
        &self,
        spirv: &[u32],
    ) -> Result<DynamicShaderModule, AllocationError> {
        debug_assert!(!spirv.is_empty());
        self.0
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
            .map(|handle| DynamicShaderModule::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_module<Module: ThinHandle<Handle = vk::ShaderModule>>(
        &self,
        module: Module,
    ) -> &Self {
        self.0.destroy_shader_module(module.into_handle(), None);
        self
    }
    pub unsafe fn create_semaphore(&self) -> Result<Semaphore<Unsignaled>, AllocationError> {
        self.0
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .map(|handle| Semaphore::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_semaphore(&self, semaphore: Semaphore<Unsignaled>) -> &Self {
        self.0.destroy_semaphore(semaphore.handle(), None);
        self
    }
    /// Create a fence in the given state, [`Signaled`] or [`Unsignaled`].
    /// ```no_run
    /// # use fzvk::*;
    /// # let device : Device = todo!();
    /// # unsafe {
    /// let signaled = device.create_fence::<Unsignaled>().unwrap();
    /// # }
    /// ```
    pub unsafe fn create_fence<State: KnownFenceState>(
        &self,
    ) -> Result<Bound<Fence<State>>, AllocationError> {
        self.0
            .create_fence(
                &vk::FenceCreateInfo::default().flags(State::CREATION_FLAGS),
                None,
            )
            .map(|handle| Fence::from_handle_unchecked(handle))
            .map(|handle| self.bind(handle))
            .map_err(AllocationError::from_vk)
    }
    /// Set each fence to the "Unsignaled" state. It is valid to reset an
    /// already unsignaled fence.
    pub unsafe fn reset_fence<State: KnownFenceState>(
        &self,
        fence: Fence<State>,
    ) -> Result<Fence<Unsignaled>, AllocationError> {
        self.reset_fences([fence]).map(|[fence]| fence)
    }
    /// Set each fence to the "Unsignaled" state. It is valid to reset an
    /// already unsignaled fence.
    pub unsafe fn reset_fences<const N: usize, State: KnownFenceState>(
        &self,
        fences: [Fence<State>; N],
    ) -> Result<[Fence<Unsignaled>; N], AllocationError> {
        self.0
            .reset_fences(ThinHandle::handles_of(&fences))
            .map_err(AllocationError::from_vk)?;
        Ok(fences.map(|f| Fence::with_state(f)))
    }
    /// Poll the fence. If the result is [`FencePoll::Unsignaled`], the result
    /// *may* be immediately out-of-date.
    pub unsafe fn fence_status(
        &self,
        fence: Fence<Pending>,
    ) -> Result<FencePoll<Fence<Signaled>, Fence<Pending>>, SubmitError> {
        let status = self
            .0
            .get_fence_status(fence.handle())
            .map_err(SubmitError::from_vk)?;
        Ok(if status {
            FencePoll::Signaled(fence.with_state())
        } else {
            FencePoll::Unsignaled(fence.with_state())
        })
    }
    /// Convinience wrapper around [`Self::wait_all_fences_timeout`] for once
    /// fence.
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
    /// Wait for every fence in the array to become signaled, or fail after some
    /// timeout. If the result is [`FencePoll::Unsignaled`], the result may
    /// become immediately out-of-date.
    pub unsafe fn wait_all_fences_timeout<const N: usize>(
        &self,
        fences: [Fence<Pending>; N],
        timeout: Timeout,
    ) -> Result<FencePoll<[Fence<Signaled>; N], [Fence<Pending>; N]>, SubmitError> {
        if N == 0 {
            // We can't easily prove to the compiler that [] is valid in this
            // block
            return Ok(FencePoll::Signaled(core::array::from_fn(
                |_| unreachable!(),
            )));
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
    /// Wait for every fence in the array to become signaled. Convinience
    /// wrapper around [`Self::wait_all_fences_timeout`] with simpler types.
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
    /// Returns `Ok(true)` if one or more fences were signaled, `Ok(false)` if
    /// not.
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
        // there's no extra type-information we can communicate here, unlike the
        // other family.
        self.wait_any_fence_timeout(fences, Timeout::Infinite)
    }
    /// Destroy a fence. The bound [`KnownFenceState`] is used to ensure that
    /// the fence is not in use within a currently-executing submission.
    pub unsafe fn destroy_fence<State: KnownFenceState>(&self, fence: Fence<State>) -> &Self {
        self.0.destroy_fence(fence.into_handle(), None);
        self
    }
    /// Create a [`CommandPool`] from which [`CommandBuffer`]s may be allocated.
    pub unsafe fn create_command_pool(
        &self,
        info: &vk::CommandPoolCreateInfo,
    ) -> Result<CommandPool, AllocationError> {
        self.0
            .create_command_pool(info, None)
            .map(|handle| CommandPool::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    /// Destroy a [`CommandPool`].
    /// # Safety
    /// Any [`CommandBuffer`]s allocated from this pool do not need to be
    /// [freed](Self::free_command_buffers) prior to this call, however they
    /// become invalid to access and should be dropped.
    pub unsafe fn destroy_command_pool(&self, pool: CommandPool) -> &Self {
        self.0.destroy_command_pool(pool.handle(), None);
        self
    }
    /// Allocate [`CommandBuffer`]s from the given pool.
    ///
    /// The constant `N` is the number of command buffers to allocate. Use this
    /// along with a destructuring to create several buffers at once, which may
    /// be more efficient:
    /// ```no_run
    /// # use fzvk::*;
    /// # let device : Device = todo!();
    /// # let mut pool : CommandPool = todo!();
    /// # unsafe {
    /// let [single_buffer] = device.allocate_command_buffers(&mut pool, Primary).unwrap();
    /// let [buffer_a, buffer_b] = device.allocate_command_buffers(&mut pool, Primary).unwrap();
    /// # }
    /// ```
    pub unsafe fn allocate_command_buffers<const N: usize, Level: CommandBufferLevel>(
        &self,
        pool: &mut CommandPool,
        _: Level,
    ) -> Result<[CommandBuffer<Level>; N], AllocationError> {
        // This is a const-time branch, and does not exist at runtime.
        if N == 0 {
            return Ok(core::array::from_fn(|_| unreachable!()));
        }
        let vec = self
            .0
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(N as _)
                    .command_pool(pool.handle())
                    .level(Level::LEVEL),
            )
            .map_err(AllocationError::from_vk)?;
        let array = <[vk::CommandBuffer; N]>::try_from(vec).unwrap();
        Ok(array.map(|h| CommandBuffer::from_handle_unchecked(h)))
    }
    /// Free [`CommandBuffer`]s back to the given pool. It is not necessary to
    /// do this before destroying the command buffer.
    pub unsafe fn free_command_buffers<const N: usize, Level: CommandBufferLevel>(
        &self,
        pool: &mut CommandPool,
        buffers: [CommandBuffer<Level>; N],
    ) -> &Self {
        // This is a const-time branch, and does not exist at runtime.
        if N != 0 {
            self.0
                .free_command_buffers(pool.handle(), ThinHandle::handles_of(&buffers));
        }
        self
    }
    /// Create a buffer. The `BufferUsageFlags` are passed at compile time:
    ///
    /// ```no_run
    /// # use fzvk::*;
    /// # use std::num::NonZero;
    /// # let device: Device = todo!();
    /// # unsafe {
    /// let buffer = device.create_buffer(
    ///         (Storage, TransferSrc, TransferDst),
    ///         NonZero::new(1024).unwrap(),
    ///         SharingMode::Concurrent(SharingFamilies::from_array(&[0, 1])),
    ///     );
    /// # }
    /// ```
    pub unsafe fn create_buffer<Usage: BufferUsage>(
        &self,
        _usage: Usage,
        size: NonZero<u64>,
        sharing: SharingMode,
    ) -> Result<memory::Virtual<Buffer<Usage>>, AllocationError> {
        self.0
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .usage(Usage::FLAGS)
                    .size(size.get())
                    .sharing_mode(sharing.mode())
                    .queue_family_indices(sharing.family_slice()),
                None,
            )
            .map(|handle| memory::Virtual::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_buffer(&self, buffer: impl ThinHandle<Handle = vk::Buffer>) -> &Self {
        self.0.destroy_buffer(buffer.handle(), None);
        self
    }
    /// Create an image. The `vkImageUsageFlags`, Dimensionality, Aspect mask,
    /// and Multisampled-ness are passed at compile time:
    ///
    /// ```no_run
    /// # use fzvk::*;
    /// # use core::num::NonZero;
    /// # let device: Device = todo!();
    /// # unsafe {
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
    ///         MipCount::ONE,
    ///         format::R8G8B8A8_UNORM,
    ///         // Makes a non-multisampled image.
    ///         // You can also use the MultiSampled enum.
    ///         SingleSampled,
    ///         vk::ImageTiling::OPTIMAL,
    ///         SharingMode::Exclusive
    ///     ).unwrap();
    /// # }
    /// ```
    ///
    /// See: [`ImageUsage`], [`Extent`], [`ColorFormat`],
    /// [`DepthStencilFormat`], [`MultiSampled`].
    pub unsafe fn create_image<
        Usage: ImageUsage,
        Ext: Extent,
        Format: format::Format,
        Samples: ImageSamples,
    >(
        &self,
        _usage: Usage,
        extent: Ext,
        mip_levels: MipCount,
        format: Format,
        samples: Samples,
        tiling: vk::ImageTiling,
        sharing: SharingMode,
    ) -> Result<memory::Virtual<Image<Usage, Ext::Dim, Format, Samples>>, AllocationError> {
        self.0
            .create_image(
                &vk::ImageCreateInfo::default()
                    .extent(extent.extent())
                    .array_layers(extent.layers().get())
                    .queue_family_indices(sharing.family_slice())
                    .sharing_mode(sharing.mode())
                    .usage(Usage::FLAGS)
                    .image_type(Ext::Dim::IMAGE_TYPE)
                    .format(Format::FORMAT)
                    .mip_levels(mip_levels.0.get())
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .samples(samples.flag())
                    .tiling(tiling),
                None,
            )
            .map(|handle| memory::Virtual::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_image(&self, image: impl ThinHandle<Handle = vk::Image>) -> &Self {
        self.0.destroy_image(image.into_handle(), None);
        self
    }
    pub unsafe fn memory_requirements<Resource: memory::HasMemoryRequirements>(
        &self,
        virt: &Resource,
    ) -> vk::MemoryRequirements {
        virt.memory_requirements(self)
    }
    pub unsafe fn bind_memory<Resource, Access>(
        &self,
        virt: memory::Virtual<Resource>,
        memory: &mut memory::Memory<Access>,
        offset: u64,
    ) -> Result<<memory::Virtual<Resource> as memory::HasMemoryRequirements>::Bound, vk::Result>
    where
        Access: memory::MemoryAccess,
        Resource: memory::HasMemoryRequirements,
        memory::Virtual<Resource>: memory::BindMemory<Access>,
    {
        use memory::BindMemory;
        virt.bind_memory(self, memory, offset)
    }
    pub unsafe fn free_memory<Access: memory::MemoryAccess>(
        &self,
        memory: memory::Memory<Access>,
    ) -> &Self {
        self.0.free_memory(memory.handle(), None);
        self
    }
    pub unsafe fn memory_commitment(
        &self,
        memory: &memory::Memory<memory::LazilyAllocated>,
    ) -> u64 {
        self.0.get_device_memory_commitment(memory.handle())
    }
    /// Attempt to map a memory region. A host-accessible pointer is returned
    /// referencing the memory.
    ///
    /// Cache control of memory visible through the pointer is left to the
    /// client code. Use [`flush_memory`](Self::flush_memory) and
    /// [`invalidate_memory`](Self::invalidate_memory) to ensure that reads and
    /// writes observe expected values.
    ///
    /// # Errors
    /// In the case an error occurs, the memory region is guaranteed to still be
    /// in the un-mapped state.
    pub unsafe fn map_memory(
        &self,
        memory: memory::Memory<memory::HostVisible>,
        range: impl core::ops::RangeBounds<u64>,
    ) -> Result<
        (memory::Memory<memory::HostMapped>, *mut u8),
        (memory::Memory<memory::HostVisible>, vk::Result),
    > {
        let (offset, size) = memory::range_to_offset_size(range);
        match self
            .0
            .map_memory(memory.handle(), offset, size, vk::MemoryMapFlags::empty())
        {
            Ok(ptr) => Ok((memory.with_state(), ptr.cast())),
            Err(err) => Err((memory, err)),
        }
    }
    /// Flush the host cache of the given range of mapped memory. The range in
    /// interpreted as relative to the opaque memory object, not relative to the
    /// mapped region.
    ///
    /// This makes previous writes to the memory *available* to the host memory
    /// domain, which can then be made *available* to the device memory domain
    /// via the execution of a corresponding
    /// [`Host::WRITE`](sync::barrier::Host::WRITE) pipeline
    /// [barrier](Self::barrier). **This requires device synchronization** -
    /// cache control is only half the equation!
    pub unsafe fn flush_memory<'this>(
        &'this self,
        memory: &'_ mut memory::Memory<memory::HostMapped>,
        range: impl core::ops::RangeBounds<u64>,
    ) -> Result<&'this Self, vk::Result> {
        let (offset, size) = memory::range_to_offset_size(range);
        self.0
            .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                .memory(memory.handle())
                .offset(offset)
                .size(size)])
            .map(|()| self)
    }
    /// Invalidate the host cache of the given range of mapped memory. The range
    /// in interpreted as relative to the opaque memory object, not relative to
    /// the mapped region.
    ///
    /// This makes previous device writes to the memory, which have been made
    /// *available* to the host memory domain via a corresponding
    /// [`Host::READ`](sync::barrier::Host::READ) pipeline
    /// [barrier](Self::barrier), *visible* through the memory's mapped pointer.
    /// **This requires device synchronization** via a [`Fence`] - cache control
    /// is only half the equation!
    pub unsafe fn invalidate_memory<'this>(
        &'this self,
        memory: &'_ mut memory::Memory<memory::HostMapped>,
        range: impl core::ops::RangeBounds<u64>,
    ) -> Result<&'this Self, vk::Result> {
        let (offset, size) = memory::range_to_offset_size(range);
        self.0
            .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                .memory(memory.handle())
                .offset(offset)
                .size(size)])
            .map(|()| self)
    }
    pub unsafe fn unmap_memory(
        &self,
        memory: memory::Memory<memory::HostMapped>,
    ) -> memory::Memory<memory::HostVisible> {
        self.0.unmap_memory(memory.handle());
        memory.with_state()
    }
    /// See [`Self::create_image_view_mutable_format`] to utilize
    /// `MUTABLE_FORMAT` image capabilities.
    pub unsafe fn create_image_view<
        Usage: ImageUsage,
        ImageDim: Dimensionality,
        ViewDim: CanView<ImageDim>,
        Format: format::Format,
        Samples: ImageSamples,
        ViewAspect: format::aspect::AspectMask,
    >(
        &self,
        image: &Image<Usage, ImageDim, Format, Samples>,
        // FIXME: Sux.
        dim: ViewDim,
        component_mapping: ComponentMapping,
        subresource_range: ImageDim::SubresourceRange,
        aspect_ty: format::aspect::ViewAspects<ViewAspect>,
    ) -> Result<ImageView<Usage, ViewDim, Format, Samples, ViewAspect>, AllocationError>
    where
        Format::AspectMask: format::aspect::AspectSupersetOf<ViewAspect>,
    {
        // Safety - Image and view statically forced to have same format.
        self.create_image_view_mutable_format::<Usage, ImageDim, ViewDim, Format, Format, Samples, ViewAspect>(
            image,
            dim,
            component_mapping,
            subresource_range,
            aspect_ty,
        )
    }
    /// Safety - If the `ImageFormat` and `ViewFormat` differ, the image must
    /// have been created with the `MUTABLE_FORMAT` flag.
    ///
    /// See [`Self::create_image_view`] for a safer constrained version.
    pub unsafe fn create_image_view_mutable_format<
        Usage: ImageUsage,
        ImageDim: Dimensionality,
        ViewDim: CanView<ImageDim>,
        ImageFormat: format::Format,
        ViewFormat: format::CompatibleWith<ImageFormat>,
        Samples: ImageSamples,
        ViewAspect: format::aspect::AspectMask,
    >(
        &self,
        image: &Image<Usage, ImageDim, ImageFormat, Samples>,
        // FIXME: Sux.
        _dim: ViewDim,
        component_mapping: ComponentMapping,
        subresource_range: ImageDim::SubresourceRange,
        _aspect_ty: format::aspect::ViewAspects<ViewAspect>,
    ) -> Result<ImageView<Usage, ViewDim, ViewFormat, Samples, ViewAspect>, AllocationError>
    where
        ImageFormat::AspectMask: format::aspect::AspectSupersetOf<ViewAspect>,
    {
        self.0
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.handle())
                    .format(ViewFormat::FORMAT)
                    .view_type(ViewDim::VIEW_TYPE)
                    .subresource_range(subresource_range.subresource_range(ViewAspect::ALL_FLAGS))
                    .components(component_mapping.into()),
                None,
            )
            .map(|handle| ThinHandle::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    pub unsafe fn destroy_image_view<
        Usage: ImageUsage,
        Dim: Dimensionality,
        Format: format::Format,
        Samples: ImageSamples,
        Aspect: format::aspect::AspectMask,
    >(
        &self,
        view: ImageView<Usage, Dim, Format, Samples, Aspect>,
    ) -> &Self {
        self.0.destroy_image_view(view.into_handle(), None);
        self
    }
    pub unsafe fn create_sampler(
        &self,
        min_filter: sampler::Filter,
        mag_filter: sampler::Filter,
        mipmap_mode: sampler::Filter,
        lod: impl core::ops::RangeBounds<f32>,
        lod_bias: f32,
        address_mode: sampler::AddressMode3,
        border_color: sampler::BorderColor,
        anisotropy: Option<sampler::Anisotropy>,
        compare_op: Option<sampler::CompareOp>,
    ) -> Result<sampler::Sampler, AllocationError> {
        use core::ops::Bound;
        // Weirdly, the spec does not mention valid ranges, or even that NaN or
        // Infinity are disallowed. Yippee?
        let min_lod = match lod.start_bound().cloned() {
            Bound::Unbounded => 0.0,
            // FIXME: is this even meaningful lol.
            Bound::Excluded(x) => x.next_up(),
            Bound::Included(x) => x,
        };
        let max_lod = match lod.end_bound().cloned() {
            Bound::Unbounded => vk::LOD_CLAMP_NONE,
            // FIXME: is this even meaningful lol.
            Bound::Excluded(x) => x.next_down(),
            Bound::Included(x) => x,
        };
        self.0
            .create_sampler(
                &vk::SamplerCreateInfo {
                    flags: vk::SamplerCreateFlags::empty(),
                    mag_filter: mag_filter.into_vk(),
                    min_filter: min_filter.into_vk(),
                    mipmap_mode: match mipmap_mode {
                        sampler::Filter::Linear => vk::SamplerMipmapMode::LINEAR,
                        sampler::Filter::Nearest => vk::SamplerMipmapMode::NEAREST,
                    },
                    address_mode_u: address_mode.u.into_vk(),
                    address_mode_v: address_mode.v.into_vk(),
                    address_mode_w: address_mode.w.into_vk(),
                    mip_lod_bias: lod_bias,
                    anisotropy_enable: anisotropy.is_some() as u32,
                    max_anisotropy: anisotropy.map_or(0.0, sampler::Anisotropy::get),
                    compare_enable: compare_op.is_some() as u32,
                    compare_op: compare_op.unwrap_or(sampler::CompareOp::Never).into_vk(),
                    min_lod,
                    max_lod,
                    border_color: border_color.into_vk(),
                    unnormalized_coordinates: false as u32,
                    ..Default::default()
                },
                None,
            )
            .map(|handle| sampler::Sampler::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    /// Create a pipeline cache, optionally populated with some data.
    ///
    /// If the data is found ny the implementation to be incompatible (but still
    /// must be valid as defined below) the pipeline cache will be empty.
    /// # Safety
    /// If `initial_data` is not None or empty, it must be the same data
    /// retrieved from a previous successful call to
    /// [`Self::get_pipeline_cache_data`].
    ///
    /// Fixme: is this correct, or is this always safe? Vk spec is a lil
    /// conflicting:
    /// * `VUID-VkPipelineCacheCreateInfo-initialDataSize-00769`: "If
    ///   initialDataSize is not 0, pInitialData must have been retrieved from a
    ///   previous call to vkGetPipelineCacheData"
    /// * `10.7.4` Note: "[...] providing invalid pipeline cache data as input
    ///   to any Vulkan API commands will result in the provided pipeline cache
    ///   data being ignored."
    pub unsafe fn create_pipeline_cache(
        &self,
        // Option<> is more idiomatic than empty-or-not
        initial_data: Option<&[u8]>,
    ) -> Result<PipelineCache, AllocationError> {
        self.0
            .create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(initial_data.unwrap_or(&[])),
                None,
            )
            .map(|handle| PipelineCache::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    /// Merge several caches into a destination cache.
    ///
    /// The destination cache will contain its previous contents, plus all the
    /// contents of the others. The other caches are unchanged.
    pub unsafe fn merge_pipeline_caches<'a>(
        &'a self,
        into: &mut PipelineCache,
        from: &[BorrowedPipelineCache],
    ) -> Result<&'a Self, AllocationError> {
        if from.is_empty() {
            return Ok(self);
        }
        self.0
            .merge_pipeline_caches(into.handle(), ThinHandle::handles_of(from))
            .map_err(AllocationError::from_vk)?;
        Ok(self)
    }
    /// Shorthand for fetching the pipeline's size, then fetching its data.
    #[cfg(feature = "alloc")]
    pub unsafe fn get_pipeline_cache_data(
        &self,
        cache: &PipelineCache,
    ) -> Result<alloc::vec::Vec<u8>, AllocationError> {
        // AllocationError here is exhaustive, since VK_INCOMPLETE can't be
        // returned. (internally, ash does the get length -> get data dance, and
        // the length of data cannot be changed since we know noone else has
        // mutable access)
        self.0
            .get_pipeline_cache_data(cache.handle())
            .map_err(AllocationError::from_vk)
    }
    /// Get the number of bytes in the pipeline cache. If the pipeline cache is
    /// mutably used, e.g. in a merge or pipeline create call, the value may
    /// become out-of-date.
    pub unsafe fn get_pipeline_cache_data_size(
        &self,
        cache: &PipelineCache,
    ) -> Result<usize, AllocationError> {
        let mut size = core::mem::MaybeUninit::<usize>::uninit();
        let res = unsafe {
            (self.0.fp_v1_0().get_pipeline_cache_data)(
                self.0.handle(),
                cache.handle(),
                size.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        match res {
            vk::Result::SUCCESS | vk::Result::INCOMPLETE => Ok(unsafe { size.assume_init() }),
            e => Err(AllocationError::from_vk(e)),
        }
    }
    /// Get some data from the pipeline.
    ///
    /// The freshly populated subslice of data is returned, with an indication
    /// of whether the buffer was sufficient to fetch *all* of the data, or
    /// whether only some subset of it was successfully fetched.
    pub unsafe fn try_get_pipeline_cache_data<'a>(
        &'_ self,
        cache: &PipelineCache,
        data: &'a mut [core::mem::MaybeUninit<u8>],
    ) -> Result<TryGetPipelineCache<'a>, AllocationError> {
        let mut len = data.len();
        let res = unsafe {
            (self.0.fp_v1_0().get_pipeline_cache_data)(
                self.0.handle(),
                cache.handle(),
                // In-out param, our max len in, and the len actually written
                // out.
                &raw mut len,
                // Subtlety: Data ptr is *non null*, even if empty, so this
                // *will not* trigger the "get total len" functionality. It will
                // correctly be treated as a get of length zero, and will fail
                // gracefully (or succeed if actually empty?). Otherwise, we'd
                // have a big issue of zero-len slice being treated as an
                // as-big-as-the-implementation-wants slice. Woe!
                data.as_mut_ptr().cast(),
            )
        };
        match res {
            vk::Result::SUCCESS | vk::Result::INCOMPLETE => {
                assert!(len <= data.len());
                let populated = &mut data[..len];

                // Safety - the API reports how many bytes it wrote, so we can
                // assume 0..len has been written and is now init.
                let assume_init = unsafe {
                    core::mem::transmute::<&'a mut [core::mem::MaybeUninit<u8>], &'a mut [u8]>(
                        populated,
                    )
                };

                if res == vk::Result::SUCCESS {
                    Ok(TryGetPipelineCache::Ok(assume_init))
                } else {
                    Ok(TryGetPipelineCache::Incomplete(assume_init))
                }
            }
            e => Err(AllocationError::from_vk(e)),
        }
    }
    pub unsafe fn destroy_pipeline_cache(&self, cache: PipelineCache) -> &Self {
        self.0.destroy_pipeline_cache(cache.into_handle(), None);
        self
    }
    pub unsafe fn create_pipeline_layout<Constants: PushConstant>(
        &self,
    ) -> Result<PipelineLayout<Constants>, AllocationError> {
        self.0
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(Constants::RANGES),
                None,
            )
            .map(|handle| PipelineLayout::from_handle_unchecked(handle))
            .map_err(AllocationError::from_vk)
    }
    /// Create (possibly) several compute pipelines.
    ///
    /// Providing multiple pipeline creations in one invocation is a hint to the
    /// implementation that the pipelines may be related and to search for
    /// shortcuts in compilation, and thus may be more efficient than creating
    /// several pipelines individually. It is always valid to use bulk creations
    /// regardless of if the pipelines are actually correlated or not.
    ///
    /// # Errors
    /// If *any* pipeline fails to be created, the others are destroyed and the
    /// error is returned.
    pub unsafe fn create_compute_pipelines<const N: usize>(
        &self,
        cache: Option<&mut PipelineCache>,
        infos: [ComputePipelineCreateInfo; N],
    ) -> Result<[Pipeline<Compute>; N], AllocationError> {
        if N == 0 {
            // We can't easily prove to the compiler that [] is valid in this
            // block
            return Ok(core::array::from_fn(|_| unreachable!()));
        }
        let infos = infos.each_ref().map(|info| {
            vk::ComputePipelineCreateInfo::default()
                .layout(*info.layout)
                .stage(info.shader.create_info())
        });
        let mut output = core::mem::MaybeUninit::<[vk::Pipeline; N]>::uninit();
        // We call the raw form (not ash's wrapper) since it works on slices and
        // vecs while we work on arrays and can thus skip an allocation
        let res = (self.0.fp_v1_0().create_compute_pipelines)(
            self.0.handle(),
            cache
                .map(|cache| cache.handle())
                .unwrap_or(vk::PipelineCache::null()),
            N.try_into().unwrap(),
            infos.as_ptr(),
            core::ptr::null(),
            output.as_mut_ptr().cast(),
        );
        // # Safety
        // Always populates the whole array, failures are populated with NULL.
        // Complete failure (i.e. no successes) even results in all NULLs. How
        // polite!
        let output = output.assume_init();

        match res {
            vk::Result::SUCCESS => Ok(output.map(|handle| {
                debug_assert!(!handle.is_null());
                Pipeline::from_handle_unchecked(handle)
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
            .map(|handle| RenderPass::from_handle_unchecked(handle))
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
        // Safety - directly from a ThinHandle, so is known non-null.
        Ok(RecordingBuffer::from_handle_unchecked(buffer.handle()))
    }
    pub unsafe fn bind_pipeline<
        'this,
        Kind: pipeline::BindPoint,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
    >(
        &'this self,
        buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        pipeline: &'_ Pipeline<Kind>,
    ) -> &'this Self {
        self.0
            .cmd_bind_pipeline(buffer.handle(), Kind::BIND_POINT, pipeline.handle());
        self
    }
    pub unsafe fn push_constants<
        'a,
        'b,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
    >(
        &'a self,
        buffer: &'b mut RecordingBuffer<Level, MaybeInsideRender>,
        layout: &'b PipelineLayout<()>,
        stages: vk::ShaderStageFlags,
        offset_words: u32,
        constants: &'b [PushValue],
    ) -> &'a Self {
        self.0.cmd_push_constants(
            buffer.handle(),
            layout.handle(),
            stages,
            offset_words * 4,
            PushValue::bytes_of(constants),
        );
        self
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
            buffer.handle(),
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
            .end_command_buffer(buffer.handle())
            .map_err(AllocationError::from_vk)?;
        Ok(self)
    }
    pub unsafe fn barrier<
        'a,
        'images,
        const BUFFER_BARRIERS: usize,
        ImageTransitions: image::ImageTransitions<'images>,
        Level: CommandBufferLevel,
    >(
        &'a self,
        buffer: &'_ mut RecordingBuffer<Level, OutsideRender>,
        wait_for: barrier::Write,
        block: impl Into<barrier::ReadWrite>,
        buffer_barriers: [BufferBarrier<'_>; BUFFER_BARRIERS],
        image_transitions: ImageTransitions,
    ) -> ImageTransitions::AfterTransition<'images> {
        let (src_stage, src_access) = wait_for.into_stage_access();
        let (dst_stage, dst_access) = block.into().into_stage_access();
        self.0.cmd_pipeline_barrier(
            buffer.handle(),
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)],
            &buffer_barriers.map(|barrier| {
                vk::BufferMemoryBarrier::default()
                    .buffer(barrier.buffer)
                    .offset(barrier.offset)
                    .size(barrier.len)
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)
                    // Defines a "none" ownership transfer.
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            }),
            image_transitions
                .as_barriers(src_access, dst_access)
                .as_ref(),
        );
        image_transitions.into_after_transition()
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
        SuperUsage: ImageSuperset<TransferDst>,
        Dim: Dimensionality,
        Format: format::Format,
        Layout: layout::UsableAs<layout::TransferDst>,
    >(
        &'a self,
        command_buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        buffer: impl AsRef<Buffer<TransferSrc>>,
        // FIXME: DepthStencil needs a way to specify *which* of the two
        // aspects.
        image: &'_ ImageReference<'_, SuperUsage, Dim, Format, SingleSampled, Layout>,
        regions: [BufferImageCopy<Dim, Format::AspectMask>; N],
    ) -> &'a Self {
        self.0.cmd_copy_buffer_to_image(
            command_buffer.handle(),
            buffer.as_ref().handle(),
            image.handle(),
            Layout::LAYOUT,
            &regions.map(|region| vk::BufferImageCopy {
                image_offset: region.image_offset.offset(),
                image_extent: region.image_extent.extent(),
                buffer_offset: region.buffer_offset,
                // None for automatic pitch calculation is mapped to a value of
                // 0, convenient!
                buffer_row_length: region.pitch.row_pitch().map(NonZero::get).unwrap_or(0),
                buffer_image_height: region.pitch.slice_pitch().map(NonZero::get).unwrap_or(0),
                image_subresource: region
                    .layers
                    .subresource_layers(format::aspect::AspectMask::aspect_mask(&region.aspects)),
            }),
        );
        self
    }
    pub unsafe fn copy_image_to_buffer<
        'a,
        const N: usize,
        Level: CommandBufferLevel,
        MaybeInsideRender: CommandBufferState,
        SuperUsage: ImageSuperset<TransferSrc>,
        Dim: Dimensionality,
        Format: format::Format,
        Layout: layout::UsableAs<layout::TransferSrc>,
    >(
        &'a self,
        command_buffer: &'_ mut RecordingBuffer<Level, MaybeInsideRender>,
        image: &'_ ImageReference<'_, SuperUsage, Dim, Format, SingleSampled, Layout>,
        buffer: impl AsRef<Buffer<TransferDst>>,
        regions: [BufferImageCopy<Dim, Format::AspectMask>; N],
    ) -> &'a Self {
        self.0.cmd_copy_image_to_buffer(
            command_buffer.handle(),
            image.handle(),
            Layout::LAYOUT,
            buffer.as_ref().handle(),
            &regions.map(|region| vk::BufferImageCopy {
                image_offset: region.image_offset.offset(),
                image_extent: region.image_extent.extent(),
                buffer_offset: region.buffer_offset,
                // None for automatic pitch calculation is mapped to a value of
                // 0, convenient!
                buffer_row_length: region.pitch.row_pitch().map(NonZero::get).unwrap_or(0),
                buffer_image_height: region.pitch.slice_pitch().map(NonZero::get).unwrap_or(0),
                image_subresource: region
                    .layers
                    .subresource_layers(format::aspect::AspectMask::aspect_mask(&region.aspects)),
            }),
        );
        self
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
            .cmd_begin_render_pass(buffer.handle(), todo!(), vk::SubpassContents::INLINE);
        unsafe { buffer.with_state() }
    }
    pub unsafe fn draw<const N: usize, Level: CommandBufferLevel>(
        &self,
        buffer: &mut RecordingBuffer<Level, RemainingSubpasses<N>>,
        vertices: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
    ) -> &Self
    where
        SubpassCount<N>: ValidSubpassCount,
    {
        self.0.cmd_draw(
            buffer.handle(),
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
        indices: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
    ) -> &Self
    where
        SubpassCount<N>: ValidSubpassCount,
    {
        self.0.cmd_draw_indexed(
            buffer.handle(),
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
        // Shouldn't HasNextSubpass already imply this? I guess there's a reason
        // why generic const arithmetic is unstable :3c
        SubpassCount<{ N - 1 }>: ValidSubpassCount,
    {
        self.0
            .cmd_next_subpass(buffer.handle(), vk::SubpassContents::INLINE);
        unsafe { buffer.with_state() }
    }
    pub unsafe fn end_render_pass<Level: CommandBufferLevel>(
        &self,
        buffer: RecordingBuffer<Level, RemainingSubpasses<1>>,
    ) -> RecordingBuffer<Level, OutsideRender> {
        self.0.cmd_end_render_pass(buffer.handle());
        unsafe { buffer.with_state() }
    }
}
