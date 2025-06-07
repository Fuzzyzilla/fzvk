//! Fences, Semaphores, Barriers galore!
use super::{ShaderStage, ThinHandle, vk};
use core::marker::PhantomData;

pub mod barrier {
    //! Types for describing execution and/or memory dependencies.
    //!
    //! These are invoked by the user via `vkCmdPipelineBarrier` or through a
    //! wait semaphore operation.
    use super::{ShaderStage, vk};
    pub trait AccessFlags: core::ops::BitOr<Output = Self> + Copy {
        const NONE: Self;
        fn into_flags(self) -> vk::AccessFlags;
    }
    #[derive(Copy, Clone)]
    pub struct StageAccess<Access: AccessFlags> {
        stages: PipelineStages,
        access: Access,
    }
    impl<Access: AccessFlags> StageAccess<Access> {
        /// Stages, with no memory access types.
        pub fn from_stages(stages: PipelineStages) -> Self {
            Self {
                stages,
                access: AccessFlags::NONE,
            }
        }
        pub fn shader<Stage: ShaderStage>() -> Self {
            Self::from_stages(Stage::PIPE_STAGE)
        }
        pub unsafe fn shader_access<Stage: ShaderStage>(access: Access) -> Self {
            Self::from_stage_access(Stage::PIPE_STAGE, access)
        }
        /// Stages and access flags.
        /// # Safety
        /// Each access flag must be a type used by at least one of the members
        /// of `stages`.
        pub const unsafe fn from_stage_access(stages: PipelineStages, access: Access) -> Self {
            Self { stages, access }
        }
        /// Convert into a stage flags and access flags pair.
        pub fn into_stage_access(self) -> (vk::PipelineStageFlags, vk::AccessFlags) {
            (self.stages.into_flags(), self.access.into_flags())
        }
    }
    impl<Access: AccessFlags> core::ops::BitOr<StageAccess<Access>> for StageAccess<Access> {
        type Output = StageAccess<Access>;
        fn bitor(self, rhs: StageAccess<Access>) -> Self::Output {
            StageAccess {
                stages: self.stages | rhs.stages,
                access: self.access | rhs.access,
            }
        }
    }
    impl<Access: AccessFlags> core::ops::BitOr<StageAccess<Access>> for &'_ StageAccess<Access> {
        type Output = StageAccess<Access>;
        fn bitor(self, rhs: StageAccess<Access>) -> Self::Output {
            StageAccess {
                stages: self.stages | rhs.stages,
                access: self.access | rhs.access,
            }
        }
    }
    impl<Access: AccessFlags> core::ops::BitOr<&'_ StageAccess<Access>> for StageAccess<Access> {
        type Output = StageAccess<Access>;
        fn bitor(self, rhs: &'_ StageAccess<Access>) -> Self::Output {
            StageAccess {
                stages: self.stages | rhs.stages,
                access: self.access | rhs.access,
            }
        }
    }
    impl<'a, Access: AccessFlags> core::ops::BitOr<&'a StageAccess<Access>>
        for &'a StageAccess<Access>
    {
        type Output = StageAccess<Access>;
        fn bitor(self, rhs: &'a StageAccess<Access>) -> Self::Output {
            StageAccess {
                stages: self.stages | rhs.stages,
                access: self.access | rhs.access,
            }
        }
    }
    /// Description of what parts of the execution pipe should finish before
    /// signaling a condition.
    pub enum ExecutionCondition {
        /// Wait until all prior commands finish. (a.k.a. `BOTTOM_OF_PIPE`)
        All,
        /// No dependency, the wait condition is immediately satisfied. (a.k.a.
        /// `TOP_OF_PIPE`)
        None,
        /// Wait until all these stages have finished executing.
        After(PipelineStages),
    }
    impl ExecutionCondition {
        /// Convert into a stage flags.
        pub fn into_stages(self) -> vk::PipelineStageFlags {
            match self {
                ExecutionCondition::All => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                ExecutionCondition::None => vk::PipelineStageFlags::TOP_OF_PIPE,
                ExecutionCondition::After(s) => s.into_flags(),
            }
        }
    }
    /// Description of what parts of the execution pipe should block while
    /// waiting for a condition.
    pub enum ExecutionBlock {
        /// Block all future commands. (a.k.a. `TOP_OF_PIPE`)
        All,
        /// No dependency, do not block anything (a.k.a `BOTTOM_OF_PIPE`)
        None,
        /// Don't run these stages until the condition is met.
        Before(PipelineStages),
    }
    impl ExecutionBlock {
        /// Convert into a stage flags.
        pub fn into_stages(self) -> vk::PipelineStageFlags {
            match self {
                ExecutionBlock::All => vk::PipelineStageFlags::TOP_OF_PIPE,
                ExecutionBlock::None => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                ExecutionBlock::Before(s) => s.into_flags(),
            }
        }
    }
    /// A synchronization condition to wait on, and which caches need to be
    /// flushed after completion.
    #[derive(Clone, Copy)]
    pub enum MemoryCondition {
        /// An execution dependency on all prior commands. (a.k.a
        /// `BOTTOM_OF_PIPE` or `ALL_COMMANDS`, depending on whether the access
        /// is empty or not)
        All(WriteAccess),
        /// No dependency, the wait condition is immediately satisfied and no
        /// caches are flushed. (a.k.a `TOP_OF_PIPE`)
        None,
        /// An execution and memory dependency.
        ///
        /// All `stage`s must complete before the mask is satisfied, and
        /// additionally all memory types specified in `access` from all
        /// applicable `stage`s are flushed to memory.
        ///
        /// Only writes are allowed here, since flushing caches for read-only
        /// memory is a nonsensical operation.
        StageAccess(StageAccess<WriteAccess>),
    }
    impl MemoryCondition {
        /// Convert into a stage flags and access flags pair.
        pub fn into_stage_access(self) -> (vk::PipelineStageFlags, vk::AccessFlags) {
            match self {
                // Definitionally equivalent to `BOTTOM_OF_PIPE` when `access ==
                // NONE`.
                Self::All(access) => (vk::PipelineStageFlags::ALL_COMMANDS, access.0),
                Self::None => (
                    // Waiting until the top of pipe implies all commands have
                    // been submitted, which is instantly.
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    // Pseudo-stages do not access memory and thus cannot flush.
                    vk::AccessFlags::empty(),
                ),
                Self::StageAccess(sa) => sa.into_stage_access(),
            }
        }
    }
    impl From<StageAccess<WriteAccess>> for MemoryCondition {
        fn from(value: StageAccess<WriteAccess>) -> Self {
            Self::StageAccess(value)
        }
    }
    /// An execution block on some condition, and which caches need to be
    /// invalidated before continuing.
    #[derive(Clone, Copy)]
    pub enum MemoryBlock {
        /// An execution dependency on all future commands. (a.k.a `TOP_OF_PIPE`
        /// or `ALL_COMMANDS`, depending on whether the access is empty or not.)
        All(ReadWriteAccess),
        /// No dependency, do not block anything and do not invalidate any
        /// caches. (a.k.a `BOTTOM_OF_PIPE`)
        None,
        /// An execution and memory dependency.
        ///
        /// All `stage`s must wait before until the event is satisfied, and
        /// additionally all memory types specified in `access` from all
        /// applicable `stage`s are invalidated and fetched from memory.
        StageAccess(StageAccess<ReadWriteAccess>),
    }
    impl From<StageAccess<ReadWriteAccess>> for MemoryBlock {
        fn from(value: StageAccess<ReadWriteAccess>) -> Self {
            Self::StageAccess(value)
        }
    }
    impl MemoryBlock {
        /// Convert into a stage flags and access flags pair.
        pub fn into_stage_access(self) -> (vk::PipelineStageFlags, vk::AccessFlags) {
            match self {
                // Definitionally equivalent to `TOP_OF_PIPE` when `access ==
                // NONE`.
                Self::All(access) => (vk::PipelineStageFlags::ALL_COMMANDS, access.0),
                Self::None => (
                    // Blocking the bottom of pipe implies commands can do
                    // literally all of their work before blocking.
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    // Pseudo-stages do not access memory and thus cannot
                    // invalidate.
                    vk::AccessFlags::empty(),
                ),
                Self::StageAccess(sa) => sa.into_stage_access(),
            }
        }
    }

    /// A combination of one or more PipelineStages, not including the
    /// `TOP_OF_PIPE`, `BOTTOM_OF_PIPE` and `ALL_COMMANDS` pseudo-stages. For
    /// those stages, see the `All` and `None` variants of [`ExecutionBlock`]
    /// and [`ExecutionCondition`].
    #[derive(Copy, Clone)]
    pub struct PipelineStages(core::num::NonZero<u32>);
    impl PipelineStages {
        pub const DRAW_INDIRECT: Self =
            Self(core::num::NonZero::new(vk::PipelineStageFlags::DRAW_INDIRECT.as_raw()).unwrap());
        pub const VERTEX_INPUT: Self =
            Self(core::num::NonZero::new(vk::PipelineStageFlags::VERTEX_INPUT.as_raw()).unwrap());
        pub const VERTEX_SHADER: Self =
            Self(core::num::NonZero::new(vk::PipelineStageFlags::VERTEX_SHADER.as_raw()).unwrap());
        pub const TESS_CONTROL_SHADER: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER.as_raw())
                .unwrap(),
        );
        pub const TESS_EVALUATION_SHADER: Self = Self(
            core::num::NonZero::new(
                vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER.as_raw(),
            )
            .unwrap(),
        );
        pub const GEOMETRY_SHADER: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::GEOMETRY_SHADER.as_raw()).unwrap(),
        );
        pub const FRAGMENT_SHADER: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()).unwrap(),
        );
        pub const EARLY_FRAGMENT_TESTS: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()).unwrap(),
        );
        pub const LATE_FRAGMENT_TESTS: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw()).unwrap(),
        );
        pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw())
                .unwrap(),
        );
        pub const COMPUTE_SHADER: Self =
            Self(core::num::NonZero::new(vk::PipelineStageFlags::COMPUTE_SHADER.as_raw()).unwrap());
        pub const TRANSFER: Self =
            Self(core::num::NonZero::new(vk::PipelineStageFlags::TRANSFER.as_raw()).unwrap());
        // This is the final remaining psuedostage, that hasn't been abstracted
        // away, and i do not know the implications of including it here.
        pub const HOST: Self =
            Self(core::num::NonZero::new(vk::PipelineStageFlags::HOST.as_raw()).unwrap());
        /// All parts of the graphics pipeline, including future extensions.
        pub const ALL_GRAPHICS: Self =
            Self(core::num::NonZero::new(vk::PipelineStageFlags::ALL_GRAPHICS.as_raw()).unwrap());
        /// `EXT_mesh_shader`
        pub const TASK_SHADER: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::TASK_SHADER_EXT.as_raw()).unwrap(),
        );
        /// `EXT_mesh_shader`
        pub const MESH_SHADER: Self = Self(
            core::num::NonZero::new(vk::PipelineStageFlags::MESH_SHADER_EXT.as_raw()).unwrap(),
        );
    }
    impl PipelineStages {
        pub fn into_flags(self) -> vk::PipelineStageFlags {
            vk::PipelineStageFlags::from_raw(self.0.get())
        }
    }
    /// Implements BitOr on a copy type which is a newtype over an implementor
    /// of BitOr.
    macro_rules! newtype_bitor {
        ($ty:ty) => {
            // T | T
            impl ::core::ops::BitOr<$ty> for $ty {
                type Output = $ty;
                fn bitor(self, other: $ty) -> Self::Output {
                    Self::Output {
                        0: self.0 | other.0,
                    }
                }
            }
            // &T | T
            impl ::core::ops::BitOr<&$ty> for $ty {
                type Output = $ty;
                fn bitor(self, other: &$ty) -> Self::Output {
                    Self::Output {
                        0: self.0 | other.0,
                    }
                }
            }
            // T | &T
            impl ::core::ops::BitOr<$ty> for &$ty {
                type Output = $ty;
                fn bitor(self, other: $ty) -> Self::Output {
                    Self::Output {
                        0: self.0 | other.0,
                    }
                }
            }
            // &T | &T
            impl ::core::ops::BitOr<&$ty> for &$ty {
                type Output = $ty;
                fn bitor(self, other: &$ty) -> Self::Output {
                    Self::Output {
                        0: self.0 | other.0,
                    }
                }
            }
        };
    }
    newtype_bitor!(PipelineStages);

    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct ReadWriteAccess(vk::AccessFlags);
    impl ReadWriteAccess {
        pub const INDIRECT_COMMAND_READ: Self = Self(vk::AccessFlags::INDIRECT_COMMAND_READ);
        pub const INDEX_READ: Self = Self(vk::AccessFlags::INDEX_READ);
        pub const VERTEX_ATTRIBUTE_READ: Self = Self(vk::AccessFlags::VERTEX_ATTRIBUTE_READ);
        pub const UNIFORM_READ: Self = Self(vk::AccessFlags::UNIFORM_READ);
        pub const INPUT_ATTACHMENT_READ: Self = Self(vk::AccessFlags::INPUT_ATTACHMENT_READ);
        pub const SHADER_READ: Self = Self(vk::AccessFlags::SHADER_READ);
        pub const COLOR_ATTACHMENT_READ: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_READ);
        pub const DEPTH_STENCIL_ATTACHMENT_READ: Self =
            Self(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ);
        pub const TRANSFER_READ: Self = Self(vk::AccessFlags::TRANSFER_READ);
        pub const HOST_READ: Self = Self(vk::AccessFlags::HOST_READ);
        /// Equivalent to *every* read applicable to *every* stage this access
        /// is used with. Always valid to use for any stage, even those that
        /// perform no reads.
        pub const MEMORY_READ: Self = Self(vk::AccessFlags::MEMORY_READ);

        pub const SHADER_WRITE: Self = Self(WriteAccess::SHADER_WRITE.0);
        pub const COLOR_ATTACHMENT_WRITE: Self = Self(WriteAccess::COLOR_ATTACHMENT_WRITE.0);
        pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self =
            Self(WriteAccess::DEPTH_STENCIL_ATTACHMENT_WRITE.0);
        pub const TRANSFER_WRITE: Self = Self(WriteAccess::TRANSFER_WRITE.0);
        pub const HOST_WRITE: Self = Self(WriteAccess::HOST_WRITE.0);
        /// Equivalent to *every* write applicable to *every* stage this access
        /// is used with. Always valid to use for any stage, even those that
        /// perform no writes.
        pub const MEMORY_WRITE: Self = Self(WriteAccess::MEMORY_WRITE.0);

        // Common combinations:
        /// `SHADER_READ | SHADER_WRITE`
        pub const SHADER_READ_WRITE: Self = Self(vk::AccessFlags::from_raw(
            Self::SHADER_READ.0.as_raw() | Self::SHADER_WRITE.0.as_raw(),
        ));
    }
    impl From<WriteAccess> for ReadWriteAccess {
        fn from(value: WriteAccess) -> Self {
            // Every write access is a readwrite access, of course!
            Self(value.0)
        }
    }
    newtype_bitor!(ReadWriteAccess);
    impl AccessFlags for ReadWriteAccess {
        const NONE: Self = Self(vk::AccessFlags::empty());
        fn into_flags(self) -> vk::AccessFlags {
            self.0
        }
    }

    /// Access flags that do not contain any reads
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct WriteAccess(vk::AccessFlags);
    impl WriteAccess {
        pub const SHADER_WRITE: Self = Self(vk::AccessFlags::SHADER_WRITE);
        pub const COLOR_ATTACHMENT_WRITE: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self =
            Self(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
        pub const TRANSFER_WRITE: Self = Self(vk::AccessFlags::TRANSFER_WRITE);
        pub const HOST_WRITE: Self = Self(vk::AccessFlags::HOST_WRITE);
        /// Equivalent to *every* write applicable to *every* stage this access
        /// is used with. Always valid to use for any stage.
        pub const MEMORY_WRITE: Self = Self(vk::AccessFlags::MEMORY_WRITE);
    }
    newtype_bitor!(WriteAccess);
    impl AccessFlags for WriteAccess {
        const NONE: Self = Self(vk::AccessFlags::empty());
        fn into_flags(self) -> vk::AccessFlags {
            self.0
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
/// The unique set of family indices sharing a resource. Always contains 2 or
/// more indices.
pub struct SharingFamilies<'a>(&'a [u32]);
impl<'a> SharingFamilies<'a> {
    /// Wrap a set of families.
    /// # Safety
    /// * The slice must be length 2 or more.
    /// * Every element of the slice must be unique.
    pub const unsafe fn from_slice_unchecked(families: &'a [u32]) -> Self {
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
    /// * Length is < 2.
    /// * If not every element is unique.
    pub const fn from_array<const N: usize>(families: &'a [u32; N]) -> Self {
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
    /// Get a slice of the queue families. Always two or more in length, and all
    /// values are unique.
    pub fn families(self) -> &'a [u32] {
        self.0
    }
}
#[derive(Clone, Copy)]
#[repr(i32)]
/// Whether a resource can be shared between queues or is exclusive to one
/// queue.
pub enum SharingMode<'a> {
    /// The resource is owned by one queue, and changes of queue are mediated by
    /// memory barriers.
    Exclusive = vk::SharingMode::EXCLUSIVE.as_raw(),
    /// The resources is shared between several queues, which may concurrently
    /// access it (synchronization allowing).
    ///
    /// The slice *must* contain 2 or more values.
    Concurrent(SharingFamilies<'a>) = vk::SharingMode::CONCURRENT.as_raw(),
}
impl<'a> SharingMode<'a> {
    /*
    /// Create a sharing mode from a slice of families. If the number of
    /// families is zero or one, [`Self::Exclusive`]. Otherwise,
    /// [`Self::Concurrent`]
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
/// Typestate for a [`Fence`] or [`Semaphore`] which is currently unsignaled and
/// not in the process of becoming signaled.
pub struct Unsignaled;
impl FenceState for Unsignaled {}
impl KnownFenceState for Unsignaled {
    const CREATION_FLAGS: vk::FenceCreateFlags = vk::FenceCreateFlags::empty();
}
/// Typestate for a [`Fence`] or [`Semaphore`] which is eventually going to
/// become signaled due to a previous queue submission operation.
pub struct Pending;
impl FenceState for Pending {}
/// A synchronization primitive for GPU->CPU communication.
/// # Typestate
/// * `State`: Whether the fence is signaled, unsignaled, or in the process of
///   becoming signaled.
#[repr(transparent)]
#[must_use = "dropping the handle will not destroy the fence and may leak resources"]
pub struct Fence<State: FenceState>(vk::Fence, PhantomData<State>);

unsafe impl<State: FenceState> ThinHandle for Fence<State> {
    type Handle = vk::Fence;
}
crate::thin_handle! {
    /// A synchronization primitive for CPU->GPU communication and fine-grained
    /// *intra*-queue GPU->GPU dependencies.
    #[must_use = "dropping the handle will not destroy the event and may leak resources"]
    pub struct Event(vk::Event);
}

crate::typestate_enum! {
    /// Typestate indicating whether a semaphore is pending a signal.
    pub enum trait SemaphoreState {
        /// Typestate for a semaphore with a strictly increasing value instead
        /// of a boolean payload.
        pub struct Timeline,
    }
}
/// Typestate for a semaphore which the device is currently waiting on.
pub struct Waiting;
impl SemaphoreState for Pending {}
impl SemaphoreState for Waiting {}
impl SemaphoreState for Unsignaled {}

crate::thin_handle! {
    /// A synchronization primitive for GPU->GPU communication and
    /// coarse-grained inter-queue dependencies.
    /// # Typestate
    /// * [`State`](SemaphoreState): Describes whether a signal operation on
    ///   this sempahore is pending, and thus whether it can be waited on. It
    ///   always follows the flow `Unsignaled -> Pending -> Waiting ->
    ///   Unsignaled`.
    ///   * The [`Signaled`] state that [fences](Fence) have is missing, as the
    ///     act of observing a semaphore in the Signaled state immediately sets
    ///     it to the unsignaled state, thus transitioning directly from
    ///     `Waiting -> Unsignaled`.
    ///   * The [`Waiting`] typestate cannot be transitioned away from
    ///     automatically by the library. In order for a waiting sempahore to
    ///     once again become an Unsignaled semaphore, a fence representing the
    ///     operation on the semaphore must be detected by the host to have been
    ///     signaled, which cannot be proven on a typestate level. See
    ///     [`Semaphore::assume_waited`].
    #[must_use = "dropping the handle will not destroy the semaphore and may leak resources"]
    pub struct Semaphore<State: SemaphoreState>(vk::Semaphore);
}
impl Semaphore<Pending> {
    /// Construct a wait operation. When passed into a queue submission
    /// operation, the device will wait for this semaphore to be signaled before
    /// any of the stages set in `wait_stage` can begin executing.
    pub fn into_wait(self, wait_stage: barrier::PipelineStages) -> WaitSemaphore {
        WaitSemaphore {
            semaphore: self,
            wait_stage,
        }
    }
}
impl Semaphore<Waiting> {
    /// Assume that the host has observed a completed GPU-side wait operation.
    /// For example, once the fence of
    /// [`Device::submit_with_fence`](super::Device::submit_with_fence) has been
    /// observed to be [`Signaled`], all [wait semaphores](WaitSemaphore) on
    /// that operation can soundly be assumed to be successfully waited.
    ///
    /// This transitions directly to [`Unsignaled`] instead of having a
    /// [`Signaled`] state, as the device observing a semaphore in the Signaled
    /// state instantly transitions it to the [`Unsignaled`] state.
    ///
    /// A common pattern is thus:
    /// ```no_run
    /// # use fzvk::*;
    /// # let device: Device = todo!();
    /// # let mut queue: Queue = todo!();
    /// let command_buffer: CommandBuffer<Primary> = todo!();
    /// let fence: Fence<Unsignaled> = todo!();
    /// // The result of some previous device-side operation that will signal
    /// // this semaphore on completion.
    /// let signaling_semaphore : Semaphore<Pending> = todo!();
    ///
    /// # unsafe {
    /// // Submit some new work, that waits on that semaphore.
    /// let submission = device.submit_with_fence(
    ///         &mut queue,
    ///         [
    ///             signaling_semaphore
    ///                 .into_wait(vk::PipelineStageFlags::TOP_OF_PIPE)
    ///         ],
    ///         &[command_buffer.reference()],
    ///         [],
    ///         fence
    ///     )
    ///     .unwrap();
    /// // Destructure the outputs of the operation
    /// let SubmitWithFence {
    ///     waited_semaphores: [waiting_semaphore],
    ///     pending_fence: fence,
    ///     signaled_semaphores: []
    /// } = submission;
    /// // Wait for the work to complete on the device.
    /// let fence = device.wait_fence(fence).unwrap();
    /// // Since we observed the fence to now be signaled, we know that the
    /// // `wait_semaphores` have each been successfully waited on by the
    /// // device!
    /// let unsignaled_semaphore = waiting_semaphore.assume_waited();
    /// # }
    /// ```
    /// # Safety
    /// * The semaphore must have been indirectly observed to have been
    ///   successfully waited upon through external means.
    pub unsafe fn assume_waited(self) -> Semaphore<Unsignaled> {
        self.with_state()
    }
    /// Convinience wrapper around [`Self::assume_waited`] for many semaphores.
    pub unsafe fn assume_many_waited<const N: usize>(
        many: [Self; N],
    ) -> [Semaphore<Unsignaled>; N] {
        many.map(|waited| waited.with_state())
    }
}
#[must_use = "the semaphore contained will be leaked"]
pub struct WaitSemaphore {
    pub semaphore: Semaphore<Pending>,
    /// These stages must wait before beginning execution.
    pub wait_stage: barrier::PipelineStages,
}
