//! Fences, Semaphores, Barriers galore!
use super::{ThinHandle, vk};

pub mod barrier {
    use super::vk;
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct NonZeroStageFlags(core::num::NonZero<u32>);
    macro_rules! nonzero_stages {
        {$($name:ident,)+} => {
            impl NonZeroStageFlags {
                $(pub const $name : Self = Self(::core::num::NonZero::new(::ash::vk::PipelineStageFlags::$name.as_raw()).unwrap());)+
            }
        };
    }
    nonzero_stages! {
        TOP_OF_PIPE,
        DRAW_INDIRECT,
        VERTEX_INPUT,
        VERTEX_SHADER,
        TESSELLATION_CONTROL_SHADER,
        TESSELLATION_EVALUATION_SHADER,
        GEOMETRY_SHADER,
        FRAGMENT_SHADER,
        EARLY_FRAGMENT_TESTS,
        LATE_FRAGMENT_TESTS,
        COLOR_ATTACHMENT_OUTPUT,
        COMPUTE_SHADER,
        TRANSFER,
        BOTTOM_OF_PIPE,
        HOST,
        ALL_GRAPHICS,
        ALL_COMMANDS,
    }
    impl NonZeroStageFlags {
        pub fn get(self) -> vk::PipelineStageFlags {
            vk::PipelineStageFlags::from_raw(self.0.get())
        }
    }
    impl core::ops::BitOr for NonZeroStageFlags {
        type Output = Self;
        fn bitor(self, rhs: Self) -> Self::Output {
            Self(self.0 | rhs.0)
        }
    }
    /// Structure defining a set of pipeline stages and some or all of the
    /// writes they perform, used as the start of a barrier operation.
    ///
    /// This type holds a subset of the accesses available to [`ReadWrite`], and
    /// can be trivially converted using [`Into::into()`].
    ///
    /// Write barriers can be combined using the `|` operator.
    #[derive(Copy, Clone)]
    pub struct Write {
        stage: NonZeroStageFlags,
        access: vk::AccessFlags,
    }
    impl Write {
        /// Don't wait for any executions to complete and don't flush any access
        /// caches.
        #[doc(alias = "TOP_OF_PIPE")]
        pub const NOTHING: Self = Self {
            stage: NonZeroStageFlags::TOP_OF_PIPE,
            access: vk::AccessFlags::empty(),
        };
        pub fn stages(self) -> vk::PipelineStageFlags {
            self.stage.get()
        }
        pub fn accesses(self) -> vk::AccessFlags {
            self.access
        }
        pub fn into_stage_access(self) -> (vk::PipelineStageFlags, vk::AccessFlags) {
            (self.stages(), self.accesses())
        }
    }
    impl core::ops::BitOr for Write {
        type Output = Write;
        fn bitor(self, rhs: Write) -> Write {
            Write {
                stage: self.stage | rhs.stage,
                access: self.access | rhs.access,
            }
        }
    }
    impl core::ops::BitOr<ReadWrite> for Write {
        type Output = ReadWrite;
        fn bitor(self, rhs: ReadWrite) -> ReadWrite {
            ReadWrite {
                stage: self.stage | rhs.stage,
                access: self.access | rhs.access,
            }
        }
    }
    impl core::ops::BitOr<Write> for ReadWrite {
        type Output = ReadWrite;
        fn bitor(self, rhs: Write) -> ReadWrite {
            ReadWrite {
                stage: self.stage | rhs.stage,
                access: self.access | rhs.access,
            }
        }
    }
    impl core::ops::BitOr for ReadWrite {
        type Output = ReadWrite;
        fn bitor(self, rhs: ReadWrite) -> ReadWrite {
            ReadWrite {
                stage: self.stage | rhs.stage,
                access: self.access | rhs.access,
            }
        }
    }
    /// Structure defining a set of pipeline stages and some or all of the reads
    /// and writes they perform, used as the end of a barrier operation.
    ///
    /// This type holds a superset of the accesses available to [`Write`] and
    /// can be trivially converted using [`From::from()`].
    ///
    /// ReadWrite barriers can be combined using the `|` operator.
    #[derive(Copy, Clone)]
    pub struct ReadWrite {
        stage: NonZeroStageFlags,
        access: vk::AccessFlags,
    }
    /// Since [`ReadWrite`] is a superset of [`Write`], it is trivially sound to
    /// convert.
    impl From<Write> for ReadWrite {
        fn from(value: Write) -> Self {
            Self {
                stage: value.stage,
                access: value.access,
            }
        }
    }
    impl ReadWrite {
        /// Don't block the execution of any stages and don't invalidate any
        /// access caches.
        #[doc(alias = "BOTTOM_OF_PIPE")]
        pub const NOTHING: Self = Self {
            stage: NonZeroStageFlags::BOTTOM_OF_PIPE,
            access: vk::AccessFlags::empty(),
        };
        pub fn stages(self) -> vk::PipelineStageFlags {
            self.stage.get()
        }
        pub fn accesses(self) -> vk::AccessFlags {
            self.access
        }
        pub fn into_stage_access(self) -> (vk::PipelineStageFlags, vk::AccessFlags) {
            (self.stages(), self.accesses())
        }
    }
    pub trait PipelineStage {
        const STAGE: NonZeroStageFlags;
        /// Block on execution and all write operations it may perform.
        const ALL_WRITE: Write;
        /// Block on execution and all read operations it may perform.
        const ALL_READ: ReadWrite;
        /// Block on execution and all read or write operations it may perform.
        const ALL_READ_WRITE: ReadWrite = ReadWrite {
            stage: NonZeroStageFlags(
                core::num::NonZero::new(
                    Self::ALL_WRITE.stage.0.get() | Self::ALL_READ.stage.0.get(),
                )
                .unwrap(),
            ),
            access: vk::AccessFlags::from_raw(
                Self::ALL_WRITE.access.as_raw() | Self::ALL_READ.access.as_raw(),
            ),
        };
        /// Block on execution of this stage, not flushing or invalidating any
        /// access caches.
        const EXECUTE: Write = Write {
            stage: Self::STAGE,
            access: vk::AccessFlags::empty(),
        };
    }
    macro_rules! stage_accesses {
        {$($stage:ident: $nonzero_stage:ident {
            reads: {
                $($(#[$read_meta:meta])*$read_access_name:ident: $read_access_raw:ident,)*
            },
            writes: {
                $($(#[$write_meta:meta])*$write_access_name:ident: $write_access_raw:ident,)*
            },
        },)+} => {
            $(
                pub struct $stage;
                impl $stage {
                    $(
                        $(#[$read_meta])*
                        pub const $read_access_name: ReadWrite = ReadWrite {
                            stage: <$stage as PipelineStage>::STAGE,
                            access: ::ash::vk::AccessFlags::$read_access_raw,
                        };
                    )*
                    $(
                        $(#[$write_meta])*
                        pub const $write_access_name: Write = Write {
                            stage: <$stage as PipelineStage>::STAGE,
                            access: ::ash::vk::AccessFlags::$write_access_raw,
                        };
                    )*
                }
                impl PipelineStage for $stage {
                    const STAGE: NonZeroStageFlags = NonZeroStageFlags::$nonzero_stage;
                    const ALL_WRITE: Write = Write {
                        stage: Self::STAGE,
                        access: ::ash::vk::AccessFlags::from_raw(0 $(| ::ash::vk::AccessFlags::$write_access_raw.as_raw())*),
                    };
                    const ALL_READ: ReadWrite = ReadWrite {
                        stage: Self::STAGE,
                        access: ::ash::vk::AccessFlags::from_raw(0 $(| ::ash::vk::AccessFlags::$read_access_raw.as_raw())*),
                    };
                }
            )+
        };
    }
    stage_accesses! {
        DrawIndirect: DRAW_INDIRECT {
            reads: {COMMAND_READ: INDIRECT_COMMAND_READ,},
            writes: {},
        },
        VertexInput: VERTEX_INPUT {
            reads: {
                ATTRIBUTE_READ: VERTEX_ATTRIBUTE_READ,
                INDEX_READ: INDEX_READ,
            },
            writes: {},
        },
        VertexShader: VERTEX_SHADER {
            reads: {READ: SHADER_READ, UNIFORM_READ: UNIFORM_READ,},
            writes: {WRITE: SHADER_WRITE,},
        },
        TessControlShader: TESSELLATION_CONTROL_SHADER {
            reads: {READ: SHADER_READ, UNIFORM_READ: UNIFORM_READ,},
            writes: {WRITE: SHADER_WRITE,},
        },
        TessEvalShader: TESSELLATION_EVALUATION_SHADER {
            reads: {READ: SHADER_READ, UNIFORM_READ: UNIFORM_READ,},
            writes: {WRITE: SHADER_WRITE,},
        },
        GeometryShader: GEOMETRY_SHADER {
            reads: {READ: SHADER_READ, UNIFORM_READ: UNIFORM_READ,},
            writes: {WRITE: SHADER_WRITE,},
        },
        FragmentShader: FRAGMENT_SHADER {
            reads: {READ: SHADER_READ, UNIFORM_READ: UNIFORM_READ, INPUT_ATTACHMENT_READ: INPUT_ATTACHMENT_READ,},
            writes: {WRITE: SHADER_WRITE,},
        },
        EarlyFragmentTests: EARLY_FRAGMENT_TESTS {
            reads: {DEPTH_STENCIL_READ: DEPTH_STENCIL_ATTACHMENT_READ,},
            writes: {DEPTH_STENCIL_WRITE: DEPTH_STENCIL_ATTACHMENT_WRITE,},
        },
        LateFragmentTests: LATE_FRAGMENT_TESTS {
            reads: {DEPTH_STENCIL_READ: DEPTH_STENCIL_ATTACHMENT_READ,},
            writes: {DEPTH_STENCIL_WRITE: DEPTH_STENCIL_ATTACHMENT_WRITE,},
        },
        ColorOutput: COLOR_ATTACHMENT_OUTPUT {
            reads: {READ: COLOR_ATTACHMENT_READ,},
            writes: {WRITE: COLOR_ATTACHMENT_WRITE,},
        },
        ComputeShader: COMPUTE_SHADER {
            reads: {READ: SHADER_READ, UNIFORM_READ: UNIFORM_READ,},
            writes: {WRITE: SHADER_WRITE,},
        },
        Transfer: TRANSFER {
            reads: {READ: TRANSFER_READ,},
            writes: {WRITE: TRANSFER_WRITE,},
        },
        Host: HOST {
            reads: {
                /// Reads through host memory-mapped regions.
                READ: HOST_READ,
            },
            writes: {
                /// Writes through host memory-mapped regions.
                WRITE: HOST_WRITE,
            },
        },
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
crate::thin_handle! {
    /// A synchronization primitive for GPU->CPU communication.
    /// # Typestate
    /// * `State`: Whether the fence is signaled, unsignaled, or in the process
    ///   of becoming signaled.
    #[must_use = "dropping the handle will not destroy the fence and may leak resources"]
    pub struct Fence<State: FenceState>(vk::Fence);
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
    pub fn into_wait(self, wait_stage: barrier::NonZeroStageFlags) -> WaitSemaphore {
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
    ///                 .into_wait(barrier::NonZeroStageFlags::ALL_COMMANDS)
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
    pub wait_stage: barrier::NonZeroStageFlags,
}
