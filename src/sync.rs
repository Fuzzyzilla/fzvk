//! Types for describing execution and/or memory dependencies.
use super::vk;

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
    pub fn shader<Stage: super::ShaderStage>() -> Self {
        Self::from_stages(Stage::PIPE_STAGE)
    }
    pub unsafe fn shader_access<Stage: super::ShaderStage>(access: Access) -> Self {
        Self::from_stage_access(Stage::PIPE_STAGE, access)
    }
    /// Stages and access flags.
    /// # Safety
    /// Each access flag must be a type used by at least one of the members of
    /// `stages`.
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
/// Description of what parts of the execution pipe should block while waiting
/// for a condition.
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
/// A synchronization condition to wait on, and which caches need to be flushed
/// after completion.
#[derive(Clone, Copy)]
pub enum MemoryCondition {
    /// An execution dependency on all prior commands. (a.k.a `BOTTOM_OF_PIPE`
    /// or `ALL_COMMANDS`, depending on whether the access is empty or not)
    All(WriteAccess),
    /// No dependency, the wait condition is immediately satisfied and no caches
    /// are flushed. (a.k.a `TOP_OF_PIPE`)
    None,
    /// An execution and memory dependency.
    ///
    /// All `stage`s must complete before the mask is satisfied, and
    /// additionally all memory types specified in `access` from all applicable
    /// `stage`s are flushed to memory.
    ///
    /// Only writes are allowed here, since flushing caches for read-only memory
    /// is a nonsensical operation.
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
                // Waiting until the top of pipe implies all commands have been
                // submitted, which is instantly.
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
    /// An execution dependency on all future commands. (a.k.a `TOP_OF_PIPE` or
    /// `ALL_COMMANDS`, depending on whether the access is empty or not.)
    All(ReadWriteAccess),
    /// No dependency, do not block anything and do not invalidate any caches.
    /// (a.k.a `BOTTOM_OF_PIPE`)
    None,
    /// An execution and memory dependency.
    ///
    /// All `stage`s must wait before until the event is satisfied, and
    /// additionally all memory types specified in `access` from all applicable
    /// `stage`s are invalidated and fetched from memory.
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
            // Definitionally equivalent to `TOP_OF_PIPE` when `access == NONE`.
            Self::All(access) => (vk::PipelineStageFlags::ALL_COMMANDS, access.0),
            Self::None => (
                // Blocking the bottom of pipe implies commands can do literally
                // all of their work before blocking.
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
/// `TOP_OF_PIPE`, `BOTTOM_OF_PIPE` and `ALL_COMMANDS` pseudo-stages. For those
/// stages, see the `All` and `None` variants of [`ExecutionBlock`] and
/// [`ExecutionCondition`].
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
        core::num::NonZero::new(vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER.as_raw())
            .unwrap(),
    );
    pub const GEOMETRY_SHADER: Self =
        Self(core::num::NonZero::new(vk::PipelineStageFlags::GEOMETRY_SHADER.as_raw()).unwrap());
    pub const FRAGMENT_SHADER: Self =
        Self(core::num::NonZero::new(vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()).unwrap());
    pub const EARLY_FRAGMENT_TESTS: Self = Self(
        core::num::NonZero::new(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()).unwrap(),
    );
    pub const LATE_FRAGMENT_TESTS: Self = Self(
        core::num::NonZero::new(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw()).unwrap(),
    );
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(
        core::num::NonZero::new(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw()).unwrap(),
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
}
impl PipelineStages {
    pub fn into_flags(self) -> vk::PipelineStageFlags {
        vk::PipelineStageFlags::from_raw(self.0.get())
    }
}
/// Implements BitOr on a copy type which is a newtype over an implementor of
/// BitOr.
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
    /// Equivalent to *every* read applicable to *every* stage this access is
    /// used with. Always valid to use for any stage, even those that perform no
    /// reads.
    pub const MEMORY_READ: Self = Self(vk::AccessFlags::MEMORY_READ);

    pub const SHADER_WRITE: Self = Self(WriteAccess::SHADER_WRITE.0);
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(WriteAccess::COLOR_ATTACHMENT_WRITE.0);
    pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self =
        Self(WriteAccess::DEPTH_STENCIL_ATTACHMENT_WRITE.0);
    pub const TRANSFER_WRITE: Self = Self(WriteAccess::TRANSFER_WRITE.0);
    pub const HOST_WRITE: Self = Self(WriteAccess::HOST_WRITE.0);
    /// Equivalent to *every* write applicable to *every* stage this access is
    /// used with. Always valid to use for any stage, even those that perform no
    /// writes.
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
    /// Equivalent to *every* write applicable to *every* stage this access is
    /// used with. Always valid to use for any stage.
    pub const MEMORY_WRITE: Self = Self(vk::AccessFlags::MEMORY_WRITE);
}
newtype_bitor!(WriteAccess);
impl AccessFlags for WriteAccess {
    const NONE: Self = Self(vk::AccessFlags::empty());
    fn into_flags(self) -> vk::AccessFlags {
        self.0
    }
}
/*
impl StageAccess {
    /// Create an empty access for the given stage.
    pub const fn from_stage(stage: vk::PipelineStageFlags) -> Self {
        Self {
            stage,
            // "No access" is valid for all stages.
            access: vk::AccessFlags::empty(),
        }
    }
    /// Create an access flags with no associated stage. This is *always*
    /// invalid to use as-is, and must be combined with some other
    /// `StageAccesses` in such a way that it becomes valid.
    /// # Safety
    /// Must not be used to create an object with access inapplicable to all of
    /// it's stages.
    ///
    /// FIXME: This is yucky, as it *always* breaks Self's invariants, so it's
    /// private.
    const unsafe fn from_access(access: vk::AccessFlags) -> Self {
        Self {
            access,
            stage: vk::PipelineStageFlags::empty(),
        }
    }
    /// Overwrite the access flags of an existing object.
    /// # Safety
    /// Must not be used to create an object with access inapplicable to all of
    /// it's stages.
    pub const unsafe fn with_access(self, access: vk::AccessFlags) -> Self {
        Self { access, ..self }
    }
    /// Take the Union of two access specifications. This may result in an
    /// overly pessimistic barrier, as the access flags from each get applied to
    /// both sets of stages.
    pub const fn union(self, other: Self) -> Self {
        // :o this is just (self as u64) | (other as u64)
        Self {
            stage: vk::PipelineStageFlags::from_raw(self.stage.as_raw() | other.stage.as_raw()),
            access: vk::AccessFlags::from_raw(self.access.as_raw() | other.access.as_raw()),
        }
    }
}
impl core::ops::BitOr for StageAccess {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}
impl core::ops::BitOr<StageAccess> for &StageAccess {
    type Output = StageAccess;
    fn bitor(self, rhs: StageAccess) -> Self::Output {
        self.union(rhs)
    }
}
impl core::ops::BitOr<&StageAccess> for StageAccess {
    type Output = StageAccess;
    fn bitor(self, rhs: &StageAccess) -> Self::Output {
        self.union(*rhs)
    }
}
impl<'a> core::ops::BitOr<&'a StageAccess> for &'a StageAccess {
    type Output = StageAccess;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(*rhs)
    }
}
// Anonymous "module" to not pollute the namespace, lol.
const _: () = {
    use vk::AccessFlags as Access;
    use vk::PipelineStageFlags as Stage;
    impl StageAccess {
        // Raw access flags. THESE ARE INVALID TO USE AS IS. These all break
        // Self's invariants, and are thus not public. Fixme?
        const INDIRECT_COMMAND_READ: Self =
            unsafe { Self::from_access(Access::INDIRECT_COMMAND_READ) };
        const INDEX_READ: Self = unsafe { Self::from_access(Access::INDEX_READ) };
        const VERTEX_ATTRIBUTE_READ: Self =
            unsafe { Self::from_access(Access::VERTEX_ATTRIBUTE_READ) };
        const UNIFORM_READ: Self = unsafe { Self::from_access(Access::UNIFORM_READ) };
        const INPUT_ATTACHMENT_READ: Self =
            unsafe { Self::from_access(Access::INPUT_ATTACHMENT_READ) };
        const SHADER_READ: Self = unsafe { Self::from_access(Access::SHADER_READ) };
        const SHADER_WRITE: Self = unsafe { Self::from_access(Access::SHADER_WRITE) };
        const COLOR_ATTACHMENT_READ: Self =
            unsafe { Self::from_access(Access::COLOR_ATTACHMENT_READ) };
        const COLOR_ATTACHMENT_WRITE: Self =
            unsafe { Self::from_access(Access::COLOR_ATTACHMENT_WRITE) };
        const DEPTH_STENCIL_ATTACHMENT_READ: Self =
            unsafe { Self::from_access(Access::DEPTH_STENCIL_ATTACHMENT_READ) };
        const DEPTH_STENCIL_ATTACHMENT_WRITE: Self =
            unsafe { Self::from_access(Access::DEPTH_STENCIL_ATTACHMENT_WRITE) };
        const TRANSFER_WRITE: Self = unsafe { Self::from_access(Access::TRANSFER_WRITE) };
        const HOST_WRITE: Self = unsafe { Self::from_access(Access::HOST_WRITE) };
        const MEMORY_WRITE: Self = unsafe { Self::from_access(Access::MEMORY_WRITE) };

        pub const TOP_OF_PIPE: Self = Self::from_stage(Stage::TOP_OF_PIPE);

        pub const DRAW_INDIRECT: Self = Self::from_stage(Stage::DRAW_INDIRECT);
        pub const DRAW_INDIRECT_INDIRECT_READ: Self =
            Self::DRAW_INDIRECT.union(Self::INDIRECT_COMMAND_READ);

        pub const VERTEX_INPUT: Self = Self::from_stage(Stage::VERTEX_INPUT);
        pub const VERTEX_INPUT_INDEX_READ: Self = Self::VERTEX_INPUT.union(Self::INDEX_READ);
        pub const VERTEX_INPUT_ATTRIBUTE_READ: Self =
            Self::VERTEX_INPUT.union(Self::VERTEX_ATTRIBUTE_READ);

        pub const VERTEX_SHADER: Self = Self::from_stage(Stage::VERTEX_SHADER);
        pub const VERTEX_SHADER_UNIFORM_READ: Self = Self::VERTEX_SHADER.union(Self::UNIFORM_READ);
        pub const VERTEX_SHADER_READ: Self = Self::VERTEX_SHADER.union(Self::SHADER_READ);
        pub const VERTEX_SHADER_WRITE: Self = Self::VERTEX_SHADER.union(Self::SHADER_WRITE);

        pub const TESS_CONTROL_SHADER: Self = Self::from_stage(Stage::TESSELLATION_CONTROL_SHADER);
        pub const TESS_CONTROL_SHADER_UNIFORM_READ: Self =
            Self::TESS_CONTROL_SHADER.union(Self::UNIFORM_READ);
        pub const TESS_CONTROL_SHADER_READ: Self =
            Self::TESS_CONTROL_SHADER.union(Self::SHADER_READ);
        pub const TESS_CONTROL_SHADER_WRITE: Self =
            Self::TESS_CONTROL_SHADER.union(Self::SHADER_WRITE);

        pub const TESS_EVALUATION_SHADER: Self =
            Self::from_stage(Stage::TESSELLATION_EVALUATION_SHADER);
        pub const TESS_EVALUATION_SHADER_UNIFORM_READ: Self =
            Self::TESS_EVALUATION_SHADER.union(Self::UNIFORM_READ);
        pub const TESS_EVALUATION_SHADER_READ: Self =
            Self::TESS_EVALUATION_SHADER.union(Self::SHADER_READ);
        pub const TESS_EVALUATION_SHADER_WRITE: Self =
            Self::TESS_EVALUATION_SHADER.union(Self::SHADER_WRITE);

        pub const GEOMETRY_SHADER: Self = Self::from_stage(Stage::GEOMETRY_SHADER);
        pub const FRAGMENT_SHADER: Self = Self::from_stage(Stage::FRAGMENT_SHADER);
        pub const EARLY_FRAGMENT_TESTS: Self = Self::from_stage(Stage::EARLY_FRAGMENT_TESTS);
        pub const LATE_FRAGMENT_TESTS: Self = Self::from_stage(Stage::LATE_FRAGMENT_TESTS);
        pub const COLOR_ATTACHMENT_OUTPUT: Self = Self::from_stage(Stage::COLOR_ATTACHMENT_OUTPUT);
        pub const COMPUTE_SHADER: Self = Self::from_stage(Stage::COMPUTE_SHADER);
        pub const TRANSFER: Self = Self::from_stage(Stage::TRANSFER);
        pub const BOTTOM_OF_PIPE: Self = Self::from_stage(Stage::BOTTOM_OF_PIPE);
        pub const HOST: Self = Self::from_stage(Stage::HOST);
        pub const ALL_GRAPHICS: Self = Self::from_stage(Stage::ALL_GRAPHICS);
        pub const ALL_COMMANDS: Self = Self::from_stage(Stage::ALL_COMMANDS);
    }
};*/
