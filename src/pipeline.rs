//! Pipelines, Shaders, Renderpasses, Framebuffers galore!
use super::{Bool32, Pred, Satisified, ThinHandle, sync, usage::Vertex, vk};
use core::marker::PhantomData;

crate::thin_handle! {
    /// A block of shader code that has been forwarded to the implementation.
    /// Includes one or more "entry points" of possibly heterogeneous shader
    /// stages.
    ///
    /// Use [`Self::entry`] to reference a specific entry point for use in a
    /// pipeline.
    #[must_use = "dropping the handle will not deallocate the module and may leak resources"]
    pub struct ShaderModule(vk::ShaderModule);
}

/// The stage of a pipeline a shader is for.
pub trait ShaderStage {
    const FLAG: vk::ShaderStageFlags;
    const PIPE_STAGE: sync::barrier::PipelineStages;
}
impl ShaderStage for Vertex {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::VERTEX;
    const PIPE_STAGE: sync::barrier::PipelineStages = sync::barrier::PipelineStages::VERTEX_SHADER;
}
pub struct Geometry;
impl ShaderStage for Geometry {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::GEOMETRY;
    const PIPE_STAGE: sync::barrier::PipelineStages =
        sync::barrier::PipelineStages::GEOMETRY_SHADER;
}
pub struct TessControl;
impl ShaderStage for TessControl {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::TESSELLATION_CONTROL;
    const PIPE_STAGE: sync::barrier::PipelineStages =
        sync::barrier::PipelineStages::TESS_CONTROL_SHADER;
}
pub struct TessEvaluation;
impl ShaderStage for TessEvaluation {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::TESSELLATION_EVALUATION;
    const PIPE_STAGE: sync::barrier::PipelineStages =
        sync::barrier::PipelineStages::TESS_EVALUATION_SHADER;
}

pub struct TaskEXT;
impl ShaderStage for TaskEXT {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::TASK_EXT;
    const PIPE_STAGE: sync::barrier::PipelineStages = sync::barrier::PipelineStages::TASK_SHADER;
}
pub struct MeshEXT;
impl ShaderStage for MeshEXT {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::MESH_EXT;
    const PIPE_STAGE: sync::barrier::PipelineStages = sync::barrier::PipelineStages::MESH_SHADER;
}
pub struct Fragment;
impl ShaderStage for Fragment {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::FRAGMENT;
    const PIPE_STAGE: sync::barrier::PipelineStages =
        sync::barrier::PipelineStages::FRAGMENT_SHADER;
}
pub struct Compute;
impl ShaderStage for Compute {
    const FLAG: vk::ShaderStageFlags = vk::ShaderStageFlags::COMPUTE;
    const PIPE_STAGE: sync::barrier::PipelineStages = sync::barrier::PipelineStages::COMPUTE_SHADER;
}

impl ShaderModule {
    /// Reference a specific entry point within this module.
    /// # Safety
    /// * The module must contain an entry point called `name`
    /// * The entry point must be of a type consistent with `Stage`
    /// * The device must be capable of using a shader of type `Stage`
    pub unsafe fn entry<'a, Stage: ShaderStage>(
        &'a self,
        name: &'a core::ffi::CStr,
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

/// A reference to a shader module, bundled with an entry point name and the
/// stage of said entry point.
///
/// Created by [`ShaderModule::entry`].
/// # Typestates
/// * `Stage`: The type of the shader, e.g. [`Vertex`], [`Compute`]
#[derive(Copy, Clone)]
pub struct Entry<'a, Stage: ShaderStage> {
    module: vk::ShaderModule,
    entry: &'a core::ffi::CStr,
    _marker: PhantomData<(&'a ShaderModule, Stage)>,
}
impl<'a, Stage: ShaderStage> Entry<'a, Stage> {
    /// Provide constants to the shader compiler.
    ///
    /// If your shader does not use specialization, you can specialize on the
    /// unit type:
    /// ```no_run
    /// # use fzvk::*;
    /// # let module : ShaderModule = todo!();
    /// # unsafe {
    /// let specialized_entry = module.main(Compute).specialize(&());
    /// # }
    /// ```
    /// # Safety
    /// * Every constant necessary for a complete module entry point must be
    ///   populated by the [entries](Specialization::ENTRIES) of `S`.
    /// * Every types required by the shader must match the types provided by
    ///   `S`
    pub unsafe fn specialize<S: Specialization>(
        self,
        specialization: &'a S,
    ) -> SpecializedEntry<'a, Stage> {
        SpecializedEntry {
            module: self.module,
            entry: self.entry,
            specialization_info: vk::SpecializationInfo {
                data_size: core::mem::size_of::<S>(),
                p_data: (specialization as *const S).cast(),
                ..vk::SpecializationInfo::default().map_entries(S::ENTRIES)
            },
            _marker: PhantomData,
        }
    }
}

/// A reference to a shader module, bundled with an entry point name, the stage
/// of said entry point, and the specialization constants needed to make it
/// concrete.
///
/// Created by [`Entry::specialize`].
/// # Typestates
/// * `Stage`: The type of the shader, e.g. [`Vertex`], [`Compute`]
pub struct SpecializedEntry<'a, Stage: ShaderStage> {
    module: vk::ShaderModule,
    entry: &'a core::ffi::CStr,
    // &dyn but funny
    specialization_info: vk::SpecializationInfo<'a>,
    // specialization_map: &'a [vk::SpecializationMapEntry],
    // specialization_data: *const u8, specialization_size: usize,
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
///   * This req sucks uwu. Basically the layout of this structure and the
///     shader *must* agree.
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
    /*/// (Task?) -> Mesh
    Mesh(
        Option<SpecializedEntry<'a, TaskEXT>>,
        SpecializedEntry<'a, MeshEXT>,
    ),*/
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

crate::thin_handle! {
    #[must_use = "dropping the handle will not destroy the pipeline and may leak resources"]
    pub struct Pipeline<Kind: BindPoint>(vk::Pipeline);
}

crate::thin_handle! {
    #[must_use = "dropping the handle will not destroy the pipeline cache and may leak resources"]
    pub struct PipelineCache(vk::PipelineCache);
}
#[repr(transparent)]
pub struct BorrowedPipelineCache<'a>(vk::PipelineCache, PhantomData<&'a PipelineCache>);
impl<'a> From<&'a PipelineCache> for BorrowedPipelineCache<'a> {
    fn from(value: &'a PipelineCache) -> Self {
        unsafe { Self::from_handle(value.handle()) }
    }
}
/// A type usable for pipeline push constants, consisting of several static
/// ranges pointing to members of `self`.
///
/// # Safety
/// * For all ranges, `offsets + size <= size_of::<Self>()`.
pub unsafe trait PushConstant {
    const RANGES: &'static [vk::PushConstantRange];
}
unsafe impl PushConstant for () {
    const RANGES: &'static [vk::PushConstantRange] = &[];
}
unsafe impl ThinHandle for BorrowedPipelineCache<'_> {
    type Handle = vk::PipelineCache;
}
crate::thin_handle! {
    /// # Typestates
    /// - `Constants`: The set of push constant ranges needed for pipelines of
    ///   this layout.
    pub struct PipelineLayout<Constants: PushConstant>(vk::PipelineLayout);
}

pub struct ComputePipelineCreateInfo<'a> {
    pub shader: SpecializedEntry<'a, Compute>,
    // Is this OK even with VK1.3 or KHR_maintainence4? "must not be accessed"
    // during any call this is passed to. Mutably? Immutably? Wargh???? Surely
    // not mutably as then multiple pipes can't be made with a shared layout
    // similtaneously and that feels silly.
    pub layout: &'a vk::PipelineLayout,
}

/// A render pass, describing the flow of rendering operations on a Framebuffer.
///
/// # Typestates
/// * `SUBPASSES` - A non-zero integer representing the number of subpasses this
///   renderpass consists of.
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
pub struct SubpassCount<const N: usize>;
/// Implemented for all non-zero subpass counts.
pub trait ValidSubpassCount {}
impl<const N: usize> ValidSubpassCount for SubpassCount<N> where Pred<{ N > 0 }>: Satisified {}
/// Implemented for all valid subpass counts where `{Count - 1}` is also a valid
/// subpass count.
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

/// A value which can be used as a push constant.
///
/// Avoid reading back from this type, as [`Bool32`] has strict requirements.
#[derive(Clone, Copy)]
#[repr(C)]
pub union PushValue {
    pub u32: u32,
    pub i32: i32,
    pub f32: f32,
    pub bool: Bool32,
}
impl PushValue {
    pub const fn bytes_of(array: &[Self]) -> &[u8] {
        let len_bytes = core::mem::size_of_val(array);
        // Align is strictly greater, of course.
        unsafe { core::slice::from_raw_parts(array.as_ptr().cast(), len_bytes) }
    }
}
