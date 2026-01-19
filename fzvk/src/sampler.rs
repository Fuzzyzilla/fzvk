use crate::vk;
crate::thin_handle! {
    #[must_use = "dropping the handle will not destroy the view and may leak resources"]
    pub struct Sampler(vk::Sampler);
}
/// The maximum number of anisotropic samples for a [`Sampler`] to take.
// repr as "NonZero<f32>" Why? Because we work with this as Option<Self> and im
// pedantic and evil :3
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Anisotropy(core::num::NonZero<u32>);
impl Anisotropy {
    pub const ONE: Self = Self::new(1.0).unwrap();
    /// Create an anisotropy value. Returns None if the value is less than one.
    pub const fn new(value: f32) -> Option<Self> {
        if value >= 1.0 {
            Some(unsafe { Self::new_unchecked(value) })
        } else {
            None
        }
    }
    /// # Safety:
    /// `value` must be >= 1.0.
    pub const unsafe fn new_unchecked(value: f32) -> Self {
        debug_assert!(value >= 1.0);
        // Safety: Unsafe contract states value is greater than one, and the
        // only way the bits could be zero is if `value == +0.0`.
        Self(core::num::NonZero::new_unchecked(value.to_bits()))
    }
    pub const fn get(self) -> f32 {
        f32::from_bits(self.0.get())
    }
}
crate::vk_enum!(pub enum Filter: vk::Filter {
    Nearest = NEAREST,
    Linear = LINEAR,
});
crate::vk_enum!(pub enum AddressMode: vk::SamplerAddressMode {
    Repeat = REPEAT,
    MirroredRepeat = MIRRORED_REPEAT,
    ClampToEdge = CLAMP_TO_EDGE,
    ClampToBorder = CLAMP_TO_BORDER,
});
#[derive(Clone, Copy)]
pub struct AddressMode3 {
    pub u: AddressMode,
    pub v: AddressMode,
    pub w: AddressMode,
}
impl AddressMode3 {
    pub const REPEAT: Self = Self::splat(AddressMode::Repeat);
    pub const MIRRORED_REPEAT: Self = Self::splat(AddressMode::MirroredRepeat);
    pub const CLAMP_TO_EDGE: Self = Self::splat(AddressMode::ClampToEdge);
    pub const CLAMP_TO_BORDER: Self = Self::splat(AddressMode::ClampToBorder);
    /// Apply the same value to all three axes.
    pub const fn splat(mode: AddressMode) -> Self {
        Self {
            u: mode,
            v: mode,
            w: mode,
        }
    }
}
crate::vk_enum!(pub enum CompareOp: vk::CompareOp {
    Never = NEVER,
    Less = LESS,
    Equal = EQUAL,
    LessOrEqual = LESS_OR_EQUAL,
    Greater = GREATER,
    NotEqual = NOT_EQUAL,
    GreaterOrEqual = GREATER_OR_EQUAL,
    Always = ALWAYS,
});
crate::vk_enum!(pub enum BorderColor: vk::BorderColor {
    FloatTransparentBlack = FLOAT_TRANSPARENT_BLACK,
    IntTransparentBlack = INT_TRANSPARENT_BLACK,
    FloatOpqueBlack = FLOAT_OPAQUE_BLACK,
    IntOpqueBlack = INT_OPAQUE_BLACK,
    FloatOpaqueWhite = FLOAT_OPAQUE_WHITE,
    IntOpaqueWhite = INT_OPAQUE_WHITE,
});
