fzvk_shader::glsl! {
    r#"#version 460 core
    #pragma shader_stage(fragment)

    layout(set = 0, binding = 0, r8ui) uniform restrict writeonly uimage2D someStorage[3];
    // Ooopsie binding 1 is missing! (the spec allows this to be interpreted as
    // an array of arbitrary type but zero elements)
    layout(set = 0, binding = 2) uniform isampler3D textureWithASamplerInsideOfIt_Wow;
    layout(set = 0, binding = 3) uniform texture2DMS aSampledTextureWithoutASampler_Ohmygod_youCanDoThat;
    layout(set = 0, binding = 4) uniform sampler thatsALotOfSamplers[65535];

    layout(set = 2, binding = 0) uniform textureCubeArray thisIsADifferentDescriptorSet;
    layout(set = 2, binding = 1, input_attachment_index = 1) uniform subpassInput omgSubpasses;

    layout(set = 3, binding = 0) uniform UBOsToo {
        float data[];
    };
    layout(set = 3, binding = 1) buffer SSBOExtravaganza {
        float data2[];
    } anArrayOfThemEven[2];

    void main() {}
    "#
}
fn main() {}
