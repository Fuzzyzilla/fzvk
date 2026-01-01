#version 460 core
#pragma shader_stage(fragment)

#ifdef should_be_undefined
    #error should_be_undefined
#endif
#ifndef should_be_defined
    #error should_be_defined
#endif
#if should_be_true != true
    #error should_be_true
#endif
#if should_be_five != 5
    #error should_be_five
#endif

layout(set = 0, binding = 0, r8ui) uniform restrict writeonly uimage2D someStorage[3];
layout(set = 0, binding = 1) uniform sampler2DShadow wdwdw[3];
// Ooopssie binding 1 is missing! (the spec allows this to be interpreted as
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
