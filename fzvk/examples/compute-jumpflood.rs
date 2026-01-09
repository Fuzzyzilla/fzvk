//! An example demonstrating compute pipelines by executing Rong Guodong's Jump
//! Flood Algorithm for creating voronoi diagrams.
//!
//! Usage: `cargo run --example compute-jumpflood <in-path> <out-path>`
//!
//! The input and output formats are any image supported by the
//! [`Image`](::image) crate. The output consists of RG16 pairs, describing the
//! texel coordinates of the nearest bright pixel in the input image (**note**:
//! the values in the output will be small, and might appear black until the
//! levels are boosted in an external image editor).
use std::num::NonZero;

mod import {
    fzvk_shader::glsl! {
        r#"
        #version 460 core
        #extension GL_EXT_shader_explicit_arithmetic_types_int32: require
        #pragma shader_stage(compute)
        
        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        
        layout(set = 0, binding = 0, r8ui) uniform restrict readonly uimage2D reference;
        layout(set = 1, binding = 0, rg16ui) uniform restrict writeonly uimage2D target;

        const u32vec2 NOWHERE = u32vec2(65535u, 65535u);
        const uint32_t OPAQUE_THRESHOLD = uint32_t(128u);

        void main() {
            u32vec2 id = gl_GlobalInvocationID.xy;
            // Early exit if out-of-bounds.
            if (any(greaterThanEqual(id, imageSize(reference)))) { return; }

            // Read the reference image.
            uint32_t reference_value = uint32_t(imageLoad(reference, ivec2(id)).x);

            // Compare and store. If opaque, the texel should refer to itself as
            // the nearest opaque pixel. If not opaque, report the nearest as
            // "unknown".
            u32vec2 result = (reference_value > OPAQUE_THRESHOLD)
                ? id
                : NOWHERE;
            imageStore(target, ivec2(id), u32vec4(result, 0, 0));
        }
        "#
    }
}
mod pingpong {
    fzvk_shader::glsl! {
        r"#version 460 core
        #extension GL_EXT_shader_explicit_arithmetic_types_int32: require
        #pragma shader_stage(compute)

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        layout(set = 0, binding = 0, rg16ui) uniform restrict readonly uimage2D reference;
        layout(set = 1, binding = 0, rg16ui) uniform restrict writeonly uimage2D target;

        layout(push_constant) uniform Push {
            layout(offset = 0) uint32_t jump;
        };

        const u32vec2 NOWHERE = u32vec2(65535u, 65535u);
        const uint32_t UINT32_MAX = ~0;

        // For some reason integer dot product requires a device feature?!?
        uint32_t dotU32(in u32vec2 a, in u32vec2 b) {
            return a.x * b.x + a.y * b.y;
        }

        uint32_t distanceSq(in u32vec2 a, in u32vec2 b) {
            u32vec2 min = min(a, b);
            u32vec2 max = max(a, b);
            // Widening to always fit the max result.
            u32vec2 delta = u32vec2(max.x - min.x, max.y - min.y);
            return dotU32(delta, delta);
        }

        void main() {
            i32vec2 id = i32vec2(gl_GlobalInvocationID.xy);
            i32vec2 imageSize = i32vec2(imageSize(reference));
            // Early exit if out-of-bounds.
            if (any(greaterThanEqual(id, imageSize))) { return; }

            // Current closest point and it's distance.
            u32vec2 thisCell = u32vec2(imageLoad(reference, id).rg);
            uint32_t thisDistanceSq = (thisCell == NOWHERE)
                ? UINT32_MAX
                : distanceSq(thisCell, u32vec2(id));

            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    i32vec2 pos = id + i32vec2(x, y) * int32_t(jump);
                    // Skip cells outside of the image.
                    if (any(lessThan(pos, i32vec2(0)))) continue;
                    if (any(greaterThanEqual(pos, imageSize))) continue;

                    // See what this other cell thinks is closest. If its closer
                    // to us than our current one, adopt it!
                    u32vec2 otherCell = u32vec2(imageLoad(reference, pos).rg);
                    uint32_t otherDistanceSq = (otherCell == NOWHERE)
                        ? UINT32_MAX
                        : distanceSq(otherCell, u32vec2(id));
                    if (otherDistanceSq < thisDistanceSq) {
                        thisCell = otherCell;
                        thisDistanceSq = otherDistanceSq;
                    }
                }
            }

            // Store the new closest. (regardless of if we found any! So if not,
            // it acts as a copy.)
            imageStore(target, id, u32vec4(thisCell, 0, 0));
        }
        "
    }
}

trait RequiredMemoryFlags: memory::MemoryAccess {
    const REQUIRED_FLAGS: vk::MemoryPropertyFlags;
}
impl RequiredMemoryFlags for memory::HostVisible {
    const REQUIRED_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE;
}
impl RequiredMemoryFlags for memory::DeviceOnly {
    const REQUIRED_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::empty();
}

// A terrible, no-good allocation strategy.
unsafe fn allocate_naive<Access: RequiredMemoryFlags>(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    device: &ash::Device,
    requirements: vk::MemoryRequirements,
) -> Result<memory::Memory<Access>> {
    // Choose the first type that fits requirements, regardless of heap.
    let mut bits = requirements.memory_type_bits;
    let index = memory_properties
        .memory_types_as_slice()
        .iter()
        .enumerate()
        .filter(|_| {
            let pass = bits & 1 != 0;
            bits >>= 1;
            pass
        })
        .find(|(_, memory_properties)| {
            const ACCEPTED: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
                vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()
                    | vk::MemoryPropertyFlags::HOST_CACHED.as_raw()
                    | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw()
                    | vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw(),
            );
            // If any unrecognized flags, must not allocate.
            if memory_properties.property_flags.as_raw() & !ACCEPTED.as_raw() != 0 {
                return false;
            }
            // If missing any required flags, bail.
            if !memory_properties
                .property_flags
                .contains(Access::REQUIRED_FLAGS)
            {
                return false;
            }
            true
        })
        .ok_or_else(|| anyhow::anyhow!("Could not find suitable memory."))?
        .0;
    unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(index as u32),
            None,
        )
    }
    .map_err(Into::into)
    .map(|mem| unsafe { memory::Memory::from_handle(mem) }.unwrap())
}

use anyhow::Result;
use ash::vk;
use fzvk::{format::Format, *};
pub fn main() -> Result<()> {
    let [in_path, out_path] = {
        let mut args = std::env::args().skip(1);
        args.next()
            .and_then(|in_path| Some([in_path, args.next()?]))
    }
    .ok_or_else(|| anyhow::anyhow!("usage: <in-path> <out-path.rg16u>"))?;
    let image = ::image::open(in_path)?.to_luma8();
    if image.width() > u32::from(u16::MAX) || image.height() > u32::from(u16::MAX) {
        anyhow::bail!("image dimensions too large.")
    }
    // =========================================================================================
    // Create an instance.
    let ash_entry = ash::Entry::linked();
    let ash_instance = unsafe {
        let has_portability = ash_entry
            .enumerate_instance_extension_properties(None)?
            .iter()
            .any(|ext| {
                ext.extension_name_as_c_str() == Ok(ash::khr::portability_enumeration::NAME)
            });
        let instance_extensions = if has_portability {
            &[ash::khr::portability_enumeration::NAME.as_ptr()][..]
        } else {
            &[]
        };
        let flags = if has_portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            Default::default()
        };

        ash_entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .enabled_extension_names(instance_extensions)
                .flags(flags),
            None,
        )
    }?;
    // =========================================================================================
    // Find a device. We need:
    // * a compute queue
    // * storage support for R8Unorm, RG8Uint.
    let (phys_device, ash_device, queue, queue_family_index) = unsafe {
        let phys_devices = ash_instance.enumerate_physical_devices()?;

        let (phys_device, family_index, is_portability) = phys_devices
            .iter()
            // Check it has all the features we need:
            .filter(|&&phys| {
                let r8unorm = ash_instance
                    .get_physical_device_format_properties(phys, format::R8_UINT::FORMAT)
                    .optimal_tiling_features
                    // TransferSrc/Dst is implicit. Weird.
                    .intersects(vk::FormatFeatureFlags::STORAGE_IMAGE);
                if !r8unorm {
                    return false;
                }
                ash_instance
                    .get_physical_device_format_properties(phys, format::R16G16_UINT::FORMAT)
                    .optimal_tiling_features
                    // TransferSrc/Dst is implicit. Weird.
                    .intersects(vk::FormatFeatureFlags::STORAGE_IMAGE)
            })
            // Find the queue type we need, or bail:
            .filter_map(|&phys| {
                let queues = ash_instance.get_physical_device_queue_family_properties(phys);
                let queue = queues
                    .iter()
                    .position(|queue| queue.queue_flags.intersects(vk::QueueFlags::COMPUTE))?
                    as u32;
                Some((phys, queue))
            })
            // Check if we need to enable portability_subset:
            .filter_map(|(phys, family_index)| {
                let exts = ash_instance
                    .enumerate_device_extension_properties(phys)
                    .ok()?;
                let is_portability = exts.iter().any(|ext| {
                    ext.extension_name_as_c_str() == Ok(ash::khr::portability_subset::NAME)
                });
                Some((phys, family_index, is_portability))
            })
            // Take the first acceptable device, or bail.
            .next()
            .ok_or_else(|| anyhow::anyhow!("no suitable device"))?;

        // If we enabled portabilty_enumeration, we *must* enable
        // portability_subset on any device that advertises it.
        let extensions = if is_portability {
            &[ash::khr::portability_subset::NAME.as_ptr()][..]
        } else {
            &[]
        };

        let device = ash_instance.create_device(
            phys_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family_index)
                    .queue_priorities(&[1.0])])
                .enabled_extension_names(extensions),
            None,
        )?;

        let queue = device.get_device_queue(family_index, 0);

        (phys_device, device, queue, family_index)
    };

    let device = Device::from_ash(&ash_device);
    let mut queue = unsafe { Queue::from_handle(queue).unwrap() };
    // =========================================================================================
    // Import image, and take only its Alpha channel.
    let extent = Extent2D {
        width: image.width().try_into()?,
        height: image.height().try_into()?,
    };
    let output_size_bytes = NonZero::new(
        u64::from(extent.width.get())
            * u64::from(extent.height.get())
            * u64::from(format::R16G16_UINT::TEXEL_SIZE),
    )
    .unwrap();

    unsafe {
        // =========================================================================================
        // Create buffer + images objects and appropriate backing memory. We
        // will need: One host buffer, one R8Uint format image to reference, two
        // RG16Uint format images to "pingpong" between. Our operation will look
        // like:

        // Host buffer --xfer--> color

        // color --storage--> UV1

        // UV1 --storage--> UV2 --storage--> UV1 --> [...]

        // UV1 or UV2 --xfer--> host buffer.

        let buffer = device.create_buffer(
            // Source for original image, dst for output image
            (TransferSrc, TransferDst),
            output_size_bytes,
            SharingMode::Exclusive,
        )?;
        let reference_image = device.create_image(
            // Source for unpacking the buffer, storage for shader access.
            (TransferDst, Storage),
            extent,
            MipCount::ONE,
            format::R8_UINT,
            SingleSampled,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )?;
        let pingpong_array = device.create_image(
            // Source for exporting the final image, storage for shader access.
            (TransferSrc, Storage),
            // Two layers to "pingpong" between. For each of the iterations, we
            // will swap each between source and destination.
            extent.with_layers(CreateArrayCount::TWO),
            MipCount::ONE,
            // Enough space to store a non-normalized UV coordinate.
            format::R16G16_UINT,
            SingleSampled,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )?;

        // =========================================================================================
        // Allocate memory for the buffers and images.
        let memory_properties = ash_instance.get_physical_device_memory_properties(phys_device);
        // See which memory types we're compatible with.
        let mut buffer_memory = allocate_naive::<memory::HostVisible>(
            &memory_properties,
            &ash_device,
            device.memory_requirements(&buffer),
        )?;
        let mut reference_memory = allocate_naive::<memory::DeviceOnly>(
            &memory_properties,
            &ash_device,
            device.memory_requirements(&reference_image),
        )?;
        let mut pingpong_memory = allocate_naive::<memory::DeviceOnly>(
            &memory_properties,
            &ash_device,
            device.memory_requirements(&pingpong_array),
        )?;

        let buffer = device.bind_memory(buffer, &mut buffer_memory, 0)?;
        let reference_image = device.bind_memory(reference_image, &mut reference_memory, 0)?;
        let pingpong_array = device.bind_memory(pingpong_array, &mut pingpong_memory, 0)?;

        // Map and upload the image to vulkan memory.
        let (mut buffer_memory, buffer_mapping_ptr) = device
            .map_memory(buffer_memory, ..)
            .map_err(|(_, err)| err)?;

        // Convert the raw pointer into a slice, and copy the image into it.

        // WARNING: It is pivotal that this reference is short lived. It is
        // unsound to hold onto it during the time period where the device may
        // be accessing the memory (e.g. after the queue submission, before the
        // fence wait and invalidation operations).
        std::slice::from_raw_parts_mut(buffer_mapping_ptr, image.len()).copy_from_slice(&image);
        // Make the written values available to the host domain.
        device.flush_memory(&mut buffer_memory, ..(image.len() as u64))?;

        let reference_image_view = device.create_image_view(
            &reference_image,
            // FIXME: Sux
            D2,
            ComponentMapping::IDENTITY,
            SubresourceMips::ALL,
        )?;
        let pingpong_views = [
            device.create_image_view(
                &pingpong_array,
                D2,
                ComponentMapping::IDENTITY,
                SubresourceRange::range(.., 0..1),
            )?,
            device.create_image_view(
                &pingpong_array,
                D2,
                ComponentMapping::IDENTITY,
                SubresourceRange::range(.., 1..2),
            )?,
        ];

        let ash_descriptor_set_layout = ash_device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&[vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)])
                    .flags(vk::DescriptorSetLayoutCreateFlags::empty()),
                None,
            )
            .unwrap();
        let pipeline_layout = {
            let ash_pipeline_layout = ash_device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .size(4)
                            .offset(0)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)])
                        // Uses two sets of the same layout
                        .set_layouts(&[ash_descriptor_set_layout, ash_descriptor_set_layout])
                        .flags(vk::PipelineLayoutCreateFlags::empty()),
                    None,
                )
                .unwrap();
            fzvk::PipelineLayout::from_handle(ash_pipeline_layout).unwrap()
        };

        let ash_descriptor_pool = ash_device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::empty())
                    .max_sets(3)
                    .pool_sizes(&[vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(3)]),
                None,
            )
            .unwrap();

        let (ash_reference_descriptor_set, ash_pingpong_descriptor_sets) = {
            let ash_descriptor_sets = ash_device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(ash_descriptor_pool)
                        .set_layouts(&[
                            ash_descriptor_set_layout,
                            ash_descriptor_set_layout,
                            ash_descriptor_set_layout,
                        ]),
                )
                .unwrap();

            (
                ash_descriptor_sets[0],
                *ash_descriptor_sets[1..].as_array::<2>().unwrap(),
            )
        };
        ash_device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(ash_reference_descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(reference_image_view.handle())]),
                vk::WriteDescriptorSet::default()
                    .dst_set(ash_pingpong_descriptor_sets[0])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(pingpong_views[0].handle())]),
                vk::WriteDescriptorSet::default()
                    .dst_set(ash_pingpong_descriptor_sets[1])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(pingpong_views[1].handle())]),
            ],
            &[],
        );

        // =========================================================================================
        // Create the compute pipeline that will do the work Shader that takes
        // our reference image, and performs the initial population of our
        // pingpong image.
        let import_module = device.create_module::<import::Module>()?;
        let import_shader = import_module.entry(import::MAIN).specialize(&());
        // Shader that iteratively refines between the two pingpong images.
        let pingpong_module = device.create_module::<pingpong::Module>()?;
        let pingpong_shader = pingpong_module.entry(pingpong::MAIN).specialize(&());

        let [import_pipe, pingpong_pipe] = device.create_compute_pipelines(
            None,
            [
                ComputePipelineCreateInfo {
                    layout: &pipeline_layout.handle(),
                    shader: import_shader,
                },
                ComputePipelineCreateInfo {
                    layout: &pipeline_layout.handle(),
                    shader: pingpong_shader,
                },
            ],
        )?;

        // It's fine to destroy a shader while it's in use by a pipeline
        device.destroy_module(import_module);
        device.destroy_module(pingpong_module);

        let mut command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
        )?;

        // =========================================================================================
        // Record a command buffer describing the whole operation
        let [mut command_buffer] = device.allocate_command_buffers(&mut command_pool, Primary)?;
        let mut recording = device.begin_command_buffer(&mut command_pool, &mut command_buffer)?;

        // Reference our images in Undefined layout, since they haven't been
        // initialized yet.
        let reference_image_ref = reference_image.discard_layout();
        let pingpong_image_ref = pingpong_array.discard_layout();

        // Immediately transition our reference image to transfer dst layout,
        // before the transfer occurs. FIXME: allow multiple transitions per
        // barrier.
        let reference_image_ref = device
            // Transition the images from Undefined -> The formats we need.
            .barrier(
                &mut recording,
                barrier::Write::NOTHING,
                // Block transfer from occuring until the image is in the right
                // layout to accept it.
                barrier::Transfer::WRITE,
                [],
                reference_image_ref.reinitialize_as(layout::TransferDst),
            );
        // Immediately transition our pingpong image to general layout, before
        // the shader writes to it.
        let pingpong_image_ref = device.barrier(
            &mut recording,
            barrier::Write::NOTHING,
            barrier::ComputeShader::WRITE,
            [buffer.barrier()],
            pingpong_image_ref.reinitialize_as(layout::General),
        );
        // Import the image from the host buffer. The queueSubmit of this
        // commandbuffer creates an implicit domain transfer, making the host
        // writes available and visible to the device.
        device.copy_buffer_to_image(
            &mut recording,
            &buffer,
            &reference_image_ref,
            // Whole image
            [BufferImageCopy {
                buffer_offset: 0,
                image_offset: Offset::ORIGIN,
                image_extent: extent,
                pitch: BufferPitch::PACKED,
                layers: SubresourceMip::BASE_LEVEL,
            }],
        );
        let _reference_image_ref = device
            // Wait until import transfer is complete...
            .barrier(
                &mut recording,
                barrier::Transfer::WRITE,
                barrier::ComputeShader::READ | barrier::ComputeShader::WRITE,
                [],
                reference_image_ref.transition(layout::General),
            );
        device
            .bind_pipeline(&mut recording, &import_pipe)
            .push_constants(
                &mut recording,
                &pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &[0u32.into()],
            );
        ash_device.cmd_bind_descriptor_sets(
            recording.handle(),
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout.handle(),
            0,
            &[
                ash_reference_descriptor_set,
                ash_pingpong_descriptor_sets[0],
            ],
            &[],
        );
        device
            // Run a shader to initialize our first pingpong buffer with the
            // reference image
            .dispatch(
                &mut recording,
                [extent.width, extent.height, NonZero::<u32>::MIN],
            )
            .bind_pipeline(&mut recording, &pingpong_pipe);

        let mut readbuf = 0;

        // Generate [1, X/2, X/4, X/8, ..., 1]
        let mut spread = extent.width.max(extent.height).get() / 2;
        let spread_iter = std::iter::once(1).chain(std::iter::from_fn(|| {
            if spread == 0 {
                None
            } else {
                let current = spread;
                spread /= 2;
                Some(current)
            }
        }));
        // Iterate the algorithm. This for-loop is where the bulk of the work
        // takes place, funny that it's the shortest part!
        for spread in spread_iter {
            ash_device.cmd_bind_descriptor_sets(
                recording.handle(),
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout.handle(),
                0,
                &[
                    ash_pingpong_descriptor_sets[readbuf],
                    ash_pingpong_descriptor_sets[if readbuf == 0 { 1 } else { 0 }],
                ],
                &[],
            );
            device.push_constants(
                &mut recording,
                &pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &[spread.into()],
            );

            device
                // Wait until previous shader is done before executing the
                // next...
                .barrier(
                    &mut recording,
                    barrier::ComputeShader::WRITE,
                    barrier::ComputeShader::READ | barrier::ComputeShader::WRITE,
                    [],
                    (),
                );
            device.dispatch(
                &mut recording,
                [extent.width, extent.height, NonZero::<u32>::MIN],
            );
            // Swap the read image and write image for the next iteration.
            readbuf = if readbuf == 0 { 1 } else { 0 };
        }
        // Export the final image back to the host buffer
        let pingpong_image_ref = device.barrier(
            &mut recording,
            barrier::ComputeShader::WRITE,
            barrier::Transfer::READ,
            [],
            // FIXME: we only actually need to transition
            // pingpong_array[readbuf].
            pingpong_image_ref.transition(layout::TransferSrc),
        );
        // It's safe to write to the buffer after the read at the start, due to
        // the transfer -> compute shader -> transfer dependency chain.
        device
            .copy_image_to_buffer(
                &mut recording,
                &pingpong_image_ref,
                &buffer,
                [BufferImageCopy {
                    buffer_offset: 0,
                    image_offset: Offset::ORIGIN,
                    image_extent: extent,
                    pitch: BufferPitch::PACKED,
                    layers: SubresourceLayers::new(0, readbuf as u32, NonZero::<u32>::MIN),
                }],
            )
            .barrier(
                &mut recording,
                barrier::Transfer::WRITE,
                // Make the results available to the host domain.
                barrier::Host::READ,
                [buffer.barrier()],
                (),
            );

        device.end_command_buffer(recording)?;

        // =========================================================================================
        // Execute!
        let fence = device.create_fence::<Unsignaled>()?.into_inner();
        let SubmitWithFence {
            pending_fence,
            signaled_semaphores: [],
            waited_semaphores: [],
        } = device.submit_with_fence(&mut queue, [], &[command_buffer.reference()], [], fence)?;
        let fence = device.wait_fence(pending_fence)?;

        // =========================================================================================
        // We can now read the buffer's memory to get our output result! There
        // does still need to be a host cache invalidation to make the available
        // writes *visible*!
        device.invalidate_memory(&mut buffer_memory, ..)?;

        // Export UV image as RGB16U. We have to make a copy to shove the zeroed
        // B channel in, because no image format supports RG format and LA has
        // quirks! Silly!
        let image_data = {
            let mapped_slice = std::slice::from_raw_parts(
                buffer_mapping_ptr.cast_const(),
                output_size_bytes.get() as usize,
            );

            bytemuck::cast_slice::<u8, [u16; 2]>(mapped_slice)
                .iter()
                .flat_map(|&[r, g]| [r, g, 0u16])
                .collect::<Vec<u16>>()
        };
        if let Err(err) = ::image::save_buffer(
            out_path,
            bytemuck::cast_slice(&image_data),
            extent.width.get(),
            extent.height.get(),
            ::image::ColorType::Rgb16,
        ) {
            eprintln!("Failed to export image: {err}");
        };

        // =========================================================================================
        // Cleanup
        ash_device.destroy_pipeline_layout(pipeline_layout.into_handle(), None);
        ash_device.destroy_descriptor_pool(ash_descriptor_pool, None);
        ash_device.destroy_descriptor_set_layout(ash_descriptor_set_layout, None);

        let [pingpong_view_0, pingpong_view_1] = pingpong_views;
        device
            .destroy_image_view(pingpong_view_0)
            .destroy_image_view(pingpong_view_1)
            .destroy_fence(fence)
            .destroy_command_pool(command_pool)
            .destroy_pipeline(pingpong_pipe)
            .destroy_pipeline(import_pipe)
            .destroy_image_view(reference_image_view)
            .destroy_image(pingpong_array)
            .destroy_image(reference_image)
            .destroy_buffer(buffer)
            .free_memory(buffer_memory)
            .free_memory(reference_memory)
            .free_memory(pingpong_memory);

        ash_device.destroy_device(None);
        ash_instance.destroy_instance(None);
    }

    Ok(())
}
