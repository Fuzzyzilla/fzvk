//! An example demonstrating compute pipelines by executing Rong Guodong's Jump
//! Flood Algorithm for creating voronoi diagrams.
use std::num::NonZero;

/// Don't let the compiler see that we're unconditionally panic'ing, as that
/// turns off a bunch of checks!
fn todo<T>() -> T {
    todo!()
}

/*mod vertex {
    fzvk_shader::glsl! {
        r"#version 460 core
        #pragma shader_stage(compute)
        layout(constant_id = 0) const int i = -4;
        layout(constant_id = 1) const uint j = 1;
        layout(constant_id = 2) const float k = 103.5;
        layout(constant_id = 107) const bool owo = true;

        layout(local_size_x_id = 3, local_size_y = 4, local_size_z_id = 4) in;

        void main() {
        }
        "
    }
}*/
mod import {
    fzvk_shader::glsl! {
        r"#version 460 core
        #pragma shader_stage(compute)
        
        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        void main() {
        }
        "
    }
}
mod pingpong {
    fzvk_shader::glsl! {
        r"#version 460 core
        #pragma shader_stage(compute)
        
        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        void main() {
        }
        "
    }
}

use anyhow::Result;
use ash::vk;
use fzvk::{format::Format, *};
pub fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: <in-path.png>"))?;
    let image =
        png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?)).read_info()?;
    if image.info().width > u32::from(u16::MAX) || image.info().height > u32::from(u16::MAX) {
        anyhow::bail!("image dimensions too large.")
    }
    // =========================================================================================
    // Create an instance.
    let entry = ash::Entry::linked();
    let instance = unsafe {
        let has_portability = entry
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

        entry.create_instance(
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
        let phys_devices = instance.enumerate_physical_devices()?;

        let (phys_device, family_index, is_portability) = phys_devices
            .iter()
            // Check it has all the features we need:
            .filter(|&&phys| {
                let r8unorm = instance
                    .get_physical_device_format_properties(phys, format::R8_UNORM::FORMAT)
                    .optimal_tiling_features
                    // TransferSrc/Dst is implicit. Weird.
                    .intersects(vk::FormatFeatureFlags::STORAGE_IMAGE);
                if !r8unorm {
                    return false;
                }
                instance
                    .get_physical_device_format_properties(phys, format::R16G16_UINT::FORMAT)
                    .optimal_tiling_features
                    // TransferSrc/Dst is implicit. Weird.
                    .intersects(vk::FormatFeatureFlags::STORAGE_IMAGE)
            })
            // Find the queue type we need, or bail:
            .filter_map(|&phys| {
                let queues = instance.get_physical_device_queue_family_properties(phys);
                let queue = queues
                    .iter()
                    .position(|queue| queue.queue_flags.intersects(vk::QueueFlags::COMPUTE))?
                    as u32;
                Some((phys, queue))
            })
            // Check if we need to enable portability_subset:
            .filter_map(|(phys, family_index)| {
                let exts = instance.enumerate_device_extension_properties(phys).ok()?;
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

        let device = instance.create_device(
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
        width: image.info().width.try_into()?,
        height: image.info().height.try_into()?,
    };
    // Number of bytes for the whole image, A8 format is 1byte per texel.
    #[allow(clippy::identity_op)]
    let size_bytes =
        NonZero::new(u64::from(extent.width.get()) * u64::from(extent.height.get()) * 1).unwrap();

    unsafe {
        // =========================================================================================
        // Create buffer + images objects and appropriate backing memory. We
        // will need: One host buffer, one R8Unorm format image to reference,
        // two RG16Uint format images to "pingpong" between. Our operation will
        // look like:

        // Host buffer --xfer--> color

        // color --storage--> UV1

        // UV1 --storage--> UV2 --storage--> UV1 --> [...]

        // UV1 or UV2 --xfer--> host buffer.

        let buffer = device.create_buffer(
            // Source for original image, dst for output image
            (TransferSrc, TransferDst),
            size_bytes,
            SharingMode::Exclusive,
        )?;
        let reference_image = device.create_image(
            // Source for unpacking the buffer, storage for shader access.
            (TransferDst, Storage),
            extent,
            MipCount::ONE,
            format::R8_UNORM,
            SingleSampled,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )?;
        let pingpong_array = device.create_image(
            // Source for exporting the final image, storage for shader access.
            (TransferSrc, Storage),
            // Two layers to "pingpong" between. For each of the iterations, we
            // will swap each between source and destination.
            extent.with_layers(ArrayCount::TWO),
            MipCount::ONE,
            // Enough space to store a non-normalized UV coordinate.
            format::R16G16_UINT,
            SingleSampled,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )?;

        // =========================================================================================
        // Allocate memory for the buffers and images.
        let memory_info = instance.get_physical_device_memory_properties(phys_device);
        // See which memory types we're compatible with.
        let buffer_requirements = ash_device.get_buffer_memory_requirements(buffer.handle());
        let reference_image_requirements =
            ash_device.get_image_memory_requirements(reference_image.handle());
        let pingpong_array_requirements =
            ash_device.get_image_memory_requirements(pingpong_array.handle());
        let bits = buffer_requirements.memory_type_bits
            & reference_image_requirements.memory_type_bits
            & pingpong_array_requirements.memory_type_bits;

        // =========================================================================================
        // Create the compute pipeline that will do the work Shader that takes
        // our reference image, and performs the initial population of our
        // pingpong image.
        let import_module = device.create_module::<import::Module>()?;
        let import_shader = import_module.entry(import::MAIN).specialize(&());
        // Shader that iteratively refines between the two pingpong images.
        let pingpong_module = device.create_module::<pingpong::Module>()?;
        let pingpong_shader = pingpong_module.entry(pingpong::MAIN).specialize(&());
        let empty_layout = device.create_pipeline_layout::<()>()?;

        let [import_pipe, pingpong_pipe] = device.create_compute_pipelines(
            None,
            [
                ComputePipelineCreateInfo {
                    layout: &empty_layout.handle(),
                    shader: import_shader,
                },
                ComputePipelineCreateInfo {
                    layout: &empty_layout.handle(),
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

        // Barrier building blocks that we'll be using a lot! Specifies a wait
        // until all previous compute shaders have finished and flushes their
        // writes to any storage.
        let wait_compute_write =
            barrier::StageAccess::shader_access::<Compute>(barrier::WriteAccess::SHADER_WRITE);
        // Specifies a block on all future compute shaders and invalidates their
        // storage caches for reads and writes.
        let block_compute_read_write = barrier::StageAccess::shader_access::<Compute>(
            barrier::ReadWriteAccess::SHADER_READ_WRITE,
        );

        // Import the image from the host buffer
        device
            // Transition the images from Undefined -> The formats we need.
            .barrier(
                &mut recording,
                // We aren't waiting on any previous operations.
                barrier::MemoryCondition::None,
                // Block transfer from occuring until the image is in the right
                // layout to accept it.
                barrier::StageAccess::from_stage_access(
                    barrier::PipelineStages::TRANSFER,
                    barrier::ReadWriteAccess::TRANSFER_WRITE,
                )
                .into(),
                // TODO: transition images.
                // reference -> transfer dst opt
                // pingpong -> storage opt
            )
            // Copy the image from host memory
            .copy_buffer_to_image(
                &mut recording,
                &buffer,
                &reference_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                // Whole image
                [BufferImageCopy {
                    buffer_offset: 0,
                    image_offset: Offset::ORIGIN,
                    image_extent: extent,
                    pitch: BufferPitch::PACKED,
                    layers: SubresourceMip::ZERO,
                }],
            )
            // Wait until import transfer is complete...
            .barrier(
                &mut recording,
                barrier::StageAccess::from_stage_access(
                    barrier::PipelineStages::TRANSFER,
                    barrier::WriteAccess::TRANSFER_WRITE,
                )
                .into(),
                block_compute_read_write.into(),
                // Todo: Transition
                // reference -> storage opt
            )
            // TODO: bind import pipe TODO: bind descriptors of images
            // Run a shader to initialize our first pingpong buffer with the
            // reference image
            .dispatch(
                &mut recording,
                [extent.width, extent.height, NonZero::<u32>::MIN],
            );
        // TODO: bind pingpong pipe
        let mut readbuf = 0u32;

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
        for _spread in spread_iter {
            device
                // TODO: swap buffers
                // TODO: push constants
                // Wait until previous shader is done before executing the
                // next...
                .barrier(
                    &mut recording,
                    wait_compute_write.into(),
                    block_compute_read_write.into(),
                )
                .dispatch(
                    &mut recording,
                    [extent.width, extent.height, NonZero::<u32>::MIN],
                );
            // Swap the read image and write image for the next iteration.
            readbuf = if readbuf == 0 { 1 } else { 0 };
        }
        // Export the final image back to the host buffer
        device
            .barrier(
                &mut recording,
                wait_compute_write.into(),
                barrier::StageAccess::from_stage_access(
                    barrier::PipelineStages::TRANSFER,
                    barrier::ReadWriteAccess::TRANSFER_READ,
                )
                .into(),
                // TODO: Transition `pingpong[readbuf]` to TRANSFER_SRC
            )
            // It's safe to write to the buffer after the read at the start, due
            // to the transfer -> compute shader -> transfer dependency chain.
            .copy_image_to_buffer(
                &mut recording,
                &pingpong_array,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                &buffer,
                [BufferImageCopy {
                    buffer_offset: 0,
                    image_offset: Offset::ORIGIN,
                    image_extent: todo(), // wrongly expects a Layers >= 2 .
                    pitch: BufferPitch::PACKED,
                    layers: SubresourceMipArray::new(0, 0..1),
                }],
            )
            .end_command_buffer(recording)?;

        // =========================================================================================
        // Execute!
        let fence = device.create_fence::<Unsignaled>()?.into_inner();
        let SubmitWithFence { pending_fence, .. } =
            device.submit_with_fence(&mut queue, [], &[command_buffer.reference()], [], fence)?;
        let fence = device.wait_fence(pending_fence)?;
        // We can now read the buffer's memory to get our output result! A fence
        // signal operation creates an everything-to-everything memory
        // dependency, so there is no need for a final barrier for transfer
        // write -> host read.

        // =========================================================================================
        // Cleanup
        device
            .destroy_fence(fence)
            .destroy_command_pool(command_pool)
            .destroy_pipeline(pingpong_pipe)
            .destroy_pipeline(import_pipe)
            .destroy_image(pingpong_array)
            .destroy_image(reference_image)
            .destroy_buffer(buffer);

        ash_device.destroy_device(None);
        instance.destroy_instance(None);
    }

    // =========================================================================================
    // Export UV image as RGB16.

    Ok(())
}
