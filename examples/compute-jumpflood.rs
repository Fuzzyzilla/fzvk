//! An example demonstrating compute pipelines by executing Rong Guodong's Jump
//! Flood Algorithm for creating voronoi diagrams.
use std::num::NonZero;

/// Don't let the compiler see that we're unconditionally panic'ing, as that
/// turns off a bunch of checks!
fn todo<T>() -> T {
    todo!()
}

use anyhow::Result;
use ash::vk;
use fzvk::*;
pub fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: <in-png-path>"))?;
    let image =
        png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?)).read_info()?;
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
                    .get_physical_device_format_properties(phys, ColorFormat::R8Unorm.format())
                    .optimal_tiling_features
                    // TransferSrc/Dst is implicit. Weird.
                    .intersects(vk::FormatFeatureFlags::STORAGE_IMAGE);
                if !r8unorm {
                    return false;
                }
                instance
                    .get_physical_device_format_properties(phys, ColorFormat::Rg16Uint.format())
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
            ColorFormat::R8Unorm,
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
            ColorFormat::Rg16Uint,
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
        let import_module = device.create_module(&[todo()])?;
        let import_shader = import_module.main(Compute).specialize(&());
        // Shader that iteratively refines between the two pingpong images.
        let pingpong_module = device.create_module(&[todo()])?;
        let pingpong_shader = pingpong_module.main(Compute).specialize(&());

        let [pipeline] = device.create_compute_pipelines(
            None,
            [ComputePipelineCreateInfo {
                layout: todo(),
                shader: pingpong_shader,
            }],
        )?;

        // It's fine to destroy a shader while it's in use by a pipeline
        device.destroy_module(import_module);
        device.destroy_module(pingpong_module);

        let mut command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
        )?;
        let [command_buffer] = device.allocate_command_buffers(&mut command_pool, Primary)?;
    }

    // =========================================================================================
    // Record a command buffer describing the whole operation

    // =========================================================================================
    // Execute!

    // =========================================================================================
    // Export UV image as RGB16.

    Ok(())
}
