//! An example demonstrating compute pipelines by executing Rong Guodong's
//! Jump Flood Algorithm for creating voronoi diagrams.
use std::num::NonZero;

use anyhow::Result;
use ash::vk;
use fzvk::*;
pub fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: <in-png-path>"))?;
    let image =
        png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?)).read_info()?;
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
    // Find a device.
    // We need:
    // * a compute queue
    // * storage support for R8Unorm, RG8Uint.
    let (phys_device, device, queue) = unsafe {
        let phys_devices = instance.enumerate_physical_devices()?;

        let (phys_device, family_index, is_portability) = phys_devices
            .iter()
            // Check it has all the features we need:
            .filter(|&&phys| {
                let r8unorm = instance
                    .get_physical_device_format_properties(phys, vk::Format::R8_UNORM)
                    .optimal_tiling_features
                    .intersects(vk::FormatFeatureFlags::STORAGE_IMAGE);
                if !r8unorm {
                    return false;
                }
                instance
                    .get_physical_device_format_properties(phys, vk::Format::R8G8_UINT)
                    .optimal_tiling_features
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

        // If we enabled portabilty_enumeration, we *must* enable portability_subset
        // on any device that advertises it.
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

        (phys_device, device, queue)
    };

    let device = Device::from_ash(device);
    // Import image, and take only its Alpha channel.
    let extent = Extent2D {
        width: image.info().width.try_into()?,
        height: image.info().height.try_into()?,
    };
    let size_bytes =
        NonZero::new(u64::from(extent.width.get()) * u64::from(extent.height.get()) * 1).unwrap();

    // Create buffer + images objects and appropriate backing memory.
    // We will need:
    // One host buffer, one R8Unorm format image to reference, two RG16Uint format images to "pingpong" between.
    // Our operation will look like:
    // Host buffer --xfer--> color --storage--> UV1 <--storage--> UV2 --xfer--> host buffer.
    unsafe {
        let buffer = device.create_buffer(
            (TransferSrc, TransferDst),
            size_bytes,
            SharingMode::Exclusive,
        )?;
        let reference_image = device.create_image(
            (TransferDst, Storage),
            extent,
            NonZero::new(1).unwrap(),
            ColorFormat::R8Unorm,
            SingleSampled,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )?;
        let pingpong = device.create_image(
            (TransferSrc, Storage),
            extent.with_layers(ArrayCount::new(2).unwrap()),
            NonZero::new(1).unwrap(),
            ColorFormat::Rg8Uint,
            SingleSampled,
            vk::ImageTiling::OPTIMAL,
            SharingMode::Exclusive,
        )?;
    }

    // Create the pipeline that will do the work

    // Record a command buffer describing the whole operation

    // Execute!

    // Export UV image as RGB16.

    Ok(())
}
