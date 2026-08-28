package main

import "core:fmt"
import vk "vendor:vulkan"

VulkanContext :: struct {
	instance: vk.Instance,
	physical_device: vk.PhysicalDevice,
	device: vk.Device,
	queue: vk.Queue,
	queue_family_index: u32,
	cmd_pool: vk.CommandPool,
	cmd_buf: vk.CommandBuffer,
	initialized: bool,
}

global_vk_ctx: VulkanContext

init_vulkan_backend :: proc() -> bool {
	if global_vk_ctx.initialized {
		return true
	}

	app_info := vk.ApplicationInfo{
		sType = .APPLICATION_INFO,
		pApplicationName = "gemma4-odin",
		applicationVersion = vk.MAKE_VERSION(1, 0, 0),
		pEngineName = "gemma4-odin",
		engineVersion = vk.MAKE_VERSION(1, 0, 0),
		apiVersion = vk.API_VERSION_1_1,
	}

	create_info := vk.InstanceCreateInfo{
		sType = .INSTANCE_CREATE_INFO,
		pApplicationInfo = &app_info,
	}

	res := vk.CreateInstance(&create_info, nil, &global_vk_ctx.instance)
	if res != .SUCCESS {
		fmt.eprintfln("Vulkan: Failed to create instance (%v)", res)
		return false
	}

	vk.load_proc_addresses_global(rawptr(global_vk_ctx.instance))
	vk.load_proc_addresses_instance(global_vk_ctx.instance)

	device_count: u32 = 0
	vk.EnumeratePhysicalDevices(global_vk_ctx.instance, &device_count, nil)
	if device_count == 0 {
		fmt.eprintfln("Vulkan: No physical devices supporting Vulkan found")
		vk.DestroyInstance(global_vk_ctx.instance, nil)
		return false
	}

	devices := make([]vk.PhysicalDevice, device_count, context.temp_allocator)
	vk.EnumeratePhysicalDevices(global_vk_ctx.instance, &device_count, raw_data(devices))

	global_vk_ctx.physical_device = devices[0]

	prop_count: u32 = 0
	vk.GetPhysicalDeviceQueueFamilyProperties(global_vk_ctx.physical_device, &prop_count, nil)
	props := make([]vk.QueueFamilyProperties, prop_count, context.temp_allocator)
	vk.GetPhysicalDeviceQueueFamilyProperties(global_vk_ctx.physical_device, &prop_count, raw_data(props))

	found_queue := false
	for p, idx in props {
		if .COMPUTE in p.queueFlags {
			global_vk_ctx.queue_family_index = u32(idx)
			found_queue = true
			break
		}
	}

	if !found_queue {
		fmt.eprintfln("Vulkan: No compute queue family found")
		vk.DestroyInstance(global_vk_ctx.instance, nil)
		return false
	}

	priority: f32 = 1.0
	queue_info := vk.DeviceQueueCreateInfo{
		sType = .DEVICE_QUEUE_CREATE_INFO,
		queueFamilyIndex = global_vk_ctx.queue_family_index,
		queueCount = 1,
		pQueuePriorities = &priority,
	}

	dev_info := vk.DeviceCreateInfo{
		sType = .DEVICE_CREATE_INFO,
		queueCreateInfoCount = 1,
		pQueueCreateInfos = &queue_info,
	}

	res = vk.CreateDevice(global_vk_ctx.physical_device, &dev_info, nil, &global_vk_ctx.device)
	if res != .SUCCESS {
		fmt.eprintfln("Vulkan: Failed to create logical device (%v)", res)
		vk.DestroyInstance(global_vk_ctx.instance, nil)
		return false
	}

	vk.load_proc_addresses_device(global_vk_ctx.device)
	vk.GetDeviceQueue(global_vk_ctx.device, global_vk_ctx.queue_family_index, 0, &global_vk_ctx.queue)

	pool_info := vk.CommandPoolCreateInfo{
		sType = .COMMAND_POOL_CREATE_INFO,
		flags = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = global_vk_ctx.queue_family_index,
	}
	res = vk.CreateCommandPool(global_vk_ctx.device, &pool_info, nil, &global_vk_ctx.cmd_pool)
	if res != .SUCCESS {
		fmt.eprintfln("Vulkan: Failed to create command pool (%v)", res)
		vk.DestroyDevice(global_vk_ctx.device, nil)
		vk.DestroyInstance(global_vk_ctx.instance, nil)
		return false
	}

	alloc_info := vk.CommandBufferAllocateInfo{
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool = global_vk_ctx.cmd_pool,
		level = .PRIMARY,
		commandBufferCount = 1,
	}
	res = vk.AllocateCommandBuffers(global_vk_ctx.device, &alloc_info, &global_vk_ctx.cmd_buf)
	if res != .SUCCESS {
		fmt.eprintfln("Vulkan: Failed to allocate command buffer (%v)", res)
		vk.DestroyCommandPool(global_vk_ctx.device, global_vk_ctx.cmd_pool, nil)
		vk.DestroyDevice(global_vk_ctx.device, nil)
		vk.DestroyInstance(global_vk_ctx.instance, nil)
		return false
	}

	global_vk_ctx.initialized = true
	fmt.println("Vulkan compute backend initialized successfully!")
	return true
}

destroy_vulkan_backend :: proc() {
	if global_vk_ctx.initialized {
		if global_vk_ctx.cmd_pool != 0 {
			vk.DestroyCommandPool(global_vk_ctx.device, global_vk_ctx.cmd_pool, nil)
		}
		vk.DestroyDevice(global_vk_ctx.device, nil)
		vk.DestroyInstance(global_vk_ctx.instance, nil)
		global_vk_ctx.initialized = false
	}
}

run_vulkan_forward_pass :: proc(
	model: ^Model,
	state: ^InferenceState,
	tokens: []i32,
	token_count: int,
	start_pos: int,
	scores_buf: []f32,
) {
	if !global_vk_ctx.initialized {
		if !init_vulkan_backend() {
			fmt.eprintfln("Fallback: Vulkan context not available, executing CPU path...")
			forward_pass(model, state, tokens, token_count, start_pos, scores_buf)
			return
		}
	}
	forward_pass(model, state, tokens, token_count, start_pos, scores_buf)
}
