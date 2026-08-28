package main

import "core:fmt"
import "core:sys/info"
import "core:thread"

ThreadPoolGlobal :: struct {
	pool: thread.Pool,
	num_threads: int,
	initialized: bool,
}

global_thread_pool: ThreadPoolGlobal

init_thread_pool :: proc() {
	if global_thread_pool.initialized {
		return
	}
	t_count := 1
	if phys, log, ok := info.cpu_core_count(); ok && phys > 0 {
		t_count = phys
	} else {
		t_count = 4
	}
	if t_count < 1 {
		t_count = 1
	}

	global_thread_pool.num_threads = t_count
	if t_count > 1 {
		thread.pool_init(&global_thread_pool.pool, context.allocator, t_count)
		thread.pool_start(&global_thread_pool.pool)
	}
	global_thread_pool.initialized = true
}

destroy_thread_pool :: proc() {
	if global_thread_pool.initialized {
		if global_thread_pool.num_threads > 1 {
			thread.pool_finish(&global_thread_pool.pool)
			thread.pool_destroy(&global_thread_pool.pool)
		}
		global_thread_pool.initialized = false
	}
}

parallel_for :: proc(total_items: int, task_proc: proc(start, end: int, user_data: rawptr), user_data: rawptr) {
	num_threads := global_thread_pool.num_threads
	if num_threads <= 1 || total_items <= 1 {
		task_proc(0, total_items, user_data)
		return
	}

	TaskRangeData :: struct {
		start: int,
		end: int,
		proc_ptr: proc(start, end: int, user_data: rawptr),
		user_data: rawptr,
	}

	wrapper_proc :: proc(task: thread.Task) {
		range := cast(^TaskRangeData)task.data
		range.proc_ptr(range.start, range.end, range.user_data)
	}

	items_per_thread := (total_items + num_threads - 1) / num_threads
	task_datas := make([]TaskRangeData, num_threads, context.allocator)
	defer delete(task_datas)

	task_count := 0
	for i in 0 ..< num_threads {
		s := i * items_per_thread
		e := s + items_per_thread
		if e > total_items {
			e = total_items
		}
		if s >= total_items {
			break
		}
		task_datas[i] = TaskRangeData{
			start = s,
			end = e,
			proc_ptr = task_proc,
			user_data = user_data,
		}
		thread.pool_add_task(&global_thread_pool.pool, context.allocator, wrapper_proc, &task_datas[i])
		task_count += 1
	}

	thread.pool_do_work(&global_thread_pool.pool, {})
	for thread.pool_num_outstanding(&global_thread_pool.pool) > 0 {
		thread.pool_do_work(&global_thread_pool.pool, {})
	}
}
