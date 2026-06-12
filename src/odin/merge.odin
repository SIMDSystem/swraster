// merge.odin — O(n) merge of two already-sorted runs (std::inplace_merge equivalent).
//
// Used by the T&L scatter-merge and global fold. Only the smaller run is copied
// into scratch, so it never needs more than items.len / 2 elements.
//
// Buffer policy (matches Zig clearRetainingCapacity / ensureTotalCapacity):
//   - [dynamic]T is Odin's native growable buffer; reserve/append/clear use context.allocator.
//   - Pre-reserve capacity once at init (see render_config.odin caps).
//   - Per frame: reset length with clear() only — never shrink/free backing storage.
//   - append() may realloc (relocating .data); caps are sized so the frame loop never hits that.

package main

import "core:slice"

merge_sorted_runs :: proc($T: typeid, items: []T, mid: int, scratch: ^[dynamic]T, less_than: proc(T, T) -> bool) {
	n := len(items)
	if mid == 0 || mid >= n do return
	left_len := mid
	right_len := n - mid

	if left_len <= right_len {
		clear(scratch)
		append(scratch, ..items[:left_len])
		buf := scratch[:]
		i, j, k := 0, mid, 0
		for i < left_len && j < n {
			if less_than(items[j], buf[i]) {
				items[k] = items[j]
				j += 1
			} else {
				items[k] = buf[i]
				i += 1
			}
			k += 1
		}
		for i < left_len {
			items[k] = buf[i]
			i += 1
			k += 1
		}
	} else {
		clear(scratch)
		append(scratch, ..items[mid:n])
		buf := scratch[:]
		i, j, k := mid, right_len, n
		for i > 0 && j > 0 {
			if less_than(buf[j - 1], items[i - 1]) {
				items[k - 1] = items[i - 1]
				i -= 1
			} else {
				items[k - 1] = buf[j - 1]
				j -= 1
			}
			k -= 1
		}
		for j > 0 {
			items[k - 1] = buf[j - 1]
			j -= 1
			k -= 1
		}
	}
}

@(private)
render_triangle_less :: proc(a, b: Render_Triangle) -> bool {
	return a.sort_z < b.sort_z
}

@(private)
render_triangle_greater :: proc(a, b: Render_Triangle) -> bool {
	return a.sort_z > b.sort_z
}

merge_sorted_runs_render_triangle :: proc(items: ^Render_Triangle_List, mid: int, scratch: ^Render_Triangle_List, count: int = -1) {
	n := count >= 0 ? count : len(items^)
	merge_sorted_runs(Render_Triangle, items[:n], mid, scratch, render_triangle_less)
}

merge_sorted_runs_render_triangle_desc :: proc(items: ^Render_Triangle_List, mid: int, scratch: ^Render_Triangle_List, count: int = -1) {
	n := count >= 0 ? count : len(items^)
	merge_sorted_runs(Render_Triangle, items[:n], mid, scratch, render_triangle_greater)
}

append_render_triangle :: proc(list: ^Render_Triangle_List, tri: Render_Triangle) {
	append(list, tri)
}

// Reset active length; retain capacity (Zig: clearRetainingCapacity).
clear_render_triangle_list :: proc(list: ^Render_Triangle_List) {
	clear(list)
}

append_render_triangles :: proc(list: ^Render_Triangle_List, src: []Render_Triangle) {
	append(list, ..src)
}

// Grow capacity only — never shrinks.
ensure_render_triangle_capacity :: proc(list: ^Render_Triangle_List, n: int) {
	if cap(list^) < n do reserve(list, n)
}

// Init / rare resize: commit length after ensure (global IPC slots).
init_render_triangle_slots :: proc(list: ^Render_Triangle_List, n: int) {
	ensure_render_triangle_capacity(list, n)
	resize(list, n)
}

resize_f32_buffer :: proc(buf: ^[dynamic]f32, n: int, fill: f32) {
	if cap(buf^) < n do reserve(buf, n)
	resize(buf, n)
	for i in 0 ..< n {
		buf[i] = fill
	}
}

ensure_instance_depth_capacity :: proc(list: ^[dynamic]Instance_Depth, n: int) {
	if cap(list^) < n do reserve(list, n)
}

// Reset active length; retain capacity.
clear_instance_depths :: proc(list: ^[dynamic]Instance_Depth) {
	clear(list)
}

append_instance_depth :: proc(list: ^[dynamic]Instance_Depth, d: Instance_Depth) {
	append(list, d)
}

@(private="file")
instance_depth_sort_ctx: ^[dynamic]Cube_Instance

@(private="file")
instance_depth_less :: proc(a, b: Instance_Depth) -> bool {
	instances := instance_depth_sort_ctx
	trans_a := instances[a.index].type == .Torus
	trans_b := instances[b.index].type == .Torus
	if trans_a != trans_b do return !trans_a
	if trans_a do return a.depth < b.depth
	return a.depth > b.depth
}

sort_instance_depths :: proc(list: ^[dynamic]Instance_Depth, instances: ^[dynamic]Cube_Instance) {
	instance_depth_sort_ctx = instances
	slice.sort_by(list[:], instance_depth_less)
}
