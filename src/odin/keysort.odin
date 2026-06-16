// keysort.odin — sort large structs by f32 key via (key,index) pairs, gather once.

package main

import "core:slice"

Key_Idx :: struct {
	key: f32,
	idx: u32,
}

ensure_key_index_capacity :: proc(keys: ^[dynamic]Key_Idx, n: int) {
	if cap(keys^) < n do reserve(keys, n)
}

@(private)
render_triangle_sort_z :: proc(t: ^Render_Triangle) -> f32 {
	return t.sort_z
}

sort_by_key :: proc($T: typeid, items: []T, ascending: bool, keys: ^[dynamic]Key_Idx, gather: ^[dynamic]T, key_of: proc(^T) -> f32) {
	n := len(items)
	if n < 2 do return

	clear(keys)
	ensure_key_index_capacity(keys, n)
	for i in 0 ..< n {
		append(keys, Key_Idx{key = key_of(&items[i]), idx = u32(i)})
	}

	if ascending {
		slice.sort_by(keys[:], proc(a, b: Key_Idx) -> bool { return a.key < b.key })
	} else {
		slice.sort_by(keys[:], proc(a, b: Key_Idx) -> bool { return a.key > b.key })
	}

	clear(gather)
	ensure_render_triangle_capacity(gather, n)
	for ki in keys {
		append(gather, items[ki.idx])
	}
	copy(items[:n], gather[:n])
}

sort_by_key_render_triangles :: proc(items: ^Render_Triangle_List, ascending: bool, keys: ^[dynamic]Key_Idx, gather: ^Render_Triangle_List) {
	sort_by_key(Render_Triangle, items[:], ascending, keys, gather, render_triangle_sort_z)
}
