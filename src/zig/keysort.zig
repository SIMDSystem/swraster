// keysort — sort large POD structs by an f32 key.
//
// RenderTriangle is ~480 bytes; std.sort.pdq would swap whole structs O(n log n)
// times. Instead we sort (key, idx) pairs (8 bytes) and gather the structs once,
// dropping large-struct movement to ~2n.

const std = @import("std");

pub const KeyIdx = struct { key: f32, idx: u32 };

fn lessKey(_: void, a: KeyIdx, b: KeyIdx) bool {
    return a.key < b.key;
}
fn greaterKey(_: void, a: KeyIdx, b: KeyIdx) bool {
    return a.key > b.key;
}

// `keys`/`gather` are caller-owned reusable scratch (contents clobbered).
pub fn sortByKey(
    comptime T: type,
    items: []T,
    ascending: bool,
    keys: *std.ArrayList(KeyIdx),
    gather: *std.ArrayList(T),
    comptime keyOf: fn (*const T) f32,
) void {
    const n = items.len;
    if (n < 2) return;

    keys.clearRetainingCapacity();
    keys.ensureTotalCapacity(std.heap.c_allocator, n) catch unreachable;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        keys.appendAssumeCapacity(.{ .key = keyOf(&items[i]), .idx = i });
    }

    if (ascending) {
        std.sort.pdq(KeyIdx, keys.items, {}, lessKey);
    } else {
        std.sort.pdq(KeyIdx, keys.items, {}, greaterKey);
    }

    gather.clearRetainingCapacity();
    gather.ensureTotalCapacity(std.heap.c_allocator, n) catch unreachable;
    for (keys.items) |ki| {
        gather.appendAssumeCapacity(items[ki.idx]);
    }
    @memcpy(items[0..n], gather.items[0..n]);
}
