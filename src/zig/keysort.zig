// keysort.zig — sort large POD structs by a f32 key without paying the cost of
// moving the structs during the sort.
//
// RenderTriangle is ~480 bytes. Sorting an array of them with std.sort.pdq
// swaps whole structs: each swap is 3 copies of 480 bytes, and an O(n log n)
// sort performs ~n*log2(n) swaps, so the comparison-sort moves megabytes of
// payload per worker per frame. C++'s std::sort moves the structs too, which is
// exactly why the Zig T&L "local sort" stage trailed the C++ build.
//
// Instead we extract (key, index) pairs (8 bytes each), sort *those* — the swaps
// are now 8-byte moves and stay cache-resident — then gather the structs into
// sorted order exactly once. Total large-struct movement drops to ~2n (gather +
// copy-back) regardless of n, which moves strictly less data than an in-place
// comparison sort for any n > ~3.

const std = @import("std");

pub const KeyIdx = struct { key: f32, idx: u32 };

fn lessKey(_: void, a: KeyIdx, b: KeyIdx) bool {
    return a.key < b.key;
}
fn greaterKey(_: void, a: KeyIdx, b: KeyIdx) bool {
    return a.key > b.key;
}

/// Sort `items` in place by `keyOf(item)`, ascending or descending. `keys` and
/// `gather` are caller-owned scratch buffers reused across calls (their
/// contents are clobbered); `gather` must be a list of the same `T`.
pub fn sortByKey(
    comptime T: type,
    items: []T,
    ascending: bool,
    keys: *std.array_list.Managed(KeyIdx),
    gather: *std.array_list.Managed(T),
    comptime keyOf: fn (*const T) f32,
) void {
    const n = items.len;
    if (n < 2) return;

    keys.clearRetainingCapacity();
    keys.ensureTotalCapacity(n) catch unreachable;
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
    gather.ensureTotalCapacity(n) catch unreachable;
    for (keys.items) |ki| {
        gather.appendAssumeCapacity(items[ki.idx]);
    }
    @memcpy(items[0..n], gather.items[0..n]);
}
