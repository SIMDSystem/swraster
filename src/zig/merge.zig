// merge — O(n) in-place merge of two already-sorted runs.
//
// Not std.mem.sort: that stack-allocates [512]T per call (~240 KB for the
// ~480-byte RenderTriangle) and re-sorts the whole concatenation. This merges
// linearly with one scratch buffer sized to the smaller run.

const std = @import("std");

// `scratch` is reusable (contents clobbered); equal elements need no defined
// relative order.
pub fn mergeSortedRuns(
    comptime T: type,
    items: []T,
    mid: usize,
    scratch: *std.ArrayList(T),
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
) void {
    const n = items.len;
    if (mid == 0 or mid >= n) return;
    const left_len = mid;
    const right_len = n - mid;

    if (left_len <= right_len) {
        // Copy smaller left run aside, merge front-to-back: k = i + (j - mid),
        // i <= mid, so write k never overtakes read j — no unread clobber.
        scratch.clearRetainingCapacity();
        scratch.appendSlice(std.heap.c_allocator, items[0..left_len]) catch unreachable;
        const buf = scratch.items;
        var i: usize = 0; // index into buf (left run)
        var j: usize = mid; // index into items (right run)
        var k: usize = 0; // write cursor
        while (i < left_len and j < n) {
            if (lessThan(context, items[j], buf[i])) {
                items[k] = items[j];
                j += 1;
            } else {
                items[k] = buf[i];
                i += 1;
            }
            k += 1;
        }
        while (i < left_len) : (i += 1) {
            items[k] = buf[i];
            k += 1;
        }
    } else {
        // Copy smaller right run aside, merge back-to-front: k-1 >= read i-1
        // (k = i + j, j >= 1), so the unread left run is never clobbered.
        scratch.clearRetainingCapacity();
        scratch.appendSlice(std.heap.c_allocator, items[mid..n]) catch unreachable;
        const buf = scratch.items;
        var i: usize = mid; // one past current left index
        var j: usize = right_len; // count remaining in buf
        var k: usize = n; // one past write cursor
        while (i > 0 and j > 0) {
            if (lessThan(context, buf[j - 1], items[i - 1])) {
                items[k - 1] = items[i - 1];
                i -= 1;
            } else {
                items[k - 1] = buf[j - 1];
                j -= 1;
            }
            k -= 1;
        }
        while (j > 0) : (j -= 1) {
            items[k - 1] = buf[j - 1];
            k -= 1;
        }
    }
}
