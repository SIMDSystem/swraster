// merge.zig — O(n) merge of two already-sorted runs, the direct equivalent of
// C++ std::inplace_merge. Used by the T&L scatter-merge and global fold, where
// a freshly-sorted source run is appended onto an already-sorted destination
// and the whole thing must stay sorted.
//
// Why this exists: Zig's std.mem.sort is the *stable block sort*, which (a)
// stack-allocates `[512]T` on every call (RenderTriangle is ~480 bytes, so
// that's ~240 KB per call) and (b) re-sorts the entire concatenation instead
// of merging the two runs. Re-sorting two sorted runs is asymptotically and
// constant-factor far worse than a single linear merge. This routine merges in
// O(n) with a single scratch buffer sized to the *smaller* run.

const std = @import("std");

/// Merge two consecutive sorted runs `items[0..mid]` and `items[mid..]` in
/// place so the whole slice is sorted by `lessThan`. `scratch` is any reusable
/// list (its contents are clobbered); only the smaller run is copied into it,
/// so it never needs more than `items.len / 2` elements.
///
/// Order is consistent with std::inplace_merge for the orderings we use
/// (strict `<` / `>` on sort_z); equal elements need no defined relative order.
pub fn mergeSortedRuns(
    comptime T: type,
    items: []T,
    mid: usize,
    scratch: *std.array_list.Managed(T),
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
) void {
    const n = items.len;
    if (mid == 0 or mid >= n) return;
    const left_len = mid;
    const right_len = n - mid;

    if (left_len <= right_len) {
        // Copy the (smaller) left run aside, then merge front-to-back. Write
        // index k never overtakes the right read index j: k = i + (j - mid) and
        // i <= mid, so k <= j throughout — no unread element is clobbered.
        scratch.clearRetainingCapacity();
        scratch.appendSlice(items[0..left_len]) catch unreachable;
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
        // Any leftover right-run elements are already in their final slots.
    } else {
        // Copy the (smaller) right run aside, then merge back-to-front. Write
        // index k-1 is always >= the left read index i-1 (k = i + j, j >= 1),
        // so the unread left run is never clobbered.
        scratch.clearRetainingCapacity();
        scratch.appendSlice(items[mid..n]) catch unreachable;
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
        // Any leftover left-run elements are already in their final slots.
    }
}
