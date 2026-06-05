#pragma once

// keysort.h — sort large POD structs by a float key without moving the structs
// during the comparison sort. C++ mirror of src/zig/keysort.zig.
//
// RenderTriangle is ~480 bytes. std::sort relocates whole structs (~n*log2(n)
// moves of 480 bytes). Instead we extract (key, index) pairs (8 bytes each),
// sort *those* (cache-resident 8-byte moves), then gather the structs into
// sorted order exactly once. Total large-struct movement drops to ~2n
// regardless of n, which beats an in-place comparison sort for any n > ~3.

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>

struct KeyIdx {
    float    key;
    uint32_t idx;
};

// Sort `items` in place by `key_of(item)`, ascending or descending. `keys` and
// `gather` are caller-owned scratch buffers reused across calls (their contents
// are clobbered); `gather` must be a vector of the same element type.
template <typename T, typename KeyFn>
inline void sort_by_key(std::vector<T>& items, bool ascending,
                        std::vector<KeyIdx>& keys, std::vector<T>& gather,
                        KeyFn key_of) {
    const size_t n = items.size();
    if (n < 2) return;

    keys.clear();
    keys.reserve(n);
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); i++) {
        keys.push_back(KeyIdx{ key_of(items[i]), i });
    }

    if (ascending) {
        std::sort(keys.begin(), keys.end(),
                  [](const KeyIdx& a, const KeyIdx& b) { return a.key < b.key; });
    } else {
        std::sort(keys.begin(), keys.end(),
                  [](const KeyIdx& a, const KeyIdx& b) { return a.key > b.key; });
    }

    gather.clear();
    gather.reserve(n);
    for (const KeyIdx& ki : keys) gather.push_back(items[ki.idx]);
    std::copy(gather.begin(), gather.end(), items.begin());
}
