#pragma once

// Sort large POD structs by a float key. Sorting 8-byte (key,index) pairs and
// gathering once moves each ~480-byte struct only ~2n times instead of n*log n.

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>

struct KeyIdx {
    float    key;
    uint32_t idx;
};

// `keys` and `gather` are caller-owned scratch buffers (clobbered each call).
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
