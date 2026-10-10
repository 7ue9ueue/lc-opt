// Zero-filled memory in transparent huge pages (2 MiB) where the kernel allows. Never freed: the
// programs end with _exit. Linux or macOS. Measurements: lib/mem/notes.md.
//
//   auto* a = mem::huge<std::uint32_t>(n);   // n zeroed values, 2 MiB aligned
//   mem::Arena arena(bytes);                 // one mapping for several arrays
//   auto* b = arena.take<std::uint64_t>(m);  // m zeroed values, 64-byte aligned
//
// Pages fault in on first touch. An arena of B bytes holds takes whose sizes, each rounded up to
// 64 bytes, sum to at most B; it does not check.
#pragma once

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace mem {

inline constexpr std::size_t kHugePage = std::size_t(1) << 21;

// bytes rounded up to whole huge pages, 2 MiB aligned.
inline void* map_huge(std::size_t bytes) {
    bytes = (bytes + kHugePage - 1) / kHugePage * kHugePage;
    void* region = ::mmap(nullptr, bytes + kHugePage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) std::abort();
    const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region) + kHugePage - 1) & ~(kHugePage - 1);
#ifdef MADV_HUGEPAGE
    ::madvise(reinterpret_cast<void*>(aligned), bytes, MADV_HUGEPAGE);
#endif
    return reinterpret_cast<void*>(aligned);
}

// count values of T.
template <class T>
T* huge(std::size_t count) {
    return static_cast<T*>(map_huge(count * sizeof(T)));
}

// Bump allocation from one map_huge() mapping.
class Arena {
public:
    explicit Arena(std::size_t bytes) : cur_(reinterpret_cast<std::uintptr_t>(map_huge(bytes))) {}

    // count values of T; the next take starts at the next multiple of 64 bytes.
    template <class T>
    T* take(std::size_t count) {
        T* p = reinterpret_cast<T*>(cur_);
        cur_ += (count * sizeof(T) + 63) & ~std::size_t(63);
        return p;
    }

private:
    std::uintptr_t cur_;
};

}  // namespace mem
