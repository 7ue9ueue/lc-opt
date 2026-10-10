// A buffer for output text in memory the solution no longer needs and has already touched, so the
// text costs no page faults. Page-aligned: write(2) then runs 2x slower on Zen 3 only at file
// offsets 1-31 mod 4096 (lib/io/notes.md).
#pragma once

#include <cstddef>
#include <cstdint>

// Bytes bytes, page-aligned within spare[0, spare_bytes) if they fit there, else a static buffer.
template <std::size_t Bytes>
char* text_buffer(void* spare, std::size_t spare_bytes) {
    constexpr std::uintptr_t kPage = 4096;
    const auto start = reinterpret_cast<std::uintptr_t>(spare);
    const std::uintptr_t aligned = (start + kPage - 1) & ~(kPage - 1);
    if (aligned - start + Bytes <= spare_bytes) return reinterpret_cast<char*>(aligned);
    alignas(kPage) static char fallback[Bytes];
    return fallback;
}
