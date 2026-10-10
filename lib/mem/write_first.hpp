// Huge pages written before they are read. On Linux 6.8 a 2 MiB page that is read first maps the
// shared huge zero page, and the first write then splits it into 4 KiB pages; a page written first
// is a huge page. Linux 7.0 allocates a huge page either way. Measurements: lib/mem/notes.md.
//
//   auto* a = mem::huge<std::uint32_t>(len);   // huge.hpp
//   in.read(a, n);
//   mem::write_first(a, n, len);               // before a[n, len) is read
//
// write_first stores zero at each 2 MiB boundary in x[begin, end): once in every page of
// x[begin, end) but the one that holds x[begin - 1], which the caller has written. x[begin, end)
// must be zero.
#pragma once

#include <cstddef>
#include <cstdint>

#include "huge.hpp"

namespace mem {

template <class T>
void write_first(T* x, std::size_t begin, std::size_t end) {
    constexpr std::size_t kPage = kHugePage / sizeof(T);  // elements
    const std::size_t offset = reinterpret_cast<std::uintptr_t>(x) / sizeof(T) % kPage;
    for (std::size_t i = (begin + offset + kPage - 1) / kPage * kPage - offset; i < end; i += kPage) x[i] = 0;
}

}  // namespace mem
