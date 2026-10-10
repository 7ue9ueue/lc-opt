// Marks a mapped input as read once (MADV_SEQUENTIAL): when the Reader unmaps it, the kernel then
// skips marking each page accessed, 0.03-0.04 ms per 20 MB (lib/io/notes.md).
//
//   io::Reader in;
//   const auto n = in.read<std::uint32_t>();
//   io::advise_sequential(in);   // while the Reader is in the input's first page
//
// fd is the Reader's. Inputs the Reader copies (pipes, files of kMapAbove bytes or less) are left
// alone.
#pragma once

#include <sys/mman.h>
#include <sys/stat.h>

#include <cstddef>
#include <cstdint>

#include "io.hpp"

namespace io {

inline void advise_sequential(const Reader& in, int fd = 0) {
    struct stat st;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || std::size_t(st.st_size) <= detail::kMapAbove) return;
    const auto start = reinterpret_cast<std::uintptr_t>(in.scan().cur) & ~std::uintptr_t(4095);
    ::madvise(reinterpret_cast<void*>(start), std::size_t(st.st_size), MADV_SEQUENTIAL);
}

}  // namespace io
