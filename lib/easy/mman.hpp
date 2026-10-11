// <sys/mman.h>, or on Windows the part of it lib/ntt and lib/poly use: anonymous private
// mappings of zeroed pages (VirtualAlloc). showcase/bundle.py puts this file where a bundled
// header includes <sys/mman.h>. No MADV_HUGEPAGE on Windows, so the madvise calls drop out.
#pragma once

#ifdef _WIN32
#include <cstddef>

extern "C" __declspec(dllimport) void* __stdcall VirtualAlloc(void* address, std::size_t size, unsigned long type,
                                                              unsigned long protect);
extern "C" __declspec(dllimport) int __stdcall VirtualFree(void* address, std::size_t size, unsigned long type);

#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_PRIVATE 2
#define MAP_ANONYMOUS 0x20
#define MAP_FAILED (reinterpret_cast<void*>(-1))

inline void* mmap(void*, std::size_t bytes, int, int, int, long long) {
    constexpr unsigned long kCommitReserve = 0x3000, kReadWrite = 0x04;
    void* p = VirtualAlloc(nullptr, bytes, kCommitReserve, kReadWrite);
    return p ? p : MAP_FAILED;
}

inline int munmap(void* p, std::size_t) {
    constexpr unsigned long kRelease = 0x8000;
    return VirtualFree(p, 0, kRelease) ? 0 : -1;
}
#else
#include <sys/mman.h>
#endif
