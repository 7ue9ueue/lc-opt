// Tests for lib/mem: alignment, zero fill and the whole range writable, for mappings and arenas.
#include "lib/mem/huge.hpp"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void expect(bool ok, const char* what, std::size_t detail = 0) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (%zu)\n", what, detail);
}

bool aligned(const void* p, std::size_t to) { return reinterpret_cast<std::uintptr_t>(p) % to == 0; }

bool zero(const unsigned char* p, std::size_t bytes) {
    for (std::size_t i = 0; i < bytes; ++i)
        if (p[i]) return false;
    return true;
}

void test_huge() {
    for (const std::size_t bytes : {std::size_t(1), std::size_t(4096), mem::kHugePage - 1, mem::kHugePage,
                                    mem::kHugePage + 1, 5 * mem::kHugePage + 12345}) {
        auto* p = static_cast<unsigned char*>(mem::map_huge(bytes));
        const std::size_t rounded = (bytes + mem::kHugePage - 1) / mem::kHugePage * mem::kHugePage;
        expect(aligned(p, mem::kHugePage), "map_huge: aligned", bytes);
        expect(zero(p, rounded), "map_huge: zero", bytes);
        std::memset(p, 0xAB, rounded);  // the rounded size is mapped
        expect(p[rounded - 1] == 0xAB, "map_huge: writable", bytes);
    }
    auto* words = mem::huge<std::uint64_t>(1000000);
    expect(aligned(words, mem::kHugePage) && zero(reinterpret_cast<unsigned char*>(words), 8000000), "huge<T>");
    words[999999] = 1;
}

void test_arena() {
    constexpr std::size_t kSizes[] = {1, 3, 64, 100, 1000, 4096, 12345};
    std::size_t total = 0;
    for (const std::size_t n : kSizes) total += (n * sizeof(std::uint32_t) + 63) / 64 * 64;
    mem::Arena arena(total);
    unsigned char* prev_end = nullptr;
    for (const std::size_t n : kSizes) {
        auto* p = arena.take<std::uint32_t>(n);
        auto* bytes = reinterpret_cast<unsigned char*>(p);
        expect(aligned(p, 64), "arena: aligned", n);
        expect(!prev_end || bytes >= prev_end, "arena: disjoint", n);
        expect(zero(bytes, n * sizeof(std::uint32_t)), "arena: zero", n);
        std::memset(bytes, 0xCD, n * sizeof(std::uint32_t));
        prev_end = bytes + n * sizeof(std::uint32_t);
    }
    auto* first = mem::Arena(mem::kHugePage).take<char>(1);
    expect(aligned(first, mem::kHugePage), "arena: first take on a huge page");
}

}  // namespace

int main() {
    test_huge();
    test_arena();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
