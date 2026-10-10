// Tests for lib/mem: alignment, zero fill and the whole range writable, for mappings and arenas;
// write_first's stores.
#include <cstdio>
#include <cstring>

#include "lib/mem/huge.hpp"
#include "lib/mem/write_first.hpp"

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
    constexpr std::size_t kBytes[] = {1, 4096, mem::kHugePage - 1, mem::kHugePage, mem::kHugePage + 1,
                                      5 * mem::kHugePage + 12345};
    for (const std::size_t bytes : kBytes) {
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

// write_first on x = base + shift for several ranges: zero exactly at the 2 MiB boundaries in
// [begin, end). The memory starts as ones so that the stores show.
template <class T>
void test_write_first() {
    constexpr std::size_t kPage = mem::kHugePage / sizeof(T), kCount = 5 * kPage;
    T* base = mem::huge<T>(kCount + kPage);
    const std::size_t shifts[] = {0, 1, kPage / 2, kPage - 1};
    for (const std::size_t shift : shifts) {
        T* x = base + shift;
        const std::size_t ranges[][2] = {{0, kCount}, {1, kCount},           {kPage - shift, kCount},
                                         {7, 7},      {5, kPage / 3},        {kPage + 3, 3 * kPage - shift},
                                         {0, 1},      {kPage - shift + 1, kCount - 1}};
        for (const auto& [begin, end] : ranges) {
            std::memset(base, 0xFF, (kCount + kPage) * sizeof(T));
            mem::write_first(x, begin, end);
            bool ok = true;
            for (std::size_t i = 0; i < kCount; ++i) {
                const bool boundary = reinterpret_cast<std::uintptr_t>(x + i) % mem::kHugePage == 0;
                ok &= (x[i] == 0) == (boundary && i >= begin && i < end);
            }
            expect(ok, "write_first", sizeof(T) * 1000000 + shift * 1000 + begin);
        }
    }
}

}  // namespace

int main() {
    test_huge();
    test_arena();
    test_write_first<std::uint32_t>();
    test_write_first<std::uint64_t>();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
