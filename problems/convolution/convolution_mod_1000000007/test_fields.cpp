// Checks both formatters against a scalar one: fields10 for every value below 10^9, fields11 for
// every value below 1000000007.
// g++ -std=c++23 -O2 -march=x86-64-v3 -I../../.. test_fields.cpp && ./a.out
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fields10.hpp"
#include "fields11.hpp"

namespace {

constexpr std::size_t kCount = 1 << 16;

// v right-aligned in width - 1 characters, then a space.
void field(std::uint32_t v, int width, char* out) {
    std::memset(out, ' ', width);
    int i = width - 2;
    do out[i--] = char('0' + v % 10), v /= 10;
    while (v);
}

// format(text, values, kCount) for every value below end, in field width `width`.
template <class Format>
bool check(std::uint32_t end, int width, Format format) {
    std::vector<std::uint32_t> values(kCount);
    std::vector<char> text(width * kCount + 64);
    char* aligned = text.data() + (-reinterpret_cast<std::uintptr_t>(text.data()) & 15);
    char want[16];
    for (std::uint64_t base = 0; base < end; base += kCount) {
        for (std::size_t j = 0; j < kCount; ++j) values[j] = std::uint32_t(std::min<std::uint64_t>(base + j, end - 1));
        format(aligned, values.data());
        for (std::size_t j = 0; j < kCount; ++j) {
            field(values[j], width, want);
            if (std::memcmp(want, aligned + width * j, width) != 0) {
                std::printf("FAIL: width %d, %u -> '%.*s'\n", width, values[j], width, aligned + width * j);
                return false;
            }
        }
    }
    std::printf("PASS: width %d, every value below %u\n", width, end);
    return true;
}

}  // namespace

int main() {
    const bool ok10 = check(1000000000, 10, [](char* p, const std::uint32_t* x) {
        fields10::detail::format(p, x, kCount, fields10::detail::kConstants);
    });
    const bool ok11 = check(1000000007, 11, [](char* p, const std::uint32_t* x) {
        fields11::detail::format(p, x, kCount, fields11::detail::kConstants);
    });
    return ok10 && ok11 ? 0 : 1;
}
