// Checks fields11's formatter against a scalar one for every value below 1000000007.
// g++ -std=c++23 -O2 -march=x86-64-v3 -I../../.. test_fields11.cpp && ./a.out
#include <cstdio>
#include <cstring>
#include <vector>

#include "fields11.hpp"

namespace {

constexpr std::uint32_t kEnd = 1000000007;

// v right-aligned in 10 characters, then a space.
void field(std::uint32_t v, char* out) {
    std::memset(out, ' ', 11);
    int i = 9;
    do out[i--] = char('0' + v % 10), v /= 10;
    while (v);
}

}  // namespace

int main() {
    constexpr std::size_t kCount = 1 << 16;
    std::vector<std::uint32_t> values(kCount);
    alignas(16) static char text[11 * kCount + 16];
    char want[11];
    for (std::uint64_t base = 0; base < kEnd; base += kCount) {
        for (std::size_t j = 0; j < kCount; ++j) values[j] = std::uint32_t(std::min<std::uint64_t>(base + j, kEnd - 1));
        fields11::detail::format(text, values.data(), kCount, fields11::detail::kConstants);
        for (std::size_t j = 0; j < kCount; ++j) {
            field(values[j], want);
            if (std::memcmp(want, text + 11 * j, 11) != 0) {
                std::printf("FAIL: %u -> '%.11s'\n", values[j], text + 11 * j);
                return 1;
            }
        }
    }
    std::printf("PASS: every value below %u\n", kEnd);
}
