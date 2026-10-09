// Checks fields64::format against printf: boundary values (powers of 10 and of 2, and their
// neighbours, multiples of 10^8 and 10^16) and random values of every length.
// g++ -std=c++23 -O2 -march=x86-64-v3 test_fields64.cpp && ./a.out
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fields64.hpp"

namespace {

std::vector<std::uint64_t> cases() {
    std::vector<std::uint64_t> v = {0, ~0ull, ~0ull - 1};
    const auto around = [&v](std::uint64_t x) {
        for (std::uint64_t d = 0; d < 3; ++d) v.push_back(x + d), v.push_back(x - d);
    };
    for (std::uint64_t p = 1; p <= 10000000000000000000ull; p *= 10) {
        around(p);
        if (p == 10000000000000000000ull) break;
    }
    for (int b = 0; b < 64; ++b) around(std::uint64_t(1) << b);
    for (std::uint64_t k = 1; k < 1845; ++k) around(k * 10000000000000000ull), around(k * 10000000000000000ull + 99999999);
    std::mt19937_64 rng(1);
    for (int i = 0; i < 2000000; ++i) {
        const std::uint64_t x = rng();
        v.push_back(x >> (rng() % 64));
        v.push_back((x % 1845) * 10000000000000000ull + (rng() % 2) * (rng() % 100000000) * 100000000);
    }
    while (v.size() % 8) v.push_back(0);
    return v;
}

}  // namespace

int main() {
    const std::vector<std::uint64_t> values = cases();
    std::vector<char> text(fields64::kTextBytes);
    fields64::prepare(text.data());
    std::size_t checked = 0;
    for (std::size_t i = 0; i < values.size(); i += fields64::kBlock) {
        const std::size_t n = std::min(fields64::kBlock, values.size() - i);
        const std::size_t bytes = fields64::format(text.data(), values.data() + i, n, true);
        if (bytes != 21 * n + 1 || text[bytes - 1] != '\n') return std::puts("FAIL: length"), 1;
        for (std::size_t j = 0; j < n; ++j) {
            char want[32];
            std::snprintf(want, sizeof want, " %20llu", static_cast<unsigned long long>(values[i + j]));
            if (std::memcmp(want, text.data() + 21 * j, 21) != 0) {
                std::printf("FAIL: %llu -> '%.21s'\n", static_cast<unsigned long long>(values[i + j]), text.data() + 21 * j);
                return 1;
            }
            ++checked;
        }
    }
    std::printf("PASS: %zu values\n", checked);
}
