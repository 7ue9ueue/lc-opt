// Tests for lib/multimod: products for every input kind, transform length and top-level path,
// against schoolbook cyclic convolution (short) or evaluation at a root of unity (long).
#include "lib/multimod/transform.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kPrimes[][2] = {{998244353, 3}, {985661441, 3}, {976224257, 3}, {975175681, 17}, {972029953, 10},
                              {754974721, 11}, {469762049, 3}};

std::mt19937_64 rng(2026);
int failures = 0;

void expect(bool ok, const char* what, int lg, std::size_t n, std::size_t m) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s, lg %d, n %zu, m %zu\n", what, lg, n, m);
}

// Words with 32-byte alignment.
template <class T>
struct Buffer {
    explicit Buffer(std::size_t count) : data(static_cast<T*>(std::aligned_alloc(32, (count * sizeof(T) + 31) & ~31))) {
        std::fill(data, data + count, T(0));
    }
    ~Buffer() { std::free(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data;
};

enum class Kind { Padded, Bounded, Wide };

// Coefficients reduced mod p, and the expected product factor a b mod (x^len - 1, p) at check
// points: every point for short lengths, else 8 random len-th roots of unity.
struct Check {
    const multimod::Modulus& m;
    int lg;
    u32 root;  // of order 2^lg
    std::vector<u32> a, b;

    u32 eval(const std::vector<u32>& f, u32 x) const {
        u32 r = 0;
        for (std::size_t i = f.size(); i-- > 0;) r = u32((u64(r) * x + f[i]) % m.p);
        return r;
    }

    bool matches(const u32* out, u32 factor) const {
        const std::size_t len = std::size_t(1) << lg;
        for (std::size_t i = 0; i < len; ++i)
            if (out[i] >= m.p) return false;
        if (lg <= 11) {
            std::vector<u32> c(len);
            for (std::size_t i = 0; i < a.size(); ++i)
                for (std::size_t j = 0; j < b.size(); ++j) {
                    const std::size_t k = (i + j) & (len - 1);
                    c[k] = u32((c[k] + u64(a[i]) * b[j]) % m.p);
                }
            for (std::size_t i = 0; i < len; ++i)
                if (out[i] != m.multiply(c[i], factor)) return false;
            return true;
        }
        const std::vector<u32> c(out, out + len);
        for (int t = 0; t < 8; ++t) {
            const u32 x = m.power(root, u32(rng() % len));
            if (eval(c, x) != m.multiply(m.multiply(eval(a, x), eval(b, x)), factor)) return false;
        }
        return true;
    }
};

// Random inputs of n and m coefficients for the given kind, then one product. Boundary values
// (2p - 1, 2^64 - 1) lead the inputs.
void test(const multimod::Modulus& mod, u32 generator, int lg, std::size_t n, std::size_t m, Kind kind, bool in_place) {
    const std::size_t len = std::size_t(1) << lg, words = len + multimod::Transform::kPadding;
    const auto extent = [&](std::size_t count) { return std::max((count + 7) & ~std::size_t(7), len / 2); };
    Buffer<u32> tables(multimod::Transform::table_words(lg)), out(words), work(words);
    Buffer<u32> a32(extent(n)), b32(std::max(extent(m), words));
    Buffer<u64> a64(std::max(extent(n), words / 2)), b64(extent(m));
    Check check{mod, lg, mod.power(generator, (mod.p - 1) >> lg), std::vector<u32>(n), std::vector<u32>(m)};
    const auto fill = [&](u32* x32, u64* x64, std::vector<u32>& reduced) {
        for (std::size_t i = 0; i < reduced.size(); ++i) {
            if (kind == Kind::Wide) {
                x64[i] = i < 2 ? ~u64(i) : rng();
                reduced[i] = u32(x64[i] % mod.p);
            } else {
                x32[i] = i < 2 ? 2 * mod.p - 1 - u32(i) : u32(rng() % (2 * mod.p));
                reduced[i] = x32[i] % mod.p;
            }
        }
    };
    fill(a32.data, a64.data, check.a);
    fill(b32.data, b64.data, check.b);
    const u32 factor = u32(rng() % mod.p);
    const multimod::Transform transform(lg, tables.data);
    // In place: work is b's storage (a's for Wide).
    switch (kind) {
    case Kind::Padded:
        transform.multiply(multimod::Padded{a32.data, n}, multimod::Padded{b32.data, m}, out.data,
                           in_place ? b32.data : work.data, mod, factor);
        break;
    case Kind::Bounded: {
        // Exactly n and m words, so a sanitizer sees any read past them.
        std::vector<u32> a(a32.data, a32.data + n), b(b32.data, b32.data + m);
        transform.multiply(multimod::Bounded{a.data(), n}, multimod::Bounded{b.data(), m}, out.data, work.data, mod,
                           factor);
        break;
    }
    case Kind::Wide:
        transform.multiply(multimod::Wide{a64.data, n}, multimod::Wide{b64.data, m}, out.data,
                           in_place ? reinterpret_cast<u32*>(a64.data) : work.data, mod, factor);
        break;
    }
    static const char* const names[] = {"Padded", "Bounded", "Wide"};
    expect(check.matches(out.data, factor), names[int(kind)], lg, n, m);
}

}  // namespace

int main() {
    std::vector<multimod::Modulus> moduli;
    for (const auto& [p, g] : kPrimes) moduli.emplace_back(p, g);
    const auto generator = [&](std::size_t k) { return kPrimes[k][1]; };
    for (int lg = 6; lg <= multimod::kMaxLog; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        // Both halves sparse (the radix-8 top for odd log of len / 8), one dense, both dense,
        // counts not a multiple of 8, a single coefficient.
        const std::size_t sizes[][2] = {{len / 2, len / 2}, {len / 2 - 3, 5}, {len - 9, 9}, {len, len}, {1, len / 2 + 1}};
        for (int kind = 0; kind < 3; ++kind)
            for (const auto& [n, m] : sizes) {
                // Wide needs p > 2^29.
                std::size_t k = rng() % moduli.size();
                while (kind == int(Kind::Wide) && moduli[k].p < (u32(1) << 29)) k = rng() % moduli.size();
                test(moduli[k], generator(k), lg, n, m, Kind(kind), false);
                if (kind != int(Kind::Bounded) && lg <= 12) test(moduli[k], generator(k), lg, n, m, Kind(kind), true);
            }
    }
    for (std::size_t k = 0; k < moduli.size(); ++k) test(moduli[k], generator(k), 10, 513, 511, Kind::Padded, true);
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
