// Luogu P5408 第一类斯特林数·行: [n, i] for all i <= n, 1 <= n < 2^18, mod 167772161, 500 ms.
// [n, i] = [x^i] x (x + 1) ... (x + n - 1). The intended solution doubles the product with a
// Taylor shift, O(n log n). This one multiplies the n linear factors by divide and conquer,
// O(n log^2 n): the two-log route, which forum threads on the problem say needs heavy constant
// tuning to fit. The modulus is not 998244353, so products run on lib/multimod's run-time prime.
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "lib/easy/io.hpp"

// Every standard header comes before the target pragma: GCC 13 and 14 fail to inline
// std::allocator's members into code compiled under it otherwise.
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include "lib/multimod/transform.hpp"

namespace {

using Poly = std::vector<std::uint32_t>;

constexpr std::uint32_t kMod = 167772161;  // 5 2^25 + 1, primitive root 3
constexpr int kMaxLog = 18;                // the full product has n + 1 <= 2^18 coefficients
constexpr std::size_t kSchoolbook = 32;    // factors this short multiply directly

// a b mod kMod; n + m - 1 <= 2^kMaxLog, both nonempty.
class Multiplier {
public:
    Multiplier() : mod_(kMod, 3), buffers_(std::make_unique<Buffers>()) {}

    Poly operator()(const Poly& a, const Poly& b) {
        const std::size_t n = a.size(), m = b.size();
        if (std::min(n, m) <= kSchoolbook) return schoolbook(a, b);
        const int lg = std::max(6, int(std::bit_width(n + m - 2)));
        const std::size_t len = std::size_t(1) << lg;
        Buffers& s = *buffers_;
        load(a, s.a, len);
        load(b, s.b, len);
        const multimod::Transform transform(lg, s.tables);
        transform.multiply(multimod::Padded{s.a, n}, multimod::Padded{s.b, m}, s.out, s.b, mod_, 1);
        return Poly(s.out, s.out + n + m - 1);
    }

private:
    static constexpr std::size_t kWords = (std::size_t(1) << kMaxLog) + multimod::Transform::kPadding;

    struct alignas(64) Buffers {
        std::uint32_t a[kWords], b[kWords], out[kWords];
        std::uint32_t tables[2 * ((std::size_t(1) << kMaxLog) / 8 + 16)];  // Transform::table_words
    };

    // Each output sums at most kSchoolbook products below 2^56: no overflow before the reduction.
    static Poly schoolbook(const Poly& a, const Poly& b) {
        std::vector<std::uint64_t> sum(a.size() + b.size() - 1);
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size(); ++j) sum[i + j] += std::uint64_t(a[i]) * b[j];
        Poly c(sum.size());
        for (std::size_t i = 0; i < c.size(); ++i) c[i] = std::uint32_t(sum[i] % kMod);
        return c;
    }

    // f into x, zero up to max(|f| rounded up to 8, len / 2) (multimod::Padded).
    static void load(const Poly& f, std::uint32_t* x, std::size_t len) {
        std::copy(f.begin(), f.end(), x);
        const std::size_t end = std::max((f.size() + 7) & ~std::size_t(7), len / 2);
        std::fill(x + f.size(), x + end, 0u);
    }

    multimod::Modulus mod_;
    std::unique_ptr<Buffers> buffers_;
};

// (x + lo) (x + lo + 1) ... (x + hi - 1), lo < hi.
Poly rising(std::uint32_t lo, std::uint32_t hi, Multiplier& multiply) {
    if (hi - lo <= kSchoolbook) {
        Poly f = {1};
        for (std::uint32_t i = lo; i < hi; ++i) {  // f <- f (x + i)
            f.push_back(0);
            for (std::size_t k = f.size() - 1; k > 0; --k) f[k] = std::uint32_t((f[k - 1] + std::uint64_t(i) * f[k]) % kMod);
            f[0] = std::uint32_t(std::uint64_t(i) * f[0] % kMod);
        }
        return f;
    }
    const std::uint32_t mid = lo + (hi - lo) / 2;
    return multiply(rising(lo, mid, multiply), rising(mid, hi, multiply));
}

}  // namespace

int main() {
    easy::Reader in;
    const auto n = in.read<std::uint32_t>();
    Multiplier multiply;
    const Poly f = rising(0, n, multiply);
    easy::Writer out;
    for (std::uint32_t i = 0; i <= n; ++i) out.write(f[i], i == n ? '\n' : ' ');
}
