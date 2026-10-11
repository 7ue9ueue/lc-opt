// CodeChef POLYEVAL Evaluate the Polynomial, N, Q <= 2.5e5, 3 s. Values of a degree-N polynomial at
// Q points modulo 786433 = 3 2^18 + 1. The intended solution evaluates at all 786433 residues with
// an NTT of mixed radix 3 and 2, O(p log p). This one is general multipoint evaluation by the
// subproduct tree and the remainder tree (f mod prod (x - x_i), down to blocks of 64 points, then
// Horner), O(n log^2 n); each division runs its own Newton inverse. A contestant reported this
// route at 12 s on the worst case.
//
// 786433 is itself an NTT prime (2^18 | p - 1, primitive root 10), so every product is one cyclic
// convolution modulo 786433: lib/multimod's transform with the modulus chosen at run time. lib/easy
// works modulo 998244353 only.
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "lib/easy/io.hpp"

#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include "lib/multimod/transform.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Poly = std::vector<u32>;
using Span = std::span<const u32>;

constexpr u32 kMod = 786433;  // 3 2^18 + 1
constexpr u32 kRoot = 10;     // primitive root mod kMod
constexpr std::size_t kNaive = 32;  // schoolbook when a factor has at most this many terms
constexpr std::size_t kLeaf = 64;   // Horner on blocks of at most this many points
constexpr int kMinLog = 6;          // shortest multimod transform

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kMod); }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + kMod - b; }

u32 power(u32 a, u32 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

// Products < 2^40, so 64-bit sums of up to 2^24 of them do not overflow.
Poly multiply_naive(Span a, Span b) {
    std::vector<u64> sum(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) sum[i + j] += u64(a[i]) * b[j];
    Poly c(sum.size());
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = u32(sum[i] % kMod);
    return c;
}

// Cyclic convolutions modulo kMod on multimod::Transform, with buffers for the longest so far.
class Convolver {
public:
    // a b mod (x^(2^lg) - 1); a.size(), b.size() <= 2^lg, lg >= kMinLog. Valid until the next call.
    Span cyclic(Span a, Span b, int lg) {
        reserve(lg);
        const multimod::Transform transform(lg, words(tables_));
        transform.multiply(multimod::Bounded{a.data(), a.size()}, multimod::Bounded{b.data(), b.size()}, words(out_),
                           words(work_), modulus_, 1);
        return {words(out_), std::size_t(1) << lg};
    }

private:
    using Buffer = std::unique_ptr<multimod::Vec[]>;

    static Buffer vectors(std::size_t words) { return std::make_unique<multimod::Vec[]>((words + 7) / 8); }
    static u32* words(const Buffer& b) { return reinterpret_cast<u32*>(b.get()); }

    void reserve(int lg) {
        if (lg <= lg_) return;
        const std::size_t len = std::size_t(1) << lg;
        tables_ = vectors(multimod::Transform::table_words(lg));
        out_ = vectors(len + multimod::Transform::kPadding);
        work_ = vectors(len + multimod::Transform::kPadding);
        lg_ = lg;
    }

    multimod::Modulus modulus_{kMod, kRoot};
    int lg_ = 0;
    Buffer tables_, out_, work_;
};

Convolver convolver;

// a mod (x^len - 1).
Poly fold(Span a, std::size_t len) {
    if (a.size() <= len) return Poly(a.begin(), a.end());
    Poly r(a.begin(), a.begin() + std::ptrdiff_t(len));
    for (std::size_t i = len; i < a.size(); ++i) r[i % len] = (r[i % len] + a[i]) % kMod;
    return r;
}

// a b mod (x^(2^lg) - 1), all 2^lg terms.
Poly cyclic(Span a, Span b, int lg) {
    const std::size_t len = std::size_t(1) << lg;
    if (a.empty() || b.empty()) return Poly(len);
    const Poly fa = fold(a, len), fb = fold(b, len);
    if (lg < kMinLog || std::min(fa.size(), fb.size()) <= kNaive) {
        Poly c = fold(multiply_naive(fa, fb), len);
        c.resize(len);
        return c;
    }
    const Span c = convolver.cyclic(fa, fb, lg);
    return Poly(c.begin(), c.end());
}

Poly multiply(Span a, Span b) {
    if (a.empty() || b.empty()) return {};
    const std::size_t n = a.size() + b.size() - 1;
    if (std::min(a.size(), b.size()) <= kNaive) return multiply_naive(a, b);
    Poly c = cyclic(a, b, std::max(kMinLog, int(std::bit_width(n - 1))));
    c.resize(n);
    return c;
}

// 1 / f mod x^n, f[0] != 0. Newton: f g = 1 + x^k h mod x^(2k), then g <- g - x^k (g h mod x^k).
Poly inverse(Span f, std::size_t n) {
    Poly g = {power(f[0], kMod - 2)};
    for (std::size_t k = 1; k < n; k *= 2) {
        // deg(f g) <= 3k - 2: modulo x^(2k) - 1 the wrap lands below k, so h is exact.
        const Poly e = cyclic(f.first(std::min(f.size(), 2 * k)), g, std::countr_zero(2 * k));
        const Poly d = multiply(g, Span(e).subspan(k));
        g.resize(2 * k);
        for (std::size_t i = 0; i < k; ++i) g[k + i] = sub(0, d[i]);
    }
    g.resize(n);
    return g;
}

// r mod d for monic d of degree >= 1.
Poly remainder(Span r, Span d) {
    const std::size_t n = r.size(), m = d.size();
    if (n < m) return Poly(r.begin(), r.end());
    // Quotient by reversal: rev(q) = rev(r) / rev(d) mod x^k.
    const std::size_t k = n - m + 1;
    const Poly rev_d(d.rbegin(), d.rbegin() + std::ptrdiff_t(std::min(k, m)));
    const Poly rev_r(r.rbegin(), r.rbegin() + std::ptrdiff_t(k));
    Poly q = multiply(rev_r, inverse(rev_d, k));
    q.resize(k);
    std::ranges::reverse(q);
    // r - q d has degree < m - 1, so it equals its residue mod x^len - 1 for len >= m - 1.
    const int lg = int(std::bit_width(std::max<std::size_t>(m - 2, 1)));
    const Poly qd = cyclic(q, d, lg), fr = fold(r, std::size_t(1) << lg);
    Poly rem(m - 1);
    for (std::size_t i = 0; i < m - 1; ++i) rem[i] = sub(i < fr.size() ? fr[i] : 0, qd[i]);
    return rem;
}

// Multipoint evaluation by the subproduct tree (node polynomials prod (x - x_i) over a range of
// points, heap-indexed) and the remainder tree.
class Evaluator {
public:
    explicit Evaluator(Span points) : points_(points), tree_(8 * (points.size() / kLeaf + 1)) {
        if (!points.empty()) build(1, 0, points.size());
    }

    std::vector<u32> evaluate(Span f) const {
        std::vector<u32> values(points_.size());
        if (!points_.empty()) descend(1, 0, points_.size(), remainder(f, tree_[1]), values);
        return values;
    }

private:
    void build(std::size_t node, std::size_t lo, std::size_t hi) {
        if (hi - lo <= kLeaf) {
            Poly p = {1};
            for (std::size_t i = lo; i < hi; ++i) {  // p <- p (x - x_i)
                p.push_back(0);
                for (std::size_t j = p.size() - 1; j > 0; --j) p[j] = sub(p[j - 1], mul(p[j], points_[i]));
                p[0] = sub(0, mul(p[0], points_[i]));
            }
            tree_[node] = std::move(p);
            return;
        }
        const std::size_t mid = lo + (hi - lo) / 2;
        build(2 * node, lo, mid);
        build(2 * node + 1, mid, hi);
        tree_[node] = multiply(tree_[2 * node], tree_[2 * node + 1]);
    }

    // r = f mod tree_[node].
    void descend(std::size_t node, std::size_t lo, std::size_t hi, Poly r, std::vector<u32>& values) const {
        if (hi - lo <= kLeaf) {
            for (std::size_t i = lo; i < hi; ++i) {
                u32 v = 0;
                for (std::size_t j = r.size(); j-- > 0;) v = u32((u64(v) * points_[i] + r[j]) % kMod);
                values[i] = v;
            }
            return;
        }
        const std::size_t mid = lo + (hi - lo) / 2;
        Poly left = remainder(r, tree_[2 * node]), right = remainder(r, tree_[2 * node + 1]);
        r = Poly();
        descend(2 * node, lo, mid, std::move(left), values);
        descend(2 * node + 1, mid, hi, std::move(right), values);
    }

    Span points_;
    std::vector<Poly> tree_;
};

}  // namespace

int main() {
    easy::Reader in;
    Poly f(in.read<u32>() + 1);
    for (auto& c : f) c = in.read<u32>();
    Poly points(in.read<u32>());
    for (auto& x : points) x = in.read<u32>();
    const std::vector<u32> values = Evaluator(points).evaluate(f);
    easy::Writer out;
    for (const u32 v : values) out.write(v, '\n');
}
