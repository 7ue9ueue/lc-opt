// The roots in F_p of f, deg f = N <= 4000, p = 998244353 = 119 2^23 + 1, by tangent Graeffe
// transforms (Grenet, van der Hoeven and Lecerf). A round shifts f by a random t, maps the roots
// r - t to (r - t)^N for N = 2^k (graeffe.hpp): the roots in F_p land in the subgroup H of order
// (p - 1) / N, on which the result is evaluated. A simple zero z gives its root; zeros where two
// roots collide, or a multiple root sits, count as multiple. The found roots are divided out;
// rounds repeat with new shifts while multiple zeros remain, on f / gcd(f, f') once repeated
// roots are likely. Design and log: notes.md.
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

#include "lib/io/io.hpp"
#include "lib/poly/evaluation.hpp"
#include "lib/poly/gcd.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/run/early.hpp"
#include "problems/polynomial/polynomial_root_finding/graeffe.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using poly::Arena;
using poly::Transform;

constexpr u32 kP = poly::kModulus;

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
u32 add(u32 a, u32 b) { return a + b >= kP ? a + b - kP : a + b; }
u32 inverse(u32 a) { return ntt::detail::power(a, kP - 2); }

// 2^l for the subgroup of order 119 2^l a round evaluates on, for degree d: about 16 points per
// root (d + 1 <= 16 2^l, as Subgroup needs).
int subgroup_log(std::size_t d) { return std::max(0, int(std::bit_width(d)) - 4); }

class RootFinder {
public:
    explicit RootFinder(std::size_t n_max)
        : n_max_(n_max),
          arena_(words(n_max)),
          t_(arena_, log_max(n_max)),
          tree_t_(arena_, log_max(n_max)),
          graeffe_(t_, arena_, n_max),
          subgroup_(arena_, n_max) {
        const std::size_t m = shift_length(n_max), n = n_max + 1;
        f_ = arena_.take(n), g_ = arena_.take(n), b_ = arena_.take(n), x_ = arena_.take(n);
        fact_ = arena_.take(n), inv_fact_ = arena_.take(n);
        work_ = arena_.take(m), work2_ = arena_.take(m);
        found_ = arena_.take(n), found_b_ = arena_.take(n), prefix_ = arena_.take(n);
        const std::size_t m8 = (n_max + 7) / 8 * 8, count = m8 / 8;
        points_ = arena_.take(m8);
        lane_scratch_ = arena_.take(LaneTree::scratch_words(count, count));
        columns_ = arena_.take(8 * ((count + 8) / 8 * 8));
        top_scratch_ = arena_.take(TopTree::scratch_words(8, m8));
        inverse_scratch_ = arena_.take(poly::inverse_scratch(n));
        fact_[0] = 1;
        for (std::size_t i = 1; i < n; ++i) fact_[i] = mul(fact_[i - 1], u32(i));
        inv_fact_[n - 1] = inverse(fact_[n - 1]);
        for (std::size_t i = n - 1; i > 0; --i) inv_fact_[i - 1] = mul(inv_fact_[i], u32(i));
    }

    // The distinct roots of f (n coefficients, the last nonzero, n <= n_max + 1).
    void solve(std::span<const u32> f, std::vector<u32>& roots) {
        std::size_t zeros = 0;
        while (f[zeros] == 0) ++zeros;
        if (zeros) roots.push_back(0);
        d_ = f.size() - 1 - zeros;
        std::copy(f.begin() + std::ptrdiff_t(zeros), f.end(), f_.begin());
        bool squarefree = false;
        for (int round = 0; d_ > 1; ++round) {
            const u32 shift = next_shift();
            if (!shift_by(shift)) {  // f(shift) = 0
                roots.push_back(shift);
                while (horner(shift) == 0) divide_linear(shift);
                continue;
            }
            const std::size_t multiple = find(shift, roots);
            if (multiple == 0) return;
            deflate(std::span<const u32>(roots).last(found_count_));
            if (!squarefree && d_ > 1 && (round > 0 || likely_repeated(multiple))) {
                radical();
                squarefree = true;
            }
        }
        if (d_ == 1) roots.push_back(mul(kP - f_[0], inverse(f_[1])));
    }

private:
    using LaneTree = poly::ProductTree<poly::LaneLayout, poly::detail::PointSlots>;
    using TopTree = poly::ProductTree<poly::StandardLayout, poly::detail::LaneProducts>;

    static int log_max(std::size_t n_max) { return std::max(13, int(std::bit_width(2 * n_max + 1))); }
    static std::size_t shift_length(std::size_t d) { return std::max<std::size_t>(64, std::bit_ceil(2 * d + 1)); }

    static std::size_t words(std::size_t n_max) {
        const std::size_t len = roots::graeffe_length(n_max), n = n_max + 1, m8 = (n_max + 7) / 8 * 8, count = m8 / 8;
        const int lg = log_max(n_max), l = std::min(roots::Subgroup::kMaxLog, subgroup_log(n_max));
        const std::size_t sub = std::size_t(1) << l;
        std::size_t w = Transform::words(lg) + poly::TreeTransform::words(lg);
        w += 2 * Arena::footprint(2 * len) + 2 * Arena::footprint(len);                       // Graeffe
        w += 3 * Arena::footprint(n + 2 * sub + 8) + 3 * Arena::footprint(8 * sub) + 2 * Arena::footprint(2 * sub);  // Subgroup
        w += 9 * Arena::footprint(n) + 2 * Arena::footprint(shift_length(n_max));
        w += Arena::footprint(m8) + Arena::footprint(LaneTree::scratch_words(count, count)) +
             Arena::footprint(8 * ((count + 8) / 8 * 8)) + Arena::footprint(TopTree::scratch_words(8, m8)) +
             Arena::footprint(poly::inverse_scratch(n));
        w += poly::detail::HalfGcd::words(std::max<std::size_t>(n_max, 1)) + 4 * Arena::footprint(n);  // radical
        return w + 4096;
    }

    u32 next_shift() {
        state_ += 0x9E3779B97F4A7C15ull;
        u64 z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return u32((z ^ (z >> 31)) % (kP - 1)) + 1;
    }

    u32 horner(u32 x) const {
        u32 r = 0;
        for (std::size_t i = d_ + 1; i-- > 0;) r = add(mul(r, x), f_[i]);
        return r;
    }

    // f = f / (x - a) for a root a.
    void divide_linear(u32 a) {
        for (std::size_t i = d_; i > 0; --i) f_[i - 1] = add(f_[i - 1], mul(a, f_[i]));
        std::copy(f_.begin() + 1, f_.begin() + std::ptrdiff_t(d_) + 1, f_.begin());
        --d_;
    }

    // g = f(x + s) by one product: g_k k! = sum_i f_i i! s^(i-k) / (i-k)!. False if g(0) = 0.
    bool shift_by(u32 s) {
        const std::size_t d = d_, m = shift_length(d);
        const std::span<u32> x = work_.first(m), y = work2_.first(m);
        for (std::size_t i = 0; i <= d; ++i) x[i] = mul(f_[i], fact_[i]);
        u32 power = 1;
        for (std::size_t j = 0; j <= d; ++j, power = mul(power, s)) y[d - j] = mul(power, inv_fact_[j]);
        t_.forward(x.first(d + 1), 0, x);
        t_.cyclic_product(y.first(d + 1), 0, y, x);
        for (std::size_t k = 0; k <= d; ++k) g_[k] = mul(y[d + k], inv_fact_[k]);
        return g_[0] != 0;
    }

    // One round on g = f(x + s): appends the roots found to roots (found_count_ of them) and
    // returns the number of multiple zeros.
    std::size_t find(u32 s, std::vector<u32>& roots) {
        const std::size_t d = d_;
        const int l = subgroup_log(d), k = 23 - l;
        poly::derivative(g_.first(d + 1), b_.first(d));
        graeffe_.run(g_.first(d + 1), b_.first(d), k);
        for (std::size_t i = 0; i <= d; ++i) x_[i] = mul(g_[i], u32(i));
        std::size_t found = 0, multiple = 0;
        subgroup_.zeros(l, g_.first(d + 1), x_.first(d + 1), b_.first(d), [&](u32 x, u32 b) {
            if (x == 0) {
                ++multiple;
            } else {
                found_[found] = x, found_b_[found] = b, ++found;
            }
        });
        // root = x / b + s, the b inverted together: prefix_[i] = b_0 ... b_(i-1).
        u32 product = 1;
        for (std::size_t i = 0; i < found; ++i) prefix_[i] = product, product = mul(product, found_b_[i]);
        u32 inv = inverse(product);  // 1 / (b_0 ... b_(i-1)) in the loop
        for (std::size_t i = found; i-- > 0;) {
            roots.push_back(add(mul(found_[i], mul(inv, prefix_[i])), s));
            inv = mul(inv, found_b_[i]);
        }
        found_count_ = found;
        last_points_ = std::size_t(119) << l, last_found_ = found;
        return multiple;
    }

    bool likely_repeated(std::size_t multiple) const {
        // Collisions of distinct roots: about r^2 / (2 |H|) pairs for r roots in F_p.
        const double r = double(last_found_ + 2 * multiple);
        return double(multiple) > r * r / double(last_points_) + 0.5;
    }

    // f = f / prod (x - a) over the roots a, all simple.
    void deflate(std::span<const u32> found) {
        const std::size_t m = found.size();
        if (m == 0) return;
        if (m * d_ <= 4096) {
            for (u32 a : found) divide_linear(a);
            return;
        }
        const std::size_t d2 = d_ - m, n2 = d2 + 1;
        const std::span<const u32> q = root_product(found);  // prod (1 - a x) 2^32
        const std::span<u32> h = g_.first(n2), r = work_.first(shift_length(d2)), th = work2_.first(r.size());
        poly::inverse(t_, q.first(std::min(n2, q.size())), h, inverse_scratch_);
        for (std::size_t i = 0; i < n2; ++i) r[i] = f_[d_ - i];  // rev(f) mod x^n2
        t_.forward(h, 0, th);
        t_.cyclic_product(r.first(n2), 0, r, th, poly::Half::kBoth, poly::detail::kR);
        for (std::size_t i = 0; i < n2; ++i) f_[i] = r[d2 - i];
        d_ = d2;
    }

    // prod (1 - a x) over the points, padded with zeros to m8, in Montgomery form (m8 + 1 words).
    std::span<const u32> root_product(std::span<const u32> points) {
        const std::size_t m8 = (points.size() + 7) / 8 * 8, count = m8 / 8;
        std::copy(points.begin(), points.end(), points_.begin());
        std::fill(points_.begin() + std::ptrdiff_t(points.size()), points_.begin() + std::ptrdiff_t(m8), 0);
        const poly::detail::PointSlots slots{points_.data(), count};
        LaneTree lanes(tree_t_, slots, lane_scratch_);
        const LaneTree::Root lane_root = lanes.root();
        const std::size_t stride = (count + 8) / 8 * 8;
        const u32 size = u32(count + 1);
        const u32 sizes[8] = {size, size, size, size, size, size, size, size};
        u32* const out[8] = {columns_.data(),
                             columns_.data() + stride,
                             columns_.data() + 2 * stride,
                             columns_.data() + 3 * stride,
                             columns_.data() + 4 * stride,
                             columns_.data() + 5 * stride,
                             columns_.data() + 6 * stride,
                             columns_.data() + 7 * stride};
        poly::lane_columns(lane_root.coefficients.data(), lane_root.length + 1, sizes, out);
        const poly::detail::LaneProducts products{columns_.data(), stride, u32(count)};
        TopTree top(tree_t_, products, top_scratch_);
        return top.root().coefficients.first(m8 + 1);
    }

    // f = f / gcd(f, f'): the last row of the jump of deg f on (f, f') has (f / gcd) as entry 3.
    void radical() {
        const std::size_t d = d_;
        poly::derivative(f_.first(d + 1), b_.first(d));
        poly::detail::HalfGcd gcd(arena_, d);
        poly::detail::Matrix r = gcd.matrix(d, 0);
        gcd.jump(f_.data(), std::ptrdiff_t(d), b_.data(), std::ptrdiff_t(d) - 1, std::ptrdiff_t(d), r, 0x8);
        std::size_t size = r.size[3];
        while (size > 0 && r.entry[3][size - 1] == 0) --size;
        std::copy_n(r.entry[3], size, f_.data());
        d_ = size - 1;
    }

    std::size_t n_max_;
    Arena arena_;
    Transform t_;
    poly::TreeTransform tree_t_;
    roots::Graeffe graeffe_;
    roots::Subgroup subgroup_;
    std::span<u32> f_, g_, b_, x_, fact_, inv_fact_, work_, work2_, found_, found_b_, prefix_;
    std::span<u32> points_, lane_scratch_, columns_, top_scratch_, inverse_scratch_;
    std::size_t d_ = 0, found_count_ = 0, last_points_ = 1, last_found_ = 0;
    u64 state_ = 0x2545F4914F6CDD1Dull;
};

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>();
    std::vector<u32> f(n + 1);
    in.read(f.data(), n + 1);
    std::vector<u32> roots;
    roots.reserve(n + 1);
    if (n > 0) {
        RootFinder finder(n);
        finder.solve(f, roots);
    }
    io::Writer out;
    out.write(u32(roots.size()), '\n');
    out.write_array(roots.data(), roots.size(), ' ');
    out.write('\n');
}

}  // namespace

RUN_EARLY(solve)
