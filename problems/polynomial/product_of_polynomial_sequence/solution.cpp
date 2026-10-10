// The product of N polynomials mod 998244353, N <= 500000, degrees summing to D <= 500000.
// Constants multiply into a scalar. The other polynomials, by degree, largest first: those much
// larger than the rest ("big") go straight to the top tree; the rest go to 8 lanes, polynomial
// 8k + l to lane l, and one product tree computes the 8 lane products side by side
// (lib/poly/product_tree.hpp). The top tree multiplies the lane products and the big ones.
// Output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
// Memory: two huge-page arenas, no heap arrays (4 KiB page faults cost 0.43 ms per MB on lc-amd,
// huge pages 0.05).
#include <unistd.h>

#include <algorithm>
#include <array>
#include <span>
#include <string_view>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/poly/product_tree.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
namespace pd = poly::detail;

constexpr u32 kP = poly::kModulus;
constexpr std::size_t kMaxDegree = 500000;

// The product of residues, 64 at a time as 8 lanes of Montgomery products. Each lane carries
// 2^-32 per vector multiplied in, undone at the end.
class ConstantProduct {
public:
    void add(u32 x) {
        buffer_[size_++] = x;
        if (size_ == 64) flush();
    }

    u32 value() const {
        alignas(32) u32 lanes[8];
        pd::store(lanes, pd::reduce(lanes_, kP));
        u64 r = ntt::detail::power(pd::kR, u32(8 * vectors_ % (kP - 1)));
        for (u32 x : lanes) r = r * x % kP;
        for (std::size_t i = 0; i < size_; ++i) r = r * buffer_[i] % kP;
        return u32(r);
    }

private:
    void flush() {
        for (std::size_t v = 0; v < 64; v += 8) lanes_ = pd::montgomery(lanes_, pd::load(buffer_ + v));  // < 2P
        vectors_ += 8, size_ = 0;
    }

    alignas(32) u32 buffer_[64];
    std::size_t size_ = 0, vectors_ = 0;
    pd::Vec lanes_ = pd::broadcast(1);
};

constexpr std::size_t kBuckets = 256;  // degrees below this are counting-sorted

// The polynomials after N, as tokens: each one's degree d, then its d + 1 coefficients.
struct Polynomials {
    u32* tokens;
    u32* starts;                  // starts[0, small): where the nonconstant polynomials of degree < kBuckets start
    std::size_t small = 0, total = 0;  // total: sum of degrees
    std::array<u32, kBuckets> count{};  // count[d]: polynomials of degree d < kBuckets
    std::array<u32, kMaxDegree / kBuckets + 1> large;  // starts of those of degree >= kBuckets
    std::size_t large_count = 0;
    ConstantProduct constants;
};

// Reads in rounds, each up to a lower bound on the token count (2 per polynomial not yet
// located), and sorts the polynomials on the way, branch free but for large degrees.
void read_polynomials(io::Reader& in, std::size_t n, Polynomials& p) {
    u32* const tokens = p.tokens;
    std::size_t parsed = 0, need = 2 * n, at = 0, i = 0, small = 0, total = 0;
    while (need > parsed) {
        io::read_bulk(in, tokens + parsed, need - parsed);
        parsed = need;
        for (; i < n && at + 1 < parsed; ++i) {  // degree and first coefficient parsed
            const u32 d = tokens[at], constant = 0u - (d == 0);  // all ones for a constant
            p.constants.add((tokens[at + 1] & constant) | (1 & ~constant));  // a select GCC made a branch
            total += d;
            if (d >= kBuckets) [[unlikely]] {
                p.large[p.large_count++] = u32(at);
            } else {
                p.starts[small] = u32(at), small += d != 0;
                ++p.count[d];
            }
            at += d + 2;
        }
        need = at + 2 * (n - i);
    }
    p.small = small, p.total = total;
}

// a[0 .. n) to Montgomery form in place, any alignment.
void montgomery_form(u32* a, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) pd::store_unaligned(a + i, pd::to_montgomery(pd::load_unaligned(a + i)));
    for (; i < n; ++i) a[i] = u32((u64(a[i]) << 32) % kP);
}

// Lanes: slot k holds polynomials order[8k .. 8k + 8) (input coefficients at tokens[start + 1]),
// lanes past the end the constant 1.
struct Slots {
    const u32* tokens;
    const u32* order;
    std::size_t size;

    std::size_t count() const { return (size + 7) / 8; }
    u32 degree(std::size_t k) const { return tokens[order[8 * k]]; }

    poly::LaneLayout::Node load(std::size_t k, u32* c) const {
        const std::size_t d = degree(k);
        if (8 * k + 8 <= size && tokens[order[8 * k + 7]] == d) return d == 1 ? load_linear(order + 8 * k, c) : load_uniform(order + 8 * k, d, c);
        std::fill_n(c, 8 * (d + 1), 0);
        alignas(32) u32 degrees[8], leads[8];
        for (std::size_t l = 0; l < 8; ++l) {
            if (8 * k + l >= size) {
                c[l] = 1, degrees[l] = 0, leads[l] = 1;
                continue;
            }
            const u32* p = tokens + order[8 * k + l];
            degrees[l] = p[0], leads[l] = p[p[0] + 1];
            for (std::size_t j = 0; j <= p[0]; ++j) c[8 * j + l] = p[j + 1];
        }
        for (std::size_t j = 0; j <= d; ++j) pd::store(c + 8 * j, pd::to_montgomery(pd::load(c + 8 * j)));
        return {pd::load(degrees), pd::to_montgomery(pd::load(leads))};
    }

    // 8 polynomials of degree d: coefficient j of each gathered.
    poly::LaneLayout::Node load_uniform(const u32* at, std::size_t d, u32* c) const {
        const pd::Vec index = _mm256_add_epi32(pd::load_unaligned(at), pd::broadcast(1));
        const auto* base = reinterpret_cast<const int*>(tokens);
        pd::Vec v{};
        for (std::size_t j = 0; j <= d; ++j) {
            v = pd::to_montgomery(_mm256_i32gather_epi32(base, _mm256_add_epi32(index, pd::broadcast(u32(j))), 4));
            pd::store(c + 8 * j, v);
        }
        return {pd::broadcast(u32(d)), v};
    }

    // 8 polynomials of degree 1: their coefficient pairs gathered as qwords, then even and odd words.
    poly::LaneLayout::Node load_linear(const u32* at, u32* c) const {
        const pd::Vec index = _mm256_add_epi32(pd::load_unaligned(at), pd::broadcast(1));
        const auto* base = reinterpret_cast<const long long*>(tokens);
        const __m256 lo = _mm256_castsi256_ps(_mm256_i32gather_epi64(base, _mm256_castsi256_si128(index), 4));
        const __m256 hi = _mm256_castsi256_ps(_mm256_i32gather_epi64(base, _mm256_extracti128_si256(index, 1), 4));
        // shuffle_ps gives qwords (0 1), (4 5), (2 3), (6 7) of the lanes: put them in order.
        const pd::Vec c0 = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(lo, hi, 0x88)), 0xD8);
        const pd::Vec c1 = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(lo, hi, 0xDD)), 0xD8);
        const pd::Vec m1 = pd::to_montgomery(c1);
        pd::store(c, pd::to_montgomery(c0));
        pd::store(c + 8, m1);
        return {pd::broadcast(1), m1};
    }
};

// A polynomial of the top tree: coefficients in Montgomery form.
struct Item {
    const u32* c;
    u32 degree;
};

struct Items {
    std::span<const Item> items;

    std::size_t count() const { return items.size(); }
    u32 degree(std::size_t k) const { return items[k].degree; }

    poly::StandardLayout::Node load(std::size_t k, u32* c) const {
        const Item& item = items[k];
        std::copy_n(item.c, item.degree + 1, c);
        return {item.degree, item.c[item.degree]};
    }
};

using LaneTree = poly::ProductTree<poly::LaneLayout, Slots>;
using TopTree = poly::ProductTree<poly::StandardLayout, Items>;

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>();
    poly::Arena input(poly::Arena::footprint(2 * n + kMaxDegree + 64) + 2 * poly::Arena::footprint(n + 1));
    Polynomials polys;
    u32* const tokens = polys.tokens = input.take(2 * n + kMaxDegree + 64).data();
    u32* const starts = polys.starts = input.take(n + 1).data();
    read_polynomials(in, n, polys);
    const std::size_t m = polys.small, large_count = polys.large_count, total = polys.total;
    auto& count = polys.count;
    auto& large = polys.large;
    const u32 scalar = polys.constants.value();

    // order: the nonconstant polynomials by degree, largest first (counting sort below kBuckets).
    std::sort(large.begin(), large.begin() + std::ptrdiff_t(large_count), [&](u32 x, u32 y) { return tokens[x] > tokens[y]; });
    const std::size_t nonconstant = m + large_count;
    u32* order = starts;
    if (large_count || std::count_if(count.begin() + 1, count.end(), [](u32 c) { return c != 0; }) > 1) {
        order = input.take(n + 1).data();
        std::copy_n(large.begin(), large_count, order);
        for (std::size_t d = kBuckets - 1, at = large_count; d >= 1; --d) at += count[d], count[d] = u32(at - count[d]);
        for (std::size_t j = 0; j < m; ++j) order[count[tokens[starts[j]]]++] = starts[j];
    }

    // Big polynomials: while one's degree exceeds 1/64 of the rest's. Few small ones join them.
    std::size_t big = 0, rest = total;
    for (; big < nonconstant; ++big) {
        const u32 d = tokens[order[big]];
        if (u64(d) * 64 <= rest - d) break;
        rest -= d;
    }
    if (nonconstant - big < 16) big = nonconstant;

    const Slots slots{tokens, order + big, nonconstant - big};
    std::size_t lane_total = 0;  // lane 0's degree, the largest
    for (std::size_t k = 0; k < slots.count(); ++k) lane_total += slots.degree(k);
    const std::size_t lane_length = poly::LaneLayout::length(u32(lane_total)), lane_words = slots.size ? 8 * (lane_length + 1) : 0;
    const std::size_t items_count = big + 8;
    const int lg = std::max(std::countr_zero(lane_length) + 3, std::countr_zero(poly::StandardLayout::length(u32(total))));
    const std::size_t scratch = std::max(slots.size ? LaneTree::scratch_words(slots.count(), lane_total) : 0,
                                         TopTree::scratch_words(items_count, total));
    poly::Arena arena(poly::TreeTransform::words(lg) + poly::Arena::footprint(scratch) + poly::Arena::footprint(lane_words + 64));
    const poly::TreeTransform t(arena, lg);
    const std::span<u32> memory = arena.take(scratch);

    std::vector<Item> items;
    for (std::size_t i = 0; i < big; ++i) {
        u32* const c = tokens + order[i] + 1;
        montgomery_form(c, c[-1] + 1);
        items.push_back({c, c[-1]});
    }
    if (slots.size) {
        LaneTree lanes(t, slots, memory);
        const LaneTree::Root root = lanes.root();
        alignas(32) u32 degrees[8], sizes[8];
        pd::store(degrees, root.node.degree);
        u32* columns[8];
        u32* const lane_polys = arena.take(lane_words + 64).data();
        for (std::size_t l = 0, at = 0; l < 8; ++l) {
            columns[l] = lane_polys + at;
            sizes[l] = degrees[l] ? degrees[l] + 1 : 0;  // lanes of constants 1 are left out
            if (degrees[l]) items.push_back({columns[l], degrees[l]}), at += (degrees[l] + 8) / 8 * 8;
        }
        poly::lane_columns(root.coefficients.data(), root.length + 1, sizes, columns);
    }
    std::stable_sort(items.begin(), items.end(), [](const Item& x, const Item& y) { return x.degree > y.degree; });

    // The product's coefficients times scalar, from Montgomery form.
    std::span<const u32> product{&scalar, 1};
    if (!items.empty()) {
        const Items leaves{items};
        TopTree top(t, leaves, memory);
        const TopTree::Root root = top.root();
        u32* const c = root.coefficients.data();
        const pd::Factor f(pd::montgomery_scalar(scalar, 1));
        for (std::size_t i = 0; i <= total; i += 8) pd::store(c + i, pd::reduce(pd::times(pd::load(c + i), f), kP));
        product = {c, total + 1};
    }
    io::Writer out;
    fields::write(out, product.data(), product.size(), reinterpret_cast<char*>(tokens));  // tokens are no longer needed
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here). _exit skips their teardown too.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
