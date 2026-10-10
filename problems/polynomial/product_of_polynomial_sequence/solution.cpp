// The product of N polynomials mod 998244353, N <= 500000, degrees summing to D <= 500000.
// Constants multiply into a scalar. The other polynomials, by degree, largest first: those much
// larger than the rest ("big") go straight to the top tree; the rest go to 8 lanes, polynomial
// 8k + l to lane l, and one product tree computes the 8 lane products side by side
// (lib/poly/product_tree.hpp). The top tree multiplies the lane products and the big ones.
// Output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
#include <unistd.h>

#include <algorithm>
#include <span>
#include <string_view>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/poly/product_tree.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using poly::detail::Vec;

constexpr u32 kP = poly::kModulus;
constexpr std::size_t kMaxDegree = 500000;

// The tokens after N: for each polynomial its degree d, then its d + 1 coefficients. Reads in
// rounds, each up to a lower bound on the token count (2 per polynomial not yet located).
// starts[i] = the index of polynomial i's degree. Returns the token count.
std::size_t read_polynomials(io::Reader& in, std::size_t n, u32* tokens, u32* starts) {
    std::size_t parsed = 0, need = 2 * n, at = 0, i = 0;
    while (need > parsed) {
        io::read_bulk(in, tokens + parsed, need - parsed);
        parsed = need;
        for (; i < n && at < parsed; ++i) starts[i] = u32(at), at += tokens[at] + 2;
        need = at + 2 * (n - i);
    }
    return parsed;
}

// a[0 .. n) to Montgomery form in place, any alignment.
void montgomery_form(u32* a, std::size_t n) {
    namespace d = poly::detail;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) d::store_unaligned(a + i, d::to_montgomery(d::load_unaligned(a + i)));
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

    poly::ProductTree<poly::LaneLayout, Slots>::Node load(std::size_t k, u32* c) const {
        namespace v = poly::detail;
        const std::size_t d = degree(k);
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
        for (std::size_t j = 0; j <= d; ++j) v::store(c + 8 * j, v::to_montgomery(v::load(c + 8 * j)));
        return {v::load(degrees), v::to_montgomery(v::load(leads))};
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

    poly::ProductTree<poly::StandardLayout, Items>::Node load(std::size_t k, u32* c) const {
        const Item& item = items[k];
        std::copy_n(item.c, item.degree + 1, c);
        return {item.degree, item.c[item.degree]};
    }
};

// Product of the constants c[0 .. n) mod P, four chains.
u32 product(const std::vector<u32>& c) {
    u64 r[4] = {1, 1, 1, 1};
    std::size_t i = 0;
    for (; i + 4 <= c.size(); i += 4)
        for (int j = 0; j < 4; ++j) r[j] = r[j] * c[i + j] % kP;
    for (; i < c.size(); ++i) r[0] = r[0] * c[i] % kP;
    return u32(r[0] * r[1] % kP * (r[2] * r[3] % kP) % kP);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>();
    poly::Arena input(poly::Arena::footprint(2 * n + kMaxDegree + 64) + 2 * poly::Arena::footprint(n + 1));
    u32* const tokens = input.take(2 * n + kMaxDegree + 64).data();
    u32* const starts = input.take(n + 1).data();
    read_polynomials(in, n, tokens, starts);

    // Constants; the others by degree, largest first (counting sort).
    std::vector<u32> constants;
    std::size_t total = 0, largest = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const u32 d = tokens[starts[i]];
        if (d == 0) constants.push_back(tokens[starts[i] + 1]);
        total += d, largest = std::max<std::size_t>(largest, d);
    }
    std::vector<u32> count(largest + 1);
    for (std::size_t i = 0; i < n; ++i) ++count[tokens[starts[i]]];
    for (std::size_t d = largest, sum = 0; d >= 1; --d) sum += count[d], count[d] = u32(sum - count[d]);
    u32* const order = input.take(n + 1).data();
    const std::size_t nonconstant = n - constants.size();
    for (std::size_t i = 0; i < n; ++i)
        if (const u32 d = tokens[starts[i]]) order[count[d]++] = starts[i];
    const u32 scalar = product(constants);

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
    const int lg = std::max(std::countr_zero(lane_length) + 3, std::countr_zero(poly::StandardLayout::length(u32(total))));
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(u32);
    using LaneTree = poly::ProductTree<poly::LaneLayout, Slots>;
    using TopTree = poly::ProductTree<poly::StandardLayout, Items>;
    const std::size_t lane_scratch = slots.size ? LaneTree::scratch_words(lane_total) : 0;
    poly::Arena arena(poly::TreeTransform::words(lg) + poly::Arena::footprint(lane_scratch) +
                      poly::Arena::footprint(lane_words + 64) + poly::Arena::footprint(TopTree::scratch_words(total)) +
                      poly::Arena::footprint(kTextWords));
    const poly::TreeTransform t(arena, lg);

    std::vector<Item> items;
    for (std::size_t i = 0; i < big; ++i) {
        u32* const c = tokens + order[i] + 1;
        montgomery_form(c, c[-1] + 1);
        items.push_back({c, c[-1]});
    }
    if (slots.size) {
        LaneTree lanes(t, slots, arena.take(lane_scratch));
        const LaneTree::Root root = lanes.root();
        alignas(32) u32 degrees[8], sizes[8];
        poly::detail::store(degrees, root.node.degree);
        u32* columns[8];
        u32* const memory = arena.take(lane_words + 64).data();
        for (std::size_t l = 0, at = 0; l < 8; ++l) {
            columns[l] = memory + at;
            sizes[l] = degrees[l] ? degrees[l] + 1 : 0;  // lanes of constants 1 are left out
            if (degrees[l]) items.push_back({columns[l], degrees[l]}), at += (degrees[l] + 8) / 8 * 8;
        }
        poly::lane_columns(root.coefficients.data(), root.length + 1, sizes, columns);
    }
    std::stable_sort(items.begin(), items.end(), [](const Item& x, const Item& y) { return x.degree > y.degree; });

    // The product's coefficients times scalar, from Montgomery form.
    const u32 factor = poly::detail::montgomery_scalar(scalar, 1);
    std::span<const u32> product{&scalar, 1};
    std::vector<u32> single;
    if (!items.empty()) {
        const Items leaves{items};
        TopTree top(t, leaves, arena.take(TopTree::scratch_words(total)));
        const TopTree::Root root = top.root();
        u32* const c = root.coefficients.data();
        const poly::detail::Factor f(factor);
        for (std::size_t i = 0; i <= total; i += 8)
            poly::detail::store(c + i, poly::detail::reduce(poly::detail::times(poly::detail::load(c + i), f), kP));
        product = {c, total + 1};
    }
    io::Writer out;
    fields::write(out, product.data(), product.size(), reinterpret_cast<char*>(arena.take(kTextWords).data()));
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
