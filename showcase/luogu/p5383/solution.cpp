// Luogu P5383, ordinary to falling-factorial basis, n <= 10^5, mod 998244353, 2 s. With
// F(x) = sum b_k x^(k falling), sum_i F(i) x^i / i! = (sum_k b_k x^k) e^x, so b is the first n
// coefficients of (sum_i F(i) x^i / i!) e^(-x): evaluate F at 0..n-1, then one product.
// The evaluation is the textbook remainder tree: the subproduct tree of (x - i), and F reduced
// mod each node by a reversed-series inverse and two products. O(n log^2 n) with a large
// constant; the setter's own reference solution on this route exceeded the time limit.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/poly.hpp"

namespace {

using easy::kMod;
using easy::Poly;
using easy::Span;

constexpr std::size_t kLeaf = 64;  // nodes of at most this many points evaluate by Horner

// Node of the subproduct tree over the points [lo, hi): prod (x - i), monic, hi - lo + 1 coefficients.
struct Node {
    std::uint32_t lo, hi;
    Poly product;
    std::unique_ptr<Node> left, right;
};

std::unique_ptr<Node> build(std::uint32_t lo, std::uint32_t hi) {
    auto node = std::make_unique<Node>(Node{lo, hi, {}, nullptr, nullptr});
    if (hi - lo <= kLeaf) {
        node->product = {1};
        for (std::uint32_t i = lo; i < hi; ++i) node->product = easy::multiply(node->product, Poly{kMod - i, 1});
        return node;
    }
    const std::uint32_t mid = lo + (hi - lo) / 2;
    node->left = build(lo, mid);
    node->right = build(mid, hi);
    node->product = easy::multiply(node->left->product, node->right->product);
    return node;
}

// f mod t for monic t: q = rev(rev(f) / rev(t) mod x^k), k = deg f - deg t + 1; r = f - q t mod x^deg t.
Poly remainder(Span f, Span t) {
    const std::size_t m = t.size() - 1;
    if (f.size() <= m) return Poly(f.begin(), f.end());
    const std::size_t k = f.size() - m;
    const Poly rf(f.rbegin(), f.rbegin() + std::ptrdiff_t(k)), rt(t.rbegin(), t.rend());
    Poly q = easy::multiply(rf, easy::inverse(rt, k));
    q.resize(k);
    std::reverse(q.begin(), q.end());
    const Poly qt = easy::multiply(q, t.first(m));
    Poly r(f.begin(), f.begin() + std::ptrdiff_t(m));
    for (std::size_t i = 0; i < m; ++i) r[i] = r[i] >= qt[i] ? r[i] - qt[i] : r[i] + kMod - qt[i];
    return r;
}

// values[i] = f(i) for i in [node.lo, node.hi), with deg f < node.hi - node.lo.
void descend(const Node& node, Span f, std::vector<std::uint32_t>& values) {
    if (!node.left) {
        for (std::uint32_t x = node.lo; x < node.hi; ++x) {
            std::uint32_t y = 0;
            for (std::size_t j = f.size(); j--;) y = std::uint32_t((std::uint64_t(y) * x + f[j]) % kMod);
            values[x] = y;
        }
        return;
    }
    descend(*node.left, remainder(f, node.left->product), values);
    descend(*node.right, remainder(f, node.right->product), values);
}

}  // namespace

int main() {
    easy::Reader in;
    const auto n = in.read<std::uint32_t>();
    Poly f(n);
    for (auto& c : f) c = in.read<std::uint32_t>();

    std::vector<std::uint32_t> values(n);
    descend(*build(0, n), f, values);

    // inverse_factorial[i] = 1 / i!
    std::vector<std::uint32_t> inverse_factorial(n);
    std::uint32_t factorial = 1;
    for (std::uint32_t i = 1; i < n; ++i) factorial = easy::mul(factorial, i);
    inverse_factorial[n - 1] = easy::power(factorial, kMod - 2);
    for (std::uint32_t i = n - 1; i > 0; --i) inverse_factorial[i - 1] = easy::mul(inverse_factorial[i], i);

    Poly scaled(n), exp_minus(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        scaled[i] = easy::mul(values[i], inverse_factorial[i]);
        exp_minus[i] = i % 2 && inverse_factorial[i] ? kMod - inverse_factorial[i] : inverse_factorial[i];
    }
    const Poly b = easy::multiply(scaled, exp_minus);

    easy::Writer out;
    for (std::uint32_t i = 0; i < n; ++i) out.write(b[i], i + 1 < n ? ' ' : '\n');
}
