// QOJ 621 多项式指数函数: exp(f) mod x^n, n <= 10^6, mod 998244353, 2.5 s, 1 GiB.
// The intended solution is Newton iteration, O(n log n). This one uses g' = f' g, that is
// i g_i = sum_(k=1..i) k f_k g_(i-k), and fills g online by divide and conquer: each range's
// left half adds its share to the right half with one full product, O(n log^2 n).
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/poly.hpp"

namespace {

// Ranges of at most kLeaf coefficients are solved by the quadratic recurrence.
constexpr std::size_t kLeaf = 64;

class OnlineExp {
public:
    explicit OnlineExp(const easy::Poly& f) : n_(f.size()), h_(n_), inv_(n_ + 1, 1), acc_(n_), g_(n_) {
        for (std::size_t k = 0; k < n_; ++k) h_[k] = easy::mul(std::uint32_t(k), f[k]);
        for (std::uint32_t i = 2; i <= n_; ++i)
            inv_[i] = easy::mul(easy::kMod - easy::kMod / i, inv_[easy::kMod % i]);
    }

    easy::Poly run() {
        solve(0, n_);
        return g_;
    }

private:
    // On entry acc_[i] = sum over j < l of g_j h_(i-j), for i in [l, r).
    void solve(std::size_t l, std::size_t r) {
        if (r - l <= kLeaf) return leaf(l, r);
        const std::size_t m = (l + r) / 2;
        solve(l, m);
        const easy::Poly p = easy::multiply(std::span(g_).subspan(l, m - l), std::span(h_).first(r - l));
        for (std::size_t i = m; i < r; ++i) acc_[i] = (acc_[i] + p[i - l]) % easy::kMod;
        solve(m, r);
    }

    void leaf(std::size_t l, std::size_t r) {
        for (std::size_t i = l; i < r; ++i) {
            g_[i] = i == 0 ? 1 : easy::mul(acc_[i], inv_[i]);
            for (std::size_t j = i + 1; j < r; ++j)
                acc_[j] = std::uint32_t((acc_[j] + std::uint64_t(g_[i]) * h_[j - i]) % easy::kMod);
        }
    }

    std::size_t n_;
    easy::Poly h_;    // h_k = k f_k
    easy::Poly inv_;  // inv_[i] = 1 / i
    easy::Poly acc_;
    easy::Poly g_;
};

}  // namespace

int main() {
    easy::Reader in;
    const auto n = in.read<std::size_t>();
    easy::Poly f(n);
    for (auto& a : f) a = in.read<std::uint32_t>();
    const easy::Poly g = OnlineExp(f).run();
    easy::Writer out;
    for (std::size_t i = 0; i < n; ++i) out.write(g[i], i + 1 < n ? ' ' : '\n');
}
