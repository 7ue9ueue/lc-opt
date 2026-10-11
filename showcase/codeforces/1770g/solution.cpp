// Codeforces 1770G Koxia and Bracket, |s| <= 5e5, 5 s, 256 MB. Count the ways to delete the
// minimum number of characters so that s becomes balanced, mod 998244353.
// The prefix up to the last unmatched ')' loses only ')', the rest only '(' (mirrored), and the
// answer is the product. Over the prefix's ')' the state e = deleted - unmatched so far >= 0 is
// multiplied by (1 + x) per matched ')' and by (1 + x^-1), floored at 0, per unmatched one.
// Divide and conquer: on a segment with c unmatched ')', states >= c never reach the floor and
// take one product with the binomial row (1 + x)^len x^-c; states < c recurse into the halves.
// O(n log^2 n), the editorial's route, known to be tight at this limit.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/multiply.hpp"

namespace {

using easy::kMod;
using easy::Poly;

#ifndef KOXIA_LEAF
#define KOXIA_LEAF 32
#endif

constexpr std::uint32_t add(std::uint32_t a, std::uint32_t b) { return a + b >= kMod ? a + b - kMod : a + b; }

class Binomials {
public:
    explicit Binomials(std::size_t n) : fact_(n + 1), inv_fact_(n + 1) {
        fact_[0] = 1;
        for (std::size_t i = 1; i <= n; ++i) fact_[i] = easy::mul(fact_[i - 1], std::uint32_t(i));
        inv_fact_[n] = easy::power(fact_[n], kMod - 2);
        for (std::size_t i = n; i > 0; --i) inv_fact_[i - 1] = easy::mul(inv_fact_[i], std::uint32_t(i));
    }

    // C(n, j) for j = 0..n.
    Poly row(std::size_t n) const {
        Poly r(n + 1);
        for (std::size_t j = 0; j <= n; ++j) r[j] = easy::mul(fact_[n], easy::mul(inv_fact_[j], inv_fact_[n - j]));
        return r;
    }

private:
    std::vector<std::uint32_t> fact_, inv_fact_;
};

// Ways to delete exactly as many ')' as there are unmatched ones, given the ')' in order
// (true: unmatched), so that every prefix stays balanced.
class Counter {
public:
    Counter(const std::vector<bool>& unmatched, const Binomials& binomials)
        : unmatched_(unmatched), binomials_(binomials), before_(unmatched.size() + 1) {
        for (std::size_t i = 0; i < unmatched.size(); ++i) before_[i + 1] = before_[i] + unmatched[i];
    }

    std::uint32_t count() const {
        const Poly p = solve(0, unmatched_.size(), Poly{1});
        return p.empty() ? 0 : p[0];
    }

private:
    // States after ')' [l, r) from states p before them. States above the number of unmatched
    // ')' still to come cannot end at 0 and are dropped.
    Poly solve(std::size_t l, std::size_t r, Poly p) const {
        if (r - l <= KOXIA_LEAF) {
            for (std::size_t i = l; i < r; ++i) step(p, i);
            return p;
        }
        const std::size_t c = before_[r] - before_[l];
        const std::size_t keep = remaining(r) + 1;
        Poly high;
        if (p.size() > c) {
            high = easy::multiply(easy::Span(p).subspan(c), binomials_.row(r - l));
            high.resize(std::min(high.size(), keep));
            p.resize(c);
        }
        if (!p.empty()) {
            const std::size_t m = (l + r) / 2;
            p = solve(m, r, solve(l, m, std::move(p)));
        }
        if (p.size() < high.size()) std::swap(p, high);
        for (std::size_t e = 0; e < high.size(); ++e) p[e] = add(p[e], high[e]);
        return p;
    }

    void step(Poly& p, std::size_t i) const {
        if (unmatched_[i]) {  // e <- e (deleted) or e - 1 (kept; dropped below 0)
            for (std::size_t e = 0; e + 1 < p.size(); ++e) p[e] = add(p[e], p[e + 1]);
        } else {  // e <- e + 1 (deleted) or e (kept)
            p.push_back(0);
            for (std::size_t e = p.size() - 1; e > 0; --e) p[e] = add(p[e], p[e - 1]);
        }
        p.resize(std::min(p.size(), remaining(i + 1) + 1));
    }

    std::size_t remaining(std::size_t i) const { return before_.back() - before_[i]; }

    const std::vector<bool>& unmatched_;
    const Binomials& binomials_;
    std::vector<std::uint32_t> before_;  // unmatched ')' among the first i
};

// The ')' of s up to its last unmatched one, true where unmatched.
std::vector<bool> closings(std::string_view s) {
    std::vector<bool> unmatched;
    std::size_t last = 0, balance = 0;
    for (const char ch : s) {
        if (ch == '(') {
            ++balance;
        } else {
            unmatched.push_back(balance == 0);
            if (balance == 0) last = unmatched.size();
            else --balance;
        }
    }
    unmatched.resize(last);
    return unmatched;
}

// s after its last unmatched ')', reversed with brackets swapped.
std::string mirrored_tail(std::string_view s) {
    std::size_t start = 0, balance = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '(') ++balance;
        else if (balance > 0) --balance;
        else start = i + 1;
    }
    std::string t(s.rbegin(), s.rend() - std::ptrdiff_t(start));
    for (char& ch : t) ch = ch == '(' ? ')' : '(';
    return t;
}

}  // namespace

int main() {
    easy::Reader in;
    const std::string_view s = in.token();
    const Binomials binomials(s.size());
    const std::vector<bool> head = closings(s), tail = closings(mirrored_tail(s));
    const std::uint32_t answer = easy::mul(Counter(head, binomials).count(), Counter(tail, binomials).count());
    easy::Writer out;
    out.write(answer, '\n');
}
