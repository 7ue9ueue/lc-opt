// Factorization of a monic f of degree N <= 100 over F_p, p prime <= 998244353, into monic
// irreducibles with multiplicities: square-free decomposition, roots of the linear part, then
// Berlekamp (factor.hpp for odd p, gf2.hpp for p = 2). Design and log: notes.md.
#include <cstdint>
#include <vector>

#include "lib/io/io.hpp"
#include "lib/run/early.hpp"
#include "problems/polynomial/factorization_of_polynomials/factor.hpp"
#include "problems/polynomial/factorization_of_polynomials/gf2.hpp"

namespace {

void solve() {
    io::Reader in;
    io::Writer out;
    const auto n = in.read<std::uint32_t>();
    const auto p = in.read<std::uint32_t>();
    std::vector<std::uint32_t> f(n + 1);
    for (auto& a : f) a = in.read<std::uint32_t>();
    std::vector<factor::Factor> factors;
    if (n > 0) factors = p == 2 ? factor::gf2::Factorizer().run(f) : factor::OddFactorizer(p).run(f);
    out.write(factors.size(), '\n');
    for (const auto& g : factors) {
        out.write(g.multiplicity, ' ', g.coefficients.size() - 1);
        for (const auto b : g.coefficients) out.write(' ', b);
        out.write('\n');
    }
}

}  // namespace

RUN_EARLY(solve)
