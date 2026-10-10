// f(a r^i) mod 998244353 for i < M, f of N coefficients (N, M <= 2^19), by the chirp z-transform.
// With t(k) = k (k - 1) / 2, i j = t(i + j) - t(i) - t(j), so
//   f(a r^i) = r^-t(i) sum_j A_j r^t(i + j),  A_j = c_j a^j r^-t(j):
// a middle product. One cyclic convolution of length L = 2^lg >= N + M - 1 of A with the reversed
// chirp B_u = r^t(L - 1 - u) gives f(a r^i) = r^-t(i) (A B)[L - 1 - i].
// The convolution is ntt::Product's (lib/ntt/product.hpp: radix-8 top, Subtrees, bottom kernels)
// with two changes: B fills the whole length, so its first pass is a full radix-8 one; and the
// last inverse level computes only the outputs, times r^-t(i) and in reverse order. L >= 2
// max(N, M), so A and the outputs each lie in one half.
// Output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/ntt/product.hpp"
#include "lib/poly/chirp.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

using ntt::detail::add;
using ntt::detail::broadcast;
using ntt::detail::diff;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::multiply_mod;
using ntt::detail::power;
using ntt::detail::reduce;
using ntt::detail::slot;
using ntt::detail::Vec;
using poly::detail::Chirp;
using poly::detail::load;
using poly::detail::montgomery;
using poly::detail::store;

// Shoup product, < 2P for any x < 2^32.
Vec times(Vec x, const Factor& f) { return ntt::detail::multiply(x, f); }

// x mod 2P for x < 4P.
Vec low(Vec x) { return reduce(x, 2 * kP); }

// x - y + P for y < P.
Vec diff_canonical(Vec x, Vec y) { return _mm256_sub_epi32(_mm256_add_epi32(x, broadcast(kP)), y); }

Vec reverse(Vec x) { return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

std::uint32_t inverse(std::uint32_t x) { return power(x, kP - 2); }

// r^e for r != 0 and any e >= 0.
std::uint32_t power_of(std::uint32_t r, std::uint64_t e) { return power(r, std::uint32_t(e % (kP - 1))); }

// ntt::detail::forward_radix8 for a factor that fills f[0, 8q): the groups act on the halves
// u = f mod (x^(n/2) - 1) and v = f mod (x^(n/2) + 1), u_t = f_t + f_(t+4) and
// v_t = f_t - f_(t+4) for the blocks f_t = f[t q, (t + 1) q). Inputs canonical; outputs < 4P.
void forward_radix8_full(Vec* f, std::size_t q, const std::uint32_t* roots) {
    const Factor i(roots[1], roots[9]), y(roots[2], roots[10]), z(roots[3], roots[11]);
    for (std::size_t j = 0; j < q; ++j) {
        Vec u[4], v[4];  // < 2P
#pragma GCC unroll 4
        for (std::size_t t = 0; t < 4; ++t) {
            const Vec lo = f[j + t * q], hi = f[j + (t + 4) * q];
            u[t] = add(lo, hi), v[t] = diff_canonical(lo, hi);
        }
        const Vec g0 = low(add(u[0], u[2])), g1 = low(add(u[1], u[3]));
        const Vec h0 = low(diff(u[0], u[2])), ih1 = times(diff(u[1], u[3]), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
        f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
        const Vec iv2 = times(v[2], i), iv3 = times(v[3], i);
        const Vec u0 = low(add(v[0], iv2)), v0 = low(diff(v[0], iv2));
        const Vec yu1 = times(add(v[1], iv3), y), zv1 = times(diff(v[1], iv3), z);
        f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1);
        f[j + 6 * q] = add(v0, zv1), f[j + 7 * q] = diff(v0, zv1);
    }
}

// The last level and the output: with u = a[0, L/2) and w = a[L/2, L) after the halves' top
// inverse groups (< 2P), coefficient L - 1 - i of the product is (u - w)[L/2 - 1 - i] times the
// transform's scale. y[i] = (u - w)[L/2 - 1 - i] x_i / 2^32 for the terms x_i of chirp, which
// carry that scale, for i < m <= L/2 rounded up to 32.
void output(const std::uint32_t* a, std::size_t half, std::uint32_t* y, std::size_t m, Chirp chirp) {
    const std::uint32_t *u = a, *w = a + half;
    for (std::size_t i = 0; i < m; i += 32, chirp.next())
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) {
            const std::size_t at = half - 8 - i - 8 * v;
            const Vec x = reduce(diff(load(u + at), load(w + at)), 2 * kP);
            store(y + i + 8 * v, reduce(montgomery(reverse(x), chirp[v]), kP));
        }
}

// The evaluations f(a r^i), i < m, of f = c[0, n), for a, r != 0. L = 2^lg >= 2 max(n, m) with
// lg even and >= 10 (L / 8 = 2 * 4^j vectors, subtrees of at least 16). One mapping in huge
// pages; single use.
class ChirpZ {
public:
    ChirpZ(std::size_t n, std::size_t m) : n_(n), m_(m) {
        lg_ = std::max(10, int(std::bit_width(2 * std::max(n, m) - 1)));
        lg_ += lg_ % 2;
        const std::size_t len = length(), words = 2 * (len + kPadding) + 2 * ntt::detail::table_words(lg_);
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes_ = (words * sizeof(std::uint32_t) + fields::kTextBytes + kHuge - 1) / kHuge * kHuge + kHuge;
        region_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) std::abort();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region_) + kHuge - 1) & ~(kHuge - 1);
        a_ = reinterpret_cast<std::uint32_t*>(aligned);
#ifdef MADV_HUGEPAGE
        ::madvise(a_, bytes_ - kHuge, MADV_HUGEPAGE);
#endif
        b_ = a_ + len + kPadding;  // a different cache set from a at equal offsets
        roots_ = b_ + len + kPadding;
        inverse_roots_ = roots_ + ntt::detail::table_words(lg_);
        text_ = reinterpret_cast<char*>(inverse_roots_ + ntt::detail::table_words(lg_));
    }

    ~ChirpZ() { ::munmap(region_, bytes_); }

    ChirpZ(const ChirpZ&) = delete;
    ChirpZ& operator=(const ChirpZ&) = delete;

    // Room for c: n coefficients < P.
    std::uint32_t* coefficients() { return a_; }
    char* text() { return text_; }  // fields::kTextBytes bytes, 16-byte aligned

    // The m evaluations, canonical.
    const std::uint32_t* evaluate(std::uint32_t a, std::uint32_t r) {
        using namespace ntt::detail;
        const std::size_t len = length(), q = len / 64;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        // B_u = r^t(L - 1 - u) = r^t(L - 1) r^(-(L - 2) u) r^t(u); A_k = c_k a^k r^-t(k).
        const std::uint32_t r_inverse = inverse(r);
        const std::uint32_t b0 = power_of(r, std::uint64_t(len - 1) * (len - 2) / 2);
        poly::chirp(b0, power(r_inverse, std::uint32_t(len - 2)), r, {b_, len});
        poly::multiply_chirp(1, a, r_inverse, {a_, n_});
        // The outputs' factors carry the transform's scale (L / 8)^-1 2^32 and 2^32 for montgomery().
        const std::uint32_t scale = multiply_mod(multiply_mod(inverse(std::uint32_t(len / 8)), kR), kR);
        auto* av = reinterpret_cast<Vec*>(a_);
        auto* bv = reinterpret_cast<Vec*>(b_);
        forward_radix8(av, q, roots_);
        forward_radix8_full(bv, q, roots_);
        const Subtrees subtrees(roots_, inverse_roots_);
        for (std::size_t c = 0; c < 4; ++c) subtrees.visit(av + c * q, bv + c * q, q, c);
        ntt::kernels::inverse_identity(av, q, inverse_roots_);
        for (std::size_t c = 4; c < 8; ++c) subtrees.visit(av + c * q, bv + c * q, q, c);
        ntt::kernels::inverse(av + 4 * q, q, inverse_roots_ + slot(1), inverse_roots_ + slot(2));
        output(a_, len / 2, b_, m_, Chirp(scale, 1, r_inverse));
        return b_;
    }

private:
    static constexpr std::size_t kPadding = 16;  // words after each factor: the kernels read 4 bytes past

    std::size_t length() const { return std::size_t(1) << lg_; }

    std::size_t n_, m_;
    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
    char* text_;
};

// f(x) for f = c[0, n), c zero up to the next multiple of 32. Lane l of chain w sums
// c[32 v + 8 w + l] x^(32 v) by Horner's rule; the 32 sums are then weighted by x^(8 w + l).
std::uint32_t evaluate(const std::uint32_t* c, std::size_t n, std::uint32_t x) {
    const Vec x32 = broadcast(multiply_mod(power(x, 32), kR));
    Vec sum[4] = {};
    for (std::size_t v = (n + 31) / 32; v-- > 0;)
        for (int w = 0; w < 4; ++w) sum[w] = low(_mm256_add_epi32(montgomery(sum[w], x32), load(c + 32 * v + 8 * w)));
    alignas(32) std::uint32_t lanes[32];
    for (int w = 0; w < 4; ++w) store(lanes + 8 * w, sum[w]);
    std::uint32_t total = 0, term = 1;
    for (int k = 0; k < 32; ++k, term = multiply_mod(term, x)) total = (total + multiply_mod(lanes[k] % kP, term)) % kP;
    return total;
}

void write_same(std::uint32_t first, std::uint32_t rest, std::size_t count) {
    std::vector<std::uint32_t> values(count, rest);
    values[0] = first;
    alignas(64) static char text[fields::kTextBytes];
    io::Writer out;
    fields::write(out, values.data(), count, text);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    const std::uint32_t a = in.read<std::uint32_t>(), r = in.read<std::uint32_t>();
    if (a == 0 || n == 1) {  // f(0) = c_0 everywhere
        const std::uint32_t c0 = in.read<std::uint32_t>();
        return write_same(c0, c0, m);
    }
    if (r == 0 || r == 1 || m == 1) {  // f(a), then f(0) or f(a)
        std::vector<std::uint32_t> c((n + 31) / 32 * 32);
        io::read_bulk(in, c.data(), n);
        const std::uint32_t at_a = evaluate(c.data(), n, a);
        return write_same(at_a, r == 0 ? c[0] : at_a, m);
    }
    ChirpZ chirp_z(n, m);
    io::read_bulk(in, chirp_z.coefficients(), n);
    const std::uint32_t* y = chirp_z.evaluate(a, r);
    io::Writer out;
    fields::write(out, y, m, chirp_z.text());
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
