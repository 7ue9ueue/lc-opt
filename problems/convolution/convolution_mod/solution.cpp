// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1 (lib/ntt), output in fixed-width
// fields (../fixed_width.hpp). For 2^lg = 2 * 4^j (2^20 at the maximum) the top level is a radix-8
// pass here (Product); other lengths use ntt::Convolution as is.
#include <unistd.h>

#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
#include "../fixed_width.hpp"

namespace {

using ntt::detail::add;
using ntt::detail::diff;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::reduce;
using ntt::detail::Vec;

// Shoup product, < 2P for any x < 2^32.
Vec times(Vec x, const Factor& f) { return ntt::detail::multiply(x, f); }

// x - y + P for y < P.
Vec diff_canonical(Vec x, Vec y) {
    return _mm256_sub_epi32(_mm256_add_epi32(x, ntt::detail::broadcast(kP)), y);
}

// First level of a factor f[0, 4q) whose upper half f[4q, 8q) is zero. Modulo x^(n/2) - 1 and
// x^(n/2) + 1 the factor is unchanged, so one pass reads it once and writes the first radix-4
// group of each half: group 0 to f[0, 4q), group 1 to f[4q, 8q). ntt::Convolution copies the
// lower half up and runs the two groups as separate passes. Inputs canonical; outputs < 4P.
void forward_radix8(Vec* f, std::size_t q, const std::uint32_t* roots) {
    const Factor i(roots[1], roots[9]), y(roots[2], roots[10]), z(roots[3], roots[11]);
    for (std::size_t j = 0; j < q; ++j) {
        const Vec f0 = f[j], f1 = f[j + q], f2 = f[j + 2 * q], f3 = f[j + 3 * q];
        // Group 0 (twiddles 1, 1, i): every term < 2P.
        const Vec g0 = add(f0, f2), g1 = add(f1, f3);
        const Vec h0 = diff_canonical(f0, f2), ih1 = times(diff_canonical(f1, f3), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
        f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
        // Group 1 (twiddles i, y, z).
        const Vec if2 = times(f2, i), if3 = times(f3, i);
        const Vec u0 = reduce(add(f0, if2), 2 * kP), v0 = reduce(diff(f0, if2), 2 * kP);
        const Vec yu1 = times(add(f1, if3), y), zv1 = times(diff(f1, if3), z);
        f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1);
        f[j + 6 * q] = add(v0, zv1), f[j + 7 * q] = diff(v0, zv1);
    }
}

// a * b when the transform length 2^lg is 2 * 4^j >= 256 and each factor fills at most half of it.
// Same memory layout as ntt::Convolution; single use.
class Product {
public:
    static bool fits(std::size_t n, std::size_t m) {
        const int lg = log_length(n, m);
        return lg % 2 == 0 && lg >= 8 && 2 * std::max(n, m) <= std::size_t(1) << lg;
    }

    Product(std::size_t n, std::size_t m) : lg_(log_length(n, m)) {
        const std::size_t len = length(), words = 2 * (len + kPadding) + 2 * ntt::detail::table_words(lg_);
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes_ = (words * sizeof(std::uint32_t) + kHuge - 1) / kHuge * kHuge + kHuge;
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
    }

    ~Product() { ::munmap(region_, bytes_); }

    Product(const Product&) = delete;
    Product& operator=(const Product&) = delete;

    std::uint32_t* a() { return a_; }
    std::uint32_t* b() { return b_; }

    // The coefficients of a * b, canonical, in a(); b() is destroyed.
    const std::uint32_t* multiply() {
        using namespace ntt::detail;
        const std::size_t len = length(), nv = len / 8, h = nv / 2, q = nv / 8;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        auto* a = reinterpret_cast<Vec*>(a_);
        auto* b = reinterpret_cast<Vec*>(b_);
        forward_radix8(a, q, roots_);
        forward_radix8(b, q, roots_);
        // As ntt::Convolution from here: each half's subtrees and last group, the radix-2 level.
        const Recursion recursion(roots_, inverse_roots_);
        for (std::size_t c = 0; c < 4; ++c) recursion.visit(a + c * q, b + c * q, q, c);
        ntt::kernels::inverse_identity(a, q, inverse_roots_);
        for (std::size_t c = 4; c < 8; ++c) recursion.visit(a + c * q, b + c * q, q, c);
        ntt::kernels::inverse(a + h, q, inverse_roots_ + slot(1), inverse_roots_ + slot(2));
        const std::uint32_t scale = multiply_mod(power(std::uint32_t(nv), kP - 2), kR);  // undoes nv / 2^32
        alignas(32) std::uint32_t s[16] = {};  // table layout: s at entry 1
        s[1] = scale;
        s[9] = quotient(scale);
        ntt::kernels::scale_radix2(a, h, s);
        return a_;
    }

private:
    static constexpr std::size_t kPadding = 16;  // words after each factor: the kernels read 4 bytes past

    static int log_length(std::size_t n, std::size_t m) { return std::max(6, int(std::bit_width(n + m - 2))); }
    std::size_t length() const { return std::size_t(1) << lg_; }

    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
};

template <class Multiplier>
void convolve(io::Reader& in, std::size_t n, std::size_t m) {
    Multiplier product(n, m);
    in.read(product.a(), n);
    in.read(product.b(), m);
    const std::uint32_t* c = product.multiply();
    io::Writer out;
    fixed_width::write(out, c, n + m - 1);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    if (Product::fits(n, m)) return convolve<Product>(in, n, m);
    convolve<ntt::Convolution>(in, n, m);
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here): 0.15 ms less per run. _exit skips their teardown too.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
