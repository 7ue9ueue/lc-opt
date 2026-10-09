// c_k = sum over i j = k (mod P) of a_i b_j mod 998244353, P prime <= 524287.
// With a primitive root g, i = g^x for i != 0, so the nonzero part is a cyclic convolution of
// length L = P - 1 of A[x] = a[g^x] and B[x] = b[g^x]: one linear product (lib/ntt, length
// 2^20 at the maximum) folded mod x^L - 1. c_0 = a_0 sum(b) + b_0 sum(a) - a_0 b_0.
#include <unistd.h>

#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
#include "fields.hpp"

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

// First level of a factor f[0, 4q) whose upper half f[4q, 8q) is unused (read as zero): one
// pass reads the lower half once and writes the first radix-4 group of both halves of the
// transform: group 0 to f[0, 4q), group 1 to f[4q, 8q). Inputs canonical; outputs < 4P.
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

int log_length(std::size_t n) { return std::max(6, int(std::bit_width(2 * n - 2))); }

// a * b for factors of n coefficients (n = m) when the transform length 2^lg is 2 * 4^j >= 256: the top
// level is forward_radix8, the rest is lib/ntt. Same layout as ntt::Convolution; single use.
class Product {
public:
    static bool fits(std::size_t n) {
        const int lg = log_length(n);
        return lg % 2 == 0 && lg >= 8;
    }

    Product(std::size_t n, std::size_t) : lg_(log_length(n)) {
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

    ~Product() { ::munmap(region_, bytes_); }

    Product(const Product&) = delete;
    Product& operator=(const Product&) = delete;

    std::uint32_t* a() { return a_; }
    std::uint32_t* b() { return b_; }
    // fields::kTextBytes bytes for the output, 16-byte aligned, after the tables (in their huge page).
    char* text() { return text_; }

    // The coefficients of a * b, canonical, in a(); b() is destroyed. The upper halves of the
    // factors are not read.
    const std::uint32_t* multiply() {
        using namespace ntt::detail;
        const std::size_t len = length(), nv = len / 8, h = nv / 2, q = nv / 8;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        auto* a = reinterpret_cast<Vec*>(a_);
        auto* b = reinterpret_cast<Vec*>(b_);
        forward_radix8(a, q, roots_);
        forward_radix8(b, q, roots_);
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

    std::size_t length() const { return std::size_t(1) << lg_; }

    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
    char* text_;
};

char* text(Product& product) { return product.text(); }

char* text(ntt::Convolution&) {
    alignas(64) static char buffer[fields::kTextBytes];
    return buffer;
}

std::uint32_t power_mod(std::uint32_t x, std::uint32_t e, std::uint32_t m) {
    std::uint64_t r = 1 % m, y = x % m;
    for (; e; e >>= 1, y = y * y % m)
        if (e & 1) r = r * y % m;
    return std::uint32_t(r);
}

// A generator of (Z/p)^*, p prime.
std::uint32_t primitive_root(std::uint32_t p) {
    const std::uint32_t order = p - 1;
    std::uint32_t primes[20], count = 0, rest = order;
    for (std::uint32_t d = 2; d * d <= rest; ++d) {
        if (rest % d) continue;
        primes[count++] = d;
        while (rest % d == 0) rest /= d;
    }
    if (rest > 1) primes[count++] = rest;
    for (std::uint32_t g = 1;; ++g) {
        bool generates = true;
        for (std::uint32_t j = 0; j < count && generates; ++j) generates = power_mod(g, order / primes[j], p) != 1;
        if (generates) return g;
    }
}

// Multiplication by a constant h mod p < 2^31, Shoup style: x < p -> x h mod p.
class Step {
public:
    Step(std::uint32_t h, std::uint32_t p)
        : h_(_mm256_set1_epi32(int(h))),
          q_(_mm256_set1_epi32(int((std::uint64_t(h) << 32) / p))),
          p_(_mm256_set1_epi32(int(p))) {}

    Vec operator()(Vec x) const {
        const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(x, q_), 32);
        const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), q_);
        const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
        const Vec r = _mm256_sub_epi32(_mm256_mullo_epi32(x, h_), _mm256_mullo_epi32(q, p_));  // < 2p
        return _mm256_min_epu32(r, _mm256_sub_epi32(r, p_));
    }

private:
    Vec h_, q_, p_;
};

// The powers g^x, ..., g^(x + 15) mod p as two vectors, advanced by 16 per step.
class Powers {
public:
    Powers(std::uint32_t g, std::uint32_t p) : step_(power_mod(g, 16, p), p) {
        alignas(32) std::uint32_t first[16];
        first[0] = 1;
        for (int j = 1; j < 16; ++j) first[j] = std::uint32_t(std::uint64_t(first[j - 1]) * g % p);
        lo_ = _mm256_load_si256(reinterpret_cast<const Vec*>(first));
        hi_ = _mm256_load_si256(reinterpret_cast<const Vec*>(first + 8));
    }

    Vec lo() const { return lo_; }
    Vec hi() const { return hi_; }
    void advance() { lo_ = step_(lo_), hi_ = step_(hi_); }
    // g^x for the current x (lane 0 of lo).
    std::uint32_t first() const { return std::uint32_t(_mm256_cvtsi256_si32(lo_)); }

private:
    Step step_;
    Vec lo_, hi_;
};

Vec load(const std::uint32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
void store(std::uint32_t* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }

// sum + x mod P for sum, x < P.
Vec add_mod(Vec sum, Vec x) {
    const Vec s = _mm256_add_epi32(sum, x);
    return _mm256_min_epu32(s, _mm256_sub_epi32(s, ntt::detail::broadcast(kP)));
}

std::uint32_t lane_sum(Vec x) {
    alignas(32) std::uint32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<Vec*>(lanes), x);
    std::uint64_t s = 0;
    for (const std::uint32_t v : lanes) s += v;
    return std::uint32_t(s % kP);
}

struct Sums {
    std::uint32_t a, b;  // sums of a_i and b_i over i != 0, mod P
};

// A[x] = sa[g^x - 1], B[x] = sb[g^x - 1] for x < n = p - 1; A and B beyond n stay zero.
Sums gather(std::uint32_t* A, std::uint32_t* B, const std::uint32_t* sa, const std::uint32_t* sb, std::uint32_t n,
            std::uint32_t g, std::uint32_t p) {
    Powers pw(g, p);
    const int* ta = reinterpret_cast<const int*>(sa) - 1;
    const int* tb = reinterpret_cast<const int*>(sb) - 1;
    Vec suma = _mm256_setzero_si256(), sumb = suma;
    std::uint32_t x = 0;
    for (; x + 16 <= n; x += 16, pw.advance()) {
        const Vec a0 = _mm256_i32gather_epi32(ta, pw.lo(), 4), a1 = _mm256_i32gather_epi32(ta, pw.hi(), 4);
        const Vec b0 = _mm256_i32gather_epi32(tb, pw.lo(), 4), b1 = _mm256_i32gather_epi32(tb, pw.hi(), 4);
        store(A + x, a0), store(A + x + 8, a1), store(B + x, b0), store(B + x + 8, b1);
        suma = add_mod(add_mod(suma, a0), a1);
        sumb = add_mod(add_mod(sumb, b0), b1);
    }
    std::uint64_t ra = lane_sum(suma), rb = lane_sum(sumb);
    for (std::uint32_t i = pw.first(); x < n; ++x, i = std::uint32_t(std::uint64_t(i) * g % p)) {
        A[x] = sa[i - 1], B[x] = sb[i - 1];
        ra += A[x], rb += B[x];
    }
    return {std::uint32_t(ra % kP), std::uint32_t(rb % kP)};
}

// c[g^k] = d[k] + d[k + n] mod P for k < n = p - 1: the product folded mod x^n - 1.
void scatter(std::uint32_t* c, const std::uint32_t* d, std::uint32_t n, std::uint32_t g, std::uint32_t p) {
    Powers pw(g, p);
    alignas(32) std::uint32_t index[16], value[16];
    std::uint32_t k = 0;
    for (; k + 16 <= n; k += 16, pw.advance()) {
        store(index, pw.lo()), store(index + 8, pw.hi());
        store(value, add_mod(load(d + k), load(d + k + n)));
        store(value + 8, add_mod(load(d + k + 8), load(d + k + n + 8)));
        for (int j = 0; j < 16; ++j) c[index[j]] = value[j];
    }
    for (std::uint32_t i = pw.first(); k < n; ++k, i = std::uint32_t(std::uint64_t(i) * g % p)) {
        const std::uint32_t s = d[k] + d[k + n];
        c[i] = s >= kP ? s - kP : s;
    }
}

template <class Multiplier>
void convolve(io::Reader& in, std::uint32_t p) {
    const std::uint32_t n = p - 1;
    Multiplier product(n, n);
    // a_1.. and b_1.. are parsed into the factors' upper halves, which the transform never reads.
    const std::size_t half = std::size_t(1) << (log_length(n) - 1);
    std::uint32_t* sa = product.a() + half;
    std::uint32_t* sb = product.b() + half;
    const std::uint32_t a0 = in.read<std::uint32_t>();
    in.read(sa, n);
    const std::uint32_t b0 = in.read<std::uint32_t>();
    in.read(sb, n);
    const std::uint32_t g = primitive_root(p);
    const Sums sums = gather(product.a(), product.b(), sa, sb, n, g, p);
    const std::uint32_t* d = product.multiply();
    std::uint32_t* c = product.b();
    scatter(c, d, n, g, p);
    // c_0 = a_0 (b_0 + sum b) + b_0 sum a.
    c[0] = std::uint32_t((std::uint64_t(a0) * ((b0 + sums.b) % kP) + std::uint64_t(b0) * sums.a) % kP);
    io::Writer out;
    fields::write(out, c, p, text(product));
}

void solve() {
    io::Reader in;
    const std::uint32_t p = in.read<std::uint32_t>();
    if (Product::fits(p - 1)) return convolve<Product>(in, p);
    convolve<ntt::Convolution>(in, p);
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
