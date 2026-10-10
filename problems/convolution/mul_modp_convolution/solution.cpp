// c_k = sum over i j = k (mod P) of a_i b_j mod 998244353, P prime <= 524287.
// With a primitive root g, i = g^x for i != 0, so the nonzero part is a cyclic convolution of
// length n = P - 1 of A[x] = a[g^x] and B[x] = b[g^x]: one linear product (lib/ntt, length
// 2^20 at the maximum) folded mod x^n - 1. c_0 = a_0 sum(b) + b_0 sum(a) - a_0 b_0.
#include <memory>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/ntt/product.hpp"
#include "lib/run/early.hpp"
#include "../convolution_mod/fields.hpp"

namespace {

using ntt::detail::broadcast;
using ntt::detail::kP;
using ntt::detail::Vec;

Vec load(const std::uint32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
void store(std::uint32_t* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }

// sum + x mod P for sum, x < P.
Vec add_mod(Vec sum, Vec x) {
    const Vec s = _mm256_add_epi32(sum, x);
    return _mm256_min_epu32(s, _mm256_sub_epi32(s, broadcast(kP)));
}

std::uint32_t lane_sum(Vec x) {
    alignas(32) std::uint32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<Vec*>(lanes), x);
    std::uint64_t s = 0;
    for (const std::uint32_t v : lanes) s += v;
    return std::uint32_t(s % kP);
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

// The discrete-log order: g^x mod p for p prime < 2^20.
class Powers {
public:
    Powers(std::uint32_t g, std::uint32_t p) : g_(g), p_(p) {}

    std::uint32_t next(std::uint32_t y) const { return std::uint32_t(std::uint64_t(y) * g_ % p_); }

    // g^(x + j) for j = 0, 1, 4, 5, 2, 3, 6, 7: the lane order of fetch().
    Vec at(std::uint32_t x) const {
        alignas(32) std::uint32_t lanes[8];
        std::uint32_t y = power_mod(g_, x, p_);
        for (const int j : kOrder) lanes[j] = y, y = next(y);
        return _mm256_load_si256(reinterpret_cast<const Vec*>(lanes));
    }

    static constexpr int kOrder[8] = {0, 1, 4, 5, 2, 3, 6, 7};  // lane of g^(x + j)

    // Multiplies vectors of powers by g^k, Shoup style: y < p -> y g^k mod p.
    class Step {
    public:
        Step(const Powers& powers, std::uint32_t k) {
            const std::uint32_t h = power_mod(powers.g_, k, powers.p_);
            h_ = broadcast(h);
            q_ = broadcast(std::uint32_t((std::uint64_t(h) << 32) / powers.p_));
            p_ = broadcast(powers.p_);
        }

        Vec operator()(Vec y) const {
            const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(y, q_), 32);
            const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(y, 32), q_);
            const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
            const Vec r = _mm256_sub_epi32(_mm256_mullo_epi32(y, h_), _mm256_mullo_epi32(q, p_));  // < 2p
            return _mm256_min_epu32(r, _mm256_sub_epi32(r, p_));
        }

    private:
        Vec h_, q_, p_;
    };

private:
    std::uint32_t g_, p_;
};

struct Sums {
    std::uint32_t a, b;  // sums of a_i and b_i over i != 0, mod P
};

// Storage of the input pairs a_i + 2^32 b_i, 0 < i < p, in two pieces: pair i at
// base[i - 1] for i <= split, else at base[i - 1 + offset]. split is a multiple of 8.
struct Pairs {
    std::uint64_t* base;
    std::uint32_t split, offset;  // split < 2^31

    std::uint64_t* at(std::uint32_t i) const { return base + i - 1 + (i > split ? offset : 0); }
};

struct Input {
    Pairs pairs;
    std::uint32_t p, g;
};

// Reads a_1.., b_0, b_1.. (n each but b_0) into pairs; returns b_0. a, b: scratch for n + 8 words.
std::uint32_t read_pairs(io::Reader& in, std::uint32_t n, std::uint32_t* a, std::uint32_t* b, const Pairs& pairs) {
    io::read_bulk(in, a, n);
    const std::uint32_t b0 = in.read<std::uint32_t>();
    io::read_bulk(in, b, n);
    for (std::uint32_t i = 0; i < n; i += 8) {
        auto* out = reinterpret_cast<Vec*>(pairs.at(i + 1));
        const Vec x = _mm256_permute4x64_epi64(load(a + i), 0xD8), y = _mm256_permute4x64_epi64(load(b + i), 0xD8);
        _mm256_storeu_si256(out, _mm256_unpacklo_epi32(x, y));
        _mm256_storeu_si256(out + 1, _mm256_unpackhi_epi32(x, y));
    }
    return b0;
}

// Pairs i for the lanes i of index (in Powers lane order) as a and b in natural order: one load
// serves both factors.
class Fetch {
public:
    explicit Fetch(const Pairs& pairs)
        : base_(reinterpret_cast<const long long*>(pairs.base - 1)), split_(broadcast(pairs.split)),
          offset_(broadcast(pairs.offset)) {}

    void operator()(Vec index, Vec& a, Vec& b) const {
        index = _mm256_add_epi32(index, _mm256_and_si256(_mm256_cmpgt_epi32(index, split_), offset_));
        const __m256 lo = _mm256_castsi256_ps(_mm256_i32gather_epi64(base_, _mm256_castsi256_si128(index), 8));
        const __m256 hi = _mm256_castsi256_ps(_mm256_i32gather_epi64(base_, _mm256_extracti128_si256(index, 1), 8));
        a = _mm256_castps_si256(_mm256_shuffle_ps(lo, hi, 0x88));
        b = _mm256_castps_si256(_mm256_shuffle_ps(lo, hi, 0xDD));
    }

private:
    const long long* base_;
    Vec split_, offset_;
};

// A[x] = a_(g^x), B[x] = b_(g^x) for x < n = p - 1; A and B beyond n stay zero.
Sums gather(std::uint32_t* a, std::uint32_t* b, const Input& in) {
    const std::uint32_t n = in.p - 1;
    const Powers powers(in.g, in.p);
    const Powers::Step step(powers, 16);
    const Fetch fetch(in.pairs);
    Vec lo = powers.at(0), hi = powers.at(8), sum_a = _mm256_setzero_si256(), sum_b = sum_a;
    std::uint32_t x = 0;
    for (; x + 16 <= n; x += 16, lo = step(lo), hi = step(hi)) {
        Vec a0, a1, b0, b1;
        fetch(lo, a0, b0);
        fetch(hi, a1, b1);
        store(a + x, a0), store(a + x + 8, a1), store(b + x, b0), store(b + x + 8, b1);
        sum_a = add_mod(add_mod(sum_a, a0), a1);
        sum_b = add_mod(add_mod(sum_b, b0), b1);
    }
    std::uint64_t ra = lane_sum(sum_a), rb = lane_sum(sum_b);
    for (std::uint32_t y = std::uint32_t(_mm256_cvtsi256_si32(lo)); x < n; ++x, y = powers.next(y)) {
        const std::uint64_t pair = *in.pairs.at(y);
        a[x] = std::uint32_t(pair), b[x] = std::uint32_t(pair >> 32);
        ra += a[x], rb += b[x];
    }
    return {std::uint32_t(ra % kP), std::uint32_t(rb % kP)};
}

// c[g^k] = d[k] + d[k + n] mod P for k < n = p - 1: the product folded mod x^n - 1.
void scatter(std::uint32_t* c, const std::uint32_t* d, std::uint32_t p, std::uint32_t g) {
    const std::uint32_t n = p - 1;
    const Powers powers(g, p);
    const Powers::Step step(powers, 16);
    Vec lo = powers.at(0), hi = powers.at(8);
    alignas(32) std::uint32_t index[16], value[16];
    std::uint32_t k = 0;
    for (; k + 16 <= n; k += 16, lo = step(lo), hi = step(hi)) {
        store(index, lo), store(index + 8, hi);
        constexpr int kLanes = 0xD8;  // 64-bit lanes 0, 2, 1, 3: Powers lane order
        store(value, _mm256_permute4x64_epi64(add_mod(load(d + k), load(d + k + n)), kLanes));
        store(value + 8, _mm256_permute4x64_epi64(add_mod(load(d + k + 8), load(d + k + n + 8)), kLanes));
        for (int j = 0; j < 16; ++j) c[index[j]] = value[j];
    }
    for (std::uint32_t y = std::uint32_t(_mm256_cvtsi256_si32(lo)); k < n; ++k, y = powers.next(y)) {
        const std::uint32_t s = d[k] + d[k + n];
        c[y] = s >= kP ? s - kP : s;
    }
}

// A * B for factors of n coefficients when ntt::Product takes them (transform length >= 512).
// Single use.
class Product {
public:
    static bool fits(std::size_t n) { return ntt::Product::fits(n, n); }

    explicit Product(std::size_t n) : product_(n, n, fields::kTextBytes) {}

    // Room for the input until load(): the pairs in the factors' upper halves (2^lg / 4 each),
    // which ntt::Product does not read, a_i and b_i in their lower halves.
    Pairs pairs() {
        const std::size_t quarter = product_.length() / 4;
        std::uint32_t *a = product_.a(), *b = product_.b();
        return {reinterpret_cast<std::uint64_t*>(a + 2 * quarter), std::uint32_t(quarter),
                std::uint32_t((b - a) / 2 - quarter)};
    }
    std::uint32_t* scratch_a() { return product_.a(); }
    std::uint32_t* scratch_b() { return product_.b(); }
    std::uint32_t* b() { return product_.b(); }
    // fields::kTextBytes bytes for the output, after the tables (in their huge page).
    char* text() { return static_cast<char*>(product_.extra()); }

    Sums load(const Input& in) { return gather(product_.a(), product_.b(), in); }

    // The coefficients of A * B, canonical; b() is destroyed.
    const std::uint32_t* multiply() { return product_.multiply(); }

private:
    ntt::Product product_;
};

// Other lengths: gather, then ntt::Convolution.
class SmallProduct {
public:
    explicit SmallProduct(std::size_t n) : convolution_(n, n), pairs_(new std::uint64_t[n + 8]) {}

    Pairs pairs() { return {pairs_.get(), 1u << 30, 0}; }
    std::uint32_t* scratch_a() { return convolution_.a(); }
    std::uint32_t* scratch_b() { return convolution_.b(); }
    std::uint32_t* b() { return convolution_.b(); }

    char* text() {
        alignas(64) static char buffer[fields::kTextBytes];
        return buffer;
    }

    Sums load(const Input& in) { return gather(convolution_.a(), convolution_.b(), in); }
    const std::uint32_t* multiply() { return convolution_.multiply(); }

private:
    ntt::Convolution convolution_;
    std::unique_ptr<std::uint64_t[]> pairs_;
};

template <class Multiplier>
void convolve(io::Reader& in, std::uint32_t p) {
    const std::uint32_t n = p - 1;
    Multiplier product(n);
    const std::uint32_t a0 = in.read<std::uint32_t>();
    const std::uint32_t b0 = read_pairs(in, n, product.scratch_a(), product.scratch_b(), product.pairs());
    const std::uint32_t g = primitive_root(p);
    const Sums sums = product.load({product.pairs(), p, g});
    const std::uint32_t* d = product.multiply();
    std::uint32_t* c = product.b();
    scatter(c, d, p, g);
    // c_0 = a_0 (b_0 + sum b) + b_0 sum a.
    c[0] = std::uint32_t((std::uint64_t(a0) * ((b0 + sums.b) % kP) + std::uint64_t(b0) * sums.a) % kP);
    io::Writer out;
    fields::write(out, c, p, product.text());
}

void solve() {
    io::Reader in;
    const std::uint32_t p = in.read<std::uint32_t>();
    if (Product::fits(p - 1)) return convolve<Product>(in, p);
    convolve<SmallProduct>(in, p);
}

}  // namespace

RUN_EARLY(solve)
