// c_k = sum over i j = k (mod P) of a_i b_j mod 998244353, P prime <= 524287.
// With a primitive root g, i = g^x for i != 0, so the nonzero part is a cyclic convolution of
// length n = P - 1 of A[x] = a[g^x] and B[x] = b[g^x]: one linear product (lib/ntt, length
// 2^20 at the maximum) folded mod x^n - 1. c_0 = a_0 sum(b) + b_0 sum(a) - a_0 b_0.
// Since g^(x + n/2) = P - g^x, input and output keep indices s and P - s side by side ("slot" s,
// 1 <= s <= n/2): one random access serves x and x + n/2.
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

Vec load(const void* p) { return _mm256_loadu_si256(static_cast<const Vec*>(p)); }
void store(void* p, Vec x) { _mm256_storeu_si256(static_cast<Vec*>(p), x); }
Vec permute(Vec x, Vec lanes) { return _mm256_permutevar8x32_epi32(x, lanes); }

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

// The discrete-log order, eight exponents per vector: g^x mod p for an odd prime p < 2^20.
class Powers {
public:
    // The slot of each lane y = g^x and whether y is its second index (y > (p - 1) / 2).
    struct Slots {
        Vec slot, second;
    };

    Powers(std::uint32_t g, std::uint32_t p) : g_(g), p_(p) {
        const std::uint32_t h = power_mod(g, 8, p);
        h_ = broadcast(h);
        q_ = broadcast(std::uint32_t((std::uint64_t(h) << 32) / p));
        modulus_ = broadcast(p);
        half_ = broadcast((p - 1) / 2);
    }

    std::uint32_t next(std::uint32_t y) const { return std::uint32_t(std::uint64_t(y) * g_ % p_); }

    // g^(x + j) in lane j.
    Vec at(std::uint32_t x) const {
        alignas(32) std::uint32_t lanes[8];
        std::uint32_t y = power_mod(g_, x, p_);
        for (std::uint32_t& lane : lanes) lane = y, y = next(y);
        return _mm256_load_si256(reinterpret_cast<const Vec*>(lanes));
    }

    // y g^8 mod p for each lane y < p, Shoup style.
    Vec step(Vec y) const {
        const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(y, q_), 32);
        const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(y, 32), q_);
        const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
        const Vec r = _mm256_sub_epi32(_mm256_mullo_epi32(y, h_), _mm256_mullo_epi32(q, modulus_));  // < 2p
        return _mm256_min_epu32(r, _mm256_sub_epi32(r, modulus_));
    }

    Slots slots(Vec y) const {
        return {_mm256_min_epu32(y, _mm256_sub_epi32(modulus_, y)), _mm256_cmpgt_epi32(y, half_)};
    }

private:
    std::uint32_t g_, p_;
    Vec h_, q_, modulus_, half_;
};

// The input at slot s: a_s, b_s, a_(p - s), b_(p - s).
struct alignas(16) Quad {
    std::uint32_t word[4];
};

// Quads of slots 1 <= s <= (p - 1) / 2 in two pieces: slot s at base[s] for s <= split, else at
// base[s + offset]. split is a multiple of 8.
struct Quads {
    Quad* base;  // where slot 0 would be
    std::uint32_t split, offset;  // split < 2^31

    Quad* at(std::uint32_t s) const { return base + s + (s > split ? offset : 0); }
};

struct Input {
    Quads quads;
    std::uint32_t p, g;
};

struct Sums {
    std::uint32_t a, b;  // of a_i and b_i over i != 0, mod P
};

// Quads from a_i = a[i - 1] and b_i = b[i - 1], 0 < i < p; returns their sums.
Sums fold_input(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t p, const Quads& quads) {
    const std::uint32_t half = (p - 1) / 2;
    // Slots s + [0 2 4 6 | 1 3 5 7]: then in-lane unpacks give [slot j | slot j + 1].
    const Vec forward = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7), backward = _mm256_setr_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    Vec sum_a = _mm256_setzero_si256(), sum_b = sum_a;
    std::uint32_t s = 1;
    for (; s + 7 <= half; s += 8) {
        const Vec a0 = permute(load(a + s - 1), forward), b0 = permute(load(b + s - 1), forward);
        const Vec a1 = permute(load(a + p - s - 8), backward), b1 = permute(load(b + p - s - 8), backward);
        sum_a = add_mod(add_mod(sum_a, a0), a1);
        sum_b = add_mod(add_mod(sum_b, b0), b1);
        const Vec lo0 = _mm256_unpacklo_epi32(a0, b0), hi0 = _mm256_unpackhi_epi32(a0, b0);
        const Vec lo1 = _mm256_unpacklo_epi32(a1, b1), hi1 = _mm256_unpackhi_epi32(a1, b1);
        auto* out = reinterpret_cast<Vec*>(quads.at(s));
        store(out, _mm256_unpacklo_epi64(lo0, lo1));
        store(out + 1, _mm256_unpackhi_epi64(lo0, lo1));
        store(out + 2, _mm256_unpacklo_epi64(hi0, hi1));
        store(out + 3, _mm256_unpackhi_epi64(hi0, hi1));
    }
    std::uint64_t ra = lane_sum(sum_a), rb = lane_sum(sum_b);
    for (; s <= half; ++s) {
        *quads.at(s) = {{a[s - 1], b[s - 1], a[p - s - 1], b[p - s - 1]}};
        ra += a[s - 1] + a[p - s - 1], rb += b[s - 1] + b[p - s - 1];
    }
    return {std::uint32_t(ra % kP), std::uint32_t(rb % kP)};
}

// The input but the quads: a_0, b_0 and the sums.
struct Rest {
    std::uint32_t a0, b0;
    Sums sums;
};

// Reads the input after p: a_i and b_i for 0 < i < p into quads, the rest into the result.
// a, b: scratch for p - 1 words.
Rest read_input(io::Reader& in, std::uint32_t p, std::uint32_t* a, std::uint32_t* b, const Quads& quads) {
    const std::uint32_t a0 = in.read<std::uint32_t>();
    io::read_bulk(in, a, p - 1);
    const std::uint32_t b0 = in.read<std::uint32_t>();
    io::read_bulk(in, b, p - 1);
    return {a0, b0, fold_input(a, b, p, quads)};
}

// The quads of eight slots, transposed: a and b of each slot's first and second index.
class Fetch {
public:
    explicit Fetch(const Quads& quads)
        : base_(quads.base), split_(broadcast(quads.split)), offset_(broadcast(quads.offset)) {}

    void operator()(Vec slot, Vec& a0, Vec& b0, Vec& a1, Vec& b1) const {
        alignas(32) std::uint32_t index[8];
        store(index, _mm256_add_epi32(slot, _mm256_and_si256(_mm256_cmpgt_epi32(slot, split_), offset_)));
        const auto quad = [this](std::uint32_t i) { return _mm_load_si128(reinterpret_cast<const __m128i*>(base_ + i)); };
        const auto pair = [&](int j) { return _mm256_inserti128_si256(_mm256_castsi128_si256(quad(index[j])), quad(index[j + 4]), 1); };
        const Vec q04 = pair(0), q15 = pair(1), q26 = pair(2), q37 = pair(3);
        const Vec t0 = _mm256_unpacklo_epi32(q04, q15), t1 = _mm256_unpackhi_epi32(q04, q15);
        const Vec t2 = _mm256_unpacklo_epi32(q26, q37), t3 = _mm256_unpackhi_epi32(q26, q37);
        a0 = _mm256_unpacklo_epi64(t0, t2), b0 = _mm256_unpackhi_epi64(t0, t2);
        a1 = _mm256_unpacklo_epi64(t1, t3), b1 = _mm256_unpackhi_epi64(t1, t3);
    }

private:
    const Quad* base_;
    Vec split_, offset_;
};

Vec select(Vec mask, Vec yes, Vec no) { return _mm256_blendv_epi8(no, yes, mask); }

// A[x] = a_(g^x), B[x] = b_(g^x) for x < n = p - 1; A and B beyond n stay zero.
void gather(std::uint32_t* a, std::uint32_t* b, const Input& in) {
    const std::uint32_t half = (in.p - 1) / 2;
    const Powers powers(in.g, in.p);
    const Fetch fetch(in.quads);
    Vec y = powers.at(0);
    std::uint32_t x = 0;
    for (; x + 8 <= half; x += 8, y = powers.step(y)) {
        const Powers::Slots s = powers.slots(y);
        Vec a0, b0, a1, b1;
        fetch(s.slot, a0, b0, a1, b1);
        store(a + x, select(s.second, a1, a0)), store(a + x + half, select(s.second, a0, a1));
        store(b + x, select(s.second, b1, b0)), store(b + x + half, select(s.second, b0, b1));
    }
    for (std::uint32_t z = std::uint32_t(_mm256_cvtsi256_si32(y)); x < half; ++x, z = powers.next(z)) {
        const std::uint32_t* quad = in.quads.at(std::min(z, in.p - z))->word;
        const int second = z > half ? 2 : 0;
        a[x] = quad[second], b[x] = quad[second + 1];
        a[x + half] = quad[2 - second], b[x + half] = quad[3 - second];
    }
}

// Folded c: c[s] = c_s + 2^32 c_(p - s) for slots s, from c_(g^k) = d[k] + d[k + n].
void scatter(std::uint64_t* c, const std::uint32_t* d, std::uint32_t p, std::uint32_t g) {
    const std::uint32_t n = p - 1, half = n / 2;
    const Powers powers(g, p);
    const Vec order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);  // of the unpacked pairs
    alignas(32) std::uint32_t index[8];
    alignas(32) std::uint64_t value[8];
    Vec y = powers.at(0);
    std::uint32_t k = 0;
    for (; k + 8 <= half; k += 8, y = powers.step(y)) {
        const Powers::Slots s = powers.slots(y);
        const Vec c0 = add_mod(load(d + k), load(d + k + n)), c1 = add_mod(load(d + k + half), load(d + k + half + n));
        const Vec first = select(s.second, c1, c0), second = select(s.second, c0, c1);
        store(value, _mm256_unpacklo_epi32(first, second)), store(value + 4, _mm256_unpackhi_epi32(first, second));
        store(index, permute(s.slot, order));
        for (int j = 0; j < 8; ++j) c[index[j]] = value[j];
    }
    for (std::uint32_t z = std::uint32_t(_mm256_cvtsi256_si32(y)); k < half; ++k, z = powers.next(z)) {
        const std::uint64_t c0 = (d[k] + d[k + n]) % kP, c1 = (d[k + half] + d[k + half + n]) % kP;
        c[std::min(z, p - z)] = z > half ? c1 | c0 << 32 : c0 | c1 << 32;
    }
}

// Dword kHalf of w[2 (s + j), 2 (s + j) + 2) for j < 8, in lane order [0 1 4 5 | 2 3 6 7].
template <int kHalf>
Vec halves(const std::uint32_t* w, std::uint32_t s) {
    const __m256 x = _mm256_castsi256_ps(load(w + 2 * s)), y = _mm256_castsi256_ps(load(w + 2 * s + 8));
    return _mm256_castps_si256(_mm256_shuffle_ps(x, y, kHalf ? 0xDD : 0x88));
}

// c_0, .., c_(p - 1) into out from folded c (c[0] = c_0).
void unfold(std::uint32_t* out, const std::uint64_t* c, std::uint32_t p) {
    const std::uint32_t half = (p - 1) / 2;
    const auto* w = reinterpret_cast<const std::uint32_t*>(c);  // w[2 s] = c_s, w[2 s + 1] = c_(p - s)
    std::uint32_t i = 0;
    for (; i + 8 <= half + 1; i += 8) store(out + i, _mm256_permute4x64_epi64(halves<0>(w, i), 0xD8));
    for (; i <= half; ++i) out[i] = w[2 * i];
    // c_(p - s) for s = p - i - 7 .. p - i, last first.
    const Vec reverse = _mm256_setr_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    for (; i + 8 <= p; i += 8) store(out + i, permute(halves<1>(w, p - i - 7), reverse));
    for (; i < p; ++i) out[i] = w[2 * (p - i) + 1];
}

// A * B for factors of n coefficients when ntt::Product takes them (transform length >= 512).
// Single use.
class Product {
public:
    static bool fits(std::size_t n) { return ntt::Product::fits(n, n); }

    explicit Product(std::size_t n) : product_(n, n, fields::kTextBytes) {}

    // Room for the input until load(): the quads in the factors' upper halves (2^lg / 8 each),
    // which ntt::Product does not read, a_i and b_i in their lower halves.
    Quads quads() {
        const std::size_t eighth = product_.length() / 8;
        std::uint32_t *a = product_.a(), *b = product_.b();
        return {reinterpret_cast<Quad*>(a + 4 * eighth) - 1, std::uint32_t(eighth), std::uint32_t((b - a) / 4 - eighth)};
    }
    std::uint32_t* scratch_a() { return product_.a(); }
    std::uint32_t* scratch_b() { return product_.b(); }
    // Folded c after multiply(): (2^lg / 2 >= n) qwords.
    std::uint64_t* folded() { return reinterpret_cast<std::uint64_t*>(product_.b()); }
    // p words for c after the product is consumed.
    std::uint32_t* values() { return product_.a(); }
    // fields::kTextBytes bytes for the output, after the tables (in their huge page).
    char* text() { return static_cast<char*>(product_.extra()); }

    void load(const Input& in) { gather(product_.a(), product_.b(), in); }

    // The coefficients of A * B, canonical; b() is destroyed.
    const std::uint32_t* multiply() { return product_.multiply(); }

private:
    ntt::Product product_;
};

// Other lengths: gather, then ntt::Convolution.
class SmallProduct {
public:
    explicit SmallProduct(std::size_t n)
        : convolution_(n, n), quads_(new Quad[n / 2 + 1]), folded_(new std::uint64_t[n / 2 + 1]) {}

    Quads quads() { return {quads_.get(), 1u << 30, 0}; }
    std::uint32_t* scratch_a() { return convolution_.a(); }
    std::uint32_t* scratch_b() { return convolution_.b(); }
    std::uint64_t* folded() { return folded_.get(); }
    std::uint32_t* values() { return convolution_.b(); }

    char* text() {
        alignas(64) static char buffer[fields::kTextBytes];
        return buffer;
    }

    void load(const Input& in) { gather(convolution_.a(), convolution_.b(), in); }
    const std::uint32_t* multiply() { return convolution_.multiply(); }

private:
    ntt::Convolution convolution_;
    std::unique_ptr<Quad[]> quads_;
    std::unique_ptr<std::uint64_t[]> folded_;
};

template <class Multiplier>
void convolve(io::Reader& in, std::uint32_t p) {
    Multiplier product(p - 1);
    const Rest rest = read_input(in, p, product.scratch_a(), product.scratch_b(), product.quads());
    const std::uint32_t g = primitive_root(p);
    product.load({product.quads(), p, g});
    const std::uint32_t* d = product.multiply();
    std::uint64_t* c = product.folded();
    scatter(c, d, p, g);
    // c_0 = a_0 (b_0 + sum b) + b_0 sum a.
    c[0] = (std::uint64_t(rest.a0) * ((rest.b0 + rest.sums.b) % kP) + std::uint64_t(rest.b0) * rest.sums.a) % kP;
    std::uint32_t* values = product.values();
    unfold(values, c, p);
    io::Writer out;
    fields::write(out, values, p, product.text());
}

// P = 2: c_0 = a_0 (b_0 + b_1) + a_1 b_0, c_1 = a_1 b_1.
void convolve_two(io::Reader& in) {
    std::uint64_t a[2], b[2];
    for (auto* x : {a, a + 1, b, b + 1}) *x = in.read<std::uint32_t>();
    io::Writer out;
    out.write((a[0] * ((b[0] + b[1]) % kP) + a[1] * b[0]) % kP, ' ', a[1] * b[1] % kP, '\n');
}

void solve() {
    io::Reader in;
    const std::uint32_t p = in.read<std::uint32_t>();
    if (p == 2) return convolve_two(in);
    if (Product::fits(p - 1)) return convolve<Product>(in, p);
    convolve<SmallProduct>(in, p);
}

}  // namespace

RUN_EARLY(solve)
