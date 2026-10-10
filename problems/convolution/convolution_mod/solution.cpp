// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1 (lib/ntt), output in fixed-width
// fields (fields.hpp). For 2^lg = 2 * 4^j >= 1024 (2^20 at the maximum) the forward top level is a
// radix-8 pass here, the bottom stage comes from bottom.hpp and the inverse top from top.hpp
// (Product); other lengths use ntt::Convolution as is. Inputs of 9-digit or 1-digit tokens take a
// fixed-width parser.
#include <unistd.h>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
#include "bottom.hpp"
#include "top.hpp"
#include "fields.hpp"

namespace {

using ntt::detail::add;
using ntt::detail::diff;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::reduce;
using ntt::detail::slot;
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

// ntt::detail::Recursion with the bottom stage of bottom.hpp: two groups per inlined kernel and
// no leaf weight array (13% less time in the bottom stage on Zen 3). Subtrees of at least 16
// vectors, so tiles start at even group indices.
class Subtrees {
public:
    Subtrees(const std::uint32_t* roots, const std::uint32_t* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // Forward transforms of a and b, leaf products into a, inverse transform; nv = 4^j >= 16.
    void visit(Vec* a, Vec* b, std::size_t nv, std::size_t k) const {
        switch (nv) {
        case 16: return tile<16>(a, b, k);
        case 64: return tile<64>(a, b, k);
        case 256: return tile<256>(a, b, k);
        }
        const std::size_t h = nv / 4;
        forward(a, b, h, k);
        for (std::size_t t = 0; t < 4; ++t) visit(a + t * h, b + t * h, h, 4 * k + t);
        inverse(a, h, k);
    }

private:
    template <std::size_t NV>
    [[gnu::noinline]] void tile(Vec* a, Vec* b, std::size_t k) const {
        for (std::size_t h = NV / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < NV; j += 4 * h, ++g) forward(a + j, b + j, h, g);
        bottom(a, b, NV, k * (NV / 4));
        for (std::size_t h = 4; h < NV; h *= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < NV; j += 4 * h, ++g) inverse(a + j, h, g);
    }

    void forward(Vec* a, Vec* b, std::size_t h, std::size_t k) const {
        if (k == 0) {
            ntt::kernels::forward_identity(a, h, r_);
            ntt::kernels::forward_identity(b, h, r_);
            return;
        }
        const std::uint32_t *x = r_ + slot(k), *y = r_ + slot(2 * k);
        if (h == 4) return ntt::kernels::forward_pair(a, b, h, x, y);
        ntt::kernels::forward(a, h, x, y);
        ntt::kernels::forward(b, h, x, y);
    }

    void inverse(Vec* a, std::size_t h, std::size_t k) const {
        if (k == 0) return ntt::kernels::inverse_identity(a, h, ir_);
        ntt::kernels::inverse(a, h, ir_ + slot(k), ir_ + slot(2 * k));
    }

    struct alignas(64) Leaves {
        std::uint32_t window[4][16];  // [w A_t, A_t]: x^i A_t mod x^8 - w is a sliding window
        std::uint32_t coefficients[4][8];
    };

    // Groups [first, first + nv / 4) with h = 1 and their leaves, two per kernel; first is even.
    // The next two groups' forward half overlaps the current two's products and inverse.
    void bottom(Vec* a, Vec* b, std::size_t nv, std::size_t first) const {
        Leaves leaves[2][2];
        bottom_kernels::first(a, b, leaves[0], r_ + slot(first), r_ + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const std::uint32_t *ix = ir_ + slot(k), *iy = ir_ + slot(2 * k);
            if (j + 8 == nv) return bottom_kernels::last(a + j, leaves[cur], ix, iy);
            bottom_kernels::both(a + j + 8, b + j + 8, leaves[next], r_ + slot(k + 2), r_ + slot(2 * k + 4), a + j,
                                 leaves[cur], ix, iy);
        }
    }

    const std::uint32_t *r_, *ir_;
};

// a * b when the transform length 2^lg is 2 * 4^j >= 1024 and each factor fills at most half of it.
// Same memory layout as ntt::Convolution; single use.
class Product {
public:
    static bool fits(std::size_t n, std::size_t m) {
        const int lg = log_length(n, m);
        return lg % 2 == 0 && lg >= 10 && 2 * std::max(n, m) <= std::size_t(1) << lg;
    }

    Product(std::size_t n, std::size_t m) : lg_(log_length(n, m)) {
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
    // fields::kTextBytes bytes for the output, 16-byte aligned, after the tables. At 2^20 it shares
    // their huge page (9.25 of 10 MiB used), so it costs no page faults.
    char* text() { return text_; }

    // The coefficients of a * b, canonical, in a(); b() is destroyed.
    const std::uint32_t* multiply() {
        using namespace ntt::detail;
        const std::size_t len = length(), nv = len / 8, q = nv / 8;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        auto* a = reinterpret_cast<Vec*>(a_);
        auto* b = reinterpret_cast<Vec*>(b_);
        forward_radix8(a, q, roots_);
        forward_radix8(b, q, roots_);
        // As ntt::Convolution from here: each half's subtrees, then its last group, the radix-2 level
        // and the scale in one pass.
        const Subtrees subtrees(roots_, inverse_roots_);
        for (std::size_t c = 0; c < 8; ++c) subtrees.visit(a + c * q, b + c * q, q, c);
        const std::uint32_t s = multiply_mod(power(std::uint32_t(nv), kP - 2), kR);  // undoes nv / 2^32
        const std::uint32_t *z0 = inverse_roots_ + 1, *x1 = inverse_roots_ + slot(1), *y1 = inverse_roots_ + slot(2);
        const std::uint32_t twiddles[6] = {s, multiply_mod(s, *z0), multiply_mod(s, y1[0]), multiply_mod(s, y1[1]),
                                           multiply_mod(s, *x1), *x1};
        alignas(32) Vec w[12];
        for (int i = 0; i < 6; ++i) w[2 * i] = broadcast(twiddles[i]), w[2 * i + 1] = broadcast(quotient(twiddles[i]));
        top_kernels::inverse_top(a, q, w);
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
    char* text_;
};

// Input fast paths for inputs whose tokens all have 9 digits (13 of the 16 large tests) or 1 digit
// (all_same_00), each followed by one separator: token i then starts at a fixed stride, so blocks
// need no separator search. Each block is checked by its separator mask; the first block that
// fails, and everything after it, goes to io::read_bulk.
namespace fixed_width {

constexpr char Z = char(0x80);  // shuffle control: zero byte

__m256i load(const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }

// Bit i set iff byte i is a separator (at most ' ').
std::uint32_t separators(__m256i bytes) {
    return std::uint32_t(_mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_set1_epi8(' ' + 1), bytes)));
}

__m256i digits(__m256i bytes, const char (&order)[32]) {
    return _mm256_shuffle_epi8(_mm256_subs_epu8(bytes, _mm256_set1_epi8('0')), load(order));
}

// A 32-byte load at token t of a 9-digit block holds t at byte 0 and t + 2 at byte 20 (byte 4 of
// the high lane); its separator mask is kNineSeparators.
constexpr std::uint32_t kNineSeparators = 1u << 9 | 1u << 19 | 1u << 29;
alignas(32) constexpr char kNineOrder[32] = {1, 2, 3, 4, 5, 6, 7, 8, Z, 0, Z, Z, Z, Z, Z, Z,      // [d1..d8, 0, d0]
                                             5, 6, 7, 8, 9, 10, 11, 12, Z, 4, Z, Z, Z, Z, Z, Z};  // the same at byte 4
alignas(32) constexpr std::int8_t kTens[32] = {10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0, 0, 0,
                                               10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0, 0, 0};

// Per lane, dwords [d1..d4, d5..d8, d0, 0] of the tokens at bytes 0 and 20.
__m256i groups(__m256i bytes) {
    const __m256i pairs = _mm256_maddubs_epi16(digits(bytes, kNineOrder), load(kTens));
    return _mm256_madd_epi16(pairs, _mm256_setr_epi16(100, 1, 100, 1, 1, 0, 0, 0, 100, 1, 100, 1, 1, 0, 0, 0));
}

// Token values in the even dwords, [t - u - | v - w -], for groups x = [t | v] and y = [u | w].
__m256i values(__m256i x, __m256i y) {
    const __m256i halves = _mm256_madd_epi16(_mm256_packus_epi32(x, y),
                                             _mm256_setr_epi16(10000, 1, 1, 0, 10000, 1, 1, 0, 10000, 1, 1, 0, 10000, 1, 1, 0));
    return _mm256_add_epi32(halves, _mm256_mul_epu32(_mm256_srli_epi64(halves, 32), _mm256_set1_epi32(100000000)));
}

// The 8 tokens at p (80 bytes) into dst, stored in any case; true if all have 9 digits and one
// separator. Reads p[0, 82).
bool nine_digits(const char* p, std::uint32_t* dst) {
    const __m256i l0 = load(p), l1 = load(p + 10), l2 = load(p + 40), l3 = load(p + 50);
    const std::uint32_t wrong = (separators(l0) ^ kNineSeparators) | (separators(l1) ^ kNineSeparators) |
                                (separators(l2) ^ kNineSeparators) | ((separators(l3) ^ kNineSeparators) & 0x3FFFFFFF);
    const __m256i low = values(groups(l0), groups(l1)), high = values(groups(l2), groups(l3));
    const __m256i mixed = _mm256_blend_epi32(low, _mm256_slli_epi64(high, 32), 0xAA);  // tokens 0 4 1 5 | 2 6 3 7
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), _mm256_permutevar8x32_epi32(mixed, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
    return !wrong;
}

alignas(32) constexpr char kOneOrder[32] = {0, 2, 4, 6, 8, 10, 12, 14, Z, Z, Z, Z, Z, Z, Z, Z,
                                            0, 2, 4, 6, 8, 10, 12, 14, Z, Z, Z, Z, Z, Z, Z, Z};

// As nine_digits for the 16 tokens at p (32 bytes) of one digit each. Reads p[0, 32).
bool one_digit(const char* p, std::uint32_t* dst) {
    const __m256i bytes = load(p), d = digits(bytes, kOneOrder);
    auto* out = reinterpret_cast<__m256i*>(dst);
    _mm256_storeu_si256(out, _mm256_cvtepu8_epi32(_mm256_castsi256_si128(d)));
    _mm256_storeu_si256(out + 1, _mm256_cvtepu8_epi32(_mm256_extracti128_si256(d, 1)));
    return separators(bytes) == 0xAAAAAAAA;
}

// Tokens parsed from p, at most count, in blocks of Block tokens of Stride bytes each.
template <std::size_t Stride, std::size_t Block, bool (*Parse)(const char*, std::uint32_t*)>
std::size_t parse(const char* p, std::uint32_t* dst, std::size_t count) {
    std::size_t done = 0;
    while (done + Block <= count && Parse(p + Stride * done, dst + done)) done += Block;
    return done;
}

}  // namespace fixed_width

// count values into dst: the fixed-width fast paths, then io::read_bulk for the rest.
void read_values(io::Reader& in, std::uint32_t* dst, std::size_t count) {
    const char* p = in.scan().cur;
    while (static_cast<unsigned char>(*p) <= ' ') ++p;
    std::size_t stride = 10, done = fixed_width::parse<10, 8, fixed_width::nine_digits>(p, dst, count);
    if (!done) stride = 2, done = fixed_width::parse<2, 16, fixed_width::one_digit>(p, dst, count);
    if (done) {  // continue the Reader after the parsed tokens, as io::detail::read_transposed does
        const char* const stop = p + stride * done;
        const auto* block = reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(stop) & ~std::uintptr_t(63));
        in.resume({stop, block, io::detail::block_separators(block) & ~std::uint64_t(0) << (stop - block)});
    }
    io::read_bulk(in, dst + done, count - done);
}

char* text(Product& product) { return product.text(); }

char* text(ntt::Convolution&) {
    alignas(64) static char buffer[fields::kTextBytes];
    return buffer;
}

template <class Multiplier>
void convolve(io::Reader& in, std::size_t n, std::size_t m) {
    Multiplier product(n, m);
    read_values(in, product.a(), n);
    read_values(in, product.b(), m);
    const std::uint32_t* c = product.multiply();
    io::Writer out;
    fields::write(out, c, n + m - 1, text(product));
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
