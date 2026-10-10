// Cyclic convolution of f and g in K variables mod a prime p, axis lengths n_i | p - 1.
// Short axes: a direct DFT over F_p along the axis (roots of order n_i exist since n_i | p - 1).
// Long axes: regrouped into the fewest cyclic factors D_r (Chinese remainder theorem); for each
// point of the short axes' spectrum, the exact product over Z by Kronecker substitution (factor
// r padded to 2 D_r - 1), modulo three NTT primes below 2^28 (convolution_mod_1000000007's lazy
// product), the CRT straight to residues mod p, then folded back to cyclic.
#include <algorithm>
#include <array>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "../convolution_mod/fields.hpp"
#include "../convolution_mod_1000000007/product.hpp"
#include "lib/run/early.hpp"

namespace {

using multimod::Vec;
using multimod::add;
using multimod::broadcast;
using multimod::Factor;
using multimod::multiply;
using multimod::reduce;

#ifdef SHORT_LIMIT
constexpr std::size_t kShortLimit = SHORT_LIMIT;  // test hook
#else
constexpr std::size_t kShortLimit = 48;
#endif

// Arithmetic mod a prime p < 2^30 (p = 2 included).
struct Field {
    std::uint32_t p;
    std::uint64_t barrett;  // floor((2^64 - 1) / p)

    explicit Field(std::uint32_t prime) : p(prime), barrett(~std::uint64_t(0) / prime) {}

    std::uint32_t multiply(std::uint32_t x, std::uint32_t y) const {
        const std::uint64_t v = std::uint64_t(x) * y;  // < 2^60
        const auto q = std::uint64_t((unsigned __int128)v * barrett >> 64);  // v / p - 1 <= q <= v / p
        const auto r = std::uint32_t(v - q * p);
        return r >= p ? r - p : r;
    }
    std::uint32_t power(std::uint32_t x, std::uint64_t e) const {
        std::uint32_t result = 1 % p;
        for (; e; e >>= 1, x = multiply(x, x))
            if (e & 1) result = multiply(result, x);
        return result;
    }
    std::uint32_t inverse(std::uint32_t x) const { return power(x, p - 2); }
    std::uint32_t quotient(std::uint32_t w) const { return std::uint32_t((std::uint64_t(w) << 32) / p); }
};

// A generator of F_p^*, p an odd prime.
std::uint32_t primitive_root(const Field& field) {
    std::vector<std::uint32_t> primes;
    std::uint32_t rest = field.p - 1;
    for (std::uint32_t d = 2; d * d <= rest; ++d) {
        if (rest % d) continue;
        primes.push_back(d);
        while (rest % d == 0) rest /= d;
    }
    if (rest > 1) primes.push_back(rest);
    for (std::uint32_t g = 2;; ++g) {
        const bool generates = std::all_of(primes.begin(), primes.end(),
                                           [&](std::uint32_t q) { return field.power(g, (field.p - 1) / q) != 1; });
        if (generates) return g;
    }
}

// Arena bytes that a take of `words` words may use (mem::Arena rounds takes up to 64 bytes).
std::size_t take_bytes(std::size_t words) { return 4 * words + 64; }

struct Axis {
    std::size_t length, stride;
};

struct alignas(32) Lane {
    Vec v;
};

// DFT of length n over F_p on 8 lanes at once: x[j stride], j < n, in place. Values stay
// canonical (< p).
class Dft {
public:
    Dft(const Field& field, std::uint32_t root, std::size_t n) : p_(field.p), n_(n), in_(n) {
        std::uint32_t w = 1;
        for (std::size_t e = 0; e < n; ++e, w = field.multiply(w, root)) factors_.emplace_back(w, field.quotient(w), p_);
    }

    // kMasked: stores only the lanes set in keep.
    template <bool kMasked = false>
    void operator()(std::uint32_t* x, std::size_t stride, Vec keep = Vec{}) {
        const Vec p = broadcast(p_), p2 = add(p, p);
        const auto at = [&](std::size_t j) { return reinterpret_cast<Vec*>(x + j * stride); };
        const auto store = [&](std::size_t j, Vec v) {
            if constexpr (kMasked) _mm256_maskstore_epi32(reinterpret_cast<int*>(at(j)), keep, v);
            else _mm256_storeu_si256(at(j), v);
        };
        for (std::size_t j = 0; j < n_; ++j) in_[j].v = _mm256_loadu_si256(at(j));
        if (n_ == 2) {
            store(0, reduce(add(in_[0].v, in_[1].v), p));
            store(1, reduce(_mm256_sub_epi32(add(in_[0].v, p), in_[1].v), p));
            return;
        }
        Vec sum = in_[0].v;
        for (std::size_t j = 1; j < n_; ++j) sum = reduce(add(sum, in_[j].v), p);
        store(0, sum);
        for (std::size_t k = 1; k < n_; ++k) {
            Vec acc = in_[0].v;  // < 2p
            std::size_t e = 0;
            for (std::size_t j = 1; j < n_; ++j) {
                e += k;
                if (e >= n_) e -= n_;
                acc = reduce(add(acc, multiply(in_[j].v, factors_[e])), p2);
            }
            store(k, reduce(acc, p));
        }
    }

private:
    std::uint32_t p_;
    std::size_t n_;
    std::vector<Factor> factors_;  // w^e, e < n
    std::vector<Lane> in_;
};

// An axis of stride s >= 8, 8 adjacent columns per step: x[a + s j + s n b] for a < s. The last
// step of each run of s columns overlaps the one before and stores only its new lanes.
void transform_wide(std::uint32_t* x, std::size_t total, Axis axis, Dft& dft) {
    const std::size_t s = axis.stride, full = s / 8 * 8;
    const Vec lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const Vec keep = _mm256_cmpgt_epi32(lane, _mm256_set1_epi32(int(7 - (s - full))));
    for (std::size_t b = 0; b < total; b += s * axis.length) {
        for (std::size_t a = 0; a < full; a += 8) dft(x + b + a, s);
        if (full < s) dft.operator()<true>(x + b + s - 8, s, keep);
    }
}

// Rows of m words hold every axis of stride < 8; rows_total is a multiple of 8 m. Eight rows at
// a time are interleaved into a buffer (word c of row l at 8 c + l) and transformed there.
void transform_narrow(std::uint32_t* x, std::size_t rows_total, std::size_t m, const std::vector<Axis>& axes,
                      std::vector<Dft>& dfts) {
    std::vector<Lane> buffer(m);
    auto* t = reinterpret_cast<std::uint32_t*>(buffer.data());
    for (std::size_t r = 0; r < rows_total; r += 8 * m) {
        for (std::size_t l = 0; l < 8; ++l)
            for (std::size_t c = 0; c < m; ++c) t[8 * c + l] = x[r + l * m + c];
        for (std::size_t i = 0; i < axes.size(); ++i) {
            const std::size_t s = axes[i].stride;
            for (std::size_t b = 0; b < m; b += s * axes[i].length)
                for (std::size_t a = 0; a < s; ++a) dfts[i](t + 8 * (b + a), 8 * s);
        }
        for (std::size_t l = 0; l < 8; ++l)
            for (std::size_t c = 0; c < m; ++c) x[r + l * m + c] = t[8 * c + l];
    }
}

// Words per row of the axes of stride < 8 (1 if none).
std::size_t narrow_row(const std::vector<Axis>& axes) {
    std::size_t m = 1;
    for (const Axis& axis : axes)
        if (axis.stride < 8) m = std::max(m, axis.stride * axis.length);
    return m;
}

// The DFT along every short axis, forward (root = generator) or inverse (root = its inverse).
// x holds total words, writable up to total rounded up to a multiple of 8 narrow_row(axes).
void transform_short(std::uint32_t* x, std::size_t total, const std::vector<Axis>& axes, const Field& field,
                     std::uint32_t root) {
    std::vector<Axis> narrow;
    std::vector<Dft> narrow_dfts;
    for (const Axis& axis : axes) {
        Dft dft(field, field.power(root, (field.p - 1) / axis.length), axis.length);
        if (axis.stride >= 8) {
            transform_wide(x, total, axis, dft);
        } else {
            narrow.push_back(axis);
            narrow_dfts.push_back(std::move(dft));
        }
    }
    const std::size_t m = narrow_row(axes);
    if (!narrow.empty()) transform_narrow(x, (total + 8 * m - 1) / (8 * m) * (8 * m), m, narrow, narrow_dfts);
}

constexpr int kPrimes = 3;
// Every coefficient over Z is below 2^18 p^2 < 2^77.8; the primes' product M is about 2^83.96.
constexpr std::array<std::array<std::uint32_t, 2>, kPrimes> kPrimeList = {{
    {268042241, 3}, {265420801, 11}, {264634369, 11}}};  // prime, generator; inputs < p <= 10^9 < 4 prime

constexpr std::uint32_t multiply_mod(std::uint64_t x, std::uint64_t y, std::uint32_t p) {
    return std::uint32_t(x % p * (y % p) % p);
}

constexpr std::uint32_t inverse_mod(std::uint32_t x, std::uint32_t p) {
    std::uint32_t result = 1;
    for (std::uint32_t e = p - 2; e; e >>= 1, x = multiply_mod(x, x, p))
        if (e & 1) result = multiply_mod(result, x, p);
    return result;
}

// 1 / M_k mod p_k with M_k = M / p_k: the transform for prime k returns y_k = c / M_k mod p_k.
constexpr std::array<std::uint32_t, kPrimes> kCrtScale = [] {
    std::array<std::uint32_t, kPrimes> scale{};
    for (int k = 0; k < kPrimes; ++k) {
        std::uint32_t other = 1;
        for (int j = 0; j < kPrimes; ++j)
            if (j != k) other = multiply_mod(other, kPrimeList[j][0], kPrimeList[k][0]);
        scale[k] = inverse_mod(other, kPrimeList[k][0]);
    }
    return scale;
}();

using Residues = std::array<std::uint32_t*, kPrimes>;

// c = sum_k y_k M_k - t M with t = floor(sum_k y_k / p_k) (the sum's fraction is c / M < 2^-5
// for c < 2^78.8, a folded sum of two coefficients included). Modulo p, times a factor z, with
// R = 2^32: s = sum_k y_k (M_k R z mod p) + t (-M R z mod p) < (3 2^28 + 3) p < R p, and
// s / R mod p = c z mod p (Montgomery reduction; p odd).
struct Crt {
    std::array<std::uint32_t, kPrimes> place{};  // M_k R z mod p
    std::uint32_t wrap = 0;                      // -M R z mod p
    std::uint32_t neg_inverse = 0;               // -1 / p mod 2^32
    std::uint32_t p = 0;

    Crt(const Field& field, std::uint32_t z) : p(field.p) {
        const std::uint32_t rz = field.multiply(std::uint32_t((std::uint64_t(1) << 32) % p), z);
        std::uint32_t modulus = 1;  // M mod p
        for (int k = 0; k < kPrimes; ++k) {
            std::uint32_t others = rz;
            for (int j = 0; j < kPrimes; ++j)
                if (j != k) others = field.multiply(others, kPrimeList[j][0] % p);
            place[k] = others;
            modulus = field.multiply(modulus, kPrimeList[k][0] % p);
        }
        wrap = field.multiply((p - modulus) % p, rz);
        std::uint32_t x = p;  // Newton: correct to 3, 6, 12, 24, 48 bits
        for (int i = 0; i < 4; ++i) x *= 2 - p * x;
        neg_inverse = 0 - x;
    }

    // c z mod p of coefficients [0, count) into out; the residues are readable up to count rounded up to 8.
    void reconstruct(const Residues& y, std::size_t count, std::uint32_t* out) const {
        for (std::size_t i = 0; i < count; i += 8) {
            const auto load = [&](int k) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(y[k] + i)); };
            _mm256_store_si256(reinterpret_cast<Vec*>(out + i), combine(load(0), load(1), load(2)));
        }
    }

    // (c_j + c_{j + n}) z mod p for j < n into out[0, n); the residues are readable up to 2n + 7.
    // In place: out may be the storage of a residue array.
    void reconstruct_folded(const Residues& y, std::size_t n, std::uint32_t* out) const {
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) _mm256_storeu_si256(reinterpret_cast<Vec*>(out + i), folded(y, i, n));
        if (i == n) return;
        const Vec keep = _mm256_cmpgt_epi32(broadcast(std::uint32_t(n - i)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        _mm256_maskstore_epi32(reinterpret_cast<int*>(out + i), keep, folded(y, i, n));
    }

private:
    // (c_j + c_{j + n}) z mod p for j in [i, i + 8): residues folded per prime, then one CRT.
    Vec folded(const Residues& y, std::size_t i, std::size_t n) const {
        const auto fold = [&](int k) {
            const auto load = [&](std::size_t j) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(y[k] + j)); };
            return reduce(add(load(i), load(i + n)), broadcast(kPrimeList[k][0]));
        };
        return combine(fold(0), fold(1), fold(2));
    }

    // c z mod p from canonical residues y_k of 8 coefficients, canonical.
    Vec combine(Vec y0, Vec y1, Vec y2) const {
        const auto inv = [](int k) { return _mm256_set1_ps(1.0f / float(kPrimeList[k][0])); };
        const auto term = [](Vec v) { return _mm256_cvtepi32_ps(v); };
        const __m256 sum = _mm256_fmadd_ps(term(y0), inv(0), _mm256_fmadd_ps(term(y1), inv(1), _mm256_set1_ps(0.5f)));
        const Vec t = _mm256_cvttps_epi32(_mm256_fmadd_ps(term(y2), inv(2), sum));  // < 3
        const auto set = [](std::uint32_t c) { return _mm256_set1_epi64x(c); };
        // s / R mod p in the high dword of each qword, < 2p; x: the even dwords.
        const auto montgomery = [&](Vec x0, Vec x1, Vec x2, Vec tx) {
            const Vec s = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(x0, set(place[0])), _mm256_mul_epu32(x1, set(place[1]))),
                _mm256_add_epi64(_mm256_mul_epu32(x2, set(place[2])), _mm256_mul_epu32(tx, set(wrap))));
            const Vec m = _mm256_mul_epu32(s, set(neg_inverse));
            return _mm256_add_epi64(s, _mm256_mul_epu32(m, set(p)));
        };
        const auto odd = [](Vec x) { return _mm256_srli_epi64(x, 32); };
        const Vec even_sum = montgomery(y0, y1, y2, t), odd_sum = montgomery(odd(y0), odd(y1), odd(y2), odd(t));
        const Vec u = _mm256_blend_epi32(_mm256_srli_epi64(even_sum, 32), odd_sum, 0xAA);
        return reduce(u, broadcast(p));
    }
};

// 1 / x mod m, gcd(x, m) = 1.
std::size_t unit_inverse(std::size_t x, std::size_t m) {
    std::int64_t a = std::int64_t(x), b = std::int64_t(m), u = 1, v = 0;
    while (b) {
        const std::int64_t t = a / b;
        a -= t * b, u -= t * v;
        std::swap(a, b), std::swap(u, v);
    }
    return std::size_t((u % std::int64_t(m) + std::int64_t(m)) % std::int64_t(m));
}

// The group Z_n1 x ... x Z_nk of some axes as the fewest cyclic factors Z_D1 x ... x Z_Dr: the
// prime-power parts of the n_i, the largest power of each prime to factor 0, the next to factor
// 1, and so on (Chinese remainder theorem). Point (i_1..i_k) maps to y_r = sum_i i_i w[i][r] mod
// D_r; a convolution over the axes is one over the factors.
struct CyclicFactors {
    std::vector<std::size_t> length;               // D_r
    std::vector<std::vector<std::size_t>> weight;  // w[i][r]: 1 mod the parts of n_i in D_r, 0 mod the rest

    explicit CyclicFactors(const std::vector<Axis>& axes) : weight(axes.size()) {
        struct Part {
            std::size_t prime, power, axis;
        };
        std::vector<Part> parts;
        for (std::size_t i = 0; i < axes.size(); ++i) {
            std::size_t n = axes[i].length;
            for (std::size_t q = 2; n > 1; ++q) {
                if (q * q > n) q = n;
                if (n % q) continue;
                std::size_t power = 1;
                while (n % q == 0) n /= q, power *= q;
                parts.push_back({q, power, i});
            }
        }
        std::sort(parts.begin(), parts.end(),
                  [](const Part& x, const Part& y) { return x.prime != y.prime ? x.prime < y.prime : x.power > y.power; });
        std::vector<std::size_t> factor(parts.size());
        for (std::size_t j = 0; j < parts.size(); ++j) {
            factor[j] = j > 0 && parts[j - 1].prime == parts[j].prime ? factor[j - 1] + 1 : 0;
            if (factor[j] == length.size()) length.push_back(1);
            length[factor[j]] *= parts[j].power;
        }
        for (auto& w : weight) w.assign(length.size(), 0);
        for (std::size_t j = 0; j < parts.size(); ++j) {
            const std::size_t d = length[factor[j]], q = parts[j].power, rest = d / q;
            std::size_t& w = weight[parts[j].axis][factor[j]];
            w = (w + rest * unit_inverse(rest % q, q)) % d;
        }
    }

    // Kronecker extent: the product of 2 D_r - 1.
    std::size_t padded() const {
        std::size_t r = 1;
        for (const std::size_t d : length) r *= 2 * d - 1;
        return r;
    }
};

// 8 x 8 transpose: v[k][l] <- v[l][k].
inline void transpose8(Vec (&v)[8]) {
    Vec t[8], u[8];
    for (int i = 0; i < 8; i += 2) {
        t[i] = _mm256_unpacklo_epi32(v[i], v[i + 1]);
        t[i + 1] = _mm256_unpackhi_epi32(v[i], v[i + 1]);
    }
    for (int i = 0; i < 8; i += 4) {
        u[i] = _mm256_unpacklo_epi64(t[i], t[i + 2]);
        u[i + 1] = _mm256_unpackhi_epi64(t[i], t[i + 2]);
        u[i + 2] = _mm256_unpacklo_epi64(t[i + 1], t[i + 3]);
        u[i + 3] = _mm256_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int i = 0; i < 4; ++i) {
        v[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
        v[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
    }
}

// Two axes x and y of coprime lengths as one cyclic factor of length D = n_x n_y. Place
// j = r n_y + c (r < n_x, c < n_y) holds the point i_x = (r + a c) mod n_x, i_y = c, with
// a = 1 / n_y mod n_x: then i_x = a j mod n_x and i_y = j mod n_y, a group isomorphism. As an
// n_x x n_y matrix, A[r][c] = F[c][(r + a c) mod n_x] with F[c] the row of f along x at i_y = c:
// a transpose of rotated rows. With x of stride 1 and n_x, n_y >= 8 it runs in 8 x 8 blocks,
// column blocks outer (the 8 rows of F stay in L1).
class SkewedTranspose {
public:
    SkewedTranspose(Axis x, Axis y)
        : x_(x), y_(y), a_(unit_inverse(y.length % x.length, x.length)),
          wide_(x.stride == 1 && x.length >= 8 && y.length >= 8 ? y.length / 8 * 8 : 0) {}

    // place[r n_y + c] <- f[i_x s_x + c s_y].
    void gather(const std::uint32_t* f, std::uint32_t* place) const {
        for (std::size_t c0 = 0; c0 < wide_; c0 += 8) {
            const std::uint32_t* rows = f + c0 * y_.stride;
            std::size_t start[8];
            first_starts(c0, start);
            for (std::size_t r0 = 0; r0 < x_.length; r0 += 8) {
                Vec v[8];
                for (std::size_t l = 0; l < 8; ++l) v[l] = load(rows + l * y_.stride, start[l]);
                transpose8(v);
                const std::size_t count = std::min<std::size_t>(8, x_.length - r0);
                for (std::size_t k = 0; k < count; ++k)
                    _mm256_storeu_si256(reinterpret_cast<Vec*>(place + (r0 + k) * y_.length + c0), v[k]);
                advance(start);
            }
        }
        for (std::size_t r = 0; r < x_.length; ++r)
            for_row(r, wide_, y_.length, [&](std::size_t c, std::size_t at) { place[r * y_.length + c] = f[at]; });
    }

    // f[i_x s_x + c s_y] <- place[r n_y + c].
    void scatter(const std::uint32_t* place, std::uint32_t* f) const {
        for (std::size_t c0 = 0; c0 < wide_; c0 += 8) {
            std::uint32_t* rows = f + c0 * y_.stride;
            std::size_t start[8];
            first_starts(c0, start);
            for (std::size_t r0 = 0; r0 < x_.length; r0 += 8) {
                const std::size_t count = std::min<std::size_t>(8, x_.length - r0);
                Vec v[8];
                for (std::size_t k = 0; k < 8; ++k)
                    v[k] = k < count ? _mm256_loadu_si256(reinterpret_cast<const Vec*>(place + (r0 + k) * y_.length + c0))
                                     : _mm256_setzero_si256();
                transpose8(v);
                for (std::size_t l = 0; l < 8; ++l) store(rows + l * y_.stride, start[l], count, v[l]);
                advance(start);
            }
        }
        for (std::size_t r = 0; r < x_.length; ++r)
            for_row(r, wide_, y_.length, [&](std::size_t c, std::size_t at) { f[at] = place[r * y_.length + c]; });
    }

private:
    static Vec lanes() { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
    // Lanes k < bound.
    static Vec below(std::size_t bound) { return _mm256_cmpgt_epi32(broadcast(std::uint32_t(bound)), lanes()); }

    // Row starts (a c) mod n_x of columns c0 + l.
    void first_starts(std::size_t c0, std::size_t (&start)[8]) const {
        for (std::size_t l = 0; l < 8; ++l) start[l] = a_ * (c0 + l) % x_.length;
    }

    void advance(std::size_t (&start)[8]) const {
        for (std::size_t& s : start) {
            s += 8;
            s = s >= x_.length ? s - x_.length : s;
        }
    }

    // row[(s + k) mod n_x], k < 8.
    Vec load(const std::uint32_t* row, std::size_t s) const {
        if (s + 8 <= x_.length) [[likely]]
            return _mm256_loadu_si256(reinterpret_cast<const Vec*>(row + s));
        const Vec low = below(x_.length - s);
        return _mm256_or_si256(_mm256_maskload_epi32(reinterpret_cast<const int*>(row + s), low),
                               _mm256_maskload_epi32(reinterpret_cast<const int*>(row + s - x_.length),
                                                     _mm256_andnot_si256(low, below(8))));
    }

    // row[(s + k) mod n_x] <- v[k], k < count.
    void store(std::uint32_t* row, std::size_t s, std::size_t count, Vec v) const {
        if (count == 8 && s + 8 <= x_.length) [[likely]] {
            _mm256_storeu_si256(reinterpret_cast<Vec*>(row + s), v);
            return;
        }
        const Vec valid = below(count), low = below(x_.length - s);
        _mm256_maskstore_epi32(reinterpret_cast<int*>(row + s), _mm256_and_si256(valid, low), v);
        _mm256_maskstore_epi32(reinterpret_cast<int*>(row + s - x_.length), _mm256_andnot_si256(low, valid), v);
    }

    // visit(c, offset of place r n_y + c in f) for c in [first, last).
    template <class Visit>
    void for_row(std::size_t r, std::size_t first, std::size_t last, Visit visit) const {
        std::size_t i = (r + a_ * first) % x_.length;
        for (std::size_t c = first; c < last; ++c) {
            visit(c, i * x_.stride + c * y_.stride);
            i += a_;
            i = i >= x_.length ? i - x_.length : i;
        }
    }

    Axis x_, y_;
    std::size_t a_;
    std::size_t wide_;  // columns done in 8 x 8 blocks
};

// Transform length for a Kronecker extent (lazy::Product needs 2^9 at least).
int transform_log(std::size_t padded) { return std::max(9, int(std::bit_width(padded - 1))); }

// Products over the long axes, one per point of the short axes' spectrum. The two factors of a
// product are the halves of one pair array of 2^lg words, each zero past its places; the pair then
// holds the last prime's residues (lazy::Product in place). Direct (one long axis, of stride 1):
// the caller's array holds pairs [row r of f | row r of g], and the products run on them in place.
class LongProduct {
public:
    // The long axes' cyclic factors have a padded extent of at most 2^kMaxLog.
    LongProduct(const std::vector<Axis>& axes, const Field& field, std::uint32_t scale, mem::Arena& arena)
        : crt_(field, scale), p_(field.p), axes_(axes) {
        CyclicFactors factors(axes);
        lengths_ = std::move(factors.length);
        weights_ = std::move(factors.weight);
        padded_ = 1;
        for (const std::size_t d : lengths_) {
            strides_.push_back(padded_);
            padded_ *= 2 * d - 1;
        }
        lg_ = transform_log(padded_);
        const std::size_t words = (std::size_t(1) << lg_) + lazy::kPadding;
        if (!is_direct(axes)) pair_ = arena.take<std::uint32_t>(words);
        for (int k = 0; k + 1 < kPrimes; ++k) residues_[k] = arena.take<std::uint32_t>(words);
        work_ = arena.take<std::uint32_t>(words);
        tables_ = arena.take<std::uint32_t>(lazy::Product::table_words(lg_));
        for (int k = 0; k < kPrimes; ++k) moduli_.emplace_back(kPrimeList[k][0], kPrimeList[k][1]);
    }

    static std::size_t arena_bytes(const std::vector<Axis>& axes) {
        const int lg = transform_log(CyclicFactors(axes).padded());
        const std::size_t words = (std::size_t(1) << lg) + lazy::kPadding;
        return (kPrimes + 1) * take_bytes(words) + take_bytes(lazy::Product::table_words(lg));
    }

    static bool is_direct(const std::vector<Axis>& axes) { return axes.size() == 1 && axes[0].stride == 1; }

    // Words per pair in the direct layout: the transform length.
    static std::size_t pair_words(const std::vector<Axis>& axes) {
        return std::size_t(1) << transform_log(CyclicFactors(axes).padded());
    }

    // Direct: pair r < count of x (rows of n values, zero up to half the pair) into x[r n, r n + n),
    // times the scale. For r >= 1 that range lies in pairs < r (n <= half a pair); for r = 0 the
    // CRT runs in place. x is readable up to count pairs plus lazy::kPadding words.
    void multiply_rows(std::uint32_t* x, std::size_t count) {
        const std::size_t n = lengths_[0];
        for (std::size_t r = 0; r < count; ++r) {
            products(x + 2 * r * half());
            crt_.reconstruct_folded(residues_, n, x + r * n);
        }
    }

    // Two long axes of coprime lengths and no short ones, f and g still in the input: read(x)
    // parses D values into x, f then g; write(x) prints D values. Each is staged in an array of
    // the product that is free at that time (f in b, g in work), the result in work.
    template <class Read, class Write>
    void multiply_input(Read read, Write write) {
        const SkewedTranspose transpose(axes_[0], axes_[1]);
        std::uint32_t *a = pair_, *b = pair_ + half();
        read(b);
        transpose.gather(b, a);
        read(work_);
        transpose.gather(work_, b);
        products(pair_);
        crt_.reconstruct_folded(residues_, lengths_[0], a);
        transpose.scatter(a, work_);
        write(work_);
    }

    // f[base + offset] <- (f * g)[base + offset] over the long axes, times the scale. The pair is
    // zero between calls but at the points' places; the last call (last = true) leaves it dirty.
    void multiply(std::uint32_t* f, const std::uint32_t* g, std::size_t base, bool last) {
        if (lengths_.size() == 1) return multiply_cyclic(f, g, base, last);
        std::uint32_t *a = pair_, *b = pair_ + half();
        for_each_point([&](std::size_t offset, std::size_t place) {
            a[place] = f[base + offset];
            b[place] = g[base + offset];
        });
        products(pair_);
        crt_.reconstruct(residues_, (padded_ + 7) & ~std::size_t(7), a);
        fold(a);
        for_each_point([&](std::size_t offset, std::size_t place) { f[base + offset] = a[place]; });
        if (!last) std::memset(pair_, 0, 2 * half() * sizeof(std::uint32_t));
    }

private:
    std::size_t half() const { return std::size_t(1) << (lg_ - 1); }

    // One cyclic factor: places are a permutation of [0, D), and the CRT folds as it goes. Only
    // the halves' tails past D need clearing.
    void multiply_cyclic(std::uint32_t* f, const std::uint32_t* g, std::size_t base, bool last) {
        const std::size_t d = lengths_[0];
        std::uint32_t *a = pair_, *b = pair_ + half();
        if (axes_.size() == 2) {
            const SkewedTranspose transpose(axes_[0], axes_[1]);
            transpose.gather(f + base, a);
            transpose.gather(g + base, b);
            products(pair_);
            crt_.reconstruct_folded(residues_, d, a);
            transpose.scatter(a, f + base);
        } else {
            for_each_point([&](std::size_t offset, std::size_t place) {
                a[place] = f[base + offset];
                b[place] = g[base + offset];
            });
            products(pair_);
            crt_.reconstruct_folded(residues_, d, a);
            for_each_point([&](std::size_t offset, std::size_t place) { f[base + offset] = a[place]; });
        }
        if (last) return;
        std::memset(a + d, 0, (half() - d) * sizeof(std::uint32_t));
        std::memset(b + d, 0, (half() - d) * sizeof(std::uint32_t));
    }

    // The product of pair's halves (readable 8 words past each) mod every prime: residues 0 and 1,
    // then the last prime's in place, which becomes residues_[2].
    void products(std::uint32_t* pair) {
        const lazy::Product product(lg_, tables_);
        for (int k = 0; k < kPrimes; ++k)
            product.multiply(pair, pair + half(), k + 1 < kPrimes ? residues_[k] : pair, work_, moduli_[k],
                             kCrtScale[k]);
        residues_[kPrimes - 1] = pair;
    }

    // visit(offset in f, place in the Kronecker array) for every point of the long axes, axis 0
    // innermost. After n_i steps along axis i, every y_r is back where it started.
    template <class Visit>
    void for_each_point(Visit visit) const {
        const std::size_t factors = lengths_.size(), n0 = axes_[0].length, s0 = axes_[0].stride;
        std::vector<std::size_t> index(axes_.size()), y(factors), z(factors);
        std::size_t offset = 0;
        for (;;) {
            if (factors == 1) {
                const std::size_t d = lengths_[0], w = weights_[0][0];
                for (std::size_t j = 0, at = offset, place = y[0]; j < n0; ++j, at += s0) {
                    visit(at, place);
                    place += w;
                    place = place >= d ? place - d : place;
                }
            } else {
                z = y;
                for (std::size_t j = 0, at = offset; j < n0; ++j, at += s0) {
                    std::size_t place = 0;
                    for (std::size_t r = 0; r < factors; ++r) place += z[r] * strides_[r];
                    visit(at, place);
                    step(z, 0);
                }
            }
            std::size_t i = 1;
            for (; i < axes_.size(); ++i) {
                offset += axes_[i].stride;
                step(y, i);
                if (++index[i] < axes_[i].length) break;
                index[i] = 0;
                offset -= axes_[i].length * axes_[i].stride;
            }
            if (i == axes_.size()) return;
        }
    }

    // One step along axis i.
    void step(std::vector<std::size_t>& y, std::size_t i) const {
        for (std::size_t r = 0; r < y.size(); ++r) {
            y[r] += weights_[i][r];
            if (y[r] >= lengths_[r]) y[r] -= lengths_[r];
        }
    }

    // Index D + j of a factor onto j, for every factor in turn.
    void fold(std::uint32_t* c) const {
        const std::uint32_t p = p_;
        for (std::size_t r = 0; r < lengths_.size(); ++r) {
            const std::size_t span = strides_[r] * (2 * lengths_[r] - 1), shift = lengths_[r] * strides_[r];
            for (std::size_t o = 0; o < padded_; o += span)
                for (std::size_t i = o; i < o + span - shift; ++i) {
                    const std::uint32_t sum = c[i] + c[i + shift];
                    c[i] = sum >= p ? sum - p : sum;
                }
        }
    }

    Crt crt_;
    std::uint32_t p_;
    std::vector<Axis> axes_;
    std::vector<std::size_t> lengths_, strides_;   // per cyclic factor: D_r, Kronecker stride
    std::vector<std::vector<std::size_t>> weights_;  // CyclicFactors::weight
    std::size_t padded_ = 0;
    int lg_ = 0;
    std::uint32_t *pair_ = nullptr, *work_, *tables_;  // pair_: none when direct
    Residues residues_{};
    std::vector<multimod::Modulus> moduli_;
};

// Offsets in f of every point of the given axes.
std::vector<std::uint32_t> points(const std::vector<Axis>& axes) {
    std::vector<std::uint32_t> result = {0};
    for (const Axis& axis : axes) {
        const std::size_t count = result.size();
        for (std::size_t i = 1; i < axis.length; ++i)
            for (std::size_t t = 0; t < count; ++t) result.push_back(std::uint32_t(result[t] + i * axis.stride));
    }
    return result;
}

// Modeled time in ns of a split into long and short axes, from lc-bench (EPYC 7B13): a product
// costs about 10.5 ns per transform word below 2^18 words and 12 from there (three primes; its
// arrays leave L2), plus 0.6 ns per word once (page faults); gather and scatter 2 ns per point
// unless direct; a short DFT 0.5 ns per element and axis length (three passes).
double modeled_cost(const std::vector<Axis>& long_axes, const std::vector<Axis>& axes, std::size_t total) {
    std::size_t long_total = 1, short_sum = 0;
    for (const Axis& axis : long_axes) long_total *= axis.length;
    for (const Axis& axis : axes) short_sum += axis.length;
    for (const Axis& axis : long_axes) short_sum -= axis.length;
    double cost = 0.5 * double(total) * double(short_sum);
    if (long_axes.empty()) return cost;
    const int lg = transform_log(CyclicFactors(long_axes).padded());
    const double words = double(std::size_t(1) << lg);
    cost += (lg < 18 ? 10.5 : 12.0) * double(total / long_total) * words + 0.6 * words;
    return LongProduct::is_direct(long_axes) ? cost : cost + 2.0 * double(total);
}

// Long axes: those above kShortLimit, the shortest dropped while the transform would exceed
// 2^kMaxLog. If any remain, a shorter axis joins them when that lowers the modeled cost.
std::vector<Axis> pick_long(const std::vector<Axis>& axes, std::size_t total) {
    std::vector<Axis> chosen;
    for (const Axis& axis : axes)
        if (axis.length > kShortLimit) chosen.push_back(axis);
    const auto fits = [](const std::vector<Axis>& s) {
        return CyclicFactors(s).padded() <= (std::size_t(1) << multimod::kMaxLog);
    };
    while (!fits(chosen)) {
        const auto shortest = std::min_element(chosen.begin(), chosen.end(),
                                               [](const Axis& x, const Axis& y) { return x.length < y.length; });
        chosen.erase(shortest);
    }
    if (chosen.empty()) return chosen;
    for (const Axis& axis : axes) {
        if (std::any_of(chosen.begin(), chosen.end(), [&](const Axis& c) { return c.stride == axis.stride; })) continue;
        std::vector<Axis> grown = chosen;
        grown.push_back(axis);
        std::sort(grown.begin(), grown.end(), [](const Axis& x, const Axis& y) { return x.stride < y.stride; });
        if (fits(grown) && modeled_cost(grown, axes, total) < modeled_cost(chosen, axes, total)) chosen = grown;
    }
    return chosen;
}

void solve() {
    io::Reader in;
    const std::uint32_t p = in.read<std::uint32_t>();
    const std::size_t k = in.read<std::uint32_t>();
    std::vector<Axis> axes;
    std::size_t total = 1;
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t n = in.read<std::uint32_t>();
        axes.push_back({n, total});
        total *= n;
    }
    const std::vector<Axis> long_axes = pick_long(axes, total);
    std::vector<Axis> short_axes;
    std::size_t short_total = 1;
    for (const Axis& axis : axes)
        if (std::none_of(long_axes.begin(), long_axes.end(), [&](const Axis& l) { return l.stride == axis.stride; })) {
            short_axes.push_back(axis);
            short_total *= axis.length;
        }
    const Field field(p);
    if (short_axes.empty() && long_axes.size() == 2 && CyclicFactors(long_axes).length.size() == 1) {
        // No arrays in input order: f, g and the result are staged in the product's arrays.
        mem::Arena arena(take_bytes(fields::kTextBytes / 4 + 1) + LongProduct::arena_bytes(long_axes));
        char* text = arena.take<char>(fields::kTextBytes);
        io::Writer out;
        LongProduct(long_axes, field, 1, arena)
            .multiply_input([&](std::uint32_t* x) { io::read_bulk(in, x, total); },
                            [&](const std::uint32_t* x) { fields::write(out, x, total, text); });
        return;
    }

    // Direct product (one long axis, of stride 1): f holds pairs [row r of f | row r of g], each row
    // of n values zero up to half the pair. Else g follows f.
    const bool direct = LongProduct::is_direct(long_axes);
    const std::size_t n = direct ? long_axes[0].length : total, rows = total / n;
    const std::size_t pair = direct ? LongProduct::pair_words(long_axes) : 0;
    const std::size_t block = 8 * narrow_row(short_axes);
    const std::size_t padded_total = (total + block - 1) / block * block;
    const std::size_t words = direct ? std::max(rows * pair + lazy::kPadding, padded_total) : 2 * padded_total;
    mem::Arena arena(take_bytes(words) + take_bytes(fields::kTextBytes / 4 + 1) +
                     (long_axes.empty() ? 0 : LongProduct::arena_bytes(long_axes)));
    auto* f = arena.take<std::uint32_t>(words);
    auto* g = direct ? f + pair / 2 : f + padded_total;
    const std::size_t gap = direct ? pair : n;  // between rows
    for (std::uint32_t* x : {f, g})
        for (std::size_t r = 0; r < rows; ++r) io::read_bulk(in, x + r * gap, n);

    const std::uint32_t scale = field.inverse(std::uint32_t(short_total % p));
    const std::uint32_t root = short_axes.empty() ? 1 : primitive_root(field);
    if (direct) {
        std::vector<Axis> spread = short_axes;  // in the layout of pairs, f and g at once
        for (Axis& axis : spread) axis.stride = axis.stride / n * pair;
        transform_short(f, rows * pair, spread, field, root);
        LongProduct(long_axes, field, scale, arena).multiply_rows(f, rows);
    } else {
        transform_short(f, total, short_axes, field, root);
        transform_short(g, total, short_axes, field, root);
        if (long_axes.empty()) {
            for (std::size_t i = 0; i < total; ++i) f[i] = field.multiply(field.multiply(f[i], g[i]), scale);
        } else {
            LongProduct product(long_axes, field, scale, arena);
            const std::vector<std::uint32_t> bases = points(short_axes);
            for (std::size_t i = 0; i < bases.size(); ++i) product.multiply(f, g, bases[i], i + 1 == bases.size());
        }
    }
    transform_short(f, total, short_axes, field, field.inverse(root));

    io::Writer out;
    fields::write(out, f, total, arena.take<char>(fields::kTextBytes));
}

}  // namespace

RUN_EARLY(solve)
