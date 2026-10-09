// Cyclic convolution of f and g in K variables mod a prime p, axis lengths n_i | p - 1.
// Short axes (n_i <= kShortLimit): a direct DFT over F_p along the axis (roots of order n_i
// exist since n_i | p - 1). Long axes: for each point of the short axes' spectrum, the exact
// product over Z by Kronecker substitution (long axis i padded to 2 n_i - 1), modulo three NTT
// primes (transform.hpp), the CRT straight to residues mod p, then folded back to cyclic.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <vector>

#include "lib/io/io.hpp"
#include "fields.hpp"
#include "transform.hpp"

namespace {

using multimod::Vec;
using multimod::detail::add;
using multimod::detail::broadcast;
using multimod::detail::Factor;
using multimod::detail::multiply;
using multimod::detail::reduce;

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

// Bump allocation in one mapping with transparent huge pages; memory starts zeroed. Never freed:
// the program ends with _exit.
class Arena {
public:
    explicit Arena(std::size_t bytes) {
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes = (bytes + kHuge - 1) / kHuge * kHuge + kHuge;
        void* region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region == MAP_FAILED) std::abort();
        cur_ = (reinterpret_cast<std::uintptr_t>(region) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
        ::madvise(reinterpret_cast<void*>(cur_), bytes - kHuge, MADV_HUGEPAGE);
#endif
    }

    template <class T>
    T* take(std::size_t count) {
        T* p = reinterpret_cast<T*>(cur_);
        cur_ += (count * sizeof(T) + 63) & ~std::size_t(63);
        return p;
    }

    static std::size_t bytes(std::size_t words) { return 4 * words + 64; }

private:
    std::uintptr_t cur_;
};

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

// Rows of m words hold every axis of stride < 8; rows_total is a multiple of 8 m. Eight rows at a time are interleaved into a
// buffer (word c of row l at 8 c + l) and transformed along those axes there.
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
// Every coefficient over Z is below 2^18 p^2 < 2^78; the primes' product M is about 2^89.6.
constexpr std::array<std::array<std::uint32_t, 2>, kPrimes> kPrimeList = {{
    {998244353, 3}, {985661441, 3}, {976224257, 3}}};  // prime, generator; inputs < p < 2 * prime

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

// c = sum_k y_k M_k - t M with t = floor(sum_k y_k / p_k) (the sum's fraction is c / M < 2^-10).
// Modulo p, times a factor z, with R = 2^32: s = sum_k y_k (M_k R z mod p) + t (-M R z mod p)
// < (3 2^30 + 3) p < R p, and s / R mod p = c z mod p (Montgomery reduction; p odd).
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

    // c z mod p of coefficients [i, i + 8) into out; the residues are readable 8 words past.
    void reconstruct(const std::array<std::uint32_t*, kPrimes>& y, std::size_t count, std::uint32_t* out) const {
        for (std::size_t i = 0; i < count; i += 8) {
            const auto load = [&](int k) { return _mm256_load_si256(reinterpret_cast<const Vec*>(y[k] + i)); };
            const Vec y0 = load(0), y1 = load(1), y2 = load(2);
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
            _mm256_store_si256(reinterpret_cast<Vec*>(out + i), reduce(u, broadcast(p)));
        }
    }
};

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
            w = (w + rest * inverse(rest % q, q)) % d;
        }
    }

    // Kronecker extent: the product of 2 D_r - 1.
    std::size_t padded() const {
        std::size_t r = 1;
        for (const std::size_t d : length) r *= 2 * d - 1;
        return r;
    }

private:
    // 1 / x mod m, gcd(x, m) = 1.
    static std::size_t inverse(std::size_t x, std::size_t m) {
        std::int64_t a = std::int64_t(x), b = std::int64_t(m), u = 1, v = 0;
        while (b) {
            const std::int64_t t = a / b;
            a -= t * b, u -= t * v;
            std::swap(a, b), std::swap(u, v);
        }
        return std::size_t((u % std::int64_t(m) + std::int64_t(m)) % std::int64_t(m));
    }
};

// Transform length for a Kronecker extent.
int transform_log(std::size_t padded) { return std::max(6, int(std::bit_width(padded - 1))); }

// Products over the long axes, one per point of the short axes' spectrum.
class LongProduct {
public:
    // The long axes' cyclic factors have a padded extent of at most 2^kMaxLog.
    LongProduct(const std::vector<Axis>& axes, const Field& field, std::uint32_t scale, Arena& arena)
        : crt_(field, scale), p_(field.p) {
        const CyclicFactors factors(axes);
        std::vector<std::size_t> place_stride;
        padded_ = 1;
        for (const std::size_t d : factors.length) {
            folds_.push_back({d, padded_});
            place_stride.push_back(padded_);
            padded_ *= 2 * d - 1;
        }
        // Every point, axis 0 fastest. After n_i steps along axis i, y_r is back where it started.
        std::size_t count = 1;
        for (const Axis& axis : axes) count *= axis.length;
        std::vector<std::size_t> index(axes.size()), y(factors.length.size());
        std::size_t offset = 0;
        for (std::size_t t = 0; t < count; ++t) {
            std::size_t place = 0;
            for (std::size_t r = 0; r < y.size(); ++r) place += y[r] * place_stride[r];
            offsets_.push_back(std::uint32_t(offset));
            places_.push_back(std::uint32_t(place));
            extent_ = std::max(extent_, place + 1);
            for (std::size_t i = 0; i < axes.size(); ++i) {
                offset += axes[i].stride;
                for (std::size_t r = 0; r < y.size(); ++r) {
                    y[r] += factors.weight[i][r];
                    if (y[r] >= factors.length[r]) y[r] -= factors.length[r];
                }
                if (++index[i] < axes[i].length) break;
                index[i] = 0;
                offset -= axes[i].length * axes[i].stride;
            }
        }
        lg_ = transform_log(padded_);
        const std::size_t words = (std::size_t(1) << lg_) + multimod::Transform::kPadding;
        a_ = arena.take<std::uint32_t>(words);
        b_ = arena.take<std::uint32_t>(words);
        for (auto& r : residues_) r = arena.take<std::uint32_t>(words);
        tables_ = arena.take<std::uint32_t>(multimod::Transform::table_words(lg_));
        for (int k = 0; k < kPrimes; ++k) moduli_.emplace_back(kPrimeList[k][0], kPrimeList[k][1]);
    }

    static std::size_t arena_bytes(const std::vector<Axis>& axes) {
        const int lg = transform_log(CyclicFactors(axes).padded());
        const std::size_t words = (std::size_t(1) << lg) + multimod::Transform::kPadding;
        return (2 + kPrimes) * Arena::bytes(words) + Arena::bytes(multimod::Transform::table_words(lg));
    }

    // f[base + offset] <- (f * g)[base + offset] over the long axes, times the scale. Between
    // calls, a and b are zero but at places_; the last call (last = true) leaves them dirty.
    void multiply(std::uint32_t* f, const std::uint32_t* g, std::size_t base, bool last) {
        for (std::size_t t = 0; t < offsets_.size(); ++t) {
            a_[places_[t]] = f[base + offsets_[t]];
            b_[places_[t]] = g[base + offsets_[t]];
        }
        const multimod::Transform transform(lg_, tables_);
        // The last prime's product goes to the work array and transforms b in place.
        std::uint32_t* const work = residues_[kPrimes - 1];
        for (int k = 0; k < kPrimes; ++k)
            transform.multiply(a_, extent_, b_, extent_, residues_[k], k + 1 < kPrimes ? work : b_, moduli_[k],
                               kCrtScale[k]);
        const std::size_t count = (padded_ + 7) & ~std::size_t(7);
        crt_.reconstruct(residues_, count, a_);
        fold(a_);
        for (std::size_t t = 0; t < offsets_.size(); ++t) f[base + offsets_[t]] = a_[places_[t]];
        if (last) return;
        std::memset(a_, 0, count * sizeof(std::uint32_t));
        std::memset(b_, 0, (std::size_t(1) << lg_) * sizeof(std::uint32_t));
    }

private:
    struct Fold {
        std::size_t length, stride;  // D and the Kronecker stride of a cyclic factor
    };

    // Index D + j of a factor onto j, for every factor in turn.
    void fold(std::uint32_t* c) const {
        const std::uint32_t p = p_;
        for (const Fold& fold : folds_) {
            const std::size_t span = fold.stride * (2 * fold.length - 1), shift = fold.length * fold.stride;
            for (std::size_t o = 0; o < padded_; o += span)
                for (std::size_t i = o; i < o + span - shift; ++i) {
                    const std::uint32_t sum = c[i] + c[i + shift];
                    c[i] = sum >= p ? sum - p : sum;
                }
        }
    }

    Crt crt_;
    std::uint32_t p_;
    std::vector<std::uint32_t> offsets_, places_;  // index in f, index in the Kronecker array
    std::vector<Fold> folds_;
    std::size_t extent_ = 0, padded_ = 0;
    int lg_ = 0;
    std::uint32_t *a_, *b_, *tables_;
    std::array<std::uint32_t*, kPrimes> residues_{};
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

// Modeled time of a split into long and short axes, from lc-amd: a product costs about 13 ns per
// transform word (three primes), a short DFT 0.5 ns per element and axis length (three passes).
double modeled_cost(const std::vector<Axis>& long_axes, const std::vector<Axis>& axes, std::size_t total) {
    std::size_t long_total = 1, short_sum = 0;
    for (const Axis& axis : long_axes) long_total *= axis.length;
    for (const Axis& axis : axes) short_sum += axis.length;
    for (const Axis& axis : long_axes) short_sum -= axis.length;
    const double product =
        long_axes.empty() ? 0 : 13.0 * double(total / long_total) * double(std::size_t(1) << transform_log(CyclicFactors(long_axes).padded()));
    return product + 0.5 * double(total) * double(short_sum);
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

    const std::size_t rows = 8 * narrow_row(short_axes);
    const std::size_t padded_total = (total + rows - 1) / rows * rows;
    Arena arena(2 * Arena::bytes(padded_total) + Arena::bytes(fields::kTextBytes / 4 + 1) +
                (long_axes.empty() ? 0 : LongProduct::arena_bytes(long_axes)));
    auto* f = arena.take<std::uint32_t>(padded_total);
    auto* g = arena.take<std::uint32_t>(padded_total);
    in.read(f, total);
    in.read(g, total);

    const Field field(p);
    const std::uint32_t scale = field.inverse(std::uint32_t(short_total % p));
    const std::uint32_t root = short_axes.empty() ? 1 : primitive_root(field);
    transform_short(f, total, short_axes, field, root);
    transform_short(g, total, short_axes, field, root);
    if (long_axes.empty()) {
        for (std::size_t i = 0; i < total; ++i) f[i] = field.multiply(field.multiply(f[i], g[i]), scale);
    } else {
        LongProduct product(long_axes, field, scale, arena);
        const std::vector<std::uint32_t> bases = points(short_axes);
        for (std::size_t i = 0; i < bases.size(); ++i) product.multiply(f, g, bases[i], i + 1 == bases.size());
    }
    transform_short(f, total, short_axes, field, field.inverse(root));

    io::Writer out;
    fields::write(out, f, total, arena.take<char>(fields::kTextBytes));
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
