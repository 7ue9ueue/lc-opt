// Submit as aplusb to check the judge's instruction set (see AGENTS.md, Instruction sets).
// Prints a + b + failed, where bit i of `failed` marks the i-th failed check (listed on stderr).
// Each probe has its own target attribute, so the file builds and runs on any x86-64 CPU.
#include <immintrin.h>

#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>

#define STR_(x) #x
#define STR(x) STR_(x)
// A defined ISA macro expands to 1; an undefined one stays its own name.
#define DEFINED(macro) (STR(macro)[0] == '1')

static uint64_t seed;  // from the input, so no probe folds to a constant
static volatile uint64_t sink;

__attribute__((target("avx2,fma"))) static void probe_avx2_fma() {
    __m256i v = _mm256_set1_epi64x(seed);
    __m256d d = _mm256_castsi256_pd(_mm256_mul_epu32(v, v));
    sink = _mm256_extract_epi64(_mm256_castpd_si256(_mm256_fmadd_pd(d, d, d)), 1);
}

__attribute__((target("bmi2,popcnt"))) static void probe_bmi2_popcnt() {
    unsigned long long high;
    sink = _pdep_u64(seed, ~seed) + _pext_u64(seed, ~seed) + _mulx_u64(seed, ~seed, &high) + high
         + _mm_popcnt_u64(seed);
}

static void probe_adx() {
    uint64_t x = seed;
    asm volatile("adcx %1, %0\n\tadox %1, %0" : "+r"(x) : "r"(~seed) : "cc");
    sink = x;
}

__attribute__((target("sse4.2"))) static void probe_crc32() { sink = _mm_crc32_u64(seed, ~seed); }

__attribute__((target("pclmul"))) static void probe_pclmul() {
    __m128i v = _mm_set1_epi64x(seed);
    sink = _mm_cvtsi128_si64(_mm_clmulepi64_si128(v, ~v, 0x01));
}

__attribute__((target("avx2,vpclmulqdq"))) static void probe_vpclmulqdq() {
    __m256i v = _mm256_set1_epi64x(seed);
    sink = _mm256_extract_epi64(_mm256_clmulepi64_epi128(v, ~v, 0x01), 3);
}

__attribute__((target("aes"))) static void probe_aes() {
    __m128i v = _mm_set1_epi64x(seed);
    sink = _mm_cvtsi128_si64(_mm_aesenc_si128(v, ~v));
}

__attribute__((target("avx2,vaes"))) static void probe_vaes() {
    __m256i v = _mm256_set1_epi64x(seed);
    sink = _mm256_extract_epi64(_mm256_aesenc_epi128(v, ~v), 3);
}

__attribute__((target("avx512f"))) static void probe_avx512f() {
    __m512i v = _mm512_set1_epi64(seed);
    sink = _mm_cvtsi128_si64(_mm512_castsi512_si128(_mm512_add_epi64(v, v)));
}

static sigjmp_buf on_sigill;

static void jump_back(int) { siglongjmp(on_sigill, 1); }

static bool runs(void (*probe)()) {
    if (sigsetjmp(on_sigill, 1)) return false;
    probe();
    return true;
}

int main() {
    long long a, b;
    if (std::scanf("%lld %lld", &a, &b) != 2) return 1;
    seed = uint64_t(a) * 0x9e3779b97f4a7c15 ^ uint64_t(b);
    std::signal(SIGILL, jump_back);

    uint64_t failed = 0;
    int bit = 0;
    auto check = [&](const char* claim, bool holds) {
        if (!holds) failed |= uint64_t(1) << bit;
        std::fprintf(stderr, "%2d %-34s %s\n", bit++, claim, holds ? "ok" : "FAILED");
    };
    check("-march=native defines __znver3__", DEFINED(__znver3__));
    check("cpuid says znver3", __builtin_cpu_is("znver3"));
    check("-march=native defines __AVX2__", DEFINED(__AVX2__));
    check("-march=native defines __FMA__", DEFINED(__FMA__));
    check("-march=native defines __BMI__", DEFINED(__BMI__));
    check("-march=native defines __BMI2__", DEFINED(__BMI2__));
    check("-march=native defines __LZCNT__", DEFINED(__LZCNT__));
    check("-march=native defines __POPCNT__", DEFINED(__POPCNT__));
    check("-march=native defines __ADX__", DEFINED(__ADX__));
    check("-march=native defines __SSE4_2__", DEFINED(__SSE4_2__));
    check("-march=native defines __PCLMUL__", DEFINED(__PCLMUL__));
    check("-march=native defines __VPCLMULQDQ__", DEFINED(__VPCLMULQDQ__));
    check("-march=native defines __AES__", DEFINED(__AES__));
    check("-march=native defines __VAES__", DEFINED(__VAES__));
    check("-march=native omits __AVX512F__", !DEFINED(__AVX512F__));
    check("avx2, fma run", runs(probe_avx2_fma));
    check("pdep, pext, mulx, popcnt run", runs(probe_bmi2_popcnt));
    check("adcx, adox run", runs(probe_adx));
    check("crc32 runs", runs(probe_crc32));
    check("pclmulqdq runs", runs(probe_pclmul));
    check("vpclmulqdq ymm runs", runs(probe_vpclmulqdq));
    check("aesenc runs", runs(probe_aes));
    check("vaesenc ymm runs", runs(probe_vaes));
    check("avx512f faults", !runs(probe_avx512f));

    std::printf("%lld\n", a + b + (long long)failed);
}
