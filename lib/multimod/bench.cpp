// Timing of Transform::multiply for each input kind at one length, both inputs half the length
// (lg 20: the largest case of the problems that use it). Arrays in 2 MiB pages, as there.
//   g++ -O2 -std=c++23 -march=native -I. lib/multimod/bench.cpp -o bench && ./bench [lg] [reps]
// Prints, per kind, the median and minimum ms of one product per prime (3 primes; 5 for Wide).
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "lib/multimod/transform.hpp"

namespace {

constexpr std::uint32_t kPrimes[][2] = {{998244353, 3}, {985661441, 3}, {976224257, 3}, {975175681, 17},
                                       {972029953, 10}};

// Bump allocation from one mapping, 2 MiB aligned, page-rounded blocks.
class Arena {
public:
    explicit Arena(std::size_t bytes) {
        void* base = mmap(nullptr, bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) std::abort();
        madvise(base, bytes + kHuge, MADV_HUGEPAGE);
        next_ = reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(base) + kHuge - 1) & ~(kHuge - 1));
    }

    template <class T>
    T* take(std::size_t count) {
        T* block = reinterpret_cast<T*>(next_);
        next_ += (count * sizeof(T) + 4095) & ~std::size_t(4095);
        return block;
    }

private:
    static constexpr std::uintptr_t kHuge = std::uintptr_t(1) << 21;
    char* next_;
};

template <class Run>
void time(const char* name, int reps, Run run) {
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        const auto start = std::chrono::steady_clock::now();
        run();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%-8s %8.3f %8.3f\n", name, ms[ms.size() / 2], ms[0]);
}

}  // namespace

int main(int argc, char** argv) {
    const int lg = argc > 1 ? std::atoi(argv[1]) : 20, reps = argc > 2 ? std::atoi(argv[2]) : 21;
    if (lg < 6 || lg > multimod::kMaxLog || reps < 1) return 1;
    const std::size_t len = std::size_t(1) << lg, half = len / 2, words = len + multimod::Transform::kPadding;
    Arena arena((std::size_t(64) << lg) + 12 * 4096);  // 49 words of 4 bytes per point, 11 blocks
    auto* a64 = arena.take<std::uint64_t>(words);
    auto* b64 = arena.take<std::uint64_t>(words);
    auto* a32 = arena.take<std::uint32_t>(words);
    auto* b32 = arena.take<std::uint32_t>(words);
    auto* tables = arena.take<std::uint32_t>(multimod::Transform::table_words(lg));
    auto* work = arena.take<std::uint32_t>(words);
    std::uint32_t* out[5];
    for (auto& o : out) o = arena.take<std::uint32_t>(words);
    std::mt19937_64 rng(1);
    for (std::size_t i = 0; i < half; ++i) {
        a64[i] = rng(), b64[i] = rng();
        a32[i] = std::uint32_t(rng() % 1000000007), b32[i] = std::uint32_t(rng() % 1000000007);
    }
    std::vector<multimod::Modulus> moduli;
    for (const auto& [p, g] : kPrimes) moduli.emplace_back(p, g);
    const multimod::Transform transform(lg, tables);
    std::printf("lg %d, %d reps; kind, median ms, min ms\n", lg, reps);
    time("Padded", reps, [&] {
        for (int k = 0; k < 3; ++k)
            transform.multiply(multimod::Padded{a32, half}, multimod::Padded{b32, half}, out[k], work, moduli[k], 1);
    });
    time("Bounded", reps, [&] {
        for (int k = 0; k < 3; ++k)
            transform.multiply(multimod::Bounded{a32, half}, multimod::Bounded{b32, half}, out[k], work, moduli[k], 1);
    });
    time("Wide", reps, [&] {
        for (int k = 0; k < 5; ++k)
            transform.multiply(multimod::Wide{a64, half}, multimod::Wide{b64, half}, out[k], work, moduli[k], 1);
    });
}
