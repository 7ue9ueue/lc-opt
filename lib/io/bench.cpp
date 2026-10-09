// In-memory throughput of lib/io: ns per token. The input arrives through a pipe, so the Reader
// holds it in an already-touched heap buffer and only parsing is timed. Output goes to /dev/null.
// Build: g++ -O2 -std=c++23 -march=native -I. lib/io/bench.cpp
#include "lib/io/io.hpp"

#include <sys/wait.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokens = std::size_t(1) << 20;
constexpr int kRounds = 15;

int pipe_with(const std::string& text) {
    int fds[2];
    if (::pipe(fds) != 0) std::abort();
    if (::fork() == 0) {
        ::close(fds[0]);
        for (std::size_t done = 0; done < text.size();) {
            const ssize_t put = ::write(fds[1], text.data() + done, text.size() - done);
            if (put <= 0) ::_exit(1);
            done += std::size_t(put);
        }
        ::_exit(0);
    }
    ::close(fds[1]);
    return fds[0];
}

double now_ns() {
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Timing {
    double median, min;  // ns per token over rounds
};

Timing time_rounds(const std::function<double()>& round) {
    std::vector<double> t;
    for (int r = 0; r < kRounds; ++r) t.push_back(round() / double(kTokens));
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t[0]};
}

volatile std::uint64_t sink;

template <class T>
std::vector<T> make(std::mt19937_64& rng, const std::function<T(std::mt19937_64&)>& gen) {
    std::vector<T> v(kTokens);
    for (auto& x : v) x = gen(rng);
    return v;
}

template <class T>
std::string join(const std::vector<T>& v) {
    std::string s;
    char buf[32];
    for (std::size_t i = 0; i < v.size(); ++i) {
        s.append(buf, std::to_chars(buf, buf + 32, v[i]).ptr);
        s += i % 2 ? '\n' : ' ';
    }
    return s;
}

template <class T>
void bench(const char* name, const std::function<T(std::mt19937_64&)>& gen) {
    std::mt19937_64 rng(1);
    const std::vector<T> v = make<T>(rng, gen);
    const std::string text = join(v);
    const Timing scalar = time_rounds([&] {
        io::Reader in(pipe_with(text));
        ::wait(nullptr);
        const double t0 = now_ns();
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < kTokens; ++i) sum += std::uint64_t(in.read<T>());
        const double t1 = now_ns();
        sink = sum;
        return t1 - t0;
    });
    Timing bulk{};
    if constexpr (std::same_as<T, std::uint32_t>) {
        std::vector<std::uint32_t> got(kTokens);
        bulk = time_rounds([&] {
            io::Reader in(pipe_with(text));
            ::wait(nullptr);
            const double t0 = now_ns();
            in.read(got.data(), kTokens);
            const double t1 = now_ns();
            sink = got[kTokens / 2];
            return t1 - t0;
        });
    }
    const int null = ::open("/dev/null", O_WRONLY);
    const Timing write = time_rounds([&] {
        io::Writer out(null);
        const double t0 = now_ns();
        for (std::size_t i = 0; i < kTokens; ++i) out.write(v[i], ' ');
        out.flush();
        return now_ns() - t0;
    });
    Timing array{};
    if constexpr (std::same_as<T, std::uint32_t>) {
        array = time_rounds([&] {
            io::Writer out(null);
            const double t0 = now_ns();
            out.write_array(v.data(), v.size(), ' ');
            out.flush();
            return now_ns() - t0;
        });
    }
    ::close(null);
    std::printf("%-8s %5.2f B  read %5.2f (%5.2f)  bulk %5.2f (%5.2f)  write %5.2f (%5.2f)  array %5.2f (%5.2f)\n",
                name, double(text.size()) / double(kTokens), scalar.median, scalar.min, bulk.median, bulk.min,
                write.median, write.min, array.median, array.min);
}

}  // namespace

int main() {
    std::printf("ns per token: median (min) of %d rounds of %zu tokens\n", kRounds, kTokens);
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;
    bench<u32>("mod", [](auto& r) { return u32(r() % 998244353); });
    bench<u32>("digit", [](auto& r) { return u32(r() % 10); });
    bench<u32>("mixed32", [](auto& r) { return u32(r() >> (r() % 32)); });
    bench<u32>("u32", [](auto& r) { return u32(r()); });
    bench<u64>("1e18", [](auto& r) { return u64(r() % 1000000000000000001); });
    bench<u64>("mixed64", [](auto& r) { return u64(r() >> (r() % 64)); });
    bench<std::int64_t>("i64", [](auto& r) { return std::int64_t(r()); });
    bench<int>("int", [](auto& r) { return int(r() % 2000001) - 1000000; });
}
