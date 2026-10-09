// Tests for lib/io. Build: g++ -O2 -std=c++23 -march=x86-64-v3 -I. lib/io/test.cpp
#include "lib/io/io.hpp"

#include <sys/wait.h>

#include <charconv>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            if (++failures > 20) std::exit(1);                                      \
        }                                                                           \
    } while (0)

std::mt19937_64 rng(12345);

std::string temp_path() {
    static int counter = 0;
    return "/tmp/lc_io_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++);
}

// A file holding text, opened for reading at the given offset.
int file_with(const std::string& text, std::size_t offset = 0) {
    const std::string path = temp_path();
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    ::unlink(path.c_str());
    CHECK(::write(fd, text.data(), text.size()) == ssize_t(text.size()));
    ::lseek(fd, off_t(offset), SEEK_SET);
    return fd;
}

// A pipe fed with text by a child process.
int pipe_with(const std::string& text) {
    int fds[2];
    CHECK(::pipe(fds) == 0);
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

void reap() {
    while (::waitpid(-1, nullptr, WNOHANG) > 0) {
    }
}

template <class T>
std::string text_of(T x) {
    char buf[32];
    return {buf, std::to_chars(buf, buf + 32, x).ptr};
}

// Random value of T with a random number of digits.
template <class T>
T random_value() {
    using L = std::numeric_limits<T>;
    const int digits = L::digits10 + 1;
    const int n = int(rng() % digits) + 1;
    unsigned long long magnitude = 0;
    for (int i = 0; i < n; ++i) magnitude = magnitude * 10 + rng() % 10;
    T x;
    if (n == digits) x = T(rng() % 2 ? L::max() - T(rng() % 1000) : T(rng() % 1000));
    else x = T(magnitude);
    if constexpr (L::is_signed)
        if (rng() % 2) x = x == L::min() ? x : T(-x);
    return x;
}

template <class T>
std::vector<T> values_of() {
    using L = std::numeric_limits<T>;
    std::vector<T> v = {0, 1, 9, L::max(), L::min(), T(L::max() - 1), T(L::min() + 1)};
    for (unsigned long long p = 10; p - 1 <= (unsigned long long)L::max(); p *= 10) {
        v.push_back(T(p - 1));
        if (p <= (unsigned long long)L::max()) v.push_back(T(p));
        if constexpr (L::is_signed) v.push_back(T(-T(p - 1)));
        if (p > (unsigned long long)L::max() / 10) break;
    }
    for (int i = 0; i < 20000; ++i) v.push_back(random_value<T>());
    return v;
}

// Separators: single spaces and newlines (Library Checker format) or any whitespace.
std::string separator(bool irregular) {
    if (!irregular) return rng() % 8 ? " " : "\n";
    static const char* const kinds[] = {" ", "\n", "\t", "\r\n", "  ", " \n ", "\v\f"};
    return kinds[rng() % 7];
}

template <class T>
std::string join(const std::vector<T>& v, bool irregular) {
    std::string s = irregular ? " \n" : "";
    for (const T& x : v) s += text_of(x) + separator(irregular);
    return s;
}

template <class T>
void test_scalar() {
    const std::vector<T> v = values_of<T>();
    for (int source = 0; source < 3; ++source)
        for (const bool irregular : {false, true}) {
            const std::string text = join(v, irregular);
            const int fd = source == 0 ? file_with(text) : source == 1 ? pipe_with(text) : file_with("xx " + text, 3);
            io::Reader in(fd);
            for (const T& x : v) CHECK(in.read<T>() == x);
            ::close(fd);
        }
    reap();
}

// Every distance between the end of the data and a page boundary, with and without a final
// separator, for read and mapped files: the last token must still parse.
void test_page_ends() {
    for (const std::size_t page_end : {std::size_t(4096), std::size_t(20 * 4096)}) {
        for (std::size_t size = page_end - 80; size <= page_end + 8; ++size) {
            for (const bool newline : {false, true}) {
                std::string text(size - newline - 10, ' ');
                text += "4294967295";
                if (newline) text += '\n';
                const int fd = file_with(text);
                io::Reader in(fd);
                CHECK(in.read<std::uint32_t>() == 4294967295u);
                ::close(fd);
            }
        }
    }
}

// 64 zero bytes before the input and after its end, for read, mapped and piped input.
void test_padding() {
    for (const std::size_t size : {std::size_t(10), std::size_t(5000), std::size_t(200000)})
        for (int source = 0; source < 2; ++source) {
            const std::string text = "7" + std::string(size - 2, ' ') + "8";
            const int fd = source ? pipe_with(text) : file_with(text);
            io::Reader in(fd);
            const std::string_view first = in.word();  // checked before the next read
            bool zeros = first == "7";
            for (int i = 1; i <= 64; ++i) zeros &= first.data()[-i] == 0;
            const std::string_view last = in.word();
            zeros &= last == "8";
            for (int i = 1; i <= 64; ++i) zeros &= last.data()[i] == 0;
            CHECK(zeros);
            ::close(fd);
        }
    reap();
}

// A custom loop on scan() state, then the Reader again after resume().
void test_scan() {
    const int fd = file_with("12 345\n6789 0 77 8\n");
    io::Reader in(fd);
    CHECK(in.read<int>() == 12);
    io::Reader::Scan scan = in.scan();
    std::vector<std::string> tokens;
    for (int i = 0; i < 3; ++i) {
        while (!scan.separators) {
            scan.block += 64;
            scan.separators = io::detail::block_separators(scan.block);
        }
        const char* end = scan.block + std::countr_zero(scan.separators);
        scan.separators &= scan.separators - 1;
        tokens.emplace_back(scan.cur, end);
        scan.cur = end + 1;
    }
    CHECK((tokens == std::vector<std::string>{"345", "6789", "0"}));
    in.resume(scan);
    CHECK(in.read<int>() == 77);
    CHECK(in.word() == "8");
    ::close(fd);
}

void test_words() {
    std::string text = "abc 1 x\n";
    std::vector<std::string> words = {std::string(700000, 'w')};
    text += words[0] + ' ';
    for (int i = 0; i < 2000; ++i) {
        std::string w(rng() % 70 + 1, 'a');
        for (char& c : w) c = char('!' + rng() % 94);
        words.push_back(w);
        text += w + separator(true);
    }
    const int fd = file_with(text);
    io::Reader in(fd);
    CHECK(in.word() == "abc");
    CHECK(in.read<int>() == 1);
    CHECK(in.read<char>() == 'x');
    for (const std::string& w : words) CHECK(in.word() == w);
    ::close(fd);
}

// Bulk reads: sizes around the chunk threshold, uneven token lengths across a chunk (unbalanced
// streams), irregular whitespace (fallback path), and scalar reads before and after.
void test_bulk() {
    const std::size_t threshold = (std::size_t(1) << 16) + 64;
    for (const std::size_t count : {std::size_t(0), std::size_t(1), std::size_t(1000), threshold,
                                    threshold + 1, std::size_t(200000), std::size_t(1) << 20})
        for (int shape = 0; shape < 6; ++shape) {
            std::vector<std::uint32_t> v(count);
            for (std::size_t i = 0; i < count; ++i) {
                switch (shape) {
                    case 0: v[i] = std::uint32_t(rng() % 998244353); break;
                    case 1: v[i] = std::uint32_t(rng() % 10); break;
                    case 2: v[i] = random_value<std::uint32_t>(); break;
                    case 3: v[i] = i < count / 3 ? std::uint32_t(rng() % 10) : std::uint32_t(rng()); break;
                    default: v[i] = std::uint32_t(rng() >> (rng() % 33)); break;
                }
            }
            std::string text = "7 ";
            for (std::size_t i = 0; i < count; ++i) {
                text += text_of(v[i]);
                // shape 5: rare irregular separators somewhere in the middle
                text += shape == 5 && rng() % 50000 == 0 ? "\r\n " : rng() % 16 ? " " : "\n";
            }
            text += "-5\n";
            for (const bool pipe : {false, true}) {
                const int fd = pipe ? pipe_with(text) : file_with(text);
                io::Reader in(fd);
                CHECK(in.read<int>() == 7);
                std::vector<std::uint32_t> got(count + 1, 0xDEADBEEF);
                in.read(got.data(), count);
                for (std::size_t i = 0; i < count; ++i)
                    if (got[i] != v[i]) {
                        CHECK(got[i] == v[i]);
                        break;
                    }
                CHECK(got[count] == 0xDEADBEEF);
                CHECK(in.read<long long>() == -5);
                ::close(fd);
            }
        }
    reap();
}

// Consecutive bulk reads must hand over the cursor exactly.
void test_bulk_split() {
    std::vector<std::uint32_t> v(300000);
    for (auto& x : v) x = std::uint32_t(rng() % 1000000000);
    const std::string text = join(v, false);
    const int fd = file_with(text);
    io::Reader in(fd);
    std::vector<std::uint32_t> got(v.size());
    std::size_t done = 0;
    for (const std::size_t part : {std::size_t(70000), std::size_t(5), std::size_t(100000), std::size_t(1)}) {
        in.read(got.data() + done, part);
        done += part;
    }
    in.read(got.data() + done, v.size() - done);
    CHECK(got == v);
    ::close(fd);
}

std::string read_all(int fd) {
    std::string s;
    char buf[1 << 16];
    ::lseek(fd, 0, SEEK_SET);
    for (ssize_t got; (got = ::read(fd, buf, sizeof buf)) > 0;) s.append(buf, std::size_t(got));
    return s;
}

template <class T>
void write_all_values(io::Writer& out, std::string& expected) {
    const std::vector<T> v = values_of<T>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i % 2) {
            out.write(v[i], ' ');
        } else {
            out.write(v[i]);
            out.write(' ');
        }
        expected += text_of(v[i]) + ' ';
    }
}

void test_writer() {
    const int fd = file_with("");
    std::string expected;
    {
        io::Writer out(fd);
        write_all_values<std::uint32_t>(out, expected);
        write_all_values<std::int32_t>(out, expected);
        write_all_values<std::uint64_t>(out, expected);
        write_all_values<std::int64_t>(out, expected);
        write_all_values<std::int8_t>(out, expected);
        write_all_values<std::uint16_t>(out, expected);
        out.write('\n');
        out.write(std::string_view("hello"));
        out.write(std::string(200000, 'z'));
        out.write(std::string_view(""));
        out.write("Yes\n", -7, ' ', std::string("abc"), 'x', std::uint64_t(18446744073709551615u), '\n');
        out.write(1, std::string(100000, 'y'), 2);
        expected += "\nhello" + std::string(200000, 'z') + "Yes\n-7 abcx18446744073709551615\n";
        expected += "1" + std::string(100000, 'y') + "2";
        for (std::uint32_t x = 0; x < 2000000; ++x) {
            out.write(x, '\n');
            expected += text_of(x) + '\n';
        }
    }
    CHECK(read_all(fd) == expected);
    ::close(fd);
}

// Values of at most 16 digits through the MaxDigits paths, both ways.
void test_max_digits() {
    std::vector<std::uint64_t> u;
    std::vector<long long> v;
    for (std::uint64_t p = 1; p <= 1000000000000000; p *= 10) u.insert(u.end(), {p - 1, p, 10 * p - 1});
    for (int i = 0; i < 100000; ++i) u.push_back(rng() % 10000000000000000 >> (rng() % 50));
    for (const std::uint64_t x : u) v.push_back(rng() % 2 ? -(long long)x : (long long)x);
    const int fd = file_with("");
    {
        io::Writer out(fd);
        for (const std::uint64_t x : u) out.write<16>(x, ' ');
        for (const long long x : v) out.write<16>(x, '\n');
        out.write_array<16>(u.data(), u.size(), ' ');
    }
    std::string expected;
    for (const std::uint64_t x : u) expected += text_of(x) + ' ';
    for (const long long x : v) expected += text_of(x) + '\n';
    for (std::size_t i = 0; i < u.size(); ++i) expected += text_of(u[i]) + (i + 1 < u.size() ? " " : "");
    CHECK(read_all(fd) == expected);
    ::close(fd);
    const int in_fd = file_with(expected);
    io::Reader in(in_fd);
    for (const std::uint64_t x : u) CHECK((in.read<std::uint64_t, 16>() == x));
    for (const long long x : v) CHECK((in.read<long long, 16>() == x));
    std::vector<std::uint64_t> got(u.size());
    in.read<16>(got.data(), got.size());
    CHECK(got == u);
    ::close(in_fd);
}

// Every input of the vector divisions and digit groups used by the array formatter.
void test_vector_arithmetic() {
    alignas(32) std::uint32_t lanes[8];
    std::uint64_t wrong = 0;
    for (std::uint64_t x = 0; x < (std::uint64_t(1) << 32); x += 8) {
        const __m256i v = _mm256_add_epi32(_mm256_set1_epi32(int(x)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), io::detail::divide<1441151881, 57>(v));
        for (std::uint32_t k = 0; k < 8; ++k) wrong += lanes[k] != std::uint32_t((x + k) / 100000000);
        if (x < 100000000) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), io::detail::divide<3518437209, 45>(v));
            for (std::uint32_t k = 0; k < 8; ++k) wrong += lanes[k] != std::uint32_t((x + k) / 10000);
        }
    }
    CHECK(wrong == 0);
    for (std::uint32_t g = 0; g < 10000; g += 8) {
        const __m256i v = _mm256_add_epi32(_mm256_set1_epi32(int(g)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), io::detail::quad_digits(v));
        for (std::uint32_t k = 0; k < 8; ++k) CHECK(lanes[k] == io::detail::kQuad[g + k]);
    }
}

void test_write_array() {
    const int fd = file_with("");
    std::string expected;
    {
        io::Writer out(fd);
        for (std::size_t count : {0, 1, 7, 8, 9, 16, 17, 100, 300000}) {
            std::vector<std::uint32_t> v(count);
            for (auto& x : v) x = rng() % 3 ? random_value<std::uint32_t>() : std::uint32_t(rng() % 10);
            if (count == 300000)
                for (std::size_t i = 0; i < 4000; ++i) v[i] = i % 2 ? 4294967295u : 0;
            const char separator = count % 2 ? ' ' : '\n';
            out.write('[');
            out.write_array(v.data(), v.size(), separator);
            out.write(']');
            expected += '[';
            for (std::size_t i = 0; i < count; ++i) expected += text_of(v[i]) + (i + 1 < count ? std::string(1, separator) : "");
            expected += ']';
            std::vector<long long> w(count);
            for (auto& x : w) x = random_value<long long>();
            out.write_array(w.data(), w.size(), ',');
            for (std::size_t i = 0; i < count; ++i) expected += text_of(w[i]) + (i + 1 < count ? "," : "");
        }
    }
    CHECK(read_all(fd) == expected);
    ::close(fd);
}

}  // namespace

int main() {
    test_scalar<std::uint32_t>();
    test_scalar<std::int32_t>();
    test_scalar<std::uint64_t>();
    test_scalar<std::int64_t>();
    test_scalar<std::uint16_t>();
    test_scalar<std::int8_t>();
    test_page_ends();
    test_padding();
    test_scan();
    test_words();
    test_bulk();
    test_bulk_split();
    test_max_digits();
    test_writer();
    test_vector_arithmetic();
    test_write_array();
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures != 0;
}
