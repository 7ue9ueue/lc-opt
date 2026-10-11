// Portable buffered I/O on stdin and stdout (Linux, macOS, Windows), for contest code that must
// build on judges where lib/io does not (lib/io needs Linux).
//
//   easy::Reader in;                          // reads all of stdin
//   const auto n = in.read<int>();            // optional '-', then digits
//   const std::string_view s = in.token();    // next run of bytes > ' '
//   easy::Writer out;                         // flushes in its destructor
//   out.write(n, ' ', s, '\n');               // integers, chars, strings
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <type_traits>
#include <vector>

namespace easy {

class Reader {
public:
    Reader() {
        std::size_t size = 0;
        data_.resize(1 << 16);
        for (;;) {
            size += std::fread(data_.data() + size, 1, data_.size() - size, stdin);
            if (size < data_.size()) break;
            data_.resize(2 * data_.size());
        }
        data_.resize(size);
        data_.push_back(0);
    }

    std::string_view token() {
        while (pos_ + 1 < data_.size() && byte(pos_) <= ' ') ++pos_;
        const std::size_t start = pos_;
        while (pos_ + 1 < data_.size() && byte(pos_) > ' ') ++pos_;
        return {data_.data() + start, pos_ - start};
    }

    template <std::integral T>
    T read() {
        const std::string_view s = token();
        std::size_t i = 0;
        bool negative = false;
        if constexpr (std::is_signed_v<T>) negative = !s.empty() && s[0] == '-', i = negative;
        std::make_unsigned_t<T> x = 0;
        for (; i < s.size(); ++i) x = x * 10 + std::make_unsigned_t<T>(s[i] - '0');
        return negative ? T(-x) : T(x);
    }

private:
    unsigned char byte(std::size_t i) const { return static_cast<unsigned char>(data_[i]); }

    std::vector<char> data_;
    std::size_t pos_ = 0;
};

class Writer {
public:
    Writer() { buffer_.reserve(kFlush + 64); }
    ~Writer() { flush(); }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    template <class... Ts>
    void write(const Ts&... xs) {
        (put(xs), ...);
        if (buffer_.size() >= kFlush) flush();
    }

    void flush() {
        std::fwrite(buffer_.data(), 1, buffer_.size(), stdout);
        std::fflush(stdout);
        buffer_.clear();
    }

private:
    static constexpr std::size_t kFlush = 1 << 16;

    void put(char c) { buffer_.push_back(c); }
    void put(std::string_view s) { buffer_.insert(buffer_.end(), s.begin(), s.end()); }
    void put(const char* s) { put(std::string_view(s)); }

    template <std::integral T>
    void put(T x) {
        std::make_unsigned_t<T> u = std::make_unsigned_t<T>(x);
        if constexpr (std::is_signed_v<T>)
            if (x < 0) buffer_.push_back('-'), u = std::make_unsigned_t<T>(0) - u;
        char digits[20];
        int k = 0;
        do digits[k++] = char('0' + u % 10), u /= 10;
        while (u);
        while (k) buffer_.push_back(digits[--k]);
    }

    std::vector<char> buffer_;
};

}  // namespace easy
