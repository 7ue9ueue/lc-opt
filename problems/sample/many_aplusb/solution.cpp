// A + B in decimal, never in binary. Per line "a b\n" (a, b <= 10^18):
//   1. Each number's digits, as byte values 0..9, right-aligned at byte 30 of a 32-byte vector:
//      one load that ends at its separator, a saturating subtract of '0' (the separator, byte 31,
//      becomes 0) and a mask that clears the bytes before the number.
//   2. The two vectors added bytewise: 0..18 per digit.
//   3. Carries from two masks of the byte-reversed sum, bit i = byte 31 - i: generate (> 9) and
//      propagate (= 9). carry = ((generate << 1) + propagate) ^ propagate marks the digits a carry
//      enters, as in a binary addition. Then digit + carry, minus 10 if over 9.
//   4. Digits of the sum: the longer number's, plus one if a carry enters the digit above it.
//      The text is bytes 31 - digits..31, with the newline at byte 31; two stores write it.
// The line loop is inline assembly: 57 instructions per line, 70 from the same steps in
// intrinsics (notes.md).
#include "lib/io/io.hpp"

#include <algorithm>
#include <cstdlib>

namespace {

struct Constants {
    __m256i zero = _mm256_set1_epi8('0');
    __m256i nine = _mm256_set1_epi8(9);
    __m256i ten = _mm256_set1_epi8(10);
    __m256i space = _mm256_set1_epi8(' ' + 1);
    __m256i reverse = _mm256_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,  //
                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    // Spreads the carry mask over the bytes: byte j gets bit 31 - j.
    __m256i spread = _mm256_setr_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,  //
                                      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i bit = _mm256_set1_epi64x(std::int64_t(0x0102040810204080));
    __m256i text = _mm256_setr_epi8('0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '\n');
};

// Writes the sums of `lines` >= 1 lines at p and returns the end; at most 21 bytes per line, and
// stores reach 16 bytes past the end. scan starts at the first line and ends after the last.
char* add_lines(char* p, io::Reader::Scan& scan, std::uint64_t lines, const Constants& c) {
    std::uint64_t r1, r2, r3, r4;
    asm volatile(
        ".Lline%=:\n\t"
        "test %[mask], %[mask]\n\t"
        "jz .Lrefill_a%=\n"
        ".Lhave_a%=:\n\t"
        "tzcnt %[mask], %[r1]\n\t"
        "blsr %[mask], %[mask]\n\t"
        "add %[block], %[r1]\n\t"                  // r1: end of a
        "test %[mask], %[mask]\n\t"
        "jz .Lrefill_b%=\n"
        ".Lhave_b%=:\n\t"
        "tzcnt %[mask], %[r2]\n\t"
        "blsr %[mask], %[mask]\n\t"
        "add %[block], %[r2]\n\t"                  // r2: end of b
        "lea 1(%[r1]), %[r3]\n\t"
        "sub %[cur], %[r3]\n\t"                    // r3: length of a, plus 1
        "mov %[r2], %[r4]\n\t"
        "sub %[r1], %[r4]\n\t"                     // r4: length of b, plus 1
        "lea 1(%[r2]), %[cur]\n\t"                 // next line
        // Step 1: digits.
        "vmovdqu -31(%[r1]), %%ymm0\n\t"
        "vpsubusb %[zero], %%ymm0, %%ymm0\n\t"
        "vpand (%[keep],%[r3]), %%ymm0, %%ymm0\n\t"
        "vmovdqu -31(%[r2]), %%ymm1\n\t"
        "vpsubusb %[zero], %%ymm1, %%ymm1\n\t"
        "vpand (%[keep],%[r4]), %%ymm1, %%ymm1\n\t"
        // Step 2: sum.
        "vpaddb %%ymm1, %%ymm0, %%ymm0\n\t"
        "cmp %[r3], %[r4]\n\t"
        "cmovb %[r3], %[r4]\n\t"                   // r4: longer length, plus 1
        // Step 3: carries.
        "vpshufb %[reverse], %%ymm0, %%ymm1\n\t"
        "vpermq $0x4e, %%ymm1, %%ymm1\n\t"
        "vpcmpgtb %[nine], %%ymm1, %%ymm2\n\t"
        "vpcmpeqb %[nine], %%ymm1, %%ymm1\n\t"
        "vpmovmskb %%ymm2, %k[r1]\n\t"             // generate
        "vpmovmskb %%ymm1, %k[r2]\n\t"             // propagate
        "lea (%q[r2],%q[r1],2), %k[r1]\n\t"
        "xor %k[r2], %k[r1]\n\t"                   // r1: carries
        "bt %k[r4], %k[r1]\n\t"                    // a carry above the longer number?
        "adc $-1, %k[r4]\n\t"                      // r4: digits of the sum
        "vmovd %k[r1], %%xmm1\n\t"
        "vpbroadcastd %%xmm1, %%ymm1\n\t"
        "vpshufb %[spread], %%ymm1, %%ymm1\n\t"
        "vpand %[bit], %%ymm1, %%ymm1\n\t"
        "vpcmpeqb %[bit], %%ymm1, %%ymm1\n\t"      // -1 where a carry enters
        "vpsubb %%ymm1, %%ymm0, %%ymm0\n\t"        // 0..19
        "vpsubb %[ten], %%ymm0, %%ymm1\n\t"
        "vpminub %%ymm1, %%ymm0, %%ymm0\n\t"       // minus 10 if over 9; otherwise the difference wraps
        // Step 4: text and stores.
        "vpaddb %[text], %%ymm0, %%ymm0\n\t"
        "mov $31, %k[r3]\n\t"
        "sub %k[r4], %k[r3]\n\t"                   // r3: first byte of the text
        "cmp $16, %k[r3]\n\t"
        "jae .Lshort%=\n\t"
        "vpshufb (%[window],%q[r3]), %%xmm0, %%xmm1\n\t"
        "vmovdqu %%xmm1, (%[p])\n\t"
        "neg %q[r3]\n\t"
        "vextracti128 $1, %%ymm0, 16(%[p],%q[r3])\n"
        ".Lstored%=:\n\t"
        "lea 1(%[p],%q[r4]), %[p]\n\t"
        "dec %[lines]\n\t"
        "jnz .Lline%=\n\t"
        "jmp .Ldone%=\n"
        ".Lshort%=:\n\t"                           // at most 15 digits: all in the high half
        "vextracti128 $1, %%ymm0, %%xmm0\n\t"
        "vpshufb -16(%[window],%q[r3]), %%xmm0, %%xmm0\n\t"
        "vmovdqu %%xmm0, (%[p])\n\t"
        "jmp .Lstored%=\n"
        ".Lrefill_a%=:\n\t"                        // the next block's separators
        "add $64, %[block]\n\t"
        "vpcmpgtb (%[block]), %[space], %%ymm1\n\t"
        "vpmovmskb %%ymm1, %k[r1]\n\t"
        "vpcmpgtb 32(%[block]), %[space], %%ymm1\n\t"
        "vpmovmskb %%ymm1, %k[mask]\n\t"
        "shl $32, %[mask]\n\t"
        "or %[r1], %[mask]\n\t"
        "jz .Lrefill_a%=\n\t"
        "jmp .Lhave_a%=\n"
        ".Lrefill_b%=:\n\t"
        "add $64, %[block]\n\t"
        "vpcmpgtb (%[block]), %[space], %%ymm1\n\t"
        "vpmovmskb %%ymm1, %k[r2]\n\t"
        "vpcmpgtb 32(%[block]), %[space], %%ymm1\n\t"
        "vpmovmskb %%ymm1, %k[mask]\n\t"
        "shl $32, %[mask]\n\t"
        "or %[r2], %[mask]\n\t"
        "jz .Lrefill_b%=\n\t"
        "jmp .Lhave_b%=\n"
        ".Ldone%=:\n"
        : [cur] "+r"(scan.cur), [block] "+r"(scan.block), [mask] "+r"(scan.separators), [p] "+r"(p),
          [lines] "+r"(lines), [r1] "=&r"(r1), [r2] "=&r"(r2), [r3] "=&r"(r3), [r4] "=&r"(r4)
        : [keep] "r"(io::detail::kKeep.data()), [window] "r"(io::detail::kShiftWindow), [zero] "x"(c.zero),
          [nine] "x"(c.nine), [ten] "x"(c.ten), [space] "x"(c.space), [reverse] "x"(c.reverse),
          [spread] "x"(c.spread), [bit] "x"(c.bit), [text] "x"(c.text)
        : "xmm0", "xmm1", "xmm2", "cc", "memory");
    return p;
}

}  // namespace

int main() {
    io::Reader in;
    io::Writer out;
    auto t = in.read<std::uint32_t>();
    io::Reader::Scan scan = in.scan();
    const Constants c;
    constexpr std::uint32_t kLines = 1024;  // per buffer reservation
    while (t) {
        const std::uint32_t lines = std::min(t, kLines);
        t -= lines;
        out.write_with(kLines * 21 + 16, [&](char* p) { return add_lines(p, scan, lines, c); });
    }
    out.flush();
    std::_Exit(0);  // skips the exit handlers: 0.03 ms
}
