// I/O floor of any problem: maps the input and touches every cache line of it, then writes an
// output of the expected size from a buffer in huge pages, as the solutions do. Computes nothing.
// The runner gives no arguments, so the case comes from stdin's path (/in/CASE.in) and the output
// size from the file sizes/CASE in the working directory. Used by tools/speed.py.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static size_t output_size(void) {
    char path[4096];
    ssize_t len = readlink("/proc/self/fd/0", path, sizeof path - 1);
    if (len < 0) exit(3);
    path[len] = '\0';
    char *name = strrchr(path, '/');
    char *dot = strrchr(path, '.');
    if (!name || !dot || dot < name) exit(3);
    *dot = '\0';
    char sizes[4200];
    snprintf(sizes, sizeof sizes, "sizes/%s", name + 1);
    FILE *f = fopen(sizes, "r");
    unsigned long long size;
    if (!f || fscanf(f, "%llu", &size) != 1) exit(3);
    fclose(f);
    return size;
}

int main(void) {
    size_t out_bytes = output_size();

    struct stat st;
    if (fstat(0, &st) != 0) return 4;
    unsigned char sum = 0;
    if (st.st_size > 0) {
        const unsigned char *in = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, 0, 0);
        if (in == MAP_FAILED) return 4;
        for (off_t i = 0; i < st.st_size; i += 64) sum ^= in[i];
    }

    if (out_bytes > 0) {
        const size_t huge = (size_t)1 << 21;
        const size_t bytes = (out_bytes + huge - 1) / huge * huge + huge;
        char *region = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region == MAP_FAILED) return 5;
        char *out = (char *)(((uintptr_t)region + huge - 1) & ~(huge - 1));
        madvise(out, bytes - huge, MADV_HUGEPAGE);
        memset(out, '0' + (sum & 1), out_bytes);
        for (size_t done = 0; done < out_bytes;) {
            ssize_t put = write(1, out + done, out_bytes - done);
            if (put <= 0) return 6;
            done += put;
        }
    }
    return 0;
}
