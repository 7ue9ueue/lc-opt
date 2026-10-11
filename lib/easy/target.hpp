// AVX2 for the code after this header, on judges that compile without -march (Codeforces, Luogu,
// QOJ, CodeChef). Include it, or a lib/easy header, after every standard header.
#pragma once

// Every standard header lib/ntt and lib/poly use comes before the target pragma: GCC 13 and 14
// fail to inline std::allocator's members into code compiled under it otherwise.
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC diagnostic ignored "-Wpsabi"  // AVX vector arguments without -mavx: all callers share the pragma
