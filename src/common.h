#pragma once

#include "index/graph_index.h"

namespace deepsearch {

enum class IndexType { BRUTEFORCE, HNSW, IVF };

enum class QuantizerType { FP32, SQ4, SQ8, PQ };

enum class DistanceType { L2, IP, COSINE };

// 类型别名保持兼容性
using Graph = index::DenseGraph<int>;

inline constexpr size_t upper_div(size_t x, size_t y) {
  return (x + y - 1) / y;
}

#if defined(__clang__)

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE __attribute__((always_inline))

#elif defined(__GNUC__)

#define FAST_BEGIN                     \
  _Pragma("GCC push_options") _Pragma( \
      "GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAST_END _Pragma("GCC pop_options")
#define GLASS_INLINE [[gnu::always_inline]]

#endif

}  // namespace deepsearch
