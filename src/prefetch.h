#pragma once

#if defined(__x86_64__)
#include <xmmintrin.h>  // SSE
#endif

namespace deepsearch {

inline void prefetch_L1(const void* address) {
#if defined(__x86_64__)
  _mm_prefetch((const char*)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

inline void prefetch_L2(const void* address) {
#if defined(__x86_64__)
  _mm_prefetch((const char*)address, _MM_HINT_T1);
#else
  __builtin_prefetch(address, 0, 2);
#endif
}

inline void prefetch_L3(const void* address) {
#if defined(__x86_64__)
  _mm_prefetch((const char*)address, _MM_HINT_T2);
#else
  __builtin_prefetch(address, 0, 1);
#endif
}

template <auto PrefetchFunc>
inline void mem_prefetch(char* ptr, const int num_lines) {
  switch (num_lines) {
    default:
      [[fallthrough]];
    case 28:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 27:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 26:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 25:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 24:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 23:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 22:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 21:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 20:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 19:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 18:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 17:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 16:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 15:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 14:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 13:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 12:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 11:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 10:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 9:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 8:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 7:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 6:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 5:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 4:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 3:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 2:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 1:
      PrefetchFunc(ptr);
      ptr += 64;
      [[fallthrough]];
    case 0:
      break;
  }
}

}  // namespace deepsearch
