#include "distance_functions.h"

#include <cmath>

namespace deepsearch {
namespace simd {

// Global function pointers definition
L2SqrFunc L2Sqr = nullptr;
IPFunc IP = nullptr;
CosineFunc CosineDistance = nullptr;
L2SqrSQ8Func L2SqrSQ8_ext = nullptr;
L2SqrSQ4Func L2SqrSQ4 = nullptr;
IPSQ8Func IPSQ8_ext = nullptr;

// Initialize SIMD function pointers
void initializeSIMDFunctions() {
  auto level = SIMDCapabilities::getOptimalSIMD();

  switch (level) {
    case SIMDCapabilities::Level::AVX512:
      L2Sqr = detail::L2Sqr_avx512;
      IP = detail::IP_avx512;
      L2SqrSQ8_ext = detail::L2SqrSQ8_avx512;
      IPSQ8_ext = detail::IPSQ8_avx512;
      L2SqrSQ4 = detail::L2SqrSQ4_avx2;  // Fallback to AVX2 for SQ4
      break;

    case SIMDCapabilities::Level::AVX2:
      L2Sqr = detail::L2Sqr_avx2;
      IP = detail::IP_avx2;
      L2SqrSQ4 = detail::L2SqrSQ4_avx2;
      L2SqrSQ8_ext = L2SqrSQ8_ref;  // Fallback to reference
      IPSQ8_ext = IPSQ8_ref;
      break;

    case SIMDCapabilities::Level::SSE:
      L2Sqr = detail::L2Sqr_sse;
      IP = detail::IP_sse;
      L2SqrSQ8_ext = L2SqrSQ8_ref;
      L2SqrSQ4 = L2SqrSQ4_ref;
      IPSQ8_ext = IPSQ8_ref;
      break;

    case SIMDCapabilities::Level::NEON:
      L2Sqr = detail::L2Sqr_neon;
      IP = detail::IP_neon;
      L2SqrSQ8_ext = detail::L2SqrSQ8_neon;
      L2SqrSQ4 = detail::L2SqrSQ4_neon;
      IPSQ8_ext = IPSQ8_ref;
      break;

    default:
      L2Sqr = L2SqrRef;
      IP = IPRef;
      L2SqrSQ8_ext = L2SqrSQ8_ref;
      L2SqrSQ4 = L2SqrSQ4_ref;
      IPSQ8_ext = IPSQ8_ref;
      break;
  }

  // Cosine distance always uses optimized L2 and IP
  CosineDistance = CosineDistanceRef;
}

// Reference implementations
float L2SqrRef(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float t = *pVect1 - *pVect2;
    res += t * t;
    pVect1++;
    pVect2++;
  }
  return res;
}

float IPRef(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    res += (*pVect1) * (*pVect2);
    pVect1++;
    pVect2++;
  }
  return res;
}

float CosineDistanceRef(const float* pVect1, const float* pVect2, size_t qty) {
  // float dot_product = IP(pVect1, pVect2, qty);
  // 确保向量已经归一化
  // float norm1 = std::sqrt(L2Sqr(pVect1, pVect1, qty));
  // float norm2 = std::sqrt(L2Sqr(pVect2, pVect2, qty));

  // if (norm1 == 0.0f || norm2 == 0.0f) {
  //   return 1.0f;  // Maximum distance for zero vectors
  // }

  float cosine_similarity = IP(pVect1, pVect2, qty);
  return 1.0f - cosine_similarity;
}

float L2SqrSQ8_ref(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float diff = static_cast<float>(pVect1[i]) - static_cast<float>(pVect2[i]);
    res += diff * diff;
  }
  return res;
}

float L2SqrSQ4_ref(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  float res = 0;
  size_t qty_bytes = (qty + 1) / 2;

  for (size_t i = 0; i < qty_bytes; i++) {
    uint8_t byte1 = pVect1[i];
    uint8_t byte2 = pVect2[i];

    float val1_low = static_cast<float>(byte1 & 0x0F);
    float val1_high = static_cast<float>((byte1 >> 4) & 0x0F);
    float val2_low = static_cast<float>(byte2 & 0x0F);
    float val2_high = static_cast<float>((byte2 >> 4) & 0x0F);

    float diff_low = val1_low - val2_low;
    float diff_high = val1_high - val2_high;

    res += diff_low * diff_low + diff_high * diff_high;
  }

  return res;
}

float IPSQ8_ref(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    res += static_cast<float>(pVect1[i]) * static_cast<float>(pVect2[i]);
  }
  return res;
}

// SIMD implementations
namespace detail {

#ifdef __SSE__
float L2Sqr_sse(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  const float* pEnd1 = pVect1 + (qty4 << 2);
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    __m128 v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    __m128 diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    float t = pVect1[i - (qty4 << 2)] - pVect2[i - (qty4 << 2)];
    res += t * t;
  }

  return res;
}

float IP_sse(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  const float* pEnd1 = pVect1 + (qty4 << 2);
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    __m128 v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    res += pVect1[i - (qty4 << 2)] * pVect2[i - (qty4 << 2)];
  }

  return res;
}
#else
float L2Sqr_sse(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_sse(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}
#endif

#ifdef __AVX2__
float L2Sqr_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty8 = qty >> 3;

  const float* pEnd1 = pVect1 + (qty8 << 3);
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    __m256 diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  res += reduce_add_f32x8(sum);

  // Handle remaining elements
  for (size_t i = qty8 << 3; i < qty; i++) {
    float t = pVect1[i - (qty8 << 3)] - pVect2[i - (qty8 << 3)];
    res += t * t;
  }

  return res;
}

float IP_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty8 = qty >> 3;

  const float* pEnd1 = pVect1 + (qty8 << 3);
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum = _mm256_fmadd_ps(v1, v2, sum);
  }

  res += reduce_add_f32x8(sum);

  // Handle remaining elements
  for (size_t i = qty8 << 3; i < qty; i++) {
    res += pVect1[i - (qty8 << 3)] * pVect2[i - (qty8 << 3)];
  }

  return res;
}

float L2SqrSQ4_avx2(const void* pVect1v, const void* pVect2v,
                    const void* qty_ptr) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
  size_t qty = *static_cast<const size_t*>(qty_ptr);

  float res = 0;
  size_t qty_bytes = (qty + 1) / 2;
  size_t qty16 = qty_bytes >> 4;

  __m256 sum = _mm256_set1_ps(0);

  for (size_t i = 0; i < qty16; i++) {
    // Load 16 bytes (32 4-bit values)
    __m128i v1_packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect1));
    __m128i v2_packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect2));

    // Unpack 4-bit values to 8-bit
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i v1_low = _mm_and_si128(v1_packed, mask);
    __m128i v1_high = _mm_and_si128(_mm_srli_epi16(v1_packed, 4), mask);
    __m128i v2_low = _mm_and_si128(v2_packed, mask);
    __m128i v2_high = _mm_and_si128(_mm_srli_epi16(v2_packed, 4), mask);

    // Convert to float and compute differences
    __m256i v1_16_low = _mm256_cvtepu8_epi16(v1_low);
    __m256i v2_16_low = _mm256_cvtepu8_epi16(v2_low);

    __m256 v1_f_low = _mm256_cvtepi32_ps(
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1_16_low)));
    __m256 v2_f_low = _mm256_cvtepi32_ps(
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2_16_low)));

    __m256 diff_low = _mm256_sub_ps(v1_f_low, v2_f_low);
    sum = _mm256_fmadd_ps(diff_low, diff_low, sum);

    pVect1 += 16;
    pVect2 += 16;
  }

  res += reduce_add_f32x8(sum);

  // Handle remaining elements
  size_t remaining_bytes = qty_bytes - (qty16 << 4);
  for (size_t i = 0; i < remaining_bytes; i++) {
    uint8_t byte1 = pVect1[i];
    uint8_t byte2 = pVect2[i];

    float val1_low = static_cast<float>(byte1 & 0x0F);
    float val1_high = static_cast<float>((byte1 >> 4) & 0x0F);
    float val2_low = static_cast<float>(byte2 & 0x0F);
    float val2_high = static_cast<float>((byte2 >> 4) & 0x0F);

    float diff_low = val1_low - val2_low;
    float diff_high = val1_high - val2_high;

    res += diff_low * diff_low + diff_high * diff_high;
  }

  return res;
}
#else
float L2Sqr_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}

float L2SqrSQ4_avx2(const void* pVect1v, const void* pVect2v, size_t qty) {
  return L2SqrSQ4_ref(pVect1v, pVect2v, qty);
}
#endif

#ifdef __AVX512F__
float L2Sqr_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty16 = qty >> 4;

  const float* pEnd1 = pVect1 + (qty16 << 4);
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    __m512 diff = _mm512_sub_ps(v1, v2);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    float t = pVect1[i - (qty16 << 4)] - pVect2[i - (qty16 << 4)];
    res += t * t;
  }

  return res;
}

float IP_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty16 = qty >> 4;

  const float* pEnd1 = pVect1 + (qty16 << 4);
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    sum = _mm512_fmadd_ps(v1, v2, sum);
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    res += pVect1[i - (qty16 << 4)] * pVect2[i - (qty16 << 4)];
  }

  return res;
}

float L2SqrSQ8_avx512(const void* pVect1v, const void* pVect2v,
                      const void* qty_ptr) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
  size_t qty = *static_cast<const size_t*>(qty_ptr);

  float res = 0;
  size_t qty16 = qty >> 4;

  __m512 sum = _mm512_set1_ps(0);

  for (size_t i = 0; i < qty16; i++) {
    __m128i v1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect1));
    __m128i v2_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect2));

    __m512i v1_32 = _mm512_cvtepu8_epi32(v1_8);
    __m512i v2_32 = _mm512_cvtepu8_epi32(v2_8);

    __m512 v1_f = _mm512_cvtepi32_ps(v1_32);
    __m512 v2_f = _mm512_cvtepi32_ps(v2_32);

    __m512 diff = _mm512_sub_ps(v1_f, v2_f);
    sum = _mm512_fmadd_ps(diff, diff, sum);

    pVect1 += 16;
    pVect2 += 16;
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    float diff = static_cast<float>(pVect1[i - (qty16 << 4)]) -
                 static_cast<float>(pVect2[i - (qty16 << 4)]);
    res += diff * diff;
  }

  return res;
}

float IPSQ8_avx512(const void* pVect1v, const void* pVect2v,
                   const void* qty_ptr) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
  size_t qty = *static_cast<const size_t*>(qty_ptr);

  float res = 0;
  size_t qty16 = qty >> 4;

  __m512 sum = _mm512_set1_ps(0);

  for (size_t i = 0; i < qty16; i++) {
    __m128i v1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect1));
    __m128i v2_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect2));

    __m512i v1_32 = _mm512_cvtepu8_epi32(v1_8);
    __m512i v2_32 = _mm512_cvtepu8_epi32(v2_8);

    __m512 v1_f = _mm512_cvtepi32_ps(v1_32);
    __m512 v2_f = _mm512_cvtepi32_ps(v2_32);

    sum = _mm512_fmadd_ps(v1_f, v2_f, sum);

    pVect1 += 16;
    pVect2 += 16;
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    res += static_cast<float>(pVect1[i - (qty16 << 4)]) *
           static_cast<float>(pVect2[i - (qty16 << 4)]);
  }

  return res;
}
#else
float L2Sqr_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}

float L2SqrSQ8_avx512(const void* pVect1v, const void* pVect2v, size_t qty) {
  return L2SqrSQ8_ref(pVect1v, pVect2v, qty);
}

float IPSQ8_avx512(const void* pVect1v, const void* pVect2v, size_t qty) {
  return IPSQ8_ref(pVect1v, pVect2v, qty);
}
#endif

#ifdef __ARM_NEON
float L2Sqr_neon(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  float32x4_t sum = vdupq_n_f32(0);

  for (size_t i = 0; i < qty4; i++) {
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    float32x4_t v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    float32x4_t diff = vsubq_f32(v1, v2);
    sum = vfmaq_f32(sum, diff, diff);
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    float t = pVect1[i - (qty4 << 2)] - pVect2[i - (qty4 << 2)];
    res += t * t;
  }

  return res;
}

float L2SqrSQ4_neon(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  size_t qty_bytes = (qty + 1) / 2;
  int32_t sum = 0;

  // 处理主循环（每次处理16字节=32个4-bit数）
  size_t i = 0;
  const size_t step = 16;
  if (qty_bytes >= step) {
    // 初始化32位累加器（4个向量寄存器）
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    for (; i <= qty_bytes - step; i += step) {
      // 加载16字节数据
      uint8x16_t vec0 = vld1q_u8(pVect1 + i);
      uint8x16_t vec1 = vld1q_u8(pVect2 + i);

      // 分离低4位和高4位
      uint8x16_t low0 = vandq_u8(vec0, vdupq_n_u8(0x0F));
      uint8x16_t high0 = vshrq_n_u8(vec0, 4);
      uint8x16_t low1 = vandq_u8(vec1, vdupq_n_u8(0x0F));
      uint8x16_t high1 = vshrq_n_u8(vec1, 4);

      // 扩展为16位整数
      uint16x8_t low0_low = vmovl_u8(vget_low_u8(low0));
      uint16x8_t low0_high = vmovl_u8(vget_high_u8(low0));
      uint16x8_t high0_low = vmovl_u8(vget_low_u8(high0));
      uint16x8_t high0_high = vmovl_u8(vget_high_u8(high0));

      uint16x8_t low1_low = vmovl_u8(vget_low_u8(low1));
      uint16x8_t low1_high = vmovl_u8(vget_high_u8(low1));
      uint16x8_t high1_low = vmovl_u8(vget_low_u8(high1));
      uint16x8_t high1_high = vmovl_u8(vget_high_u8(high1));

      // 计算差值（转换为有符号）
      int16x8_t diff_low0 = vsubq_s16(vreinterpretq_s16_u16(low0_low),
                                      vreinterpretq_s16_u16(low1_low));
      int16x8_t diff_low1 = vsubq_s16(vreinterpretq_s16_u16(low0_high),
                                      vreinterpretq_s16_u16(low1_high));
      int16x8_t diff_high0 = vsubq_s16(vreinterpretq_s16_u16(high0_low),
                                       vreinterpretq_s16_u16(high1_low));
      int16x8_t diff_high1 = vsubq_s16(vreinterpretq_s16_u16(high0_high),
                                       vreinterpretq_s16_u16(high1_high));

      // 计算平方
      int16x8_t sq_low0 = vmulq_s16(diff_low0, diff_low0);
      int16x8_t sq_low1 = vmulq_s16(diff_low1, diff_low1);
      int16x8_t sq_high0 = vmulq_s16(diff_high0, diff_high0);
      int16x8_t sq_high1 = vmulq_s16(diff_high1, diff_high1);

      // 合并高低4位的平方和
      int16x8_t total_sq0 = vaddq_s16(sq_low0, sq_high0);
      int16x8_t total_sq1 = vaddq_s16(sq_low1, sq_high1);

      // 扩展为32位并累加
      acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(total_sq0)));
      acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(total_sq0)));
      acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(total_sq1)));
      acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(total_sq1)));
    }

    // 合并累加器
    int32x4_t sum32 = vaddq_s32(acc0, acc1);
    sum32 = vaddq_s32(sum32, acc2);
    sum32 = vaddq_s32(sum32, acc3);

    // 水平求和
    int32x2_t sum2 = vadd_s32(vget_low_s32(sum32), vget_high_s32(sum32));
    sum += vget_lane_s32(sum2, 0) + vget_lane_s32(sum2, 1);
  }

  // 处理尾部剩余字节（标量处理）
  for (; i < qty_bytes; i++) {
    uint8_t b1 = pVect1[i], b2 = pVect2[i];
    int d1 = (b1 & 0x0F) - (b2 & 0x0F);
    int d2 = (b1 >> 4) - (b2 >> 4);
    sum += d1 * d1 + d2 * d2;
  }

  return static_cast<float>(sum);
}

float L2SqrSQ8_neon(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  float32x4_t acc = vdupq_n_f32(0.0f);  // 浮点累加器初始化为0
  size_t i = 0;

  // 每次处理16个元素（128位）
  const size_t step = 16;
  if (qty >= step) {
    for (; i <= qty - step; i += step) {
      // 加载16个字节数据
      uint8x16_t v1 = vld1q_u8(pVect1 + i);
      uint8x16_t v2 = vld1q_u8(pVect2 + i);

      // 将8位无符号整数扩展为16位有符号整数
      uint16x8_t v1_low = vmovl_u8(vget_low_u8(v1));
      uint16x8_t v1_high = vmovl_u8(vget_high_u8(v1));
      uint16x8_t v2_low = vmovl_u8(vget_low_u8(v2));
      uint16x8_t v2_high = vmovl_u8(vget_high_u8(v2));

      // 计算差值（转为有符号整数）
      int16x8_t diff_low = vsubq_s16(vreinterpretq_s16_u16(v1_low),
                                     vreinterpretq_s16_u16(v2_low));
      int16x8_t diff_high = vsubq_s16(vreinterpretq_s16_u16(v1_high),
                                      vreinterpretq_s16_u16(v2_high));

      // 计算平方并扩展为32位整数
      int32x4_t sq_low_low =
          vmull_s16(vget_low_s16(diff_low), vget_low_s16(diff_low));
      int32x4_t sq_low_high =
          vmull_s16(vget_high_s16(diff_low), vget_high_s16(diff_low));
      int32x4_t sq_high_low =
          vmull_s16(vget_low_s16(diff_high), vget_low_s16(diff_high));
      int32x4_t sq_high_high =
          vmull_s16(vget_high_s16(diff_high), vget_high_s16(diff_high));

      // 将32位整数平方值转为浮点数
      float32x4_t fsq0 = vcvtq_f32_s32(sq_low_low);
      float32x4_t fsq1 = vcvtq_f32_s32(sq_low_high);
      float32x4_t fsq2 = vcvtq_f32_s32(sq_high_low);
      float32x4_t fsq3 = vcvtq_f32_s32(sq_high_high);

      // 累加到浮点累加器
      acc = vaddq_f32(acc, fsq0);
      acc = vaddq_f32(acc, fsq1);
      acc = vaddq_f32(acc, fsq2);
      acc = vaddq_f32(acc, fsq3);
    }
  }

  // 水平求和
  float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
  float sum = vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);

  // 处理尾部剩余元素（标量处理）
  for (; i < qty; i++) {
    float diff = static_cast<float>(pVect1[i]) - static_cast<float>(pVect2[i]);
    sum += diff * diff;
  }

  return sum;
}

float IP_neon(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  float32x4_t sum = vdupq_n_f32(0);

  for (size_t i = 0; i < qty4; i++) {
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    float32x4_t v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    sum = vfmaq_f32(sum, v1, v2);
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    res += pVect1[i - (qty4 << 2)] * pVect2[i - (qty4 << 2)];
  }

  return res;
}
#else
float L2Sqr_neon(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_neon(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}
#endif

}  // namespace detail

}  // namespace simd
}  // namespace deepsearch

// 在文件末尾添加静态初始化器
namespace {
struct SIMDInitializer {
  SIMDInitializer() { deepsearch::simd::initializeSIMDFunctions(); }
};

// 在程序启动时自动初始化
[[maybe_unused]] SIMDInitializer simd_init;
}  // namespace
