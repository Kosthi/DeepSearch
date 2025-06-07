#pragma once

#include <cstddef>

#include "simd_utils.h"

namespace deepsearch {
namespace simd {

/**
 * Function pointer types for distance functions
 */
using L2SqrFunc = float (*)(const float*, const float*, size_t);
using IPFunc = float (*)(const float*, const float*, size_t);
using CosineFunc = float (*)(const float*, const float*, size_t);
using L2SqrSQ8Func = float (*)(const void*, const void*, size_t);
using L2SqrSQ4Func = float (*)(const void*, const void*, size_t);
using IPSQ8Func = float (*)(const void*, const void*, size_t);

/**
 * Global function pointers (initialized once at startup)
 */
extern L2SqrFunc L2Sqr;
extern IPFunc IP;
extern CosineFunc CosineDistance;
extern L2SqrSQ8Func L2SqrSQ8_ext;
extern L2SqrSQ4Func L2SqrSQ4;
extern IPSQ8Func IPSQ8_ext;

/**
 * Initialize all SIMD function pointers based on CPU capabilities
 * This should be called once at program startup
 */
void initializeSIMDFunctions();

/**
 * Reference implementations (always available)
 */
namespace ref {
float L2Sqr(const float* pVect1, const float* pVect2, size_t qty);
float IP(const float* pVect1, const float* pVect2, size_t qty);
float CosineDistance(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ8(const void* pVect1v, const void* pVect2v, size_t qty);
float L2SqrSQ4(const void* pVect1v, const void* pVect2v, size_t qty);
float IPSQ8(const void* pVect1v, const void* pVect2v, size_t qty);
}  // namespace ref

/**
 * Internal implementation functions (not for direct use)
 */

// SSE implementations
namespace sse {
float L2Sqr(const float* pVect1, const float* pVect2, size_t qty);
float IP(const float* pVect1, const float* pVect2, size_t qty);
}  // namespace sse

// AVX2 implementations
namespace avx2 {
float L2Sqr(const float* pVect1, const float* pVect2, size_t qty);
float IP(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ4(const void* pVect1v, const void* pVect2v, size_t qty);
}  // namespace avx2

// AVX512 implementations
namespace avx512 {
float L2Sqr(const float* pVect1, const float* pVect2, size_t qty);
float IP(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ8(const void* pVect1v, const void* pVect2v, size_t qty);
float IPSQ8(const void* pVect1v, const void* pVect2v, size_t qty);
}  // namespace avx512

// NEON implementations
namespace neon {
float L2Sqr(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ4(const void* pVect1v, const void* pVect2v, size_t qty);
float L2SqrSQ8(const void* pVect1v, const void* pVect2v, size_t qty);
float IP(const float* pVect1, const float* pVect2, size_t qty);
}  // namespace neon

}  // namespace simd
}  // namespace deepsearch
