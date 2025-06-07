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
float L2SqrRef(const float* pVect1, const float* pVect2, size_t qty);
float IPRef(const float* pVect1, const float* pVect2, size_t qty);
float CosineDistanceRef(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ8_ref(const void* pVect1v, const void* pVect2v, size_t qty);
float L2SqrSQ4_ref(const void* pVect1v, const void* pVect2v, size_t qty);
float IPSQ8_ref(const void* pVect1v, const void* pVect2v, size_t qty);

/**
 * Internal implementation functions (not for direct use)
 */
namespace detail {
// SSE implementations
float L2Sqr_sse(const float* pVect1, const float* pVect2, size_t qty);
float IP_sse(const float* pVect1, const float* pVect2, size_t qty);

// AVX2 implementations
float L2Sqr_avx2(const float* pVect1, const float* pVect2, size_t qty);
float IP_avx2(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ4_avx2(const void* pVect1v, const void* pVect2v, size_t qty);

// AVX512 implementations
float L2Sqr_avx512(const float* pVect1, const float* pVect2, size_t qty);
float IP_avx512(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ8_avx512(const void* pVect1v, const void* pVect2v, size_t qty);
float IPSQ8_avx512(const void* pVect1v, const void* pVect2v, size_t qty);

// NEON implementations
float L2Sqr_neon(const float* pVect1, const float* pVect2, size_t qty);
float L2SqrSQ4_neon(const void* pVect1v, const void* pVect2v, size_t qty);
float L2SqrSQ8_neon(const void* pVect1v, const void* pVect2v, size_t qty);
float IP_neon(const float* pVect1, const float* pVect2, size_t qty);
}  // namespace detail

}  // namespace simd
}  // namespace deepsearch
