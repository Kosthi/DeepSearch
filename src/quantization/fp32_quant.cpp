#include "fp32_quant.h"

#include <core/factory.h>
#include <distance/computers.h>

#include <cstring>
#include <stdexcept>

#include "allocator.h"

namespace deepsearch {
namespace quantization {

FP32Quantizer::FP32Quantizer(core::DistanceType distanceType, size_t dim)
    : d(dim), d_align(do_align(dim, kAlign)) {
  // 使用FP32模板特化的距离计算器
  distance_computer_ = distance::DistanceComputerFactory::create<float>(
      distanceType, dim, QuantizerType::FP32);
}

FP32Quantizer::~FP32Quantizer() {
  if (codes) {
    free(codes);
  }
}

void FP32Quantizer::train(const float* data, size_t n, size_t dim) {
  if (dim != d) {
    throw std::invalid_argument("Dimension mismatch");
  }

  size_t total_size = n * d_align * sizeof(float);
  codes = (char*)alloc2M(total_size);

  for (size_t i = 0; i < n; ++i) {
    encode(data + i * d, reinterpret_cast<float*>(get_data(i)));
  }
}

void FP32Quantizer::encode(const float* input, float* output) const {
  std::memcpy(output, input, d * sizeof(float));
  // 填充对齐部分为0
  for (size_t i = d; i < d_align; ++i) {
    output[i] = 0.0f;
  }
}

void FP32Quantizer::decode(const float* input, float* output) const {
  std::memcpy(output, input, d * sizeof(float));
}

const char* FP32Quantizer::get_data(size_t index) const {
  return codes + index * d_align * sizeof(float);
}

char* FP32Quantizer::get_data(size_t index) {
  return codes + index * d_align * sizeof(float);
}

float FP32Quantizer::compute_distance(const float* a, const float* b) const {
  return distance_computer_->compute(a, b);
}

void FP32Quantizer::prefetch_data(size_t index, int lines) const {
  mem_prefetch<prefetch_L1>(const_cast<char*>(get_data(index)), lines);
}

void FP32Quantizer::encode_query(const float* query) {
  ensure_thread_query_initialized();
  encode(query, query_.get());
}

float FP32Quantizer::compute_query_distance(size_t index) const {
  const float* data_code = reinterpret_cast<const float*>(get_data(index));
  return distance_computer_->compute(query_.get(), data_code);
}

float FP32Quantizer::compute_query_distance(const float* code) const {
  return distance_computer_->compute(query_.get(), code);
}

void FP32Quantizer::ensure_thread_query_initialized() {
  if (query_ == nullptr) {
    query_.reset(static_cast<float*>(alloc64B(d_align * sizeof(float))));
  }
}

}  // namespace quantization
}  // namespace deepsearch
