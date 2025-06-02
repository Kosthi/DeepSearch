#include "sq8_quant.h"

#include <cmath>
#include <stdexcept>

#include "allocator.h"
#include "distance/computers.h"

namespace deepsearch {
namespace quantization {

SQ8Quantizer::SQ8Quantizer(core::DistanceType distanceType, size_t dim,
                           std::shared_ptr<FP32Quantizer> reorder_quantizer)
    : d(dim),
      d_align(do_align(dim, kAlign)),
      scale_(d_align),
      offset_(d_align),
      reorder_quantizer_(std::move(reorder_quantizer)) {
  // 使用SQ8模板特化的距离计算器
  distance_computer_ = distance::DistanceComputerFactory::create<uint8_t>(
      distanceType, dim, QuantizerType::SQ8);
}

SQ8Quantizer::~SQ8Quantizer() {
  if (codes) {
    free(codes);
  }
}

void SQ8Quantizer::train(const float* data, size_t n, size_t dim) {
  if (dim != d) {
    throw std::invalid_argument("Dimension mismatch");
  }

  // 计算每个维度的最小值和最大值
  std::vector<float> min_vals(d, HUGE_VALF);
  std::vector<float> max_vals(d, -HUGE_VALF);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      float val = data[i * d + j];
      min_vals[j] = std::min(min_vals[j], val);
      max_vals[j] = std::max(max_vals[j], val);
    }
  }

  // 计算量化参数
  for (size_t j = 0; j < d; ++j) {
    offset_[j] = min_vals[j];
    scale_[j] = (max_vals[j] - min_vals[j]) / 255.0f;
    if (scale_[j] == 0.0f) {
      scale_[j] = 1.0f;  // 避免除零
    }
  }

  // 填充对齐部分
  for (size_t j = d; j < d_align; ++j) {
    offset_[j] = 0.0f;
    scale_[j] = 1.0f;
  }

  // 分配内存并编码数据
  size_t total_size = n * d_align * sizeof(uint8_t);
  codes = (char*)alloc2M(total_size);

  for (size_t i = 0; i < n; ++i) {
    encode(data + i * d, reinterpret_cast<uint8_t*>(get_data(i)));
  }

  // 如果有精排量化器，也需要训练它
  if (reorder_quantizer_) {
    reorder_quantizer_->train(data, n, dim);
  }
}

void SQ8Quantizer::encode(const float* input, uint8_t* output) const {
  for (size_t j = 0; j < d; ++j) {
    float normalized = (input[j] - offset_[j]) / scale_[j];
    normalized = std::max(0.0f, std::min(255.0f, normalized));
    output[j] = static_cast<uint8_t>(std::round(normalized));
  }

  // 填充对齐部分为0
  for (size_t j = d; j < d_align; ++j) {
    output[j] = 0;
  }
}

void SQ8Quantizer::decode(const uint8_t* input, float* output) const {
  for (size_t j = 0; j < d; ++j) {
    output[j] = input[j] * scale_[j] + offset_[j];
  }
}

const char* SQ8Quantizer::get_data(size_t index) const {
  return codes + index * d_align * sizeof(uint8_t);
}

char* SQ8Quantizer::get_data(size_t index) {
  return codes + index * d_align * sizeof(uint8_t);
}

float SQ8Quantizer::compute_distance(const uint8_t* a, const uint8_t* b) const {
  return distance_computer_->compute(a, b);
}

void SQ8Quantizer::prefetch_data(size_t index, int lines) const {
  mem_prefetch<prefetch_L1>(const_cast<char*>(get_data(index)), lines);
}

void SQ8Quantizer::encode_query(const float* query) {
  ensure_thread_query_initialized();
  encode(query, query_.get());
}

float SQ8Quantizer::compute_query_distance(size_t index) const {
  const uint8_t* data_code = reinterpret_cast<const uint8_t*>(get_data(index));
  return distance_computer_->compute(query_.get(), data_code);
}

float SQ8Quantizer::compute_query_distance(const uint8_t* code) const {
  return distance_computer_->compute(query_.get(), code);
}

void SQ8Quantizer::ensure_thread_query_initialized() {
  if (query_ == nullptr) {
    query_.reset(static_cast<uint8_t*>(alloc64B(d_align * sizeof(uint8_t))));
  }
}

}  // namespace quantization
}  // namespace deepsearch
