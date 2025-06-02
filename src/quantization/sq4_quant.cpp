#include "sq4_quant.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "allocator.h"
#include "distance/computers.h"

namespace deepsearch {
namespace quantization {

SQ4Quantizer::SQ4Quantizer(core::DistanceType distanceType, size_t dim,
                           std::shared_ptr<FP32Quantizer> reorder_quantizer)
    : d(dim),
      d_align(do_align(dim, kAlign)),
      reorder_quantizer_(std::move(reorder_quantizer)) {
  // 使用SQ4模板特化的距离计算器
  distance_computer_ = distance::DistanceComputerFactory::create<uint8_t>(
      distanceType, dim, QuantizerType::SQ4);
}

SQ4Quantizer::~SQ4Quantizer() {
  if (codes) {
    free(codes);
  }
}

void SQ4Quantizer::train(const float* data, size_t n, size_t dim) {
  if (dim != d) {
    throw std::invalid_argument("Dimension mismatch");
  }

  // 计算全局最小值和最大值
  float min_val = HUGE_VALF;
  float max_val = -HUGE_VALF;

  for (size_t i = 0; i < n * d; ++i) {
    min_val = std::min(min_val, data[i]);
    max_val = std::max(max_val, data[i]);
  }

  offset_ = min_val;
  scale_ = (max_val - min_val) / 15.0f;  // 4位量化，范围0-15
  if (scale_ == 0.0f) {
    scale_ = 1.0f;  // 避免除零
  }

  // 分配内存并编码数据
  size_t total_size = n * (d_align / 2) * sizeof(uint8_t);
  codes = (uint8_t*)alloc2M(total_size);

  for (size_t i = 0; i < n; ++i) {
    encode(data + i * d, reinterpret_cast<uint8_t*>(get_data(i)));
  }

  // 如果有精排量化器，也需要训练它
  if (reorder_quantizer_) {
    reorder_quantizer_->train(data, n, dim);
  }
}

void SQ4Quantizer::encode(const float* input, uint8_t* output) const {
  std::memset(output, 0, d_align / 2);

  for (size_t j = 0; j < d; ++j) {
    float normalized = (input[j] - offset_) / scale_;
    normalized = std::max(0.0f, std::min(15.0f, normalized));
    uint8_t quantized = static_cast<uint8_t>(std::round(normalized));

    if (j % 2 == 0) {
      output[j / 2] |= quantized;  // 低4位
    } else {
      output[j / 2] |= (quantized << 4);  // 高4位
    }
  }
}

void SQ4Quantizer::decode(const uint8_t* input, float* output) const {
  for (size_t j = 0; j < d; ++j) {
    uint8_t quantized;
    if (j % 2 == 0) {
      quantized = input[j / 2] & 0x0F;  // 低4位
    } else {
      quantized = (input[j / 2] >> 4) & 0x0F;  // 高4位
    }
    output[j] = quantized * scale_ + offset_;
  }
}

const char* SQ4Quantizer::get_data(size_t index) const {
  return reinterpret_cast<const char*>(codes + index * (d_align / 2));
}

char* SQ4Quantizer::get_data(size_t index) {
  return reinterpret_cast<char*>(codes + index * (d_align / 2));
}

float SQ4Quantizer::compute_distance(const uint8_t* a, const uint8_t* b) const {
  return distance_computer_->compute(a, b);
}

void SQ4Quantizer::encode_query(const float* query) {
  ensure_thread_query_initialized();
  encode(query, get_query().get());
}

float SQ4Quantizer::compute_query_distance(size_t index) const {
  const uint8_t* data_code = reinterpret_cast<const uint8_t*>(get_data(index));
  return distance_computer_->compute(get_query().get(), data_code);
}

float SQ4Quantizer::compute_query_distance(const uint8_t* code) const {
  return distance_computer_->compute(get_query().get(), code);
}

void SQ4Quantizer::ensure_thread_query_initialized() {
  auto& query = get_query();
  if (query == nullptr) {
    query.reset(static_cast<uint8_t*>(alloc64B(d_align * sizeof(uint8_t))));
  }
}

void SQ4Quantizer::prefetch_data(size_t index, int lines) const {
  mem_prefetch<prefetch_L1>(const_cast<char*>(get_data(index)), lines);
}

}  // namespace quantization
}  // namespace deepsearch
