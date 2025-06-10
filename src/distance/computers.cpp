#include "computers.h"

#include "simd/distance_functions.h"

namespace deepsearch {
namespace distance {

// 余弦距离计算器实现
template <typename T>
float CosineDistanceComputer<T>::compute(const T* a, const T* b) const {
  if constexpr (std::is_same_v<T, float>) {
    // 使用新的统一SIMD接口
    return simd::CosineDistance(a, b, dim_);
  }
  return computeGeneric(a, b);
}

template <typename T>
float CosineDistanceComputer<T>::computeGeneric(const T* a, const T* b) const {
  float dot_product = 0;
  float norm_a = 0;
  float norm_b = 0;

  for (size_t i = 0; i < dim_; ++i) {
    float val_a = static_cast<float>(a[i]);
    float val_b = static_cast<float>(b[i]);
    dot_product += val_a * val_b;
    norm_a += val_a * val_a;
    norm_b += val_b * val_b;
  }

  float norm_product = std::sqrt(norm_a * norm_b);
  if (norm_product == 0) return 1.0;  // 避免除零

  return 1.0 - (dot_product / norm_product);
}

std::vector<std::string> DistanceComputerFactory::getSupportedTypes() {
  return {"L2", "IP", "COSINE"};
}

bool DistanceComputerFactory::isTypeSupported(DistanceType type) {
  return type == DistanceType::L2 || type == DistanceType::IP ||
         type == DistanceType::COSINE;
}

// 显式实例化
template class L2DistanceComputer<float>;
template class IPDistanceComputer<float>;
template class CosineDistanceComputer<float>;

// 添加 uint8_t 类型的实例化
template class L2DistanceComputer<uint8_t>;
template class IPDistanceComputer<uint8_t>;
template class CosineDistanceComputer<uint8_t>;

template std::unique_ptr<core::DistanceComputerTemplate<float>>
DistanceComputerFactory::create<float>(DistanceType, size_t);

// 添加 uint8_t 类型的工厂方法实例化
template std::unique_ptr<core::DistanceComputerTemplate<uint8_t>>
DistanceComputerFactory::create<uint8_t>(DistanceType, size_t);

}  // namespace distance
}  // namespace deepsearch
