#pragma once

#include <memory>
#include <string>

#include "core/interfaces.h"
#include "searcher/searcher.h"
#include "simd/distance_functions.h"

namespace deepsearch {
namespace distance {

using deepsearch::QuantizerType;

// 模板化的L2距离计算器
template <typename T, QuantizerType Quantizer = QuantizerType::FP32>
class L2DistanceComputer : public core::DistanceComputerTemplate<T> {
 public:
  explicit L2DistanceComputer(size_t dim) : dim_(dim) {}

  float compute(const T* a, const T* b) const override {
    // 首先根据数据类型T进行分支
    if constexpr (std::is_same_v<T, float>) {
      // float类型数据，使用标准L2 SIMD函数
      return simd::L2Sqr(a, b, dim_);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      // uint8_t类型数据，根据量化类型选择优化函数
      if constexpr (Quantizer == QuantizerType::SQ8) {
        return simd::L2SqrSQ8_ext(a, b, dim_);
      } else if constexpr (Quantizer == QuantizerType::SQ4) {
        return simd::L2SqrSQ4(a, b, dim_);
      } else {
        return computeGeneric(a, b);
      }
    } else {
      // 其他类型，回退到通用实现
      return computeGeneric(a, b);
    }
  }

  std::string name() const override {
    if constexpr (Quantizer == QuantizerType::SQ8) {
      return "L2Distance_SQ8";
    } else if constexpr (Quantizer == QuantizerType::SQ4) {
      return "L2Distance_SQ4";
    } else {
      return "L2Distance_FP32";
    }
  }

 private:
  size_t dim_;

  float computeGeneric(const T* a, const T* b) const {
    float result = 0;
    for (size_t i = 0; i < dim_; ++i) {
      float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
      result += diff * diff;
    }
    return result;
  }
};

// IP距离计算器
template <typename T, QuantizerType Quantizer = QuantizerType::FP32>
class IPDistanceComputer : public core::DistanceComputerTemplate<T> {
 public:
  explicit IPDistanceComputer(size_t dim) : dim_(dim) {}

  float compute(const T* a, const T* b) const override {
    // 首先根据数据类型T进行分支
    if constexpr (std::is_same_v<T, float>) {
      // float类型数据，使用标准L2 SIMD函数
      return 1.0f - simd::IP(a, b, dim_);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      // uint8_t类型数据，根据量化类型选择优化函数
      if constexpr (Quantizer == QuantizerType::SQ8) {
        return 1.0f - simd::IPSQ8_ext(a, b, dim_);
      } else if constexpr (Quantizer == QuantizerType::SQ4) {
        return computeGeneric(a, b);
      } else {
        return computeGeneric(a, b);
      }
    } else {
      // 其他类型，回退到通用实现
      return computeGeneric(a, b);
    }
  }

  std::string name() const override {
    if constexpr (Quantizer == QuantizerType::SQ8) {
      return "IPDistance_SQ8";
    } else {
      return "IPDistance_FP32";
    }
  }

 private:
  size_t dim_;

  float computeGeneric(const T* a, const T* b) const {
    float result = 0;
    for (size_t i = 0; i < dim_; ++i) {
      result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return 1.0f - result;
  }
};

// COSINE距离计算器（暂不支持量化优化）
template <typename T>
class CosineDistanceComputer : public core::DistanceComputerTemplate<T> {
 public:
  explicit CosineDistanceComputer(size_t dim) : dim_(dim) {}

  float compute(const T* a, const T* b) const override;
  std::string name() const override { return "CosineDistance"; }

 private:
  size_t dim_;
  float computeGeneric(const T* a, const T* b) const;
};

// 距离计算器工厂
class DistanceComputerFactory {
 public:
  static std::vector<std::string> getSupportedTypes();
  static bool isTypeSupported(DistanceType type);

  // 原有方法保持兼容
  template <typename T>
  static std::unique_ptr<core::DistanceComputerTemplate<T>> create(
      DistanceType type, size_t dim) {
    return create<T>(type, dim, QuantizerType::FP32);
  }

  // 新的模板化工厂方法
  template <typename T>
  static std::unique_ptr<core::DistanceComputerTemplate<T>> create(
      DistanceType type, size_t dim, QuantizerType quant_type) {
    switch (quant_type) {
      case QuantizerType::SQ8:
        return createWithQuant<T, QuantizerType::SQ8>(type, dim);
      case QuantizerType::SQ4:
        return createWithQuant<T, QuantizerType::SQ4>(type, dim);
      case QuantizerType::FP32:
      default:
        return createWithQuant<T, QuantizerType::FP32>(type, dim);
    }
  }

 private:
  template <typename T, QuantizerType Quantizer>
  static std::unique_ptr<core::DistanceComputerTemplate<T>> createWithQuant(
      DistanceType type, size_t dim) {
    switch (type) {
      case DistanceType::L2:
        return std::make_unique<L2DistanceComputer<T, Quantizer>>(dim);
      case DistanceType::IP:
        return std::make_unique<IPDistanceComputer<T, Quantizer>>(dim);
      case DistanceType::COSINE:
        return std::make_unique<CosineDistanceComputer<T>>(dim);
      default:
        throw std::invalid_argument("Unsupported distance type");
    }
  }
};

}  // namespace distance
}  // namespace deepsearch
