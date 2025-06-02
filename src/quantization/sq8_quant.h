#pragma once

#include <memory>
#include <vector>

#include "core/interfaces.h"
#include "quantizer.h"
#include "simd/distance_functions.h"

namespace deepsearch {
namespace quantization {

class FP32Quantizer;

class SQ8Quantizer : public QuantizerBase<float, uint8_t> {
 public:
  using data_type = uint8_t;
  constexpr static int kAlign = 16;

  SQ8Quantizer() = default;

  // 修改构造函数，传入FP32量化器作为精排器
  explicit SQ8Quantizer(
      core::DistanceType distanceType, size_t dim,
      std::shared_ptr<FP32Quantizer> reorder_quantizer = nullptr);

  ~SQ8Quantizer() override;

  // 实现基类接口
  void train(const float* data, size_t n, size_t dim) override;
  void encode(const float* input, uint8_t* output) const override;
  void decode(const uint8_t* input, float* output) const override;

  size_t code_size() const override { return d_align * sizeof(uint8_t); }
  size_t dimension() const override { return d; }
  std::string name() const override { return "SQ8Quantizer"; }

  const char* get_data(size_t index) const override;
  char* get_data(size_t index) override;

  // 距离计算（直接在编码空间）
  float compute_distance(const uint8_t* a, const uint8_t* b) const;
  void prefetch_data(size_t index, int lines = 1) const;

  // 重排序接口 - 使用FP32量化器进行精排
  template <typename Pool>
  void reorder(const Pool& pool, const float* query, int* dst, int k) const {
    // if (reorder_quantizer_) {
    //   // 使用FP32量化器进行精确重排序
    //   std::vector<std::pair<int, float>> candidates;
    //   candidates.reserve(std::min(k, pool.size()));
    //
    //   for (int i = 0; i < std::min(k, pool.size()); ++i) {
    //     int id = pool.id(i);
    //     if (id >= 0) {
    //       // 从FP32量化器获取原始浮点数据
    //       const float* fp32_data =
    //           reinterpret_cast<const
    //           float*>(reorder_quantizer_->get_data(id));
    //       // 使用FP32量化器的距离计算方法
    //       float exact_dist =
    //           reorder_quantizer_->compute_distance(query, fp32_data);
    //       candidates.emplace_back(id, exact_dist);
    //     }
    //   }
    //
    //   std::sort(
    //       candidates.begin(), candidates.end(),
    //       [](const auto& a, const auto& b) { return a.second < b.second; });
    //
    //   int result_size = std::min(k, static_cast<int>(candidates.size()));
    //   for (int i = 0; i < result_size; ++i) {
    //     dst[i] = candidates[i].first;
    //   }
    //   for (int i = result_size; i < k; ++i) {
    //     dst[i] = -1;
    //   }
    // } else {
    //   // 如果没有精排器，使用简单的ID复制
    //   for (int i = 0; i < k; ++i) {
    //     dst[i] = (i < pool.size()) ? pool.id(i) : -1;
    //   }
    // }
  }

  // 新增：查询编码接口
  void encode_query(const float* query) override;
  float compute_query_distance(size_t index) const override;
  float compute_query_distance(const uint8_t* code) const override;

  void ensure_thread_query_initialized() override;

 private:
  size_t d, d_align;
  char* codes = nullptr;
  std::vector<float> scale_, offset_;
  std::shared_ptr<FP32Quantizer> reorder_quantizer_;
  std::unique_ptr<core::DistanceComputerTemplate<uint8_t>> distance_computer_;
};

}  // namespace quantization
}  // namespace deepsearch
