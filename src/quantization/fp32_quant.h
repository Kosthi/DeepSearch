#pragma once

#include "common.h"
#include "core/interfaces.h"
#include "quantizer.h"

namespace deepsearch {
namespace quantization {

class FP32Quantizer : public QuantizerBase<float, float> {
 public:
  using data_type = float;
  constexpr static int kAlign = 16;

  FP32Quantizer() = default;
  explicit FP32Quantizer(DistanceType metric, size_t dim);
  ~FP32Quantizer() override;

  // 实现基类接口
  void train(const float* data, size_t n, size_t dim) override;
  void encode(const float* input, float* output) const override;
  void decode(const float* input, float* output) const override;

  size_t code_size() const override { return d_align * sizeof(float); }
  size_t dimension() const override { return d; }
  std::string name() const override { return "FP32Quantizer"; }

  const char* get_data(size_t index) const override;
  char* get_data(size_t index) override;

  // 距离计算
  float compute_distance(const float* a, const float* b) const;
  void encode_query(const float* query) override;
  float compute_query_distance(size_t index) const override;
  float compute_query_distance(const float* code) const override;

  void ensure_thread_query_initialized() override;

  void prefetch_data(size_t index, int lines = 1) const;

  // 重排序接口，模板函数定义在头文件中
  template <typename Pool>
  inline void reorder(const Pool& pool, int* dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

 private:
  size_t d, d_align;
  char* codes = nullptr;
  std::unique_ptr<core::DistanceComputerTemplate<float>> distance_computer_;
};

}  // namespace quantization
}  // namespace deepsearch
