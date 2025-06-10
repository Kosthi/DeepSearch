#pragma once

#include <string>
#include <thread>
#include <vector>

#include "builder/builder.h"
#include "common.h"
#include "neighbor.h"
#include "quantization/quantizer.h"
#include "utils.h"

namespace deepsearch::quantization {
class SQ4Quantizer;
class SQ8Quantizer;
class FP32Quantizer;
}  // namespace deepsearch::quantization

namespace deepsearch {
namespace searcher {

using namespace quantization;

// 基础搜索器接口
class SearcherBase {
 public:
  virtual ~SearcherBase() = default;
  virtual void SetData(const float* data, int n, int dim) = 0;
  virtual void Optimize(int num_threads = 0) = 0;
  virtual void Search(const float* q, int k, int* dst) const = 0;
  virtual void SetEf(int ef) = 0;
  virtual int GetEf() const = 0;
  virtual std::string GetQuantizerName() const = 0;
};

// 模板化的 Searcher - 编译时确定量化器类型
template <typename QuantizerType>
class Searcher : public SearcherBase {
 public:
  Searcher() = default;

  // 构造函数：接受图和量化器
  explicit Searcher(const Graph& graph,
                    std::unique_ptr<QuantizerType> quantizer)
      : graph_(graph),
        quantizer_(std::move(quantizer)),
        graph_po_(graph.max_degree() / 16) {}

  // 从图构建器创建
  template <typename T>
  static std::unique_ptr<Searcher<QuantizerType>> from_builder(
      std::unique_ptr<builder::Builder<T>> builder, const T* data, size_t n,
      size_t dim, std::unique_ptr<QuantizerType> quantizer) {
    auto graph = builder->build(data, n, dim);
    return std::make_unique<Searcher<QuantizerType>>(std::move(graph),
                                                     std::move(quantizer));
  }

  ~Searcher() override = default;

  // 实现基类接口
  void SetData(const float* data, int n, int dim) override;
  void Optimize(int num_threads = 0) override;
  void Search(const float* q, int k, int* dst) const override;

  template <typename Pool, typename Quant>
  void SearchImpl(Pool& pool, const Quant& quant) const {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph_.prefetch_neighbors(u, graph_po_);
      for (int i = 0; i < po_; ++i) {
        int to = graph_.at(u, i);
        // if (to == -1) {
        //   break;
        // }
        quant->prefetch_data(to, pl_);
      }
      for (int i = 0; i < graph_.max_degree(); ++i) {
        int v = graph_.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po_ < graph_.max_degree() && graph_.at(u, i + po_) != -1) {
          int to = graph_.at(u, i + po_);
          quant->prefetch_data(to, pl_);
        }
        if (pool.vis.get(v)) {
          continue;
        }
        pool.vis.set(v);
        auto cur_dist = quant->compute_query_distance(v);
        pool.insert(v, cur_dist);
      }
    }
  }

  void SetEf(int ef) override { ef_ = ef; }
  int GetEf() const override { return ef_; }
  std::string GetQuantizerName() const override {
    return quantizer_ ? quantizer_->name() : "Unknown";
  }

  // 模板特化的高性能搜索接口
  void SearchFast(const float* q, int k, int* dst) const;

  // 获取量化器（用于高级用法）
  QuantizerType* get_quantizer() const { return quantizer_.get(); }

 private:
  Graph graph_;
  std::unique_ptr<QuantizerType> quantizer_;

  // 数据维度信息
  int nb_ = 0;  // 数据点数量
  int d_ = 0;   // 数据维度

  // Search parameters
  int ef_ = 32;
  int po_ = 1;
  int pl_ = 1;

  // Optimization parameters
  static constexpr int kOptimizePoints = 1000;
  static constexpr int kTryPos = 10;
  static constexpr int kTryPls = 5;
  static constexpr int kTryK = 10;
  int sample_points_num_ = 0;
  std::vector<float> optimize_queries_;
  int graph_po_{};
};

// 类型别名，方便使用
using FP32Searcher = Searcher<quantization::FP32Quantizer>;
using SQ8Searcher = Searcher<quantization::SQ8Quantizer>;
using SQ4Searcher = Searcher<quantization::SQ4Quantizer>;

// 搜索器工厂
class SearcherFactory {
 public:
  enum class SearcherType { FP32, SQ8, SQ4 };

  // 创建基类指针（运行时多态）
  // static std::unique_ptr<SearcherBase> create(
  //     SearcherType type, const Graph& graph,
  //     core::DistanceType metric = core::DistanceType::L2,
  //     size_t dimension = 128);

  // 创建具体类型（编译时确定）
  template <typename QuantizerType>
  static std::unique_ptr<Searcher<QuantizerType>> create(
      const Graph& graph, std::unique_ptr<QuantizerType> quantizer) {
    return std::make_unique<Searcher<QuantizerType>>(graph,
                                                     std::move(quantizer));
  }

  // 便捷创建方法
  static std::unique_ptr<FP32Searcher> createFP32(const Graph& graph,
                                                  DistanceType metric,
                                                  size_t dimension);

  static std::unique_ptr<SQ8Searcher> createSQ8(const Graph& graph,
                                                DistanceType metric,
                                                size_t dimension);

  static std::unique_ptr<SQ4Searcher> createSQ4(const Graph& graph,
                                                DistanceType metric,
                                                size_t dimension);
};

}  // namespace searcher
}  // namespace deepsearch

// 包含模板实现
#include "searcher_impl.h"
