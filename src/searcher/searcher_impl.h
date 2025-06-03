#pragma once

#include "core/logger.h"
#include "quantization/fp32_quant.h"
#include "quantization/sq4_quant.h"
#include "quantization/sq8_quant.h"
#include "searcher.h"

namespace deepsearch {
namespace searcher {

template <typename QuantizerType>
void Searcher<QuantizerType>::SetData(const float* data, int n, int dim) {
  auto logger = core::LogManager::instance().get_logger("searcher");
  nb_ = n;
  d_ = dim;

  if (!quantizer_) {
    logger->error("Quantizer not initialized");
    throw std::runtime_error("Quantizer not initialized");
  }

  logger->info("Starting quantizer training for {} points", n);
  auto t1 = std::chrono::high_resolution_clock::now();
  quantizer_->train(data, n, dim);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto training_time = std::chrono::duration<double>(t2 - t1).count();
  logger->info("Quantizer training completed in {:.2f}s", training_time);

  sample_points_num_ = std::min(kOptimizePoints, nb_ - 1);
  std::vector<int> sample_points(sample_points_num_);
  std::mt19937 rng;
  GenRandom(rng, sample_points.data(), sample_points_num_, nb_);

  optimize_queries_.resize(sample_points_num_ * d_);
  for (int i = 0; i < sample_points_num_; ++i) {
    std::memcpy(optimize_queries_.data() + i * d_, data + sample_points[i] * d_,
                d_ * sizeof(float));
  }
}

template <typename QuantizerType>
void Searcher<QuantizerType>::Optimize(int num_threads) {
  auto logger = core::LogManager::instance().get_logger("searcher");

  if (num_threads == 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
  }

  logger->info("Starting search optimization with {} threads", num_threads);

  std::vector<int> try_pos(
      std::min(kTryPos, static_cast<int>(graph_.max_degree())));
  std::vector<int> try_pls(
      std::min(kTryPls, (int)upper_div(quantizer_->code_size(), 64)));
  std::iota(try_pos.begin(), try_pos.end(), 1);
  std::iota(try_pls.begin(), try_pls.end(), 1);
  std::vector<int> dummy_dst(kTryK);
  logger->info("=============Start optimization=============");
  {  // warmup
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < sample_points_num_; ++i) {
      Search(optimize_queries_.data() + i * d_, kTryK, dummy_dst.data());
    }
  }

  double min_ela = std::numeric_limits<double>::max();
  int best_po = 0, best_pl = 0;
  for (auto try_po : try_pos) {
    for (auto try_pl : try_pls) {
      this->po_ = try_po;
      this->pl_ = try_pl;
      auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
      for (int i = 0; i < sample_points_num_; ++i) {
        Search(optimize_queries_.data() + i * d_, kTryK, dummy_dst.data());
      }

      auto ed = std::chrono::high_resolution_clock::now();
      auto ela = std::chrono::duration<double>(ed - st).count();
      if (ela < min_ela) {
        min_ela = ela;
        best_po = try_po;
        best_pl = try_pl;
      }
    }
  }
  this->po_ = 1;
  this->pl_ = 1;
  auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
  for (int i = 0; i < sample_points_num_; ++i) {
    Search(optimize_queries_.data() + i * d_, kTryK, dummy_dst.data());
  }
  auto ed = std::chrono::high_resolution_clock::now();
  double baseline_ela = std::chrono::duration<double>(ed - st).count();
  logger->info(
      "settint best po = {}, best pl = {}, gaining {:.2f} performance "
      "improvement",
      best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));

  this->po_ = best_po;
  this->pl_ = best_pl;
}

template <typename QuantizerType>
void Searcher<QuantizerType>::Search(const float* q, int k, int* dst) const {
  // if (!quantizer_) {
  //   throw std::runtime_error("Quantizer not initialized");
  // }

  // 编码查询向量
  quantizer_->encode_query(q);
  // 创建候选池
  LinearPool<float> pool(nb_, std::max(k, ef_), k);

  // 导航到第 0 层
  graph_.initialize_search(pool, quantizer_);
  // 搜索第 0 层
  SearchImpl(pool, quantizer_);

  // 使用量化器重排序获得最终结果
  quantizer_->reorder(pool, q, dst, k);
}

template <typename QuantizerType>
void Searcher<QuantizerType>::SearchFast(const float* q, int k,
                                         int* dst) const {
  // 高性能版本，避免虚函数调用
  // if constexpr (std::is_same_v<QuantizerType,
  //                              quantization::FP32Quantizer<float, uint8_t>>)
  //                              {
  //   // FP32 特化实现
  //   SearchFP32Impl(q, k, dst);
  // } else if constexpr (std::is_same_v<QuantizerType,
  // quantization::SQ8Quantizer<
  //                                                        float, uint8_t>>) {
  //   // SQ8 特化实现
  //   SearchSQ8Impl(q, k, dst);
  // } else if constexpr (std::is_same_v<QuantizerType,
  // quantization::SQ4Quantizer<
  //                                                        float, uint8_t>>) {
  //   // SQ4 特化实现
  //   SearchSQ4Impl(q, k, dst);
  // } else {
  //   // 通用实现
  //   Search(q, k, dst);
  // }
}

}  // namespace searcher
}  // namespace deepsearch
