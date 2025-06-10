#include "hnsw_index.h"

#include "builder/builder_factory.h"
#include "searcher/searcher.h"

namespace deepsearch {
namespace index {

template <typename T>
HNSWIndex<T>::HNSWIndex(DistanceType distance_type, size_t dimension,
                        QuantizerType quantizer,
                        const builder::BuilderConfig& config)
    : dim_(dimension),
      n_total_(0),
      distance_type_(distance_type),
      quantizer_type_(quantizer) {
  builder_ = builder::BuilderFactory<T>::create(IndexType::HNSW, distance_type_,
                                                dimension, config);
}

template <typename T>
void HNSWIndex<T>::build(const float* data, size_t n, size_t dim) {
  if (dim != dim_) {
    throw std::invalid_argument("Dimension mismatch");
  }

  if (!builder_) {
    throw std::runtime_error("Builder not initialized");
  }

  // 使用构建器构建图
  graph_ = builder_->build(data, n, dim);
  n_total_ = n;

  // 创建搜索器
  switch (quantizer_type_) {
    case QuantizerType::FP32: {
      searcher_ =
          searcher::SearcherFactory::createFP32(graph_, distance_type_, dim_);
      break;
    }
    case QuantizerType::SQ4: {
      searcher_ =
          searcher::SearcherFactory::createSQ4(graph_, distance_type_, dim_);
      break;
    }
    case QuantizerType::SQ8: {
      searcher_ =
          searcher::SearcherFactory::createSQ8(graph_, distance_type_, dim_);
      break;
    }
    default: {
    } break;
  }

  searcher_->SetData(data, n, dim);
  searcher_->Optimize(1);
}

template <typename T>
void HNSWIndex<T>::search(const float* query, int k, int* dst) const {
  searcher_->Search(query, k, dst);
}

// template <typename T>
// std::vector<std::vector<int>> HNSWIndex<T>::batch_search(const T* queries,
//                                                          size_t nq, size_t
//                                                          dim, size_t k) const
//                                                          {
//   if (dim != dim_) {
//     throw std::invalid_argument("Dimension mismatch");
//   }
//
//   std::vector<std::vector<int>> results(nq);
//
//   for (size_t i = 0; i < nq; ++i) {
//     results[i] = search(queries + i * dim, k);
//   }
//
//   return results;
// }

template <typename T>
void HNSWIndex<T>::save(const std::string& path) const {
  graph_.save(path);
}

template <typename T>
void HNSWIndex<T>::load(const std::string& path) {
  graph_.load(path);
  searcher_ =
      searcher::SearcherFactory::createFP32(graph_, distance_type_, dim_);
}

template <typename T>
void HNSWIndex<T>::add_points(const T* data, const size_t* labels, size_t n) {
  throw std::runtime_error("not implement!");
  // if (!builder_) {
  //   throw std::runtime_error("Builder not initialized");
  // }
  //
  // builder_->add_points(data, labels, n);
  // n_total_ += n;
  //
  // // 重新创建搜索器
  // if (searcher_) {
  //   graph_ = builder_->build(nullptr, 0, dim_);  // 从builder获取更新后的图
  //   auto quantizer =
  //       quantization::QuantizerFactory::create<T>("FP32", distance_type_,
  //       dim_);
  //   searcher_ =
  //       searcher::SearcherFactory::createFP32(graph_, distance_type_, dim_);
  //   // searcher_->SetEf(config_.ef);
  // }
}

template <typename T>
void HNSWIndex<T>::remove_points(const size_t* labels, size_t n) {
  throw std::runtime_error("not implement!");

  // if (!builder_) {
  //   throw std::runtime_error("Builder not initialized");
  // }
  //
  // builder_->remove_points(labels, n);
  // n_total_ = (n_total_ > n) ? n_total_ - n : 0;
  //
  // // 重新创建搜索器
  // if (searcher_) {
  //   graph_ = builder_->build(nullptr, 0, dim_);  // 从builder获取更新后的图
  //   auto quantizer = quantization::QuantizerFactory::create<T>(
  //       QuantizerType::SQ8, distance_type_, dim_);
  //   searcher_ =
  //       searcher::SearcherFactory::createFP32(graph_, distance_type_, dim_);
  //   searcher_->SetEf(config_.ef);
  // }
}

template class HNSWIndex<float>;

}  // namespace index
}  // namespace deepsearch
