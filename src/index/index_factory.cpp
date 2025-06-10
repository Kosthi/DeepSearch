#include "index_factory.h"

#include <stdexcept>

#include "common.h"
#include "core/logger.h"
#include "hnsw_index.h"
#include "ivf_index.h"

namespace deepsearch {
namespace index {

template <typename T>
std::unique_ptr<GraphBasedIndex<T>> IndexFactory<T>::createGraphIndex(
    IndexType type, DistanceType distance_type, size_t dimension,
    QuantizerType quantizer, const builder::BuilderConfig& config) {
  switch (type) {
    case IndexType::HNSW: {
      return std::make_unique<HNSWIndex<T>>(distance_type, dimension, quantizer,
                                            config);
    }
    default:
      throw std::invalid_argument("Unsupported graph index type: " +
                                  std::to_string(static_cast<int>(type)));
  }
}

template <typename T>
std::unique_ptr<PartitionBasedIndex<T>> IndexFactory<T>::createPartitionIndex(
    IndexType type, DistanceType distance_type, size_t dimension,
    QuantizerType quantizer, const builder::BuilderConfig& config) {
  switch (type) {
    case IndexType::IVF: {
      return std::make_unique<IVFIndex<T>>(distance_type, dimension, quantizer,
                                           config);
    }
    default:
      throw std::invalid_argument("Unsupported partition index type: " +
                                  std::to_string(static_cast<int>(type)));
  }
}

template <typename T>
std::unique_ptr<IndexBase<T>> IndexFactory<T>::createIndex(
    IndexType type, DistanceType distance_type, size_t dimension,
    QuantizerType quantizer, const builder::BuilderConfig& config) {
  switch (type) {
    case IndexType::HNSW:
      return createGraphIndex(type, distance_type, dimension, quantizer,
                              config);
    case IndexType::IVF:
      return createPartitionIndex(type, distance_type, dimension, quantizer,
                                  config);
    case IndexType::BRUTEFORCE:
      // TODO: 实现暴力搜索索引
      throw std::runtime_error("BRUTEFORCE index not implemented yet");
    default:
      throw std::invalid_argument("Unknown index type: " +
                                  std::to_string(static_cast<int>(type)));
  }
}

template <typename T>
IndexType IndexFactory<T>::parseType(const std::string& name) {
  std::string lower_name = name;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                 ::tolower);

  if (lower_name == "hnsw") {
    return IndexType::HNSW;
  } else if (lower_name == "ivf") {
    return IndexType::IVF;
  } else if (lower_name == "bruteforce" || lower_name == "brute_force") {
    return IndexType::BRUTEFORCE;
  } else {
    throw std::invalid_argument("Unknown index type name: " + name);
  }
}

template <typename T>
std::string IndexFactory<T>::typeName(IndexType type) {
  switch (type) {
    case IndexType::HNSW:
      return "HNSW";
    case IndexType::IVF:
      return "IVF";
    case IndexType::BRUTEFORCE:
      return "BRUTEFORCE";
    default:
      return "UNKNOWN";
  }
}

// 显式实例化模板
template class IndexFactory<float>;
// template class IndexFactory<int>;

}  // namespace index
}  // namespace deepsearch
