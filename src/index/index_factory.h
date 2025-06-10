#pragma once

#include <algorithm>
#include <any>
#include <cctype>
#include <memory>
#include <string>

#include "common.h"
#include "hnsw_index.h"
#include "index.h"
#include "ivf_index.h"

namespace deepsearch {
namespace index {

// 前向声明配置结构
struct HNSWIndexConfig;
struct IVFIndexConfig;

template <typename T>
class IndexFactory {
 public:
  // 创建图索引
  static std::unique_ptr<GraphBasedIndex<T>> createGraphIndex(
      IndexType type, DistanceType distance_type, size_t dimension,
      QuantizerType quantizer = QuantizerType::FP32,
      const builder::BuilderConfig& config = {});

  // 创建划分索引
  static std::unique_ptr<PartitionBasedIndex<T>> createPartitionIndex(
      IndexType type, DistanceType distance_type, size_t dimension,
      QuantizerType quantizer = QuantizerType::FP32,
      const builder::BuilderConfig& config = {});

  // 通用创建接口
  static std::unique_ptr<IndexBase<T>> createIndex(
      IndexType type, DistanceType distance_type, size_t dimension,
      QuantizerType quantizer = QuantizerType::FP32,
      const builder::BuilderConfig& config = {});

  // 解析索引类型
  static IndexType parseType(const std::string& name);
  static std::string typeName(IndexType type);
};

}  // namespace index
}  // namespace deepsearch
