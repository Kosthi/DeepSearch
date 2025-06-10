#pragma once

#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "builder/builder_config.h"
#include "core/interfaces.h"
#include "distance/computers.h"
#include "quantization/quantizer.h"

namespace deepsearch {
namespace index {

// 倒排列表项
struct InvertedListItem {
  size_t id;                  // 原始向量ID
  std::vector<uint8_t> code;  // 量化后的编码

  InvertedListItem(size_t id, const std::vector<uint8_t>& code)
      : id(id), code(code) {}
};

template <typename T>
class IVFIndex : public PartitionBasedIndex<T> {
 public:
  explicit IVFIndex(DistanceType distance_type, size_t dimension,
                    QuantizerType quantizer,
                    const builder::BuilderConfig& config = {});
  ~IVFIndex() override = default;

  // 实现基类接口
  void build(const float* data, size_t n, size_t dim) override;
  void search(const float* query, int k, int* dst,
              size_t nprobe = 0) const override;
  // std::vector<std::vector<int>> batch_search(const T* queries, size_t nq,
  //                                            size_t dim, size_t k,
  //                                            size_t nprobe = 0) const
  //                                            override;

  void save(const std::string& path) const override {}
  void load(const std::string& path) override {}

  void set_nprobe(size_t nprobe) override {
    // config_.nprobe = std::min(nprobe, config_.nlist);
  }
  size_t get_nprobe() const override {
    // return config_.nprobe;
    return 0;
  }
  size_t get_nlist() const override {
    // return config_.nlist;
    return 0;
  }

  size_t size() const override { return n_total_; }
  size_t dimension() const override { return dim_; }
  std::string name() const override { return "IVF"; }

  // IVF特有接口
  void add_vectors(const T* vectors, const size_t* ids, size_t n);
  void remove_vectors(const size_t* ids, size_t n);
  const std::vector<InvertedListItem>& get_inverted_list(
      size_t cluster_id) const;
  const T* get_centroid(size_t cluster_id) const;

 private:
  void train_centroids(const T* data, size_t n);
  void build_inverted_lists(const T* data, size_t n);
  size_t assign_to_cluster(const T* vector) const;
  std::vector<size_t> search_clusters(const T* query, size_t nprobe) const;
  void encode_vector(const T* vector, size_t cluster_id,
                     std::vector<uint8_t>& code) const;
  void decode_vector(const std::vector<uint8_t>& code, size_t cluster_id,
                     T* vector) const;

  size_t dim_;                        // 向量维度
  size_t n_total_;                    // 总向量数量
  builder::IVFBuilderConfig config_;  // IVF配置
  DistanceType distance_type_;        // 距离类型

  // 聚类中心
  std::vector<T> centroids_;  // 形状: [nlist, dim]

  // 倒排列表
  std::vector<std::vector<InvertedListItem>> inverted_lists_;

  // 距离计算器
  std::unique_ptr<core::DistanceComputerTemplate<T>> distance_computer_;

  // 内部量化器（用于压缩存储）
  std::unique_ptr<quantization::QuantizerBase<T, uint8_t>> quantizer_;

  // 向量ID映射
  std::unordered_map<size_t, std::pair<size_t, size_t>>
      id_to_location_;  // id -> (cluster, index)
};

}  // namespace index
}  // namespace deepsearch
