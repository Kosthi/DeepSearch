#pragma once

#include "builder/builder.h"
#include "builder/builder_config.h"
#include "core/interfaces.h"
#include "searcher/searcher.h"

namespace deepsearch {
namespace index {

// HNSW索引实现
template <typename T>
class HNSWIndex : public GraphBasedIndex<T> {
 public:
  explicit HNSWIndex(DistanceType distance_type, size_t dimension,
                     QuantizerType quantizer,
                     const builder::BuilderConfig& config = {});
  ~HNSWIndex() override = default;

  void SetData(const float* data, int n, int dim) override {
    searcher_->SetData(data, n, dim);
    searcher_->Optimize();
  }

  // 实现基类接口
  void build(const float* data, size_t n, size_t dim) override;
  void search(const float* q, int k, int* dst) const override;

  void save(const std::string& path) const override;
  void load(const std::string& path) override;

  size_t size() const override { return n_total_; }
  size_t dimension() const override { return dim_; }
  std::string name() const override { return "HNSW"; }

  // 图索引特有接口
  void add_points(const T* data, const size_t* labels, size_t n) override;
  void remove_points(const size_t* labels, size_t n) override;
  void set_ef(int ef) override { searcher_->SetEf(ef); }

  size_t max_degree() const { return 0; }
  const T* neighbors(size_t node_id) const { return 0; };
  size_t degree(size_t node_id) const { return 0; }
  void prefetch_neighbors(size_t node_id, int lines = 1) const {}
  const std::vector<size_t>& entry_points() const { return {}; }
  // int get_ef() const override { return config_.ef; }

 private:
  size_t dim_;
  size_t n_total_;
  // HNSWIndexConfig config_;
  DistanceType distance_type_;
  QuantizerType quantizer_type_;

  std::unique_ptr<builder::GraphBuilder<T>> builder_;
  std::unique_ptr<searcher::SearcherBase> searcher_;
  Graph graph_;
};

}  // namespace index
}  // namespace deepsearch
