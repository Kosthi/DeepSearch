#pragma once

#include "algorithm/hnsw_wrapper.h"
#include "index.h"

namespace deepsearch {
namespace index {

template <typename node_t>
class DenseGraph {
 public:
  DenseGraph() = default;
  explicit DenseGraph(size_t num_nodes, size_t max_degree);
  DenseGraph(const DenseGraph& other);
  DenseGraph& operator=(const DenseGraph& other);
  DenseGraph(DenseGraph&& other) noexcept;
  DenseGraph& operator=(DenseGraph&& other) noexcept;
  ~DenseGraph();

  // 构建接口
  void initialize(size_t num_nodes, size_t max_degree);
  void set_neighbors(size_t node_id, const node_t* neighbors, size_t count);
  void add_edge(size_t from, size_t to);
  void remove_edge(size_t from, size_t to);

  // 实现基类接口
  size_t num_nodes() const { return num_nodes_; }
  size_t max_degree() const { return max_degree_; }
  const node_t* neighbors(size_t node_id) const;
  size_t degree(size_t node_id) const;
  void prefetch_neighbors(size_t node_id, int lines = 1) const;
  const std::vector<size_t>& entry_points() const { return entry_points_; }
  void save(const std::string& filename) const;
  void load(const std::string& filename);
  GraphMetadata metadata() const;

  // 兼容性接口（保持向后兼容）
  node_t at(size_t i, size_t j) const { return data_[i * max_degree_ + j]; }
  node_t& at(size_t i, size_t j) { return data_[i * max_degree_ + j]; }
  const node_t* edges(size_t u) const { return neighbors(u); }
  node_t* edges(size_t u) { return data_ + max_degree_ * u; }

  // 搜索初始化
  template <typename Pool, typename Quant>
  void initialize_search(Pool& pool, const Quant& quant) const {
    if (initializer_) {
      initializer_->initialize(pool, quant);
    } else {
      for (auto ep : entry_points_) {
        pool.insert(ep, quant->compute_query_distance(ep));
      }
    }
  }

  // 设置初始化器和入口点
  void set_initializer(std::unique_ptr<algorithm::HnswWrapper> init) {
    initializer_ = std::move(init);
  }
  void set_entry_points(const std::vector<size_t>& eps) { entry_points_ = eps; }

 private:
  void cleanup();
  void copy_from(const DenseGraph& other);

  size_t num_nodes_ = 0;
  size_t max_degree_ = 0;
  node_t* data_ = nullptr;
  std::vector<size_t> degrees_;  // 每个节点的实际度数
  std::vector<size_t> entry_points_;
  std::unique_ptr<algorithm::HnswWrapper> initializer_;
  GraphMetadata metadata_;
};

}  // namespace index
}  // namespace deepsearch
