#pragma once

#include <vector>

namespace deepsearch {
namespace index {

constexpr int EMPTY_ID = -1;

// 图元数据
struct GraphMetadata {
  size_t num_nodes = 0;
  size_t max_degree = 0;
  size_t total_edges = 0;
  std::string builder_name;
  std::string distance_type;
  std::vector<int> entry_points;
};

// 抽象索引基类
template <typename T>
class IndexBase {
 public:
  virtual ~IndexBase() = default;

  // 核心接口
  virtual void build(const float* data, size_t n, size_t dim) = 0;
  virtual void search(const float* query, int k, int* dst) const = 0;
  // virtual std::vector<std::vector<int>> batch_search(const T* queries,
  //                                                    size_t nq, size_t dim,
  //                                                    size_t k) const = 0;

  // 索引管理
  virtual void save(const std::string& path) const = 0;
  virtual void load(const std::string& path) = 0;

  // 信息接口
  virtual size_t size() const = 0;
  virtual size_t dimension() const = 0;
  virtual std::string name() const = 0;
  virtual std::string type() const = 0;  // "graph" 或 "partition"
};

// 图索引基类
template <typename node_t>
class GraphBasedIndex : public IndexBase<node_t> {
 public:
  virtual ~GraphBasedIndex() = default;

  [[nodiscard]] std::string type() const override { return "graph"; }

  virtual void SetData(const float* data, int n, int dim) = 0;
  // 图特有接口
  virtual void add_points(const node_t* data, const size_t* labels,
                          size_t n) = 0;
  virtual void remove_points(const size_t* labels, size_t n) = 0;
  // 基本访问
  // virtual size_t num_nodes() const = 0;
  virtual size_t max_degree() const = 0;
  virtual const node_t* neighbors(size_t node_id) const = 0;
  virtual size_t degree(size_t node_id) const = 0;

  // 搜索支持
  virtual void prefetch_neighbors(size_t node_id, int lines = 1) const = 0;
  virtual const std::vector<size_t>& entry_points() const = 0;

  // 序列化
  virtual void save(const std::string& filename) const = 0;
  virtual void load(const std::string& filename) = 0;

  // 元数据
  // virtual GraphMetadata metadata() const = 0;
  virtual void set_ef(int ef) = 0;
  // virtual int get_ef() const = 0;
};

// 划分索引基类
template <typename T>
class PartitionBasedIndex : public IndexBase<T> {
 public:
  virtual ~PartitionBasedIndex() = default;

  std::string type() const override { return "partition"; }

  virtual void search(const float* query, int k, int* dst,
                      size_t nprobe = 0) const = 0;

  // virtual std::vector<std::vector<int>> batch_search(
  //     const T* queries, size_t nq, size_t dim, size_t k,
  //     size_t nprobe = 1) const = 0 {
  //   return {{}};
  // }

  // 基类搜索接口的默认实现
  void search(const float* q, int k, int* dst) const override {
    search(q, k, dst, get_nprobe());
  }

  // std::vector<std::vector<int>> batch_search(const T* queries, size_t nq,
  //                                            size_t dim,
  //                                            size_t k) const override {
  //   return batch_search(queries, nq, dim, k, get_nprobe());
  // }

  // 划分特有接口
  virtual void set_nprobe(size_t nprobe) = 0;
  virtual size_t get_nprobe() const = 0;
  virtual size_t get_nlist() const = 0;
};

}  // namespace index
}  // namespace deepsearch
