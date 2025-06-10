#pragma once

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deepsearch {
namespace core {

// 距离类型枚举
// enum class DistanceType {
//   L2,
//   IP,  // Inner Product
//   COSINE,
//   HAMMING
// };

// 搜索结果结构
struct SearchResult {
  size_t label;    // 向量标签/ID
  float distance;  // 距离值

  SearchResult() : label(0), distance(0.0f) {}
  SearchResult(size_t l, float d) : label(l), distance(d) {}
};

// 搜索参数结构
struct SearchParams {
  size_t ef = 50;                                          // HNSW搜索参数
  size_t num_threads = 1;                                  // 线程数
  bool use_prefetch = true;                                // 是否使用预取
  size_t batch_size = 1000;                                // 批处理大小
  std::unordered_map<std::string, std::any> extra_params;  // 额外参数
};

// 模板化的搜索引擎接口
template <typename T>
class SearchEngineTemplate {
 public:
  virtual ~SearchEngineTemplate() = default;

  // 构建索引
  virtual void build(const T* data, size_t n, size_t dim) = 0;

  // 搜索最近邻
  virtual std::vector<int> search(const T* query, size_t k) const = 0;

  // 批量搜索
  virtual std::vector<std::vector<int>> batch_search(const T* queries,
                                                     size_t nq, size_t dim,
                                                     size_t k) const {
    std::vector<std::vector<int>> results(nq);
    for (size_t i = 0; i < nq; ++i) {
      results[i] = search(queries + i * dim, k);
    }
    return results;
  }

  // 保存和加载索引
  virtual void save(const std::string& path) const = 0;
  virtual void load(const std::string& path) = 0;

  // 获取索引信息
  virtual size_t size() const = 0;
  virtual size_t dimension() const = 0;
};

// 模板化的距离计算接口
template <typename T>
class DistanceComputerTemplate {
 public:
  virtual ~DistanceComputerTemplate() = default;
  virtual float compute(const T* a, const T* b) const = 0;
  virtual std::string name() const = 0;
};

// 量化器接口
template <typename InputType, typename CodeType>
class Quantizer {
 public:
  virtual ~Quantizer() = default;
  virtual void train(const InputType* data, size_t n, size_t dim) = 0;
  virtual void encode(const InputType* input, CodeType* output) const = 0;
  virtual void decode(const CodeType* input, InputType* output) const = 0;
  virtual size_t code_size() const = 0;
  virtual size_t dimension() const = 0;
};

// 图存储接口
template <typename NodeType>
class GraphStorage {
 public:
  virtual ~GraphStorage() = default;
  virtual void init(size_t n, size_t k) = 0;
  virtual const NodeType* edges(size_t u) const = 0;
  virtual NodeType* edges(size_t u) = 0;
  virtual NodeType at(size_t i, size_t j) const = 0;
  virtual NodeType& at(size_t i, size_t j) = 0;
  virtual void save(const std::string& path) const = 0;
  virtual void load(const std::string& path) = 0;
  virtual size_t num_nodes() const = 0;
  virtual size_t max_degree() const = 0;
};

}  // namespace core
}  // namespace deepsearch
