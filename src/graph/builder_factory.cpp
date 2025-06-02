#include "builder_factory.h"

#include <stdexcept>

#include "core/exceptions.h"

namespace deepsearch {
namespace graph {

// 静态成员初始化
template <typename T>
std::unordered_map<std::string, BuilderType> BuilderFactory<T>::type_map_ = {
    {"hnsw", BuilderType::HNSW},
    {"bruteforce", BuilderType::BRUTEFORCE},
    {"random", BuilderType::RANDOM}};

template <typename T>
std::unique_ptr<GraphBuilder<T>> BuilderFactory<T>::create(
    BuilderType type, core::DistanceType distance_type, size_t dimension,
    const BuilderConfig& config) {
  switch (type) {
    case BuilderType::HNSW: {
      auto builder = std::make_unique<HNSWBuilder>(distance_type, dimension);
      builder->configure(config);
      return std::move(builder);
    }
    case BuilderType::BRUTEFORCE:
      // TODO: 实现暴力搜索构建器
      throw std::runtime_error("BruteForce builder not implemented yet");
    case BuilderType::RANDOM:
      // TODO: 实现随机图构建器
      throw std::runtime_error("Random builder not implemented yet");
    default:
      throw std::invalid_argument("Unknown builder type");
  }
}

template <typename T>
std::vector<BuilderType> BuilderFactory<T>::supported_types() {
  return {BuilderType::HNSW};  // 目前只支持HNSW
}

template <typename T>
std::string BuilderFactory<T>::type_name(BuilderType type) {
  switch (type) {
    case BuilderType::HNSW:
      return "hnsw";
    case BuilderType::BRUTEFORCE:
      return "bruteforce";
    case BuilderType::RANDOM:
      return "random";
    default:
      return "unknown";
  }
}

template <typename T>
BuilderType BuilderFactory<T>::parse_type(const std::string& name) {
  auto it = type_map_.find(name);
  if (it != type_map_.end()) {
    return it->second;
  }
  throw std::invalid_argument("Unknown builder type: " + name);
}

// 显式实例化常用类型
template class BuilderFactory<float>;
// template class BuilderFactory<double>;
// template class BuilderFactory<int>;

}  // namespace graph
}  // namespace deepsearch
