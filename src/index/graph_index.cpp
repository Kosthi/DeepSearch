#include "graph_index.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "algorithm/hnsw_wrapper.h"
#include "prefetch.h"

namespace deepsearch {
namespace index {

// 构造函数
template <typename node_t>
DenseGraph<node_t>::DenseGraph(size_t num_nodes, size_t max_degree)
    : num_nodes_(num_nodes), max_degree_(max_degree) {
  initialize(num_nodes, max_degree);
}

// 拷贝构造函数
template <typename node_t>
DenseGraph<node_t>::DenseGraph(const DenseGraph& other) {
  copy_from(other);
}

// 拷贝赋值运算符
template <typename node_t>
DenseGraph<node_t>& DenseGraph<node_t>::operator=(const DenseGraph& other) {
  if (this != &other) {
    cleanup();
    copy_from(other);
  }
  return *this;
}

// 移动构造函数
template <typename node_t>
DenseGraph<node_t>::DenseGraph(DenseGraph&& other) noexcept
    : num_nodes_(other.num_nodes_),
      max_degree_(other.max_degree_),
      data_(other.data_),
      degrees_(std::move(other.degrees_)),
      entry_points_(std::move(other.entry_points_)),
      initializer_(std::move(other.initializer_)),
      metadata_(std::move(other.metadata_)) {
  other.num_nodes_ = 0;
  other.max_degree_ = 0;
  other.data_ = nullptr;
}

// 移动赋值运算符
template <typename node_t>
DenseGraph<node_t>& DenseGraph<node_t>::operator=(DenseGraph&& other) noexcept {
  if (this != &other) {
    cleanup();

    num_nodes_ = other.num_nodes_;
    max_degree_ = other.max_degree_;
    data_ = other.data_;
    degrees_ = std::move(other.degrees_);
    entry_points_ = std::move(other.entry_points_);
    initializer_ = std::move(other.initializer_);
    metadata_ = std::move(other.metadata_);

    other.num_nodes_ = 0;
    other.max_degree_ = 0;
    other.data_ = nullptr;
  }
  return *this;
}

// 析构函数
template <typename node_t>
DenseGraph<node_t>::~DenseGraph() {
  cleanup();
}

// 初始化图
template <typename node_t>
void DenseGraph<node_t>::initialize(size_t num_nodes, size_t max_degree) {
  cleanup();

  num_nodes_ = num_nodes;
  max_degree_ = max_degree;

  if (num_nodes > 0 && max_degree > 0) {
    size_t total_size = num_nodes * max_degree;
    data_ = static_cast<node_t*>(
        std::aligned_alloc(64, total_size * sizeof(node_t)));
    if (!data_) {
      throw std::bad_alloc();
    }

    // 初始化为 EMPTY_ID
    std::fill(data_, data_ + total_size, static_cast<node_t>(EMPTY_ID));

    degrees_.resize(num_nodes, 0);
  }

  // 初始化元数据
  metadata_.num_nodes = num_nodes;
  metadata_.max_degree = max_degree;
  metadata_.total_edges = 0;
}

// 设置邻居
template <typename node_t>
void DenseGraph<node_t>::set_neighbors(size_t node_id, const node_t* neighbors,
                                       size_t count) {
  if (node_id >= num_nodes_) {
    throw std::out_of_range("Node ID out of range");
  }
  if (count > max_degree_) {
    throw std::invalid_argument("Too many neighbors");
  }

  node_t* node_neighbors = data_ + node_id * max_degree_;

  // 复制邻居
  std::memcpy(node_neighbors, neighbors, count * sizeof(node_t));

  // 填充剩余位置为 EMPTY_ID
  for (size_t i = count; i < max_degree_; ++i) {
    node_neighbors[i] = static_cast<node_t>(EMPTY_ID);
  }

  degrees_[node_id] = count;
}

// 添加边
template <typename node_t>
void DenseGraph<node_t>::add_edge(size_t from, size_t to) {
  if (from >= num_nodes_ || to >= num_nodes_) {
    throw std::out_of_range("Node ID out of range");
  }

  node_t* neighbors = data_ + from * max_degree_;
  size_t& degree = degrees_[from];

  // 检查是否已存在
  for (size_t i = 0; i < degree; ++i) {
    if (neighbors[i] == static_cast<node_t>(to)) {
      return;  // 边已存在
    }
  }

  // 添加新边
  if (degree < max_degree_) {
    neighbors[degree] = static_cast<node_t>(to);
    degree++;
    metadata_.total_edges++;
  }
}

// 移除边
template <typename node_t>
void DenseGraph<node_t>::remove_edge(size_t from, size_t to) {
  if (from >= num_nodes_ || to >= num_nodes_) {
    throw std::out_of_range("Node ID out of range");
  }

  node_t* neighbors = data_ + from * max_degree_;
  size_t& degree = degrees_[from];

  for (size_t i = 0; i < degree; ++i) {
    if (neighbors[i] == static_cast<node_t>(to)) {
      // 移动后续元素
      for (size_t j = i; j < degree - 1; ++j) {
        neighbors[j] = neighbors[j + 1];
      }
      neighbors[degree - 1] = static_cast<node_t>(EMPTY_ID);
      degree--;
      metadata_.total_edges--;
      break;
    }
  }
}

// 获取邻居
template <typename node_t>
const node_t* DenseGraph<node_t>::neighbors(size_t node_id) const {
  if (node_id >= num_nodes_) {
    throw std::out_of_range("Node ID out of range");
  }
  return data_ + node_id * max_degree_;
}

// 获取度数
template <typename node_t>
size_t DenseGraph<node_t>::degree(size_t node_id) const {
  if (node_id >= num_nodes_) {
    throw std::out_of_range("Node ID out of range");
  }
  return degrees_[node_id];
}

// 预取邻居
template <typename node_t>
void DenseGraph<node_t>::prefetch_neighbors(size_t node_id, int lines) const {
  auto nei = data_ + node_id * max_degree_;
  mem_prefetch<prefetch_L1>(reinterpret_cast<char*>(nei), lines);
}

// 保存图
template <typename node_t>
void DenseGraph<node_t>::save(const std::string& filename) const {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  if (initializer_) {
    initializer_->save(file);
  }

  // 写入基本信息
  file.write(reinterpret_cast<const char*>(&num_nodes_), sizeof(num_nodes_));
  file.write(reinterpret_cast<const char*>(&max_degree_), sizeof(max_degree_));

  // 写入图数据
  if (data_) {
    size_t total_size = num_nodes_ * max_degree_;
    file.write(reinterpret_cast<const char*>(data_),
               total_size * sizeof(node_t));
  }

  // 写入度数
  size_t degrees_size = degrees_.size();
  file.write(reinterpret_cast<const char*>(&degrees_size),
             sizeof(degrees_size));
  if (degrees_size > 0) {
    file.write(reinterpret_cast<const char*>(degrees_.data()),
               degrees_size * sizeof(size_t));
  }

  // 写入入口点
  size_t entry_points_size = entry_points_.size();
  file.write(reinterpret_cast<const char*>(&entry_points_size),
             sizeof(entry_points_size));
  if (entry_points_size > 0) {
    file.write(reinterpret_cast<const char*>(entry_points_.data()),
               entry_points_size * sizeof(size_t));
  }

  // 写入元数据
  file.write(reinterpret_cast<const char*>(&metadata_.total_edges),
             sizeof(metadata_.total_edges));

  size_t builder_name_size = metadata_.builder_name.size();
  file.write(reinterpret_cast<const char*>(&builder_name_size),
             sizeof(builder_name_size));
  if (builder_name_size > 0) {
    file.write(metadata_.builder_name.c_str(), builder_name_size);
  }

  size_t distance_type_size = metadata_.distance_type.size();
  file.write(reinterpret_cast<const char*>(&distance_type_size),
             sizeof(distance_type_size));
  if (distance_type_size > 0) {
    file.write(metadata_.distance_type.c_str(), distance_type_size);
  }
}

// 加载图
template <typename node_t>
void DenseGraph<node_t>::load(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  cleanup();

  initializer_ = std::make_unique<algorithm::HnswWrapper>(1000000, 16);
  initializer_->load(file);

  // 读取基本信息
  file.read(reinterpret_cast<char*>(&num_nodes_), sizeof(num_nodes_));
  file.read(reinterpret_cast<char*>(&max_degree_), sizeof(max_degree_));

  // 分配内存并读取图数据
  if (num_nodes_ > 0 && max_degree_ > 0) {
    size_t total_size = num_nodes_ * max_degree_;
    data_ = static_cast<node_t*>(
        std::aligned_alloc(64, total_size * sizeof(node_t)));
    if (!data_) {
      throw std::bad_alloc();
    }
    file.read(reinterpret_cast<char*>(data_), total_size * sizeof(node_t));
  }

  // 读取度数
  size_t degrees_size;
  file.read(reinterpret_cast<char*>(&degrees_size), sizeof(degrees_size));
  degrees_.resize(degrees_size);
  if (degrees_size > 0) {
    file.read(reinterpret_cast<char*>(degrees_.data()),
              degrees_size * sizeof(size_t));
  }

  // 读取入口点
  size_t entry_points_size;
  file.read(reinterpret_cast<char*>(&entry_points_size),
            sizeof(entry_points_size));
  entry_points_.resize(entry_points_size);
  if (entry_points_size > 0) {
    file.read(reinterpret_cast<char*>(entry_points_.data()),
              entry_points_size * sizeof(size_t));
  }

  // 读取元数据
  file.read(reinterpret_cast<char*>(&metadata_.total_edges),
            sizeof(metadata_.total_edges));
  metadata_.num_nodes = num_nodes_;
  metadata_.max_degree = max_degree_;

  size_t builder_name_size;
  file.read(reinterpret_cast<char*>(&builder_name_size),
            sizeof(builder_name_size));
  if (builder_name_size > 0) {
    metadata_.builder_name.resize(builder_name_size);
    file.read(&metadata_.builder_name[0], builder_name_size);
  }

  size_t distance_type_size;
  file.read(reinterpret_cast<char*>(&distance_type_size),
            sizeof(distance_type_size));
  if (distance_type_size > 0) {
    metadata_.distance_type.resize(distance_type_size);
    file.read(&metadata_.distance_type[0], distance_type_size);
  }
}

// 获取元数据
template <typename node_t>
GraphMetadata DenseGraph<node_t>::metadata() const {
  GraphMetadata meta = metadata_;
  meta.num_nodes = num_nodes_;
  meta.max_degree = max_degree_;
  meta.entry_points.clear();
  meta.entry_points.insert(meta.entry_points.end(), entry_points_.begin(),
                           entry_points_.end());
  return meta;
}

// 清理资源
template <typename node_t>
void DenseGraph<node_t>::cleanup() {
  if (data_) {
    std::free(data_);
    data_ = nullptr;
  }
  num_nodes_ = 0;
  max_degree_ = 0;
  degrees_.clear();
  entry_points_.clear();
  initializer_.reset();
}

// 拷贝数据
template <typename node_t>
void DenseGraph<node_t>::copy_from(const DenseGraph& other) {
  num_nodes_ = other.num_nodes_;
  max_degree_ = other.max_degree_;
  degrees_ = other.degrees_;
  entry_points_ = other.entry_points_;
  metadata_ = other.metadata_;

  if (other.data_ && num_nodes_ > 0 && max_degree_ > 0) {
    size_t total_size = num_nodes_ * max_degree_;
    data_ = static_cast<node_t*>(
        std::aligned_alloc(64, total_size * sizeof(node_t)));
    if (!data_) {
      throw std::bad_alloc();
    }
    std::memcpy(data_, other.data_, total_size * sizeof(node_t));
  } else {
    data_ = nullptr;
  }

  if (other.initializer_) {
    // 注意：这里需要根据 HnswInitializer 的具体实现来决定如何拷贝
    // 暂时设为 nullptr，可能需要实现 clone 方法
    initializer_ =
        std::make_unique<algorithm::HnswWrapper>(*other.initializer_);
  }
}

// 显式实例化常用类型
template class DenseGraph<int>;
template class DenseGraph<uint32_t>;

}  // namespace index
}  // namespace deepsearch
