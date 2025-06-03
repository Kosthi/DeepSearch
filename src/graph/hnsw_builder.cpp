#include <chrono>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "builder.h"
#include "common.h"
#include "core/logger.h"
#include "graph/builder.h"
#include "hnswlib/hnswlib.h"

namespace deepsearch {
namespace graph {

HNSWBuilder::HNSWBuilder(core::DistanceType distance_type, size_t dim)
    : distance_type_(distance_type), dim_(dim) {
  // 根据距离类型创建空间
  switch (distance_type_) {
    case core::DistanceType::L2:
      space_ = std::make_unique<hnswlib::L2Space>(dim);
      break;
    case core::DistanceType::IP:
      space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
      break;
    default:
      throw std::invalid_argument("Unsupported distance type");
  }
}

void HNSWBuilder::configure(const BuilderConfig& config) {
  config_ = config;
  initialize_hnsw();
}

void HNSWBuilder::initialize_hnsw() {
  if (config_.max_elements > 0) {
    hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(), config_.max_elements, config_.M, config_.ef_construction,
        config_.random_seed);
  }
}

Graph HNSWBuilder::build(const float* data, size_t n, size_t dim) {
  auto logger = core::LogManager::instance().get_logger("hnsw_builder");

  if (dim != dim_) {
    logger->error("Dimension mismatch: expected {}, got {}", dim_, dim);
    throw std::invalid_argument("Dimension mismatch");
  }

  // 重新配置 HNSW 以适应数据大小
  if (n > config_.max_elements) {
    logger->warn("Data size {} exceeds max_elements {}, reconfiguring", n,
                 config_.max_elements);
    config_.max_elements = n;
    initialize_hnsw();
  }

  // 设置 OpenMP 线程数（可配置）
  int num_threads = std::min(static_cast<int>(omp_get_max_threads()),
                             static_cast<int>(std::max(1UL, n / 1000)));
  omp_set_num_threads(num_threads);
  logger->info("Building HNSW index with {} threads for {} points", num_threads,
               n);

  std::atomic<int> cnt{0};
  auto start = std::chrono::high_resolution_clock::now();

  // 添加第一个点
  if (n > 0) {
    hnsw_->addPoint(data, 0);
    current_size_ = 1;
    cnt++;
  }

  // TODO: 多线程环境下内外部ID可能不一一对应，而搜索器返回的是内部id
  // see: https://github.com/zilliztech/pyglass/pull/11/files
  // #pragma omp parallel for schedule(dynamic)
  for (int i = 1; i < n; ++i) {
    hnsw_->addPoint(data + i * dim, i);
    int cur = cnt += 1;
    if (cur % 10000 == 0) {
      logger->info("HNSW building progress: [{}/{}] ({:.1f}%)", cur, n,
                   100.0 * cur / n);
    }
  }

  current_size_ = cnt;
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();

  logger->info(
      "HNSW index built successfully: {} points in {:.2f}s ({:.0f} points/sec)",
      n, duration, n / duration);

  return extract_graph();
}

Graph HNSWBuilder::extract_graph() {
  if (!hnsw_ || current_size_ == 0) {
    return Graph();
  }

  // 创建图结构
  Graph graph(current_size_, 2 * config_.M);

  // 提取0层连接
  for (size_t i = 0; i < current_size_; ++i) {
    int* edges = reinterpret_cast<int*>(hnsw_->get_linklist0(i));
    size_t degree = std::min(static_cast<size_t>(edges[0]), 2 * config_.M);

    std::vector<int> neighbors;
    for (size_t j = 1; j <= degree; ++j) {
      neighbors.push_back(edges[j]);
    }

    graph.set_neighbors(i, neighbors.data(), neighbors.size());
  }

  // 创建并设置初始化器
  auto initializer =
      std::make_unique<HnswInitializer>(current_size_, config_.M);
  initializer->ep = hnsw_->enterpoint_node_;

  for (size_t i = 0; i < current_size_; ++i) {
    int level = hnsw_->element_levels_[i];
    initializer->levels[i] = level;

    if (level > 0) {
      initializer->lists[i].assign(level * config_.M, -1);
      for (int j = 1; j <= level; ++j) {
        int* edges = reinterpret_cast<int*>(hnsw_->get_linklist(i, j));
        for (int k = 1; k <= edges[0]; ++k) {
          initializer->at(j, i, k - 1) = edges[k];
        }
      }
    }
  }

  graph.set_initializer(std::move(initializer));
  graph.set_entry_points({hnsw_->enterpoint_node_});

  return graph;
}

void HNSWBuilder::add_points(const float* data, const size_t* labels,
                             size_t n) {
  if (!hnsw_) {
    throw std::runtime_error("HNSW not initialized");
  }

  for (size_t i = 0; i < n; ++i) {
    hnsw_->addPoint(data + i * dim_, labels[i]);
    current_size_++;
  }
}

void HNSWBuilder::remove_points(const size_t* labels, size_t n) {
  if (!hnsw_) {
    throw std::runtime_error("HNSW not initialized");
  }

  for (size_t i = 0; i < n; ++i) {
    hnsw_->markDelete(labels[i]);
  }
}

}  // namespace graph
}  // namespace deepsearch
