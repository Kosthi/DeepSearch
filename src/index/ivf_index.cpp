#include "ivf_index.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <random>

#include "distance/computers.h"
#include "quantization/quantizer.h"

namespace deepsearch {
namespace index {

template <typename T>
IVFIndex<T>::IVFIndex(DistanceType distance_type, size_t dimension,
                      QuantizerType quantizer,
                      const builder::BuilderConfig& config)
    : dim_(dimension),
      n_total_(0),
      config_(config.partition_builder_config.ivf_builder_config),
      distance_type_(distance_type) {
  // 初始化距离计算器
  distance_computer_ =
      distance::DistanceComputerFactory::create<T>(distance_type, dimension);

  // // 初始化内部量化器
  // if (config_.quantizer_type == "sq8") {
  //   quantizer_ = quantization::QuantizerFactory<T, uint8_t>::create(
  //       quantization::QuantizerType::SQ8, distance_type, dimension);
  // } else if (config_.quantizer_type == "sq4") {
  //   quantizer_ = quantization::QuantizerFactory<T, uint8_t>::create(
  //       quantization::QuantizerType::SQ4, distance_type, dimension);
  // } else {
  //   // 默认使用FP32（实际上不压缩）
  //   quantizer_ = quantization::QuantizerFactory<T, uint8_t>::create(
  //       quantization::QuantizerType::FP32, distance_type, dimension);
  //   ;
  // }

  centroids_.resize(config_.nlist * dim_);
  inverted_lists_.resize(config_.nlist);
}

template <typename T>
void IVFIndex<T>::build(const float* data, size_t n, size_t dim) {
  if (dim != dim_) {
    throw std::invalid_argument("Dimension mismatch");
  }

  n_total_ = n;

  // 训练聚类中心
  train_centroids(data, n);

  // 训练内部量化器
  if (quantizer_) {
    quantizer_->train(data, n, dim);
  }

  // 构建倒排列表
  build_inverted_lists(data, n);
}

template <typename T>
void IVFIndex<T>::train_centroids(const T* data, size_t n) {
  std::mt19937 rng(config_.random_seed);
  std::uniform_int_distribution<size_t> dist(0, n - 1);

  // K-means++ 初始化
  std::vector<bool> selected(n, false);

  // 选择第一个中心
  size_t first_idx = dist(rng);
  std::memcpy(centroids_.data(), data + first_idx * dim_, dim_ * sizeof(T));
  selected[first_idx] = true;

  // 选择剩余中心
  for (size_t c = 1; c < config_.nlist; ++c) {
    std::vector<float> distances(n);
    float total_dist = 0.0f;

    // 计算每个点到最近中心的距离
    for (size_t i = 0; i < n; ++i) {
      if (selected[i]) {
        distances[i] = 0.0f;
        continue;
      }

      float min_dist = std::numeric_limits<float>::max();
      for (size_t j = 0; j < c; ++j) {
        float dist = distance_computer_->compute(data + i * dim_,
                                                 centroids_.data() + j * dim_);
        min_dist = std::min(min_dist, dist);
      }
      distances[i] = min_dist;
      total_dist += min_dist;
    }

    // 按概率选择下一个中心
    std::uniform_real_distribution<float> prob_dist(0.0f, total_dist);
    float target = prob_dist(rng);
    float cumsum = 0.0f;

    for (size_t i = 0; i < n; ++i) {
      if (selected[i]) continue;
      cumsum += distances[i];
      if (cumsum >= target) {
        std::memcpy(centroids_.data() + c * dim_, data + i * dim_,
                    dim_ * sizeof(T));
        selected[i] = true;
        break;
      }
    }
  }

  // K-means迭代优化
  std::vector<size_t> assignments(n);
  std::vector<size_t> cluster_counts(config_.nlist);
  std::vector<T> new_centroids(config_.nlist * dim_, 0);

  for (size_t iter = 0; iter < config_.max_iter; ++iter) {
    // 分配步骤
    std::fill(cluster_counts.begin(), cluster_counts.end(), 0);
    std::fill(new_centroids.begin(), new_centroids.end(), T(0));

    for (size_t i = 0; i < n; ++i) {
      assignments[i] = assign_to_cluster(data + i * dim_);
      cluster_counts[assignments[i]]++;

      // 累加到新中心
      const T* vector = data + i * dim_;
      T* centroid = new_centroids.data() + assignments[i] * dim_;
      for (size_t j = 0; j < dim_; ++j) {
        centroid[j] += vector[j];
      }
    }

    // 更新步骤
    float max_change = 0.0f;
    for (size_t i = 0; i < config_.nlist; ++i) {
      if (cluster_counts[i] > 0) {
        T* old_centroid = centroids_.data() + i * dim_;
        T* new_centroid = new_centroids.data() + i * dim_;

        // 计算平均值
        for (size_t j = 0; j < dim_; ++j) {
          new_centroid[j] /= static_cast<T>(cluster_counts[i]);
        }

        // 计算变化量
        float change = distance_computer_->compute(old_centroid, new_centroid);
        max_change = std::max(max_change, change);

        // 更新中心
        std::memcpy(old_centroid, new_centroid, dim_ * sizeof(T));
      }
    }

    // 检查收敛
    if (max_change < config_.tolerance) {
      break;
    }
  }
}

template <typename T>
void IVFIndex<T>::build_inverted_lists(const T* data, size_t n) {
  // 清空现有列表
  for (auto& list : inverted_lists_) {
    list.clear();
  }
  id_to_location_.clear();

  // 分配向量到聚类
  for (size_t i = 0; i < n; ++i) {
    size_t cluster_id = assign_to_cluster(data + i * dim_);

    // 编码向量
    std::vector<uint8_t> code;
    encode_vector(data + i * dim_, cluster_id, code);

    // 添加到倒排列表
    size_t list_index = inverted_lists_[cluster_id].size();
    inverted_lists_[cluster_id].emplace_back(i, code);

    // 更新ID映射
    id_to_location_[i] = {cluster_id, list_index};
  }
}

template <typename T>
size_t IVFIndex<T>::assign_to_cluster(const T* vector) const {
  float min_dist = std::numeric_limits<float>::max();
  size_t best_cluster = 0;

  for (size_t i = 0; i < config_.nlist; ++i) {
    const T* centroid = centroids_.data() + i * dim_;
    float dist = distance_computer_->compute(vector, centroid);
    if (dist < min_dist) {
      min_dist = dist;
      best_cluster = i;
    }
  }

  return best_cluster;
}

template <typename T>
std::vector<size_t> IVFIndex<T>::search_clusters(const T* query,
                                                 size_t nprobe) const {
  if (nprobe == 0) nprobe = config_.nprobe;
  nprobe = std::min(nprobe, config_.nlist);

  std::vector<std::pair<float, size_t>> distances;
  distances.reserve(config_.nlist);

  for (size_t i = 0; i < config_.nlist; ++i) {
    const T* centroid = centroids_.data() + i * dim_;
    float dist = distance_computer_->compute(query, centroid);
    distances.emplace_back(dist, i);
  }

  // 部分排序，只需要前nprobe个
  std::partial_sort(distances.begin(), distances.begin() + nprobe,
                    distances.end());

  std::vector<size_t> result;
  result.reserve(nprobe);
  for (size_t i = 0; i < nprobe; ++i) {
    result.push_back(distances[i].second);
  }

  return result;
}

template <typename T>
void IVFIndex<T>::search(const float* query, int k, int* dst,
                         size_t nprobe) const {
  // 搜索相关聚类
  auto cluster_ids = search_clusters(query, nprobe);

  // 在选中的聚类中搜索
  std::priority_queue<std::pair<float, size_t>> candidates;

  for (size_t cluster_id : cluster_ids) {
    const auto& inverted_list = inverted_lists_[cluster_id];
    const T* centroid = centroids_.data() + cluster_id * dim_;

    for (const auto& item : inverted_list) {
      float dist;

      if (quantizer_) {
        // 解码向量并计算距离
        std::vector<T> decoded(dim_);
        decode_vector(item.code, cluster_id, decoded.data());
        dist = distance_computer_->compute(query, decoded.data());
      } else {
        // 直接计算距离（假设code存储的是原始向量）
        const T* vector = reinterpret_cast<const T*>(item.code.data());
        dist = distance_computer_->compute(query, vector);
      }

      if (candidates.size() < k) {
        candidates.emplace(dist, item.id);
      } else if (dist < candidates.top().first) {
        candidates.pop();
        candidates.emplace(dist, item.id);
      }
    }
  }

  // 提取结果
  std::vector<int> results;
  results.reserve(candidates.size());

  while (!candidates.empty()) {
    results.push_back(static_cast<int>(candidates.top().second));
    candidates.pop();
  }

  std::reverse(results.begin(), results.end());
}

template <typename T>
void IVFIndex<T>::encode_vector(const T* vector, size_t cluster_id,
                                std::vector<uint8_t>& code) const {
  if (quantizer_) {
    if (config_.use_residual) {
      // 计算残差
      std::vector<T> residual(dim_);
      const T* centroid = centroids_.data() + cluster_id * dim_;
      for (size_t i = 0; i < dim_; ++i) {
        residual[i] = vector[i] - centroid[i];
      }

      // 量化残差
      code.resize(quantizer_->code_size());
      quantizer_->encode(residual.data(), code.data());
    } else {
      // 直接量化原始向量
      code.resize(quantizer_->code_size());
      quantizer_->encode(vector, code.data());
    }
  } else {
    // 不压缩，直接存储
    code.resize(dim_ * sizeof(T));
    std::memcpy(code.data(), vector, dim_ * sizeof(T));
  }
}

template <typename T>
void IVFIndex<T>::decode_vector(const std::vector<uint8_t>& code,
                                size_t cluster_id, T* vector) const {
  if (quantizer_) {
    if (config_.use_residual) {
      // 解码残差
      std::vector<T> residual(dim_);
      quantizer_->decode(code.data(), residual.data());

      // 加上聚类中心
      const T* centroid = centroids_.data() + cluster_id * dim_;
      for (size_t i = 0; i < dim_; ++i) {
        vector[i] = residual[i] + centroid[i];
      }
    } else {
      // 直接解码
      quantizer_->decode(code.data(), vector);
    }
  } else {
    // 直接复制
    std::memcpy(vector, code.data(), dim_ * sizeof(T));
  }
}

// 显式实例化
template class IVFIndex<float>;

}  // namespace index
}  // namespace deepsearch
