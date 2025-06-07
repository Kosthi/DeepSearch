#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "core/interfaces.h"
#include "distance/computers.h"
#include "graph/graph.h"
#include "neighbor.h"
#include "quantization/fp32_quant.h"
#include "quantization/sq4_quant.h"
#include "quantization/sq8_quant.h"
#include "searcher/searcher.h"
#include "simd/distance_functions.h"

using namespace deepsearch;
using namespace Catch;

class PrecisionTestFixture {
 public:
  static constexpr size_t NUM_POINTS = 10;
  static constexpr size_t DIM = 128;
  static constexpr int K = 5;

  PrecisionTestFixture() { generateTestData(); }

 private:
  void generateTestData() {
    std::mt19937 rng(42);  // 固定种子确保可重现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    data_.resize(NUM_POINTS * DIM);
    normalized_data_.resize(NUM_POINTS * DIM);

    // 生成随机数据
    for (size_t i = 0; i < NUM_POINTS * DIM; ++i) {
      data_[i] = dist(rng);
    }

    // 创建归一化版本（对IP距离很重要）
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      float norm = 0;
      for (size_t j = 0; j < DIM; ++j) {
        float val = data_[i * DIM + j];
        norm += val * val;
      }
      norm = std::sqrt(norm);

      for (size_t j = 0; j < DIM; ++j) {
        normalized_data_[i * DIM + j] = data_[i * DIM + j] / norm;
      }
    }
  }

 protected:
  std::vector<float> data_;
  std::vector<float> normalized_data_;
};

TEST_CASE_METHOD(PrecisionTestFixture, "Distance Computer Consistency Test",
                 "[precision][distance]") {
  SECTION("L2 Distance Consistency") {
    auto distance_computer = distance::DistanceComputerFactory::create<float>(
        core::DistanceType::L2, DIM);

    INFO("Testing L2 distance computer: " << distance_computer->name());

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i + 1; j < 3; ++j) {
        float computed_dist = distance_computer->compute(
            data_.data() + i * DIM, data_.data() + j * DIM);

        // 手动计算L2距离
        float manual_dist = 0;
        for (size_t d = 0; d < DIM; ++d) {
          float diff = data_[i * DIM + d] - data_[j * DIM + d];
          manual_dist += diff * diff;
        }

        INFO("Point " << i << " to Point " << j);
        INFO("Computed: " << computed_dist << ", Manual: " << manual_dist);

        REQUIRE(computed_dist == Approx(manual_dist).epsilon(1e-5));
      }
    }
  }

  SECTION("IP Distance Consistency") {
    auto distance_computer = distance::DistanceComputerFactory::create<float>(
        core::DistanceType::IP, DIM);

    INFO("Testing IP distance computer: " << distance_computer->name());

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i + 1; j < 3; ++j) {
        float computed_dist =
            distance_computer->compute(normalized_data_.data() + i * DIM,
                                       normalized_data_.data() + j * DIM);

        // 手动计算IP距离
        float manual_ip = 0;
        for (size_t d = 0; d < DIM; ++d) {
          manual_ip +=
              normalized_data_[i * DIM + d] * normalized_data_[j * DIM + d];
        }
        float manual_dist = 1.0f - manual_ip;

        INFO("Point " << i << " to Point " << j);
        INFO("Computed: " << computed_dist << ", Manual: " << manual_dist);
        INFO("IP value: " << manual_ip);

        REQUIRE(computed_dist == Approx(manual_dist).epsilon(1e-5));
      }
    }
  }
}

TEST_CASE_METHOD(PrecisionTestFixture, "Quantizer Precision Test",
                 "[precision][quantizer]") {
  SECTION("FP32 Quantizer Test") {
    auto quantizer = std::make_unique<quantization::FP32Quantizer>(
        core::DistanceType::IP, DIM);

    // 训练量化器
    quantizer->train(normalized_data_.data(), NUM_POINTS, DIM);

    // 测试编码/解码一致性
    std::vector<float> encoded(DIM);
    std::vector<float> decoded(DIM);

    for (size_t i = 0; i < 3; ++i) {
      quantizer->encode(normalized_data_.data() + i * DIM, encoded.data());
      quantizer->decode(encoded.data(), decoded.data());

      for (size_t j = 0; j < DIM; ++j) {
        REQUIRE(decoded[j] ==
                Approx(normalized_data_[i * DIM + j]).epsilon(1e-6));
      }
    }
  }

  SECTION("SQ8 Quantizer Test") {
    auto fp32_quantizer = std::make_unique<quantization::FP32Quantizer>(
        core::DistanceType::IP, DIM);
    fp32_quantizer->train(normalized_data_.data(), NUM_POINTS, DIM);

    auto quantizer = std::make_unique<quantization::SQ8Quantizer>(
        core::DistanceType::IP, DIM, std::move(fp32_quantizer));

    // 训练量化器
    quantizer->train(normalized_data_.data(), NUM_POINTS, DIM);

    // 测试编码/解码
    std::vector<uint8_t> encoded(quantizer->code_size());
    std::vector<float> decoded(DIM);

    for (size_t i = 0; i < 3; ++i) {
      quantizer->encode(normalized_data_.data() + i * DIM, encoded.data());
      quantizer->decode(encoded.data(), decoded.data());

      // SQ8量化会有精度损失，但应该在合理范围内
      float max_error = 0;
      for (size_t j = 0; j < DIM; ++j) {
        float error = std::abs(decoded[j] - normalized_data_[i * DIM + j]);
        max_error = std::max(max_error, error);
      }

      INFO("Point " << i << " max quantization error: " << max_error);
      REQUIRE(max_error < 0.1f);  // 量化误差应该小于0.1
    }
  }
}

TEST_CASE_METHOD(PrecisionTestFixture, "End-to-End Precision Test",
                 "[precision][integration]") {
  SECTION("FP32 End-to-End Test") {
    // 创建FP32搜索器
    auto quantizer = std::make_unique<quantization::FP32Quantizer>(
        core::DistanceType::IP, DIM);

    // 训练量化器
    quantizer->train(normalized_data_.data(), NUM_POINTS, DIM);

    // 计算真实的最近邻
    std::vector<std::vector<std::pair<float, size_t>>> true_neighbors(
        NUM_POINTS);

    auto distance_computer = distance::DistanceComputerFactory::create<float>(
        core::DistanceType::IP, DIM);

    for (size_t i = 0; i < NUM_POINTS; ++i) {
      for (size_t j = 0; j < NUM_POINTS; ++j) {
        if (i != j) {
          float dist =
              distance_computer->compute(normalized_data_.data() + i * DIM,
                                         normalized_data_.data() + j * DIM);
          true_neighbors[i].push_back({dist, j});
        }
      }
      std::sort(true_neighbors[i].begin(), true_neighbors[i].end());
    }

    // 测试查询精度
    int total_correct = 0;
    int total_queries = 0;

    for (size_t query_idx = 0; query_idx < NUM_POINTS; ++query_idx) {
      // 编码查询
      quantizer->encode_query(normalized_data_.data() + query_idx * DIM);

      // 计算到所有点的距离
      std::vector<std::pair<float, size_t>> computed_neighbors;
      for (size_t i = 0; i < NUM_POINTS; ++i) {
        if (i != query_idx) {
          float dist = quantizer->compute_query_distance(i);
          computed_neighbors.emplace_back(dist, i);
        }
      }
      std::sort(computed_neighbors.begin(), computed_neighbors.end());

      // 检查最近邻是否正确
      if (!true_neighbors[query_idx].empty() && !computed_neighbors.empty()) {
        size_t true_nearest = true_neighbors[query_idx][0].second;
        size_t computed_nearest = computed_neighbors[0].second;

        INFO("Query " << query_idx << ": true=" << true_nearest
                      << ", computed=" << computed_nearest);
        INFO("True distance: " << true_neighbors[query_idx][0].first);
        INFO("Computed distance: " << computed_neighbors[0].first);

        if (true_nearest == computed_nearest) {
          total_correct++;
        }
        total_queries++;
      }
    }

    double precision =
        total_queries > 0 ? (double)total_correct / total_queries : 0.0;

    INFO("Total queries: " << total_queries);
    INFO("Correct predictions: " << total_correct);
    INFO("Precision: " << precision * 100 << "%");

    // FP32应该有100%的精度
    REQUIRE(precision >= 0.9);  // 至少90%精度
  }

  SECTION("LinearPool Insert/Reorder Test") {
    using namespace searcher;

    LinearPool<float> pool(5, 10);  // capacity=5, max_size=10

    // 插入一些距离值
    pool.insert(0, 0.5f);
    pool.insert(1, 0.3f);
    pool.insert(2, 0.7f);
    pool.insert(3, 0.1f);
    pool.insert(4, 0.9f);

    INFO("Pool size after insertions: " << pool.size());
    REQUIRE(pool.size() == 5);

    // 检查排序是否正确（应该按距离从小到大排序）
    std::vector<float> distances;
    std::vector<int> ids;

    for (int i = 0; i < pool.size(); ++i) {
      distances.push_back(pool.distance(i));
      ids.push_back(pool.id(i));
      INFO("Position " << i << ": id=" << pool.id(i)
                       << ", distance=" << pool.distance(i));
    }

    // 验证距离是否按升序排列
    for (size_t i = 1; i < distances.size(); ++i) {
      REQUIRE(distances[i] >= distances[i - 1]);
    }

    // 验证最小距离的ID是否正确
    REQUIRE(ids[0] == 3);  // ID 3 对应距离 0.1
  }
}

TEST_CASE_METHOD(PrecisionTestFixture, "SIMD Function Test",
                 "[precision][simd]") {
  SECTION("SIMD IP Function Test") {
    // 测试SIMD IP函数的正确性
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i + 1; j < 3; ++j) {
        float simd_result = simd::IP(normalized_data_.data() + i * DIM,
                                     normalized_data_.data() + j * DIM, DIM);

        float manual_result = 0;
        for (size_t d = 0; d < DIM; ++d) {
          manual_result +=
              normalized_data_[i * DIM + d] * normalized_data_[j * DIM + d];
        }

        INFO("SIMD IP: " << simd_result << ", Manual IP: " << manual_result);
        REQUIRE(simd_result == Approx(manual_result).epsilon(1e-5));
      }
    }
  }

  SECTION("SIMD L2 Function Test") {
    // 测试SIMD L2函数的正确性
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i + 1; j < 3; ++j) {
        float simd_result =
            simd::L2Sqr(data_.data() + i * DIM, data_.data() + j * DIM, DIM);

        float manual_result = 0;
        for (size_t d = 0; d < DIM; ++d) {
          float diff = data_[i * DIM + d] - data_[j * DIM + d];
          manual_result += diff * diff;
        }

        INFO("SIMD L2: " << simd_result << ", Manual L2: " << manual_result);
        REQUIRE(simd_result == Approx(manual_result).epsilon(1e-5));
      }
    }
  }
}
