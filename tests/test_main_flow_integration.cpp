#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include "core/interfaces.h"
#include "graph/builder_factory.h"
#include "quantization/fp32_quant.h"
#include "searcher/searcher.h"

using namespace deepsearch;
using namespace deepsearch::core;
using namespace deepsearch::graph;
using namespace deepsearch::searcher;
using namespace deepsearch::quantization;
using namespace Catch;

class MainFlowTestFixture {
 public:
  static constexpr size_t NUM_POINTS = 100;
  static constexpr size_t DIM = 128;
  static constexpr int K = 10;
  static constexpr int M = 16;
  static constexpr int EF_CONSTRUCTION = 200;
  static constexpr int SEARCH_EF = 50;

  MainFlowTestFixture() { generateTestData(); }

  ~MainFlowTestFixture() {
    if (std::filesystem::exists(graph_path_)) {
      std::filesystem::remove(graph_path_);
    }
  }

 private:
  void generateTestData() {
    std::mt19937 rng(42);  // 固定种子确保可重现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    data_.resize(NUM_POINTS * DIM);

    // 生成随机数据
    for (size_t i = 0; i < NUM_POINTS * DIM; ++i) {
      data_[i] = dist(rng);
    }

    // 为了L2距离，我们不需要归一化
    // 但为了测试的稳定性，我们可以稍微缩放数据
    for (size_t i = 0; i < NUM_POINTS * DIM; ++i) {
      data_[i] *= 0.1f;  // 缩放到较小范围
    }

    graph_path_ = "test_graph_" + std::to_string(std::time(nullptr)) + ".bin";
  }

 protected:
  std::vector<float> data_;
  std::string graph_path_;
};

TEST_CASE_METHOD(MainFlowTestFixture,
                 "Main Flow Integration Test - L2 Distance",
                 "[integration][main_flow]") {
  SECTION("Complete Build and Search Flow") {
    // 0. 配置日志
    LogConfig logConfig;
    LoggerGuard logger_guard(logConfig);

    // 1. 按照main.cpp的流程创建builder配置
    BuilderConfig config;
    config.M = M;
    config.ef_construction = EF_CONSTRUCTION;
    config.max_elements = NUM_POINTS;

    INFO("Creating HNSW builder with L2 distance");
    auto hnsw_builder = BuilderFactory<float>::create(
        BuilderType::HNSW, DistanceType::L2, DIM, config);
    REQUIRE(hnsw_builder != nullptr);

    // 2. 建图
    INFO("Building graph with " << NUM_POINTS << " points");
    Graph graph = hnsw_builder->build(data_.data(), NUM_POINTS, DIM);

    // 3. 保存和加载图（模拟main.cpp的流程）
    INFO("Saving graph to: " << graph_path_);
    graph.save(graph_path_);
    REQUIRE(std::filesystem::exists(graph_path_));

    // 重新加载图
    Graph loaded_graph;
    loaded_graph.load(graph_path_);

    // 4. 创建量化器和搜索器（按照main.cpp的流程）
    INFO("Creating FP32 quantizer and searcher");
    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, DIM);
    auto searcher = SearcherFactory::create<FP32Quantizer>(
        loaded_graph, std::move(quantizer));
    REQUIRE(searcher != nullptr);

    // 5. 设置数据和优化
    searcher->SetData(data_.data(), NUM_POINTS, DIM);
    searcher->Optimize(1);  // 单线程
    searcher->SetEf(SEARCH_EF);

    // 6. 计算真实的最近邻（暴力搜索）
    INFO("Computing ground truth with brute force search");
    std::vector<std::vector<std::pair<float, int>>> ground_truth(NUM_POINTS);

    for (size_t i = 0; i < NUM_POINTS; ++i) {
      for (size_t j = 0; j < NUM_POINTS; ++j) {
        if (i != j) {
          // 计算L2距离
          float dist = 0;
          for (size_t d = 0; d < DIM; ++d) {
            float diff = data_[i * DIM + d] - data_[j * DIM + d];
            dist += diff * diff;
          }
          ground_truth[i].emplace_back(dist, j);
        }
      }
      std::sort(ground_truth[i].begin(), ground_truth[i].end());
    }

    // 7. 使用建图的点作为查询点进行搜索（按照main.cpp的流程）
    INFO("Performing search with build points as queries");
    std::vector<int> predictions(NUM_POINTS * K);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      searcher->Search(data_.data() + i * DIM, K, predictions.data() + i * K);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration<double>(end_time - start_time).count();
    double qps = NUM_POINTS / elapsed;
    INFO("Search completed. QPS: " << qps);

    // 8. 计算精度（按照main.cpp的方式）
    int correct_predictions = 0;

    // 显示前几个查询的详细结果（类似main.cpp）
    for (int i = 0; i < std::min(5, (int)NUM_POINTS); ++i) {
      INFO("Query " << i << ":");

      std::string pred_str = "  Pred: ";
      for (int j = 0; j < K; ++j) {
        pred_str += std::to_string(predictions[i * K + j]) + " ";
      }
      INFO(pred_str);

      std::string gt_str = "  GT:   ";
      for (int j = 0; j < std::min(K, (int)ground_truth[i].size()); ++j) {
        gt_str += std::to_string(ground_truth[i][j].second) + " ";
      }
      INFO(gt_str);
    }

    // 计算recall@K
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      std::unordered_set<int> gt_set;
      for (int j = 0; j < std::min(K, (int)ground_truth[i].size()); ++j) {
        gt_set.insert(ground_truth[i][j].second);
      }

      for (int j = 0; j < K; ++j) {
        if (gt_set.count(predictions[i * K + j])) {
          correct_predictions++;
        }
      }
    }

    double recall = (double)correct_predictions / (NUM_POINTS * K);
    INFO("Recall@" << K << " = " << recall * 100 << "%");
    INFO("Correct predictions: " << correct_predictions << "/"
                                 << (NUM_POINTS * K));

    // 9. 验证精度要求
    // 对于FP32量化器和相对简单的数据，我们期望较高的精度
    REQUIRE(recall >= 0.8);  // 至少80%的recall

    // 10. 验证搜索结果的合理性
    // 检查是否有无效的ID
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      for (int j = 0; j < K; ++j) {
        int pred_id = predictions[i * K + j];
        REQUIRE(pred_id >= 0);
        REQUIRE(pred_id < (int)NUM_POINTS);
        // REQUIRE(pred_id != (int)i);  // 不应该返回查询点自身
      }
    }

    // 11. 验证距离的单调性（搜索结果应该按距离排序）
    for (size_t i = 0; i < std::min((size_t)10, NUM_POINTS); ++i) {
      std::vector<float> distances;
      for (int j = 0; j < K; ++j) {
        int pred_id = predictions[i * K + j];
        float dist = 0;
        for (size_t d = 0; d < DIM; ++d) {
          float diff = data_[i * DIM + d] - data_[pred_id * DIM + d];
          dist += diff * diff;
        }
        distances.push_back(dist);
      }

      // 检查距离是否单调递增（或至少非递减）
      for (int j = 1; j < K; ++j) {
        if (distances[j] < distances[j - 1] - 1e-6) {  // 允许小的浮点误差
          INFO("Query " << i << ": distance[" << j << "] = " << distances[j]
                        << " < distance[" << (j - 1)
                        << "] = " << distances[j - 1]);
          FAIL("Search results are not properly sorted by distance");
        }
      }
    }

    INFO("All tests passed! Recall: " << recall * 100 << "%, QPS: " << qps);
  }
}

TEST_CASE_METHOD(MainFlowTestFixture,
                 "Main Flow Integration Test - IP Distance",
                 "[integration][main_flow]") {
  SECTION("Complete Build and Search Flow with IP Distance") {
    // 为IP距离归一化数据
    std::vector<float> normalized_data(NUM_POINTS * DIM);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      float norm = 0;
      for (size_t j = 0; j < DIM; ++j) {
        float val = data_[i * DIM + j];
        norm += val * val;
      }
      norm = std::sqrt(norm);

      for (size_t j = 0; j < DIM; ++j) {
        normalized_data[i * DIM + j] = data_[i * DIM + j] / norm;
      }
    }

    // 0. 配置日志
    LogConfig logConfig;
    LoggerGuard logger_guard(logConfig);

    // 使用IP距离重复相同的测试流程
    BuilderConfig config;
    config.M = M;
    config.ef_construction = EF_CONSTRUCTION;
    config.max_elements = NUM_POINTS;

    auto hnsw_builder = BuilderFactory<float>::create(
        BuilderType::HNSW, DistanceType::IP, DIM, config);
    REQUIRE(hnsw_builder != nullptr);

    Graph graph = hnsw_builder->build(normalized_data.data(), NUM_POINTS, DIM);

    std::string ip_graph_path =
        "test_ip_graph_" + std::to_string(std::time(nullptr)) + ".bin";
    graph.save(ip_graph_path);

    Graph loaded_graph;
    loaded_graph.load(ip_graph_path);

    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::IP, DIM);
    auto searcher = SearcherFactory::create<FP32Quantizer>(
        loaded_graph, std::move(quantizer));

    searcher->SetData(normalized_data.data(), NUM_POINTS, DIM);
    searcher->Optimize(1);
    searcher->SetEf(SEARCH_EF);

    // 计算IP距离的真实最近邻
    std::vector<std::vector<std::pair<float, int>>> ground_truth(NUM_POINTS);

    for (size_t i = 0; i < NUM_POINTS; ++i) {
      for (size_t j = 0; j < NUM_POINTS; ++j) {
        if (i != j) {
          float ip = 0;
          for (size_t d = 0; d < DIM; ++d) {
            ip += normalized_data[i * DIM + d] * normalized_data[j * DIM + d];
          }
          float dist = 1.0f - ip;  // IP距离
          ground_truth[i].emplace_back(dist, j);
        }
      }
      std::sort(ground_truth[i].begin(), ground_truth[i].end());
    }

    std::vector<int> predictions(NUM_POINTS * K);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      searcher->Search(normalized_data.data() + i * DIM, K,
                       predictions.data() + i * K);
    }

    int correct_predictions = 0;
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      std::unordered_set<int> gt_set;
      for (int j = 0; j < std::min(K, (int)ground_truth[i].size()); ++j) {
        gt_set.insert(ground_truth[i][j].second);
      }

      for (int j = 0; j < K; ++j) {
        if (gt_set.count(predictions[i * K + j])) {
          correct_predictions++;
        }
      }
    }

    double recall = (double)correct_predictions / (NUM_POINTS * K);
    INFO("IP Distance Recall@" << K << " = " << recall * 100 << "%");

    REQUIRE(recall >= 0.8);  // 至少80%的recall

    // 清理临时文件
    if (std::filesystem::exists(ip_graph_path)) {
      std::filesystem::remove(ip_graph_path);
    }
  }
}
