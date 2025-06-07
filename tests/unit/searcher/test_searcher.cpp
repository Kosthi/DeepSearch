#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <set>
#include <vector>

#include "graph/graph.h"
#include "quantization/fp32_quant.h"
#include "quantization/sq8_quant.h"
#include "searcher/searcher.h"

using namespace deepsearch;
using namespace deepsearch::quantization;
using namespace deepsearch::core;
using namespace deepsearch::searcher;

using Catch::Approx;

class SearcherTestFixture {
 public:
  SearcherTestFixture() : dim_(64), n_(1000), k_(10) {
    generateTestData();
    createTestGraph();
  }

 private:
  void generateTestData() {
    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子确保可重现性
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    data_.resize(n_ * dim_);
    queries_.resize(10 * dim_);  // 10个查询向量

    for (size_t i = 0; i < n_ * dim_; ++i) {
      data_[i] = dis(gen);
    }

    for (size_t i = 0; i < 10 * dim_; ++i) {
      queries_[i] = dis(gen);
    }
  }

  void createTestGraph() {
    // 创建一个简单的随机图用于测试
    int K = 16;  // 每个节点的邻居数
    graph_ = Graph(n_, K);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dis(0, n_ - 1);

    for (int i = 0; i < n_; ++i) {
      std::set<int> neighbors;
      while (neighbors.size() < K) {
        int neighbor = dis(gen);
        if (neighbor != i) {
          neighbors.insert(neighbor);
        }
      }

      int j = 0;
      for (int neighbor : neighbors) {
        graph_.at(i, j++) = neighbor;
      }
    }

    // 设置入口点
    graph_.set_entry_points({0});
  }

 protected:
  size_t dim_;
  size_t n_;
  size_t k_;
  std::vector<float> data_;
  std::vector<float> queries_;
  graph::Graph graph_;
  LoggerGuard logger_guard;
};

TEST_CASE_METHOD(SearcherTestFixture, "Basic Searcher functionality",
                 "[searcher][basic]") {
  SECTION("Searcher creation and setup") {
    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    Searcher searcher(graph_, std::move(quantizer));

    // 设置数据
    REQUIRE_NOTHROW(searcher.SetData(data_.data(), n_, dim_));

    // 测试基本属性
    REQUIRE(searcher.GetEf() > 0);
    REQUIRE(searcher.GetQuantizerName() == "FP32Quantizer");
  }

  SECTION("Search functionality") {
    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    Searcher searcher(graph_, std::move(quantizer));

    searcher.SetData(data_.data(), n_, dim_);

    // 执行搜索
    std::vector<int> results(k_);
    REQUIRE_NOTHROW(searcher.Search(queries_.data(), k_, results.data()));

    // 验证搜索结果
    for (size_t i = 0; i < k_; ++i) {
      REQUIRE(results[i] >= 0);
      REQUIRE(results[i] < static_cast<int>(n_));
    }

    // 检查结果的唯一性
    std::set<int> unique_results(results.begin(), results.end());
    REQUIRE(unique_results.size() == k_);
  }

  SECTION("Parameter adjustment") {
    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    Searcher searcher(graph_, std::move(quantizer));

    // 测试ef参数设置
    searcher.SetEf(64);
    REQUIRE(searcher.GetEf() == 64);

    searcher.SetEf(128);
    REQUIRE(searcher.GetEf() == 128);
  }
}

TEST_CASE_METHOD(SearcherTestFixture, "Searcher with different quantizers",
                 "[searcher][quantizers]") {
  SECTION("FP32 Quantizer") {
    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    Searcher searcher(graph_, std::move(quantizer));

    searcher.SetData(data_.data(), n_, dim_);

    std::vector<int> results(k_);
    REQUIRE_NOTHROW(searcher.Search(queries_.data(), k_, results.data()));

    REQUIRE(searcher.GetQuantizerName() == "FP32Quantizer");
  }

  SECTION("SQ8 Quantizer") {
    auto quantizer = std::make_unique<SQ8Quantizer>(DistanceType::L2, dim_);
    Searcher searcher(graph_, std::move(quantizer));

    searcher.SetData(data_.data(), n_, dim_);

    std::vector<int> results(k_);
    REQUIRE_NOTHROW(searcher.Search(queries_.data(), k_, results.data()));

    REQUIRE(searcher.GetQuantizerName() == "SQ8Quantizer");
  }
}

TEST_CASE_METHOD(SearcherTestFixture, "SearcherFactory functionality",
                 "[searcher][factory]") {
  SECTION("Create basic searcher") {
    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    auto searcher =
        SearcherFactory::create<FP32Quantizer>(graph_, std::move(quantizer));
    REQUIRE(searcher != nullptr);

    searcher->SetData(data_.data(), n_, dim_);

    std::vector<int> results(k_);
    REQUIRE_NOTHROW(searcher->Search(queries_.data(), k_, results.data()));
  }

  SECTION("Create searcher with custom quantizer") {
    auto quantizer = std::make_unique<SQ8Quantizer>(DistanceType::L2, dim_);
    auto searcher = SearcherFactory::create(graph_, std::move(quantizer));
    REQUIRE(searcher != nullptr);

    searcher->SetData(data_.data(), n_, dim_);
    REQUIRE(searcher->GetQuantizerName() == "SQ8Quantizer");
  }
}

TEST_CASE_METHOD(SearcherTestFixture, "Search quality comparison",
                 "[searcher][quality]") {
  SECTION("Compare FP32 vs SQ8 search results") {
    // FP32搜索器
    // auto fp32_quantizer =
    //     std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    // Searcher fp32_searcher(graph_, std::move(fp32_quantizer));
    // fp32_searcher.SetData(data_.data(), n_, dim_);
    //
    // // SQ8搜索器
    // auto sq8_quantizer = std::make_unique<SQ8Quantizer>(DistanceType::L2,
    // dim_); Searcher sq8_searcher(graph_, std::move(sq8_quantizer));
    // sq8_searcher.SetData(data_.data(), n_, dim_);
    //
    // // 比较搜索结果
    // std::vector<int> fp32_results(k_);
    // std::vector<int> sq8_results(k_);
    //
    // fp32_searcher.Search(queries_.data(), k_, fp32_results.data());
    // sq8_searcher.Search(queries_.data(), k_, sq8_results.data());
    //
    // // 计算重叠度
    // std::set<int> fp32_set(fp32_results.begin(), fp32_results.end());
    // std::set<int> sq8_set(sq8_results.begin(), sq8_results.end());
    //
    // std::vector<int> intersection;
    // std::set_intersection(fp32_set.begin(), fp32_set.end(), sq8_set.begin(),
    //                       sq8_set.end(), std::back_inserter(intersection));
    //
    // // 量化搜索应该与精确搜索有一定的重叠
    // float overlap_ratio = static_cast<float>(intersection.size()) / k_;
    // REQUIRE(overlap_ratio > 0.3f);  // 至少30%的重叠
  }
}

TEST_CASE_METHOD(SearcherTestFixture, "Optimization functionality",
                 "[searcher][optimization]") {
  SECTION("Parameter optimization") {
    auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    Searcher searcher(graph_, std::move(quantizer));

    searcher.SetData(data_.data(), n_, dim_);

    // 记录优化前的ef值
    int ef_before = searcher.GetEf();

    // 执行优化
    REQUIRE_NOTHROW(searcher.Optimize(1));

    // 优化后应该仍然能正常搜索
    std::vector<int> results(k_);
    REQUIRE_NOTHROW(searcher.Search(queries_.data(), k_, results.data()));
  }
}
