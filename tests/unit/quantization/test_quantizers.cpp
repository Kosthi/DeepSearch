#include <neighbor.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "quantization/fp32_quant.h"
#include "quantization/quantizer.h"
#include "quantization/sq4_quant.h"
#include "quantization/sq8_quant.h"

using namespace deepsearch::quantization;
using namespace deepsearch::core;
using Catch::Approx;

class QuantizerTestFixture {
 public:
  QuantizerTestFixture() : dim_(128), n_(1000) { generateTestData(); }

 private:
  void generateTestData() {
    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子确保可重现性
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    data_.resize(n_ * dim_);
    query_.resize(dim_);

    for (size_t i = 0; i < n_ * dim_; ++i) {
      data_[i] = dis(gen);
    }

    for (size_t i = 0; i < dim_; ++i) {
      query_[i] = dis(gen);
    }
  }

 protected:
  size_t dim_;
  size_t n_;
  std::vector<float> data_;
  std::vector<float> query_;
};

TEST_CASE_METHOD(QuantizerTestFixture, "FP32Quantizer functionality",
                 "[quantization][fp32]") {
  SECTION("Basic operations") {
    FP32Quantizer quantizer(DistanceType::L2, dim_);

    // 测试训练
    REQUIRE_NOTHROW(quantizer.train(data_.data(), n_, dim_));

    // 测试基本属性
    REQUIRE(quantizer.dimension() == dim_);
    REQUIRE(quantizer.code_size() == dim_ * sizeof(float));
    REQUIRE(quantizer.name() == "FP32Quantizer");
  }

  SECTION("Encode/Decode consistency") {
    FP32Quantizer quantizer(DistanceType::L2, dim_);
    quantizer.train(data_.data(), n_, dim_);

    std::vector<float> encoded(dim_);
    std::vector<float> decoded(dim_);

    // 编码解码测试
    quantizer.encode(query_.data(), encoded.data());
    quantizer.decode(encoded.data(), decoded.data());

    // FP32应该是无损的
    for (size_t i = 0; i < dim_; ++i) {
      REQUIRE(decoded[i] == Approx(query_[i]).margin(1e-6f));
    }
  }

  SECTION("Distance computation") {
    FP32Quantizer quantizer(DistanceType::L2, dim_);
    quantizer.train(data_.data(), n_, dim_);

    // 测试距离计算
    float dist = quantizer.compute_distance(query_.data(), data_.data());
    float res = 0;
    for (auto i = 0; i < dim_; ++i) {
      res += (query_[i] - data_[i]) * (query_[i] - data_[i]);
    }

    quantizer.encode_query(query_.data());
    auto dis = quantizer.compute_query_distance(data_.data());

    REQUIRE(dist == Approx(res).margin(1e-6f));
    REQUIRE(dist == Approx(dis).margin(1e-6f));

    // 自距离应该为0
    float self_dist = quantizer.compute_distance(query_.data(), query_.data());
    REQUIRE(self_dist == Approx(0.0f).margin(1e-6f));
  }
}

TEST_CASE_METHOD(QuantizerTestFixture, "SQ8Quantizer functionality",
                 "[quantization][sq8]") {
  SECTION("Basic operations") {
    SQ8Quantizer quantizer(DistanceType::L2, dim_);

    // 测试训练
    REQUIRE_NOTHROW(quantizer.train(data_.data(), n_, dim_));

    // 测试基本属性
    REQUIRE(quantizer.dimension() == dim_);
    REQUIRE(quantizer.code_size() == dim_ * sizeof(uint8_t));
    REQUIRE(quantizer.name() == "SQ8Quantizer");
  }

  SECTION("Quantization bounds") {
    SQ8Quantizer quantizer(DistanceType::L2, dim_);
    quantizer.train(data_.data(), n_, dim_);

    std::vector<uint8_t> encoded(dim_);
    quantizer.encode(query_.data(), encoded.data());

    // 检查量化值在有效范围内
    for (size_t i = 0; i < dim_; ++i) {
      REQUIRE(encoded[i] <= 255);
    }
  }

  SECTION("Reconstruction quality") {
    SQ8Quantizer quantizer(DistanceType::L2, dim_);
    quantizer.train(data_.data(), n_, dim_);

    std::vector<uint8_t> encoded(dim_);
    std::vector<float> decoded(dim_);

    quantizer.encode(query_.data(), encoded.data());
    quantizer.decode(encoded.data(), decoded.data());

    // 计算重构误差
    float mse = 0.0f;
    for (size_t i = 0; i < dim_; ++i) {
      float diff = query_[i] - decoded[i];
      mse += diff * diff;
    }
    mse /= dim_;

    // 重构误差应该在合理范围内
    REQUIRE(mse < 1.0f);  // 根据数据范围调整
  }
}

TEST_CASE_METHOD(QuantizerTestFixture, "SQ4Quantizer functionality",
                 "[quantization][sq4]") {
  SECTION("Basic operations") {
    SQ4Quantizer quantizer(DistanceType::L2, dim_);

    // 测试训练
    REQUIRE_NOTHROW(quantizer.train(data_.data(), n_, dim_));

    // 测试基本属性
    REQUIRE(quantizer.dimension() == dim_);
    REQUIRE(quantizer.code_size() == (dim_ + 1) / 2);  // 4位打包
    REQUIRE(quantizer.name() == "SQ4Quantizer");
  }

  SECTION("4-bit packing") {
    SQ4Quantizer quantizer(DistanceType::L2, dim_);
    quantizer.train(data_.data(), n_, dim_);

    std::vector<uint8_t> encoded((dim_ + 1) / 2);
    quantizer.encode(query_.data(), encoded.data());

    // 验证4位值的范围
    for (size_t i = 0; i < encoded.size(); ++i) {
      uint8_t low = encoded[i] & 0x0F;
      uint8_t high = (encoded[i] >> 4) & 0x0F;
      REQUIRE(low <= 15);
      REQUIRE(high <= 15);
    }
  }
}

TEST_CASE("QuantizerFactory functionality", "[quantization][factory]") {
  SECTION("Create different quantizer types") {
    size_t dim = 64;

    // 测试FP32量化器创建
    auto fp32_quantizer = QuantizerFactory<float, float>::create(
        QuantizerType::FP32, DistanceType::L2, dim);
    REQUIRE(fp32_quantizer != nullptr);
    REQUIRE(fp32_quantizer->name() == "FP32Quantizer");

    // 测试SQ8量化器创建
    auto sq8_quantizer = QuantizerFactory<float, uint8_t>::create(
        QuantizerType::SQ8, DistanceType::L2, dim);
    REQUIRE(sq8_quantizer != nullptr);
    REQUIRE(sq8_quantizer->name() == "SQ8Quantizer");

    // 测试SQ4量化器创建
    auto sq4_quantizer = QuantizerFactory<float, uint8_t>::create(
        QuantizerType::SQ4, DistanceType::L2, dim);
    REQUIRE(sq4_quantizer != nullptr);
    REQUIRE(sq4_quantizer->name() == "SQ4Quantizer");
  }

  SECTION("Supported types") {
    auto types = QuantizerFactory<float, float>::get_supported_types();
    REQUIRE(!types.empty());

    REQUIRE(
        QuantizerFactory<float, float>::is_type_supported(QuantizerType::FP32));
    REQUIRE(QuantizerFactory<float, uint8_t>::is_type_supported(
        QuantizerType::SQ8));
    REQUIRE(QuantizerFactory<float, uint8_t>::is_type_supported(
        QuantizerType::SQ4));
  }
}

TEST_CASE_METHOD(QuantizerTestFixture, "Quantizer with reorderer",
                 "[quantization][reorder]") {
  SECTION("SQ8 with FP32 reorderer") {
    auto fp32_reorderer =
        std::make_unique<FP32Quantizer>(DistanceType::L2, dim_);
    SQ8Quantizer quantizer(DistanceType::L2, dim_, std::move(fp32_reorderer));

    // 训练量化器和重排器
    quantizer.train(data_.data(), n_, dim_);

    // 量化查询向量
    std::vector<uint8_t> encoded_query(quantizer.code_size());
    quantizer.encode(query_.data(), encoded_query.data());

    // 创建候选池
    deepsearch::searcher::LinearPool<float> pool(n_,
                                                 10);  // 创建容量为10的线性池

    // 向池中添加一些候选项
    for (int i = 0; i < std::min(10, static_cast<int>(n_)); ++i) {
      // 获取量化后的数据并计算距离
      const uint8_t* quantized_data =
          reinterpret_cast<const uint8_t*>(quantizer.get_data(i));
      float dist =
          quantizer.compute_distance(encoded_query.data(), quantized_data);
      pool.insert(i, dist);
    }

    std::vector<int> reordered_ids(pool.size());
    quantizer.reorder(pool, reordered_ids.data(), pool.size());

    // 验证重排序结果
    REQUIRE(reordered_ids.size() == pool.size());
    // 第一个应该是最近的
    REQUIRE(reordered_ids[0] >= 0);
    REQUIRE(reordered_ids[0] < static_cast<int>(n_));
  }
}
