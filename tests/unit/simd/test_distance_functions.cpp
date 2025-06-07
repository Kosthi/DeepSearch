#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "simd/distance_functions.h"

using namespace deepsearch::simd;
using Catch::Approx;

class DistanceFunctionTestFixture {
 public:
  DistanceFunctionTestFixture() : dim_(128) { generateTestData(); }

 private:
  void generateTestData() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    data_a_.resize(dim_);
    data_b_.resize(dim_);

    for (size_t i = 0; i < dim_; ++i) {
      data_a_[i] = dis(gen);
      data_b_[i] = dis(gen);
    }

    // 创建归一化版本用于余弦距离测试
    normalized_a_ = data_a_;
    normalized_b_ = data_b_;
    normalizeVector(normalized_a_);
    normalizeVector(normalized_b_);
  }

  void normalizeVector(std::vector<float>& vec) {
    // 计算L2范数
    float norm = 0.0f;
    for (float val : vec) {
      norm += val * val;
    }
    norm = std::sqrt(norm);

    // 避免除零错误
    if (norm < 1e-8f) {
      // 如果范数太小，生成新的随机向量
      std::uniform_real_distribution<float> dis(0.1f, 1.0f);
      for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen_);
      }
      normalizeVector(vec);  // 递归归一化
      return;
    }

    // 执行归一化
    for (float& val : vec) {
      val /= norm;
    }
  }

 protected:
  size_t dim_;
  std::vector<float> data_a_;
  std::vector<float> data_b_;
  std::vector<float> normalized_a_;
  std::vector<float> normalized_b_;
  std::mt19937 gen_{std::random_device{}()};
};

TEST_CASE("SIMD Function Initialization", "[simd][init]") {
  // 检查函数指针是否已正确设置
  REQUIRE(L2Sqr != nullptr);
  REQUIRE(IP != nullptr);
  REQUIRE(CosineDistance != nullptr);
  REQUIRE(L2SqrSQ8_ext != nullptr);
  REQUIRE(L2SqrSQ4 != nullptr);
  REQUIRE(IPSQ8_ext != nullptr);
}

TEST_CASE_METHOD(DistanceFunctionTestFixture, "L2 Distance Functions",
                 "[simd][l2]") {
  SECTION("L2Sqr vs L2SqrRef consistency") {
    // 测试优化版本与参考实现的一致性
    float simd_result = L2Sqr(data_a_.data(), data_b_.data(), dim_);
    float ref_result = ref::L2Sqr(data_a_.data(), data_b_.data(), dim_);

    REQUIRE(simd_result == Approx(ref_result).margin(1e-5f));
  }

  SECTION("L2Sqr properties") {
    // 测试L2距离的基本性质

    // 1. 非负性
    float distance = L2Sqr(data_a_.data(), data_b_.data(), dim_);
    REQUIRE(distance >= 0.0f);

    // 2. 自距离为0
    float self_distance = L2Sqr(data_a_.data(), data_a_.data(), dim_);
    REQUIRE(self_distance == Approx(0.0f).margin(1e-6f));

    // 3. 对称性
    float dist_ab = L2Sqr(data_a_.data(), data_b_.data(), dim_);
    float dist_ba = L2Sqr(data_b_.data(), data_a_.data(), dim_);
    REQUIRE(dist_ab == Approx(dist_ba).margin(1e-6f));
  }

  SECTION("Different vector sizes") {
    // 测试不同向量大小
    std::vector<size_t> test_sizes = {1, 4, 8, 16, 32, 64, 127, 128, 129, 256};

    for (size_t size : test_sizes) {
      if (size <= data_a_.size()) {
        float simd_result = L2Sqr(data_a_.data(), data_b_.data(), size);
        float ref_result = ref::L2Sqr(data_a_.data(), data_b_.data(), size);

        INFO("Testing size: " << size);
        REQUIRE(simd_result == Approx(ref_result).margin(1e-5f));
      }
    }
  }
}

TEST_CASE_METHOD(DistanceFunctionTestFixture, "IP Distance Functions",
                 "[simd][ip]") {
  SECTION("IP vs IPRef consistency") {
    // 测试优化版本与参考实现的一致性
    float simd_result = IP(data_a_.data(), data_b_.data(), dim_);
    float ref_result = ref::IP(data_a_.data(), data_b_.data(), dim_);

    REQUIRE(simd_result == Approx(ref_result).margin(1e-5f));
  }

  SECTION("IP properties") {
    // 测试内积的基本性质

    // 1. 对称性
    float ip_ab = IP(data_a_.data(), data_b_.data(), dim_);
    float ip_ba = IP(data_b_.data(), data_a_.data(), dim_);
    REQUIRE(ip_ab == Approx(ip_ba).margin(1e-6f));

    // 2. 自内积为向量的L2范数平方
    float self_ip = IP(data_a_.data(), data_a_.data(), dim_);
    float l2_distance_self = L2Sqr(data_a_.data(), data_a_.data(), dim_);

    // IP(a,a) = ||a||²，而L2Sqr(a,a) = ||a-a||² = 0
    REQUIRE(self_ip >= 0.0f);                                 // 自内积应该非负
    REQUIRE(l2_distance_self == Approx(0.0f).margin(1e-6f));  // 自距离应该为0

    // 如果要测试范数平方，应该手动计算
    float manual_norm_sq = 0.0f;
    for (int i = 0; i < dim_; ++i) {
      manual_norm_sq += data_a_[i] * data_a_[i];
    }
    REQUIRE(self_ip == Approx(manual_norm_sq).margin(1e-5f));
  }
}

TEST_CASE_METHOD(DistanceFunctionTestFixture, "Cosine Distance Functions",
                 "[simd][cosine]") {
  SECTION("CosineDistance vs CosineDistanceRef consistency") {
    // 测试优化版本与参考实现的一致性
    float simd_result =
        CosineDistance(normalized_a_.data(), normalized_b_.data(), dim_);
    float ref_result =
        ref::CosineDistance(normalized_a_.data(), normalized_b_.data(), dim_);

    REQUIRE(simd_result == Approx(ref_result).margin(1e-5f));
  }

  SECTION("Cosine distance properties") {
    // 测试余弦距离的基本性质

    // 1. 范围 [0, 2]
    float distance =
        CosineDistance(normalized_a_.data(), normalized_b_.data(), dim_);
    REQUIRE(distance >= 0.0f);
    REQUIRE(distance <= 2.0f);

    // 2. 自距离为0（对于归一化向量）
    float self_distance =
        CosineDistance(normalized_a_.data(), normalized_a_.data(), dim_);
    REQUIRE(self_distance == Approx(0.0f).margin(1e-5f));

    // 3. 对称性
    float dist_ab =
        CosineDistance(normalized_a_.data(), normalized_b_.data(), dim_);
    float dist_ba =
        CosineDistance(normalized_b_.data(), normalized_a_.data(), dim_);
    REQUIRE(dist_ab == Approx(dist_ba).margin(1e-6f));
  }
}

TEST_CASE("Quantized Distance Functions", "[simd][quantized]") {
  // 初始化SIMD函数
  initializeSIMDFunctions();

  SECTION("SQ8 functions are callable") {
    // 创建测试数据
    std::vector<uint8_t> data_a(128, 100);
    std::vector<uint8_t> data_b(128, 150);
    size_t dim = 128;

    // 测试函数是否可调用（不会崩溃）
    REQUIRE_NOTHROW(L2SqrSQ8_ext(data_a.data(), data_b.data(), dim));
    REQUIRE_NOTHROW(IPSQ8_ext(data_a.data(), data_b.data(), dim));
  }

  SECTION("SQ4 functions are callable") {
    // 创建测试数据
    std::vector<uint8_t> data_a(64, 0x55);  // 每个字节包含两个4位值
    std::vector<uint8_t> data_b(64, 0xAA);
    size_t dim = 128;  // 128个4位值

    // 测试函数是否可调用（不会崩溃）
    REQUIRE_NOTHROW(L2SqrSQ4(data_a.data(), data_b.data(), dim));
  }
}
