#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "core/config.h"

using namespace deepsearch::core;

TEST_CASE("HNSWConfig serialization and deserialization", "[config][hnsw]") {
  HNSWConfig config;
  config.M = 32;
  config.ef_construction = 200;
  config.max_elements = 500000;
  config.allow_replace_deleted = true;
  config.random_seed = 42;

  SECTION("Basic serialization") {
    std::string serialized = config.to_string();
    REQUIRE_FALSE(serialized.empty());

    HNSWConfig config2;
    config2.from_string(serialized);

    REQUIRE(config.M == config2.M);
    REQUIRE(config.ef_construction == config2.ef_construction);
    REQUIRE(config.max_elements == config2.max_elements);
    REQUIRE(config.allow_replace_deleted == config2.allow_replace_deleted);
    REQUIRE(config.random_seed == config2.random_seed);
  }

  SECTION("Boundary values") {
    config.M = 1;
    config.ef_construction = 1;
    config.max_elements = 1;

    std::string serialized = config.to_string();
    HNSWConfig deserialized;
    deserialized.from_string(serialized);

    REQUIRE(config.M == deserialized.M);
    REQUIRE(config.ef_construction == deserialized.ef_construction);
    REQUIRE(config.max_elements == deserialized.max_elements);
  }
}

TEST_CASE("SearchConfig serialization and deserialization",
          "[config][search]") {
  SearchConfig config;
  config.ef = 100;
  config.num_threads = 8;
  config.use_prefetch = false;
  config.batch_size = 2000;

  std::string serialized = config.to_string();
  REQUIRE_FALSE(serialized.empty());

  SearchConfig config2;
  config2.from_string(serialized);

  REQUIRE(config.ef == config2.ef);
  REQUIRE(config.num_threads == config2.num_threads);
  REQUIRE(config.use_prefetch == config2.use_prefetch);
  REQUIRE(config.batch_size == config2.batch_size);
}

TEST_CASE("QuantizationConfig serialization and deserialization",
          "[config][quantization]") {
  QuantizationConfig config;
  config.nbits = 4;
  config.subvector_size = 16;
  config.num_centroids = 512;

  std::string serialized = config.to_string();
  REQUIRE_FALSE(serialized.empty());

  QuantizationConfig config2;
  config2.from_string(serialized);

  REQUIRE(config.nbits == config2.nbits);
  REQUIRE(config.subvector_size == config2.subvector_size);
  REQUIRE(config.num_centroids == config2.num_centroids);
}

TEST_CASE("ConfigManager functionality", "[config][manager]") {
  auto& manager = ConfigManager::instance();

  SECTION("Singleton pattern") {
    auto& manager2 = ConfigManager::instance();
    REQUIRE(&manager == &manager2);
  }

  SECTION("Basic operations") {
    manager.reset_to_defaults();

    REQUIRE(manager.has_config("logging"));
    REQUIRE(manager.has_config("hnsw"));
    REQUIRE(manager.has_config("search"));
    REQUIRE(manager.has_config("quantization"));

    auto names = manager.get_config_names();
    REQUIRE(names.size() == 4);
  }

  SECTION("Convenience methods") {
    HNSWConfig hnsw_config;
    hnsw_config.M = 64;
    hnsw_config.ef_construction = 400;
    manager.set_hnsw_config(hnsw_config);

    auto retrieved_hnsw = manager.get_hnsw_config();
    REQUIRE(retrieved_hnsw.M == 64);
    REQUIRE(retrieved_hnsw.ef_construction == 400);
  }

  SECTION("File I/O") {
    const std::string test_file = "test_config.ini";

    // 设置配置
    HNSWConfig hnsw_config;
    hnsw_config.M = 48;
    hnsw_config.ef_construction = 300;
    manager.set_hnsw_config(hnsw_config);

    // 保存和加载
    manager.save_to_file(test_file);
    REQUIRE(std::filesystem::exists(test_file));

    manager.reset_to_defaults();
    manager.load_from_file(test_file);

    auto loaded_hnsw = manager.get_hnsw_config();
    REQUIRE(loaded_hnsw.M == 48);
    REQUIRE(loaded_hnsw.ef_construction == 300);

    // 清理
    std::filesystem::remove(test_file);
  }
}

TEST_CASE("Config error handling", "[config][error]") {
  SECTION("Invalid string format") {
    HNSWConfig config;

    // 空字符串应该保持默认值
    config.from_string("");
    REQUIRE(config.M == 16);

    // 无效格式应该保持默认值
    config.from_string("invalid_format");
    REQUIRE(config.M == 16);

    // 部分有效格式
    config.from_string("M=32;invalid;ef_construction=200");
    REQUIRE(config.M == 32);
    REQUIRE(config.ef_construction == 200);
  }
}
