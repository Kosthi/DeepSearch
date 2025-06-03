#include "core/config.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace deepsearch {
namespace core {

// HNSWConfig 实现
std::string HNSWConfig::to_string() const {
  std::ostringstream oss;
  oss << "M=" << M << ";ef_construction=" << ef_construction
      << ";max_elements=" << max_elements
      << ";allow_replace_deleted=" << (allow_replace_deleted ? "true" : "false")
      << ";random_seed=" << random_seed;
  return oss.str();
}

void HNSWConfig::from_string(const std::string& str) {
  std::istringstream iss(str);
  std::string token;
  while (std::getline(iss, token, ';')) {
    size_t pos = token.find('=');
    if (pos != std::string::npos) {
      std::string key = token.substr(0, pos);
      std::string value = token.substr(pos + 1);

      if (key == "M") {
        M = std::stoul(value);
      } else if (key == "ef_construction") {
        ef_construction = std::stoul(value);
      } else if (key == "max_elements") {
        max_elements = std::stoul(value);
      } else if (key == "allow_replace_deleted") {
        allow_replace_deleted = (value == "true");
      } else if (key == "random_seed") {
        random_seed = std::stoul(value);
      }
    }
  }
}

// SearchConfig 实现
std::string SearchConfig::to_string() const {
  std::ostringstream oss;
  oss << "ef=" << ef << ";num_threads=" << num_threads
      << ";use_prefetch=" << (use_prefetch ? "true" : "false")
      << ";batch_size=" << batch_size;
  return oss.str();
}

void SearchConfig::from_string(const std::string& str) {
  std::istringstream iss(str);
  std::string token;
  while (std::getline(iss, token, ';')) {
    size_t pos = token.find('=');
    if (pos != std::string::npos) {
      std::string key = token.substr(0, pos);
      std::string value = token.substr(pos + 1);

      if (key == "ef") {
        ef = std::stoul(value);
      } else if (key == "num_threads") {
        num_threads = std::stoul(value);
      } else if (key == "use_prefetch") {
        use_prefetch = (value == "true");
      } else if (key == "batch_size") {
        batch_size = std::stoul(value);
      }
    }
  }
}

// QuantizationConfig 实现
std::string QuantizationConfig::to_string() const {
  std::ostringstream oss;
  oss << "nbits=" << nbits << ";subvector_size=" << subvector_size
      << ";num_centroids=" << num_centroids;
  return oss.str();
}

void QuantizationConfig::from_string(const std::string& str) {
  std::istringstream iss(str);
  std::string token;
  while (std::getline(iss, token, ';')) {
    size_t pos = token.find('=');
    if (pos != std::string::npos) {
      std::string key = token.substr(0, pos);
      std::string value = token.substr(pos + 1);

      if (key == "nbits") {
        nbits = std::stoul(value);
      } else if (key == "subvector_size") {
        subvector_size = std::stoul(value);
      } else if (key == "num_centroids") {
        num_centroids = std::stoul(value);
      }
    }
  }
}

// ConfigManager 实现
ConfigManager& ConfigManager::instance() {
  static ConfigManager instance;
  return instance;
}

void ConfigManager::load_from_file(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open config file: " + path);
  }

  std::string line;
  std::string current_section;

  while (std::getline(file, line)) {
    // 跳过空行和注释
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // 检查是否是节标题 [section_name]
    if (line[0] == '[' && line.back() == ']') {
      current_section = line.substr(1, line.length() - 2);
      continue;
    }

    // 解析配置项
    size_t pos = line.find('=');
    if (pos != std::string::npos && !current_section.empty()) {
      std::string key = line.substr(0, pos);
      std::string value = line.substr(pos + 1);

      // 根据节名创建或更新配置
      if (current_section == "hnsw") {
        auto config = get_config<HNSWConfig>("hnsw");
        if (!config) {
          config = std::make_shared<HNSWConfig>();
          register_config("hnsw", config);
        }
        config->from_string(key + "=" + value);
      } else if (current_section == "search") {
        auto config = get_config<SearchConfig>("search");
        if (!config) {
          config = std::make_shared<SearchConfig>();
          register_config("search", config);
        }
        config->from_string(key + "=" + value);
      } else if (current_section == "quantization") {
        auto config = get_config<QuantizationConfig>("quantization");
        if (!config) {
          config = std::make_shared<QuantizationConfig>();
          register_config("quantization", config);
        }
        config->from_string(key + "=" + value);
      }
    }
  }
}

void ConfigManager::save_to_file(const std::string& path) const {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot create config file: " + path);
  }

  file << "# DeepSearch Configuration File\n";
  file << "# Generated automatically\n\n";

  for (const auto& [name, config] : configs_) {
    file << "[" << name << "]\n";

    std::string config_str = config->to_string();
    std::istringstream iss(config_str);
    std::string token;

    while (std::getline(iss, token, ';')) {
      if (!token.empty()) {
        file << token << "\n";
      }
    }
    file << "\n";
  }
}

// 便捷方法实现
void ConfigManager::set_hnsw_config(const HNSWConfig& config) {
  register_config("hnsw", std::make_shared<HNSWConfig>(config));
}

void ConfigManager::set_search_config(const SearchConfig& config) {
  register_config("search", std::make_shared<SearchConfig>(config));
}

void ConfigManager::set_quantization_config(const QuantizationConfig& config) {
  register_config("quantization", std::make_shared<QuantizationConfig>(config));
}

HNSWConfig ConfigManager::get_hnsw_config() const {
  auto config = get_config<HNSWConfig>("hnsw");
  return config ? *config : HNSWConfig{};
}

SearchConfig ConfigManager::get_search_config() const {
  auto config = get_config<SearchConfig>("search");
  return config ? *config : SearchConfig{};
}

QuantizationConfig ConfigManager::get_quantization_config() const {
  auto config = get_config<QuantizationConfig>("quantization");
  return config ? *config : QuantizationConfig{};
}

void ConfigManager::reset_to_defaults() {
  configs_.clear();

  // 设置默认配置
  register_config("hnsw", std::make_shared<HNSWConfig>());
  register_config("search", std::make_shared<SearchConfig>());
  register_config("quantization", std::make_shared<QuantizationConfig>());
  register_config("logging", std::make_shared<LoggingConfig>());  // 添加这一行
}

bool ConfigManager::has_config(const std::string& name) const {
  return configs_.find(name) != configs_.end();
}

std::vector<std::string> ConfigManager::get_config_names() const {
  std::vector<std::string> names;
  for (const auto& [name, config] : configs_) {
    names.push_back(name);
  }
  return names;
}

LoggingConfig ConfigManager::get_logging_config() const {
  auto config = get_config<LoggingConfig>("logging");
  return config ? *config : LoggingConfig{};
}

// LoggingConfig 实现
std::string LoggingConfig::to_string() const {
  std::ostringstream oss;
  oss << "level=" << static_cast<int>(level)
      << ";enable_console=" << (enable_console ? "true" : "false")
      << ";enable_file=" << (enable_file ? "true" : "false")
      << ";log_dir=" << log_dir << ";log_pattern=" << log_pattern
      << ";max_file_size=" << max_file_size << ";max_files=" << max_files
      << ";async_logging=" << (async_logging ? "true" : "false");
  return oss.str();
}

void LoggingConfig::from_string(const std::string& str) {
  std::istringstream iss(str);
  std::string token;
  while (std::getline(iss, token, ';')) {
    size_t pos = token.find('=');
    if (pos != std::string::npos) {
      std::string key = token.substr(0, pos);
      std::string value = token.substr(pos + 1);

      if (key == "level") {
        level = static_cast<LogLevel>(std::stoi(value));
      } else if (key == "enable_console") {
        enable_console = (value == "true");
      } else if (key == "enable_file") {
        enable_file = (value == "true");
      } else if (key == "log_dir") {
        log_dir = value;
      } else if (key == "log_pattern") {
        log_pattern = value;
      } else if (key == "max_file_size") {
        max_file_size = std::stoull(value);
      } else if (key == "max_files") {
        max_files = std::stoull(value);
      } else if (key == "async_logging") {
        async_logging = (value == "true");
      }
    }
  }
}

}  // namespace core
}  // namespace deepsearch
