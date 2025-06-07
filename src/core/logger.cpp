#include "logger.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>

#include <filesystem>
#include <iostream>

namespace deepsearch {
namespace core {

LogManager& LogManager::instance() {
  static LogManager instance;
  return instance;
}

void LogManager::initialize(const LogConfig& config) {
  if (initialized_) {
    return;
  }

  config_ = config;

  try {
    // 创建日志目录
    if (config_.enable_file && !std::filesystem::exists(config_.log_dir)) {
      std::filesystem::create_directories(config_.log_dir);
    }

    // 初始化异步日志
    if (config_.async_logging) {
      spdlog::init_thread_pool(config_.async_queue_size,
                               config_.thread_pool_size);
    }

    // 创建sinks
    create_sinks();

    // 创建默认logger
    auto default_logger = std::make_shared<spdlog::logger>(
        "default", sinks_.begin(), sinks_.end());
    default_logger->set_level(to_spdlog_level(config_.level));
    default_logger->set_pattern(config_.log_pattern);

    loggers_["default"] = default_logger;
    spdlog::set_default_logger(default_logger);

    initialized_ = true;

    DLOG_INFO("Log system initialized successfully");

  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize log system: " << e.what() << std::endl;
    throw;
  }
}

void LogManager::create_sinks() {
  sinks_.clear();

  // 控制台sink
  if (config_.enable_console) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(to_spdlog_level(config_.level));
    console_sink->set_pattern(config_.log_pattern);
    sinks_.push_back(console_sink);
  }

  // 文件sink
  if (config_.enable_file) {
    // 使用rotating file sink
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        config_.log_dir + "/deepsearch.log", config_.max_file_size,
        config_.max_files);
    file_sink->set_level(to_spdlog_level(config_.level));
    file_sink->set_pattern(config_.log_pattern);
    sinks_.push_back(file_sink);

    // 错误日志单独文件
    auto error_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        config_.log_dir + "/deepsearch_error.log", config_.max_file_size,
        config_.max_files);
    error_sink->set_level(spdlog::level::err);
    error_sink->set_pattern(config_.log_pattern);
    sinks_.push_back(error_sink);
  }
}

std::shared_ptr<spdlog::logger> LogManager::get_logger(
    const std::string& name) {
  if (!initialized_) {
    initialize();
  }

  auto it = loggers_.find(name);
  if (it != loggers_.end()) {
    return it->second;
  }

  // 创建新的logger
  std::shared_ptr<spdlog::logger> logger;

  if (config_.async_logging) {
    logger = std::make_shared<spdlog::async_logger>(
        name, sinks_.begin(), sinks_.end(), spdlog::thread_pool(),
        spdlog::async_overflow_policy::block);
  } else {
    logger =
        std::make_shared<spdlog::logger>(name, sinks_.begin(), sinks_.end());
  }

  logger->set_level(to_spdlog_level(config_.level));
  logger->set_pattern(config_.log_pattern);

  loggers_[name] = logger;
  return logger;
}

void LogManager::set_level(LogLevel level) {
  config_.level = level;
  auto spdlog_level = to_spdlog_level(level);

  for (auto& [name, logger] : loggers_) {
    logger->set_level(spdlog_level);
  }

  for (auto& sink : sinks_) {
    sink->set_level(spdlog_level);
  }
}

void LogManager::set_module_level(const std::string& module, LogLevel level) {
  auto logger = get_logger(module);
  logger->set_level(to_spdlog_level(level));
}

void LogManager::flush_all() {
  for (auto& [name, logger] : loggers_) {
    logger->flush();
  }
}

void LogManager::shutdown() {
  if (!initialized_) {
    return;
  }

  // 首先刷新所有logger
  flush_all();

  // 清理所有logger（包括异步logger）
  for (auto& [name, logger] : loggers_) {
    if (logger) {
      logger->flush();
    }
  }

  // 清理默认logger
  spdlog::set_default_logger(nullptr);

  // 清理所有注册的logger
  loggers_.clear();
  sinks_.clear();

  // 最后关闭线程池
  if (config_.async_logging) {
    spdlog::shutdown();
  }

  initialized_ = false;
}

void LogManager::update_config(const LogConfig& config) {
  config_ = config;

  // 重新创建sinks
  create_sinks();

  // 更新所有logger
  for (auto& [name, logger] : loggers_) {
    logger->sinks() = sinks_;
    logger->set_level(to_spdlog_level(config_.level));
    logger->set_pattern(config_.log_pattern);
  }
}

spdlog::level::level_enum LogManager::to_spdlog_level(LogLevel level) const {
  switch (level) {
    case LogLevel::TRACE:
      return spdlog::level::trace;
    case LogLevel::DEBUG:
      return spdlog::level::debug;
    case LogLevel::INFO:
      return spdlog::level::info;
    case LogLevel::WARN:
      return spdlog::level::warn;
    case LogLevel::ERROR:
      return spdlog::level::err;
    case LogLevel::CRITICAL:
      return spdlog::level::critical;
    case LogLevel::OFF:
      return spdlog::level::off;
    default:
      return spdlog::level::info;
  }
}

}  // namespace core
}  // namespace deepsearch
