#pragma once

#include <spdlog/async.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace deepsearch {
namespace core {

// 日志级别枚举
enum class LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  CRITICAL = 5,
  OFF = 6
};

// 日志配置结构
struct LogConfig {
  LogLevel level = LogLevel::INFO;
  bool enable_console = true;
  bool enable_file = true;
  std::string log_dir = "logs";
  std::string log_pattern = "[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] [%t] %v";
  size_t max_file_size = 10 * 1024 * 1024;  // 10MB
  size_t max_files = 5;
  bool async_logging = true;
  size_t async_queue_size = 8192;
  size_t thread_pool_size = 1;
};

// 日志管理器类
class LogManager {
 public:
  static LogManager& instance();

  // 初始化日志系统
  void initialize(const LogConfig& config = LogConfig{});

  // 获取或创建logger
  std::shared_ptr<spdlog::logger> get_logger(
      const std::string& name = "default");

  // 设置全局日志级别
  void set_level(LogLevel level);

  // 设置模块日志级别
  void set_module_level(const std::string& module, LogLevel level);

  // 刷新所有日志
  void flush_all();

  // 关闭日志系统
  void shutdown();

  // 配置管理
  void update_config(const LogConfig& config);
  const LogConfig& get_config() const { return config_; }

 private:
  LogManager() = default;
  ~LogManager() {
    // 在析构函数中安全地清理
    try {
      if (initialized_) {
        shutdown_internal();
      }
    } catch (...) {
      // 忽略析构时的异常
    }
  }

  void shutdown_internal() {
    if (!initialized_) {
      return;
    }

    // 首先停止所有异步操作
    if (config_.async_logging) {
      // 等待异步操作完成
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 清理logger，避免异步flush
    for (auto& [name, logger] : loggers_) {
      if (logger) {
        try {
          // 同步flush，避免异步操作
          logger->flush();
        } catch (...) {
          // 忽略flush错误
        }
      }
    }

    // 清理资源
    spdlog::set_default_logger(nullptr);
    loggers_.clear();
    sinks_.clear();

    // 最后关闭线程池
    if (config_.async_logging) {
      try {
        spdlog::shutdown();
      } catch (...) {
        // 忽略shutdown错误
      }
    }

    initialized_ = false;
  }

  void create_sinks();
  spdlog::level::level_enum to_spdlog_level(LogLevel level) const;

  LogConfig config_;
  std::vector<spdlog::sink_ptr> sinks_;
  std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> loggers_;
  bool initialized_ = false;
};

class LoggerGuard {
 public:
  explicit LoggerGuard(const LogConfig& config = LogConfig{}) {
    auto& logger = LogManager::instance();
    logger.initialize(config);
  }

  ~LoggerGuard() {
    auto& logger = LogManager::instance();
    logger.shutdown();
  }

  LoggerGuard(const LoggerGuard&) = delete;
  LoggerGuard& operator=(const LoggerGuard&) = delete;
};

// 便利宏定义
#define LOG_TRACE(logger, ...) SPDLOG_LOGGER_TRACE(logger, __VA_ARGS__)
#define LOG_DEBUG(logger, ...) SPDLOG_LOGGER_DEBUG(logger, __VA_ARGS__)
#define LOG_INFO(logger, ...) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__)
#define LOG_WARN(logger, ...) SPDLOG_LOGGER_WARN(logger, __VA_ARGS__)
#define LOG_ERROR(logger, ...) SPDLOG_LOGGER_ERROR(logger, __VA_ARGS__)
#define LOG_CRITICAL(logger, ...) SPDLOG_LOGGER_CRITICAL(logger, __VA_ARGS__)

// 默认logger宏
#define DLOG_TRACE(...) \
  LOG_TRACE(deepsearch::core::LogManager::instance().get_logger(), __VA_ARGS__)
#define DLOG_DEBUG(...) \
  LOG_DEBUG(deepsearch::core::LogManager::instance().get_logger(), __VA_ARGS__)
#define DLOG_INFO(...) \
  LOG_INFO(deepsearch::core::LogManager::instance().get_logger(), __VA_ARGS__)
#define DLOG_WARN(...) \
  LOG_WARN(deepsearch::core::LogManager::instance().get_logger(), __VA_ARGS__)
#define DLOG_ERROR(...) \
  LOG_ERROR(deepsearch::core::LogManager::instance().get_logger(), __VA_ARGS__)
#define DLOG_CRITICAL(...)                                            \
  LOG_CRITICAL(deepsearch::core::LogManager::instance().get_logger(), \
               __VA_ARGS__)

// 模块化日志宏
#define MODULE_LOGGER(module) \
  deepsearch::core::LogManager::instance().get_logger(#module)
#define MLOG_TRACE(module, ...) LOG_TRACE(MODULE_LOGGER(module), __VA_ARGS__)
#define MLOG_DEBUG(module, ...) LOG_DEBUG(MODULE_LOGGER(module), __VA_ARGS__)
#define MLOG_INFO(module, ...) LOG_INFO(MODULE_LOGGER(module), __VA_ARGS__)
#define MLOG_WARN(module, ...) LOG_WARN(MODULE_LOGGER(module), __VA_ARGS__)
#define MLOG_ERROR(module, ...) LOG_ERROR(MODULE_LOGGER(module), __VA_ARGS__)
#define MLOG_CRITICAL(module, ...) \
  LOG_CRITICAL(MODULE_LOGGER(module), __VA_ARGS__)

}  // namespace core
}  // namespace deepsearch
