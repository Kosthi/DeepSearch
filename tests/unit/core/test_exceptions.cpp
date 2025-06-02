#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "core/exceptions.h"

using namespace deepsearch::core;

TEST_CASE("DeepSearchException basic functionality", "[exceptions][basic]") {
  const std::string test_message = "Test exception message";
  const std::string test_file = "test_file.cpp";
  const int test_line = 42;

  SECTION("Constructor and getters") {
    DeepSearchException exc(test_message, test_file, test_line);

    REQUIRE(exc.message() == test_message);
    REQUIRE(exc.file() == test_file);
    REQUIRE(exc.line() == test_line);
    REQUIRE(std::string(exc.what()) == test_message);
  }

  SECTION("Copy constructor and assignment") {
    DeepSearchException original(test_message, test_file, test_line);

    // 拷贝构造
    DeepSearchException copied(original);
    REQUIRE(copied.message() == original.message());
    REQUIRE(copied.file() == original.file());
    REQUIRE(copied.line() == original.line());

    // 赋值操作
    DeepSearchException assigned("temp", "temp.cpp", 1);
    assigned = original;
    REQUIRE(assigned.message() == original.message());
    REQUIRE(assigned.file() == original.file());
    REQUIRE(assigned.line() == original.line());
  }
}

TEST_CASE("Specific exception types", "[exceptions][types]") {
  SECTION("InvalidParameterException") {
    const std::string param_name = "invalid_param";

    REQUIRE_THROWS_AS(throw InvalidParameterException(param_name),
                      InvalidParameterException);

    try {
      throw InvalidParameterException(param_name, "test.cpp", 100);
    } catch (const InvalidParameterException& e) {
      REQUIRE(e.message() == "Invalid parameter: " + param_name);
      REQUIRE(e.file() == "test.cpp");
      REQUIRE(e.line() == 100);
    }
  }

  SECTION("FileIOException") {
    const std::string filename = "nonexistent_file.txt";

    REQUIRE_THROWS_AS(throw FileIOException(filename), FileIOException);

    try {
      throw FileIOException(filename, "io_test.cpp", 200);
    } catch (const FileIOException& e) {
      REQUIRE(e.message() == "File I/O error: " + filename);
      REQUIRE(e.file() == "io_test.cpp");
      REQUIRE(e.line() == 200);
    }
  }

  SECTION("MemoryException") {
    const std::string operation = "aligned_alloc";

    REQUIRE_THROWS_AS(throw MemoryException(operation), MemoryException);

    try {
      throw MemoryException(operation, "memory_test.cpp", 300);
    } catch (const MemoryException& e) {
      REQUIRE(e.message() == "Memory error in: " + operation);
      REQUIRE(e.file() == "memory_test.cpp");
      REQUIRE(e.line() == 300);
    }
  }

  SECTION("IndexException") {
    const std::string error_msg = "Index out of bounds";

    REQUIRE_THROWS_AS(throw IndexException(error_msg), IndexException);

    try {
      throw IndexException(error_msg, "index_test.cpp", 400);
    } catch (const IndexException& e) {
      REQUIRE(e.message() == "Index error: " + error_msg);
      REQUIRE(e.file() == "index_test.cpp");
      REQUIRE(e.line() == 400);
    }
  }
}

TEST_CASE("Exception macros", "[exceptions][macros]") {
  SECTION("THROW_INVALID_PARAM") {
    REQUIRE_THROWS_AS(THROW_INVALID_PARAM("test_param"),
                      InvalidParameterException);

    try {
      THROW_INVALID_PARAM("test_param");
    } catch (const InvalidParameterException& e) {
      REQUIRE(e.message().find("test_param") != std::string::npos);
      REQUIRE_FALSE(e.file().empty());
      REQUIRE(e.line() > 0);
    }
  }

  SECTION("THROW_FILE_IO_ERROR") {
    REQUIRE_THROWS_AS(THROW_FILE_IO_ERROR("test_file.dat"), FileIOException);
  }

  SECTION("THROW_MEMORY_ERROR") {
    REQUIRE_THROWS_AS(THROW_MEMORY_ERROR("allocation_failed"), MemoryException);
  }

  SECTION("THROW_INDEX_ERROR") {
    REQUIRE_THROWS_AS(THROW_INDEX_ERROR("bounds_check_failed"), IndexException);
  }
}

TEST_CASE("Exception polymorphism", "[exceptions][polymorphism]") {
  std::vector<std::unique_ptr<DeepSearchException>> exceptions;

  exceptions.push_back(std::make_unique<InvalidParameterException>("param"));
  exceptions.push_back(std::make_unique<FileIOException>("file.txt"));
  exceptions.push_back(std::make_unique<MemoryException>("alloc"));
  exceptions.push_back(std::make_unique<IndexException>("bounds"));

  for (const auto& exc : exceptions) {
    REQUIRE_FALSE(exc->message().empty());
    REQUIRE_FALSE(std::string(exc->what()).empty());
  }
}

TEST_CASE("Exception edge cases", "[exceptions][edge]") {
  SECTION("Empty parameters") {
    DeepSearchException empty_msg("", "file.cpp", 100);
    REQUIRE(empty_msg.message().empty());
    REQUIRE(empty_msg.file() == "file.cpp");
    REQUIRE(empty_msg.line() == 100);

    DeepSearchException empty_file("message", "", 200);
    REQUIRE(empty_file.message() == "message");
    REQUIRE(empty_file.file().empty());
    REQUIRE(empty_file.line() == 200);

    DeepSearchException zero_line("message", "file.cpp", 0);
    REQUIRE(zero_line.message() == "message");
    REQUIRE(zero_line.file() == "file.cpp");
    REQUIRE(zero_line.line() == 0);
  }
}
