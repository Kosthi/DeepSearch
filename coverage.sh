#!/bin/bash

set -e

echo "=== DeepSearch 代码覆盖率测试 ==="

# 清理之前的构建
echo "清理构建目录..."
rm -rf build_coverage
mkdir build_coverage
cd build_coverage

# 配置 CMake 启用覆盖率
echo "配置项目（启用覆盖率）..."
cmake .. -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON

# 构建项目
echo "构建项目..."
cmake --build . --config Debug -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# 下载测试数据（如果不存在）
if [ ! -d "../siftsmall" ]; then
    echo "下载测试数据..."
    cd ..
    wget -q -O siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
    tar -xzf siftsmall.tar.gz
    cd build_coverage
fi

# 初始化覆盖率计数器
echo "初始化覆盖率计数器..."
lcov --capture --initial --directory . --output-file coverage_base.info \
  --ignore-errors inconsistent,source,unsupported,format,unused \
  --exclude "*/build_coverage/_deps/*" \
  --exclude "*/third_party/*" \
  --exclude "*/tests/*" \
  --exclude "*/test/*"

# 运行所有单元测试
echo "运行单元测试..."
ctest --output-on-failure --verbose

# 运行集成测试
echo "运行集成测试..."
./DeepSearch ../siftsmall/siftsmall_base.fvecs ../siftsmall/siftsmall_query.fvecs ../siftsmall/siftsmall_groundtruth.ivecs graph.index 1 10 120 0

# 捕获测试后的覆盖率数据
echo "捕获覆盖率数据..."
lcov --capture --directory . --output-file coverage_test.info \
  --ignore-errors inconsistent,source,unsupported,format,unused \
  --exclude "*/build_coverage/_deps/*" \
  --exclude "*/third_party/*" \
  --exclude "*/tests/*" \
  --exclude "*/test/*"

# 合并基线和测试覆盖率数据
echo "合并覆盖率数据..."
lcov --add-tracefile coverage_base.info --add-tracefile coverage_test.info \
  --output-file coverage_combined.info \
  --ignore-errors inconsistent,source,unsupported,format,unused

# 过滤不需要的文件
echo "过滤覆盖率数据..."
lcov --remove coverage_combined.info \
  "*/build_coverage/_deps/*" \
  "*/third_party/*" \
  "*/tests/*" \
  "*/test/*" \
  "*/usr/*" \
  "*/opt/*" \
  --output-file coverage_filtered.info \
  --ignore-errors inconsistent,source,unsupported,format,unused

# 显示覆盖率摘要
echo "\n=== 覆盖率摘要 ==="
lcov --list coverage_filtered.info --ignore-errors inconsistent,source,unsupported,format,unused

# 生成 HTML 报告
echo "\n生成 HTML 报告..."
genhtml coverage_filtered.info --output-directory coverage_html --title "DeepSearch Coverage Report" --show-details --legend --ignore-errors inconsistent,source,unsupported,format,unused,category

echo "\n=== 覆盖率报告生成完成 ==="
echo "HTML 报告位置: build_coverage/coverage_html/index.html"
echo "在浏览器中打开: open build_coverage/coverage_html/index.html"
echo "\n主要覆盖率文件:"
echo "  - coverage_filtered.info: 过滤后的覆盖率数据"
echo "  - coverage_html/: HTML 报告目录"
