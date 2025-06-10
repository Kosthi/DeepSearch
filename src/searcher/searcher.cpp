#include "searcher.h"

#include "quantization/fp32_quant.h"
#include "quantization/sq4_quant.h"
#include "quantization/sq8_quant.h"

namespace deepsearch {
namespace searcher {

std::unique_ptr<FP32Searcher> SearcherFactory::createFP32(const Graph& graph,
                                                          DistanceType metric,
                                                          size_t dimension) {
  auto quantizer = std::make_unique<FP32Quantizer>(metric, dimension);
  return std::make_unique<FP32Searcher>(graph, std::move(quantizer));
}

std::unique_ptr<SQ8Searcher> SearcherFactory::createSQ8(const Graph& graph,
                                                        DistanceType metric,
                                                        size_t dimension) {
  auto fp32_quantizer = std::make_unique<FP32Quantizer>(metric, dimension);
  auto quantizer = std::make_unique<SQ8Quantizer>(metric, dimension,
                                                  std::move(fp32_quantizer));
  return std::make_unique<SQ8Searcher>(graph, std::move(quantizer));
}

std::unique_ptr<SQ4Searcher> SearcherFactory::createSQ4(const Graph& graph,
                                                        DistanceType metric,
                                                        size_t dimension) {
  auto fp32_quantizer = std::make_unique<FP32Quantizer>(metric, dimension);
  auto quantizer =
      std::make_unique<SQ4Quantizer>(metric, dimension, std::move(fp32_quantizer));
  return std::make_unique<SQ4Searcher>(graph, std::move(quantizer));
}

// 显式实例化常用类型
template class Searcher<quantization::FP32Quantizer>;
template class Searcher<quantization::SQ8Quantizer>;
template class Searcher<quantization::SQ4Quantizer>;

}  // namespace searcher
}  // namespace deepsearch
