#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "core/logger.h"
#include "graph/builder_factory.h"
#include "searcher/searcher.h"

using namespace deepsearch::core;
using namespace deepsearch::graph;
using namespace deepsearch::searcher;
using namespace deepsearch::quantization;

template <typename T>
void load_fvecs(const char* filename, T*& p, int64_t& n, int64_t& dim) {
  auto logger = LogManager::instance().get_logger("load_fvecs");
  std::ifstream fs(filename, std::ios::binary);
  int dim_32;
  fs.read((char*)&dim_32, 4);
  dim = dim_32;
  fs.seekg(0, std::ios::end);
  n = fs.tellg() / (4 + dim * sizeof(T));
  fs.seekg(0, std::ios::beg);
  logger->info("Read path: {}, nx: {}, dim: {}", filename, n, dim);
  p = reinterpret_cast<T*>(aligned_alloc(64, n * dim * sizeof(T)));
  for (int i = 0; i < n; ++i) {
    fs.seekg(4, std::ios::cur);
    fs.read((char*)&p[i * dim], dim * sizeof(T));
  }
}

int main(int argc, char** argv) {
  if (argc < 8) {
    printf(
        "Usage: ./main base_path query_path gt_path graph_path level "
        "topk search_ef num_threads\n");
    exit(-1);
  }

  // 使用RAII管理日志系统生命周期
  LogConfig logConfig;
  LoggerGuard logger_guard(logConfig);

  auto logger = LogManager::instance().get_logger("main");

  std::string base_path = argv[1];
  std::string query_path = argv[2];
  std::string gt_path = argv[3];
  std::string graph_path = argv[4];
  int level = std::stoi(argv[5]);
  int topk = std::stoi(argv[6]);
  int search_ef = std::stoi(argv[7]);
  int num_threads = 1;
  int iters = 10;
  if (argc >= 9) {
    num_threads = std::stoi(argv[8]);
  }
  if (argc >= 10) {
    iters = std::stoi(argv[9]);
  }
  float *base, *query;
  int* gt;
  int64_t N, dim, nq, gt_k;

  logger->info("Loading data from: base={}, query={}, gt={}", base_path,
               query_path, gt_path);

  load_fvecs(base_path.c_str(), base, N, dim);
  load_fvecs(query_path.c_str(), query, nq, dim);
  load_fvecs(gt_path.c_str(), gt, nq, gt_k);

  logger->info("Data loaded: N={}, dim={}, nq={}, gt_k={}", N, dim, nq, gt_k);

  BuilderConfig config;
  config.M = 16;
  config.ef_construction = 200;
  config.max_elements = N;

  logger->info(
      "Building HNSW with config: M={}, ef_construction={}, max_elements={}",
      config.M, config.ef_construction, config.max_elements);

  auto hnsw_builder = BuilderFactory<float>::create(
      BuilderType::HNSW, DistanceType::L2, dim, config);
  if (!std::filesystem::exists(graph_path)) {
    auto graph = hnsw_builder->build(base, N, dim);
    graph.save(graph_path);
    logger->info("Graph built and saved to: {}", graph_path);
  } else {
    logger->info("Loading existing graph from: {}", graph_path);
  }
  Graph graph;
  graph.load(graph_path);

  // create quantizer
  auto fp32_quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim);
  auto sq4_quantizer = std::make_unique<SQ4Quantizer>(
      DistanceType::L2, dim, std::move(fp32_quantizer));
  // create searcher
  auto searcher =
      SearcherFactory::create<SQ4Quantizer>(graph, std::move(sq4_quantizer));
  searcher->SetData(base, N, dim);
  searcher->Optimize(num_threads);
  searcher->SetEf(search_ef);

  logger->info("Starting search with {} threads, ef={}, topk={}, iterations={}",
               num_threads, search_ef, topk, iters);

  double recall;
  double best_qps = 0.0;
  for (int iter = 1; iter <= iters; ++iter) {
    logger->info("Running iteration [{}/{}]", iter, iters);
    std::vector<int> pred(nq * topk);
    auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < nq; ++i) {
      searcher->Search(query + i * dim, topk, pred.data() + i * topk);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    double qps = nq / ela;
    best_qps = std::max(qps, best_qps);
    int cnt = 0;

    // for (int i = 0; i < 10; ++i) {
    //   printf("Query %d:\n", i);
    //
    //   printf("  Pred: ");
    //   for (int j = 0; j < topk; ++j) {
    //     int pred_id = pred[i * topk + j];
    //     printf("%d ", pred_id);
    //   }
    //   printf("\n");
    //
    //   printf("  GT:   ");
    //   for (int j = 0; j < topk; ++j) {
    //     int gt_id = gt[i * gt_k + j];
    //     printf("%d ", gt_id);
    //   }
    //   printf("\n");
    // }

    for (int i = 0; i < nq; ++i) {
      std::unordered_set<int> st(gt + i * gt_k, gt + i * gt_k + topk);
      for (int j = 0; j < topk; ++j) {
        if (st.count(pred[i * topk + j])) {
          cnt++;
        }
      }
    }
    recall = (double)cnt / nq / topk;
    logger->info("Iteration {} results: Recall@{}={:.4f}, QPS={:.2f}", iter,
                 topk, recall, qps);
  }
  logger->info("Search completed. Best QPS: {:.2f}", best_qps);

  free(base);
  free(query);
  free(gt);
}
