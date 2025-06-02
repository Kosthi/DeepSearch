#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "graph/builder_factory.h"
#include "searcher/searcher.h"

using namespace deepsearch::core;
using namespace deepsearch::graph;
using namespace deepsearch::searcher;
using namespace deepsearch::quantization;

template <typename T>
void load_fvecs(const char* filename, T*& p, int64_t& n, int64_t& dim) {
  std::ifstream fs(filename, std::ios::binary);
  int dim_32;
  fs.read((char*)&dim_32, 4);
  dim = dim_32;
  fs.seekg(0, std::ios::end);
  n = fs.tellg() / (4 + dim * sizeof(T));
  fs.seekg(0, std::ios::beg);
  std::cout << "Read path: " << filename << ", nx: " << n << ", dim: " << dim
            << std::endl;
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
  load_fvecs(base_path.c_str(), base, N, dim);
  load_fvecs(query_path.c_str(), query, nq, dim);
  load_fvecs(gt_path.c_str(), gt, nq, gt_k);

  BuilderConfig config;
  config.M = 16;
  config.ef_construction = 200;
  config.max_elements = N;

  auto hnsw_builder = BuilderFactory<float>::create(
      BuilderType::HNSW, DistanceType::L2, dim, config);
  if (!std::filesystem::exists(graph_path)) {
    auto graph = hnsw_builder->build(base, N, dim);
    graph.save(graph_path);
  }
  Graph graph;
  graph.load(graph_path);

  // create quantizer
  auto quantizer = std::make_unique<FP32Quantizer>(DistanceType::L2, dim);
  // create searcher
  auto searcher =
      SearcherFactory::create<FP32Quantizer>(graph, std::move(quantizer));
  searcher->SetData(base, N, dim);
  searcher->Optimize(num_threads);
  searcher->SetEf(search_ef);

  double recall;
  double best_qps = 0.0;
  for (int iter = 1; iter <= iters; ++iter) {
    printf("iter : [%d/%d]\n", iter, iters);
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
    printf("\tRecall@%d = %.4lf, QPS = %.2lf\n", topk, recall, qps);
  }
  printf("Best QPS = %.2lf\n", best_qps);
  free(base);
  free(query);
  free(gt);
}
