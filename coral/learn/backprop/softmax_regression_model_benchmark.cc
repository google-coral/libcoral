/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <chrono>  // NOLINT(build/c++11)
#include <vector>

#include "benchmark/benchmark.h"
#include "coral/learn/backprop/layers.h"
#include "coral/learn/backprop/softmax_regression_model.h"
#include "coral/learn/backprop/test_utils.h"
#include "glog/logging.h"

namespace coral {

using ::Eigen::MatrixXf;
using ::Eigen::VectorXf;

constexpr int kTotalNumTrainingSamples = 1024;
constexpr int kTotalNumValidationSamples = 256;
constexpr int kTotalNumSamples =
    kTotalNumTrainingSamples + kTotalNumValidationSamples;
constexpr int kNumTrainingEpochs = 500;
constexpr int kBatchSize = 100;

template <int NumClass, int FeatureDim>
static void BM_SoftmaxRegressionBackprop(benchmark::State& state) {
  // For latency benchmark purposes, the distribution of training data does not
  // matter. We just use random values.
  const auto& start_time = std::chrono::steady_clock::now();
  const std::vector<int> class_sizes(NumClass, kTotalNumSamples / NumClass);
  const auto training_data = GenerateUniformRandomData(
      class_sizes, FeatureDim, kTotalNumTrainingSamples);
  std::chrono::duration<double, std::milli> time_span =
      std::chrono::steady_clock::now() - start_time;
  LOG(INFO) << "Time (ms) preparing data set:" << time_span.count();

  SoftmaxRegressionModel model(FeatureDim, NumClass);
  TrainConfig train_config = {kNumTrainingEpochs, kBatchSize,
                              /*print_every=*/-1};
  while (state.KeepRunning()) {
    model.Train(training_data, train_config, /*learning_rate=*/0.01);
  }
}
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 4, 256);
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 16, 256);
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 4, 1024);
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 16, 1024);

}  // namespace coral
