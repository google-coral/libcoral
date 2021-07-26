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

// Tool to do simple performance analysis when using multiple Edge TPU devices.
//
// Basically, it tries to run `num_inferences` inferences with 1, 2, ...,
// [Max # of Edge TPUs] available on host; and record the wall time.
//
// It does this for each model and reports speedup in the end.
//
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance

#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tflite/public/edgetpu.h"

ABSL_FLAG(int, num_requests, 30000, "Number of inference requests to run.");

namespace coral {

// Returns processing wall time in milliseconds.
double ProcessRequests(const std::string& model_name, int num_threads,
                       int num_requests) {
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  // Divide work among different threads, round up a bit if not divisible.
  int num_requests_per_thread = (num_requests + num_threads - 1) / num_threads;

  const auto start_time = std::chrono::steady_clock::now();
  for (int k = 0; k < num_threads; ++k) {
    workers.emplace_back([&, k]() {
      const auto tid = std::this_thread::get_id();
      LOG(INFO) << "thread: " << tid
                << ", # requests need to process: " << num_requests_per_thread
                << ", device: " << k;
      auto model = LoadModelOrDie(TestDataPath(model_name));
      auto tpu_context = GetEdgeTpuContextOrDie(absl::nullopt, k);
      auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
      CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
      FillRandomInt(MutableTensorData<uint8_t>(*interpreter->input_tensor(0)));

      for (int i = 0; i < num_requests_per_thread; ++i)
        CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

      LOG(INFO) << "thread: " << tid << " finished processing requests.";
    });
  }
  for (auto& worker : workers) worker.join();

  return (std::chrono::steady_clock::now() - start_time).count();
}
}  // namespace coral

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const int num_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu().size();
  CHECK_GT(num_tpus, 1) << "Need > 1 Edge TPU for the run to be meaningful";
  LOG(INFO) << "Number of TPUs detected: " << num_tpus;

  const std::vector<std::string> models_to_check = {
      "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
      "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
      "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
      "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
      "inception_v1_224_quant_edgetpu.tflite",
      "inception_v2_224_quant_edgetpu.tflite",
      "inception_v3_299_quant_edgetpu.tflite",
      "inception_v4_299_quant_edgetpu.tflite",
  };

  auto print_speedup = [](const std::vector<double>& time_vec) {
    CHECK_GT(time_vec.size(), 1);
    LOG(INFO) << "Single Edge TPU base time " << time_vec[0] << " seconds.";
    for (int i = 1; i < time_vec.size(); ++i) {
      LOG(INFO) << "# TPUs: " << (i + 1)
                << " speedup: " << time_vec[0] / time_vec[i];
    }
  };

  std::map<std::string, std::vector<double>> processing_time_map;
  for (const auto& model_name : models_to_check) {
    auto& time_vec = processing_time_map[model_name];
    time_vec.resize(num_tpus);
    // Run with max number of Edge TPUs first on purpose, otherwise, it can take
    // a long time for user to realize there is not enough Edge TPUs on host.
    for (int i = num_tpus - 1; i >= 0; --i) {
      time_vec[i] = coral::ProcessRequests(model_name,
                                           /*num_threads=*/(i + 1),
                                           absl::GetFlag(FLAGS_num_requests));
      LOG(INFO) << "Model name: " << model_name << " # TPUs: " << (i + 1)
                << " processing time: " << time_vec[i];
    }
    print_speedup(time_vec);
  }

  LOG(INFO) << "===========Summary=============";
  for (const auto& model_name : models_to_check) {
    LOG(INFO) << "----------------------";
    LOG(INFO) << "Model name: " << model_name;
    const auto& time_vec = processing_time_map[model_name];
    print_speedup(time_vec);
  }
  return 0;
}
