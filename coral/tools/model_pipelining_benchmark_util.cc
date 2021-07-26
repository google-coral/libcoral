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

// Utils to analyze model pipelining performance.
#include "coral/tools/model_pipelining_benchmark_util.h"

#include "coral/error_reporter.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/pipeline/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace coral {

using edgetpu::EdgeTpuContext;

// num_segments, latency (in ns), latencies for all segments(in ns) tuple.
using PerfStats = std::tuple<int, int64_t, std::vector<int64_t>>;

std::vector<std::shared_ptr<EdgeTpuContext>> PrepareEdgeTpuContexts(
    int num_tpus, EdgeTpuType device_type) {
  auto get_available_tpus = [](EdgeTpuType device_type) {
    const auto& all_tpus =
        edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    if (device_type == EdgeTpuType::kAny) {
      return all_tpus;
    } else {
      edgetpu::DeviceType target_type;
      if (device_type == EdgeTpuType::kPciOnly) {
        target_type = edgetpu::DeviceType::kApexPci;
      } else if (device_type == EdgeTpuType::kUsbOnly) {
        target_type = edgetpu::DeviceType::kApexUsb;
      } else {
        LOG(FATAL) << "Invalid device type";
      }
      std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> result;
      for (const auto& tpu : all_tpus) {
        if (tpu.type == target_type) {
          result.push_back(tpu);
        }
      }
      return result;
    }
  };
  const auto& available_tpus = get_available_tpus(device_type);
  CHECK_GE(available_tpus.size(), num_tpus);

  std::vector<std::shared_ptr<EdgeTpuContext>> edgetpu_contexts(num_tpus);
  for (int i = 0; i < num_tpus; ++i) {
    edgetpu_contexts[i] = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
        available_tpus[i].type, available_tpus[i].path);
    LOG(INFO) << "Device " << available_tpus[i].path << " is selected.";
  }

  return edgetpu_contexts;
}

PerfStats BenchmarkPartitionedModel(
    const std::vector<std::string>& model_segments_paths,
    const std::vector<std::shared_ptr<EdgeTpuContext>>* edgetpu_contexts,
    int num_inferences) {
  CHECK_LE(model_segments_paths.size(), edgetpu_contexts->size());
  const int num_segments = model_segments_paths.size();
  std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
      num_segments);
  std::vector<tflite::Interpreter*> interpreters(num_segments);
  std::vector<EdgeTpuErrorReporter> error_reporters(num_segments);
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    models[i] =
        tflite::FlatBufferModel::BuildFromFile(model_segments_paths[i].c_str());
    managed_interpreters[i] = coral::CreateInterpreter(
        *(models[i]), (*edgetpu_contexts)[i].get(), &error_reporters[i]);
    interpreters[i] = managed_interpreters[i].get();
  }

  // Parameter Caching before pipelining starts running.
  for (int i = 0; i < num_segments; ++i) {
    for (const int input_tensor_index : interpreters[i]->inputs()) {
      auto input_data = MutableTensorData<uint8_t>(
          *interpreters[i]->tensor(input_tensor_index));
      std::memset(input_data.data(), 0, input_data.size());
    }
    CHECK_EQ(interpreters[i]->Invoke(), kTfLiteOk)
        << error_reporters[i].message();
  }

  auto runner = absl::make_unique<PipelinedModelRunner>(interpreters);

  // Generating input tensors can be quite time consuming, pulling them out to
  // avoid polluting measurement of pipelining latency.
  std::vector<std::vector<PipelineTensor>> input_requests(num_inferences);
  for (int i = 0; i < num_inferences; ++i) {
    input_requests[i] = CreateRandomInputTensors(
        interpreters[0], runner->GetInputTensorAllocator());
  }

  auto request_producer = [&runner, &input_requests]() {
    const auto& start_time = std::chrono::steady_clock::now();
    const auto& num_inferences = input_requests.size();
    for (int i = 0; i < num_inferences; ++i) {
      CHECK(runner->Push(input_requests[i]).ok());
    }
    CHECK(runner->Push({}).ok());
    std::chrono::duration<int64_t, std::nano> time_span =
        std::chrono::steady_clock::now() - start_time;
    LOG(INFO) << "Producer thread per request latency (in ns): "
              << time_span.count() / num_inferences;
  };

  auto request_consumer = [&runner, &num_inferences]() {
    const auto& start_time = std::chrono::steady_clock::now();
    std::vector<PipelineTensor> output_tensors;
    while (runner->Pop(&output_tensors).ok() && !output_tensors.empty()) {
      FreePipelineTensors(output_tensors, runner->GetOutputTensorAllocator());
      output_tensors.clear();
    }
    LOG(INFO) << "All tensors consumed";
    std::chrono::duration<int64_t, std::nano> time_span =
        std::chrono::steady_clock::now() - start_time;
    LOG(INFO) << "Consumer thread per request latency (in ns): "
              << time_span.count() / num_inferences;
  };

  const auto& start_time = std::chrono::steady_clock::now();
  auto producer = std::thread(request_producer);
  auto consumer = std::thread(request_consumer);
  producer.join();
  consumer.join();
  std::chrono::duration<int64_t, std::nano> time_span =
      std::chrono::steady_clock::now() - start_time;

  std::vector<int64_t> segments_inference_times;
  for (auto& stats : runner->GetSegmentStats()) {
    segments_inference_times.push_back(stats.total_time_ns /
                                       stats.num_inferences);
  }

  return std::make_tuple(num_segments, time_span.count() / num_inferences,
                         segments_inference_times);
}

}  // namespace coral
