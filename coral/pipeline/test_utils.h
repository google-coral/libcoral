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

#ifndef LIBCORAL_CORAL_PIPELINE_TEST_UTILS_H_
#define LIBCORAL_CORAL_PIPELINE_TEST_UTILS_H_

#include <string>
#include <vector>

#include "coral/pipeline/allocator.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {

#ifdef __arm__
static constexpr int kNumEdgeTpuAvailable = 2;
#else
static constexpr int kNumEdgeTpuAvailable = 4;
#endif

inline std::vector<int> NumSegments() {
  // `result` looks like 2, 3, ..., kNumEdgeTpuAvailable.
  std::vector<int> result(kNumEdgeTpuAvailable - 1);
  std::generate(result.begin(), result.end(),
                [n = 2]() mutable { return n++; });
  return result;
}

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* context,
    tflite::StatefulErrorReporter* error_reporter,
    bool allocate_tensors = true);

// Constructs model segments' names based on base name and number of segments.
std::vector<std::string> SegmentsNames(const std::string& model_base_name,
                                       int num_segments);

// If `allocator` is not provided, it uses std::malloc to allocate tensors.
std::vector<PipelineTensor> CreateRandomInputTensors(
    const tflite::Interpreter* interpreter, Allocator* allocator = nullptr);

std::vector<PipelineTensor> RunInferenceWithPipelinedModel(
    const std::string& model_segment_base_name, int num_segments,
    const std::vector<PipelineTensor>& input_tensors,
    std::vector<edgetpu::EdgeTpuContext*>& edgetpu_resources,
    std::unique_ptr<PipelinedModelRunner>& pipeline_runner);

}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_TEST_UTILS_H_
