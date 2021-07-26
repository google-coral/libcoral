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

#include "coral/pipeline/test_utils.h"

#include <random>

#include "absl/strings/str_cat.h"
#include "coral/error_reporter.h"
#include "coral/pipeline/allocator.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace coral {
namespace {

// Returns deep copy of a pipeline tensor vector.
std::vector<PipelineTensor> ClonePipelineTensors(
    const std::vector<PipelineTensor>& tensors, Allocator& allocator) {
  std::vector<PipelineTensor> copy(tensors.size());
  for (int i = 0; i < tensors.size(); ++i) {
    copy[i].name = tensors[i].name;
    copy[i].buffer = allocator.Alloc(tensors[i].bytes);
    copy[i].bytes = tensors[i].bytes;
    copy[i].type = tensors[i].type;
    std::memcpy(CHECK_NOTNULL(copy[i].buffer->ptr()),
                CHECK_NOTNULL(tensors[i].buffer->ptr()), tensors[i].bytes);
  }
  return copy;
}

}  // namespace

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* context,
    tflite::StatefulErrorReporter* error_reporter, bool allocate_tensors) {
  CHECK(context);
  CHECK(error_reporter);
  auto interpreter =
      MakeEdgeTpuInterpreterOrDie(model, context, nullptr, error_reporter);
  interpreter->SetNumThreads(1);
  if (allocate_tensors) {
    CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk)
        << error_reporter->message();
  }
  return interpreter;
}

std::vector<std::string> SegmentsNames(const std::string& model_base_name,
                                       int num_segments) {
  std::vector<std::string> result(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    result[i] = absl::StrCat(model_base_name, "_segment_", i, "_of_",
                             num_segments, "_edgetpu.tflite");
  }
  return result;
}

std::vector<PipelineTensor> CreateRandomInputTensors(
    const tflite::Interpreter* interpreter, Allocator* allocator) {
  CHECK(interpreter);
  CHECK(allocator);
  std::vector<PipelineTensor> input_tensors;
  for (int input_index : interpreter->inputs()) {
    const auto* input_tensor = interpreter->tensor(input_index);
    PipelineTensor input_buffer;
    input_buffer.name = input_tensor->name;
    input_buffer.buffer = allocator->Alloc(input_tensor->bytes);
    input_buffer.bytes = input_tensor->bytes;
    input_buffer.type = input_tensor->type;
    auto* ptr = reinterpret_cast<uint8_t*>(input_buffer.buffer->ptr());
    FillRandomInt(ptr, ptr + input_buffer.bytes);
    input_tensors.push_back(input_buffer);
  }
  return input_tensors;
}

std::vector<PipelineTensor> RunInferenceWithPipelinedModel(
    const std::string& model_segment_base_name, int num_segments,
    const std::vector<PipelineTensor>& input_tensors,
    std::vector<edgetpu::EdgeTpuContext*>& edgetpu_resources,
    std::unique_ptr<PipelinedModelRunner>& pipeline_runner) {
  CHECK_GE(edgetpu_resources.size(), num_segments);
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
  std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
      num_segments);
  std::vector<tflite::Interpreter*> interpreters(num_segments);
  std::vector<EdgeTpuErrorReporter> error_reporters(num_segments);
  const auto& segments_names =
      SegmentsNames(model_segment_base_name, num_segments);

  // Construct PipelinedModelRunner.
  for (int i = 0; i < num_segments; ++i) {
    models[i] = tflite::FlatBufferModel::BuildFromFile(
        TestDataPath(segments_names[i]).c_str());
    managed_interpreters[i] = CreateInterpreter(
        *(models[i]), edgetpu_resources[i], &error_reporters[i]);
    interpreters[i] = managed_interpreters[i].get();
  }
  pipeline_runner = absl::make_unique<PipelinedModelRunner>(interpreters);

  // Run inference.
  CHECK(pipeline_runner
            ->Push(ClonePipelineTensors(
                input_tensors, *pipeline_runner->GetInputTensorAllocator()))
            .ok());
  std::vector<PipelineTensor> output_tensors;
  CHECK(pipeline_runner->Pop(&output_tensors).ok());
  return output_tensors;
}

}  // namespace coral
