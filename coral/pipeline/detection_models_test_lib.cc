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

#include "coral/pipeline/detection_models_test_lib.h"

#include <vector>

#include "absl/flags/flag.h"
#include "coral/detection/adapter.h"
#include "coral/error_reporter.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/pipeline/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

ABSL_FLAG(std::string, tpu_device_type, "any",
          "Test Edge TPU device type. Can be 'any', 'pci', or 'usb'");

namespace coral {

internal::DefaultAllocator* PipelinedSsdDetectionModelTest::input_allocator_ =
    nullptr;

void PipelinedSsdDetectionModelTest::SetUpTestSuite() {
  input_allocator_ = new internal::DefaultAllocator();
}

void PipelinedSsdDetectionModelTest::TearDownTestSuite() {
  delete input_allocator_;
}

void PipelinedSsdDetectionModelTest::TestCatMsCocoDetection(
    const std::string& model_segment_base_name, int num_segments,
    float score_threshold, float iou_threshold) {
  auto tpu_contexts = GetTpuContextCache(num_segments);
  CHECK_EQ(tpu_contexts.size(), num_segments);
  LOG(INFO) << "Testing model " << model_segment_base_name << " with "
            << num_segments << " segments...";

  const auto first_segment_name =
      absl::StrCat(model_segment_base_name, "_segment_", 0, "_of_",
                   num_segments, "_edgetpu.tflite");
  auto first_segment = tflite::FlatBufferModel::BuildFromFile(
      TestDataPath(first_segment_name).c_str());
  EdgeTpuErrorReporter error_reporter;
  auto interpreter =
      CreateInterpreter(*first_segment, tpu_contexts[0], &error_reporter);
  CHECK_EQ(interpreter->inputs().size(), 1);

  std::vector<PipelineTensor> input_tensors;
  const auto* input_tensor = interpreter->input_tensor(0);
  CHECK_EQ(input_tensor->type, kTfLiteUInt8);
  PipelineTensor input_buffer;
  input_buffer.name = input_tensor->name;
  input_buffer.buffer = input_allocator_->Alloc(input_tensor->bytes);
  input_buffer.bytes = input_tensor->bytes;
  input_buffer.type = input_tensor->type;
  auto tensor_dims = BrcdShapeToImageDims(TensorShape(*input_tensor));
  const auto input_tensor_data =
      GetInputFromImage(TestDataPath("cat.bmp"), tensor_dims);
  std::memcpy(reinterpret_cast<uint8_t*>(input_buffer.buffer->ptr()),
              input_tensor_data.data(), input_buffer.bytes);
  input_tensors.push_back(input_buffer);

  std::unique_ptr<PipelinedModelRunner> pipeline_runner;
  const auto results = RunInferenceWithPipelinedModel(
      model_segment_base_name, num_segments, input_tensors, tpu_contexts,
      pipeline_runner);
  ASSERT_EQ(results.size(), 4);
  absl::Span<const float> bboxes, ids, scores, count;
  bboxes = TensorData<float>(results[0]);
  ids = TensorData<float>(results[1]);
  scores = TensorData<float>(results[2]);
  count = TensorData<float>(results[3]);
  CHECK_EQ(count.size(), 1);
  const auto objects = GetDetectionResults(
      bboxes, ids, scores, static_cast<int>(count[0]), score_threshold,
      /*top_k=*/1);

  constexpr int kExpectedLabel = 16;
  const BBox<float> expected_box{0.1, 0.1, 1.0, 0.7};
  ASSERT_EQ(objects.size(), 1);
  const auto& object = objects[0];
  EXPECT_EQ(object.id, kExpectedLabel);
  EXPECT_GT(object.score, score_threshold);
  EXPECT_GT(IntersectionOverUnion(object.bbox, expected_box), iou_threshold)
      << "Actual " << ToString(object.bbox) << ", expected "
      << ToString(expected_box);

  FreePipelineTensors(input_tensors, input_allocator_);
  FreePipelineTensors(results, pipeline_runner->GetOutputTensorAllocator());
}

}  // namespace coral
