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

#include <memory>
#include <random>
#include <thread>  // NOLINT
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "coral/error_reporter.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/internal/default_allocator.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/pipeline/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

ABSL_FLAG(std::string, model_names, "", "Comma separated list of model names");

namespace coral {
namespace {

std::string GetModelSegmentBaseName(const std::string& model_base_name) {
  static constexpr char kPipelinedModelPrefix[] = "pipeline/";
  // It assumes that segments are under folder
  // <holistic-model-folder>/pipeline/.
  std::vector<std::string> fields = absl::StrSplit(model_base_name, '/');
  fields.insert(fields.end() - 1, kPipelinedModelPrefix);
  return absl::StrJoin(fields, "/");
}

// Tests all supported models with different number of segments.
//
// The test parameter is number of segments a model is partitioned into.
class PipelinedModelsTest : public MultipleEdgeTpuCacheTestBase,
                            public ::testing::WithParamInterface<int> {
 public:
  static void SetUpTestSuite() {
    CHECK(!absl::GetFlag(FLAGS_model_names).empty());

    input_tensors_map_ =
        new std::unordered_map<std::string, std::vector<PipelineTensor>>();

    ref_results_map_ =
        new absl::node_hash_map<std::string, std::vector<PipelineTensor>>();

    ref_tensor_allocator_ = new internal::DefaultAllocator();
  }

  static void TearDownTestSuite() {
    if (input_tensors_map_) {
      for (const auto& tensors : *input_tensors_map_) {
        for (const auto& tensor : tensors.second) {
          ref_tensor_allocator_->Free(tensor.buffer);
        }
      }
      delete input_tensors_map_;
    }

    if (ref_results_map_) {
      for (const auto& tensors : *ref_results_map_) {
        for (const auto& tensor : tensors.second) {
          ref_tensor_allocator_->Free(tensor.buffer);
        }
      }
      delete ref_results_map_;
    }

    delete ref_tensor_allocator_;
  }

  void SetUp() override {
    num_segments_ = GetParam();

    tpu_contexts_ = GetTpuContextCache(num_segments_);
    CHECK_EQ(tpu_contexts_.size(), num_segments_);

    model_list_ = absl::StrSplit(absl::GetFlag(FLAGS_model_names), ',');
    for (const auto& model_base_name : model_list_) {
      CHECK(!model_base_name.empty());
      // Calculate reference results.
      if (ref_results_map_->find(model_base_name) == ref_results_map_->end()) {
        // Construct tflite interpreter.
        const auto model_name =
            absl::StrCat(model_base_name, "_edgetpu.tflite");
        auto model = tflite::FlatBufferModel::BuildFromFile(
            TestDataPath(model_name).c_str());
        EdgeTpuErrorReporter error_reporter;
        auto interpreter =
            CreateInterpreter(*model, tpu_contexts_[0], &error_reporter);

        // Setup input tensors.
        const auto input_tensors =
            CreateRandomInputTensors(interpreter.get(), ref_tensor_allocator_);
        input_tensors_map_->insert({model_base_name, input_tensors});
        for (int i = 0; i < interpreter->inputs().size(); ++i) {
          auto* tensor = interpreter->input_tensor(i);
          std::memcpy(tensor->data.data,
                      CHECK_NOTNULL(input_tensors[i].buffer->ptr()),
                      input_tensors[i].bytes);
        }

        CHECK(interpreter->Invoke() == kTfLiteOk) << error_reporter.message();

        // Record reference results.
        std::vector<PipelineTensor> ref_results(interpreter->outputs().size());
        for (int i = 0; i < interpreter->outputs().size(); ++i) {
          auto* tensor = interpreter->output_tensor(i);
          ref_results[i].name = tensor->name;
          ref_results[i].buffer = ref_tensor_allocator_->Alloc(tensor->bytes);
          std::memcpy(CHECK_NOTNULL(ref_results[i].buffer->ptr()),
                      tensor->data.data, tensor->bytes);
          ref_results[i].bytes = tensor->bytes;
          ref_results[i].type = tensor->type;
        }
        ref_results_map_->insert({model_base_name, ref_results});
      }
    }
  }

 protected:
  // Checks that the actual results and expected results contain the same set of
  // tensors. However, the tensors may be in different orders.
  void CheckSameTensors(const std::vector<PipelineTensor>& actual_tensors,
                        const std::vector<PipelineTensor>& expected_tensors) {
    ASSERT_EQ(actual_tensors.size(), expected_tensors.size());
    for (const auto& expected : expected_tensors) {
      bool found = false;
      for (const auto& actual : actual_tensors) {
        if (actual.name == expected.name) {
          found = true;
          ASSERT_EQ(actual.type, expected.type);
          ASSERT_EQ(actual.bytes, expected.bytes);
          const auto* actual_data = CHECK_NOTNULL(
              reinterpret_cast<const uint8_t*>(actual.buffer->ptr()));
          const auto* expected_data = CHECK_NOTNULL(
              reinterpret_cast<const uint8_t*>(expected.buffer->ptr()));
          for (int j = 0; j < expected.bytes; ++j) {
            EXPECT_EQ(actual_data[j], expected_data[j]) << "Element " << j;
          }
          break;
        }
      }
      ASSERT_TRUE(found) << "Can not find tensor in results: " << expected.name;
    }
  }

  // List of models to test.
  std::vector<std::string> model_list_;

  // Key is `model_list_[i]`, value is input tensors. This map makes sure that
  // the same input tensors are used when running the model with no partition, 2
  // partitins, 3 partitions, and so on.
  static std::unordered_map<std::string, std::vector<PipelineTensor>>*
      input_tensors_map_;

  // Key is `model_list_[i]`, value is output tensors from non partitioned
  // model, recorded here as reference results.
  static absl::node_hash_map<std::string, std::vector<PipelineTensor>>*
      ref_results_map_;

  static internal::DefaultAllocator* ref_tensor_allocator_;

  std::vector<edgetpu::EdgeTpuContext*> tpu_contexts_;
  int num_segments_;

  std::unique_ptr<PipelinedModelRunner> runner_;
};

std::unordered_map<std::string, std::vector<PipelineTensor>>*
    PipelinedModelsTest::input_tensors_map_ = nullptr;
absl::node_hash_map<std::string, std::vector<PipelineTensor>>*
    PipelinedModelsTest::ref_results_map_ = nullptr;
internal::DefaultAllocator* PipelinedModelsTest::ref_tensor_allocator_ =
    nullptr;

TEST_P(PipelinedModelsTest, CheckInferenceResult) {
  for (const auto& model_base_name : model_list_) {
    LOG(INFO) << "Testing " << model_base_name << " with " << num_segments_
              << " segments.";
    const auto& output_tensors = RunInferenceWithPipelinedModel(
        GetModelSegmentBaseName(model_base_name), num_segments_,
        (*input_tensors_map_)[model_base_name], tpu_contexts_, runner_);
    const auto& expected_tensors = (*ref_results_map_)[model_base_name];
    CheckSameTensors(output_tensors, expected_tensors);
    FreePipelineTensors(output_tensors, runner_->GetOutputTensorAllocator());
  }
}

TEST_P(PipelinedModelsTest, RepeatabilityTest) {
  constexpr int kNumRuns = 10;
  for (const auto& model_base_name : model_list_) {
    LOG(INFO) << "Testing " << model_base_name << " with " << num_segments_
              << " segments.";
    const auto& expected_tensors = RunInferenceWithPipelinedModel(
        GetModelSegmentBaseName(model_base_name), num_segments_,
        (*input_tensors_map_)[model_base_name], tpu_contexts_, runner_);
    for (int i = 0; i < kNumRuns; ++i) {
      const auto& output_tensors = RunInferenceWithPipelinedModel(
          GetModelSegmentBaseName(model_base_name), num_segments_,
          (*input_tensors_map_)[model_base_name], tpu_contexts_, runner_);
      CheckSameTensors(output_tensors, expected_tensors);
      FreePipelineTensors(output_tensors, runner_->GetOutputTensorAllocator());
    }
    FreePipelineTensors(expected_tensors, runner_->GetOutputTensorAllocator());
  }
}

INSTANTIATE_TEST_CASE_P(PipelinedModelsTest, PipelinedModelsTest,
                        ::testing::ValuesIn(NumSegments()));

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
