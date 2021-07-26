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

#include <cmath>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

ABSL_FLAG(int, stress_test_runs, 500, "Number of iterations for stress test.");

namespace coral {
namespace {
class InferenceRepeatabilityTest : public EdgeTpuCacheTestBase {
 protected:
  void RepeatabilityTest(const std::string& model_path) {
    const int runs = absl::GetFlag(FLAGS_stress_test_runs);

    auto model = LoadModelOrDie(TestDataPath(model_path));
    auto interpreter =
        MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextCache());
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    FillRandomInt(MutableTensorData<uint8_t>(*interpreter->input_tensor(0)));
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    const auto num_outputs = interpreter->outputs().size();
    std::vector<std::vector<uint8_t>> reference_tensors(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      auto output = TensorData<uint8_t>(*interpreter->output_tensor(i));
      reference_tensors[i].assign(output.begin(), output.end());
    }

    int error_count = 0;
    for (int r = 0; r < runs; ++r) {
      VLOG_EVERY_N(0, 100) << "inference running iter " << r << "...";
      ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
      for (int i = 0; i < num_outputs; ++i) {
        auto tensor = TensorData<uint8_t>(*interpreter->output_tensor(i));
        for (int j = 0; j < tensor.size(); ++j) {
          if (tensor[j] != reference_tensors[i][j]) {
            VLOG(1) << "[ iteration = " << r << " ] output of tensor " << i
                    << " at position " << j << " differs from reference.\n"
                    << "( output = " << tensor[j]
                    << " reference = " << reference_tensors[i][j] << " )";
            ++error_count;
          }
        }
      }
    }
    EXPECT_EQ(error_count, 0) << "total runs " << runs;
  }
};

TEST_F(InferenceRepeatabilityTest, MobilenetV1) {
  RepeatabilityTest("mobilenet_v1_1.0_224_quant_edgetpu.tflite");
}

TEST_F(InferenceRepeatabilityTest, InceptionV2) {
  RepeatabilityTest("inception_v2_224_quant_edgetpu.tflite");
}

TEST_F(InferenceRepeatabilityTest, InceptionV4) {
  RepeatabilityTest("inception_v4_299_quant_edgetpu.tflite");
}
}  // namespace
}  // namespace coral
