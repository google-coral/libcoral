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

#include <chrono>  // NOLINT
#include <string>
#include <thread>  // NOLINT

#include "absl/flags/flag.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

ABSL_FLAG(int, stress_test_runs, 500, "Number of iterations for stress test.");
ABSL_FLAG(int, stress_with_sleep_test_runs, 200,
          "Number of iterations for stress test.");
ABSL_FLAG(int, stress_sleep_sec, 3,
          "Seconds to sleep in-between inference runs.");

namespace coral {
namespace {

class InferenceStressTest : public EdgeTpuCacheTestBase {
 protected:
  void Run(const std::string& model_name, bool sleep = false) {
    auto model = LoadModelOrDie(TestDataPath(model_name));
    auto interpreter =
        MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextCache());
    CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    FillRandomInt(MutableTensorData<uint8_t>(*interpreter->input_tensor(0)));

    const int runs = sleep ? absl::GetFlag(FLAGS_stress_with_sleep_test_runs)
                           : absl::GetFlag(FLAGS_stress_test_runs);
    const int sleep_sec = sleep ? absl::GetFlag(FLAGS_stress_sleep_sec) : 0;

    auto currTime = std::chrono::system_clock::now();
    auto elapsedTime = std::chrono::system_clock::now();
    std::chrono::duration<double> duration;

    for (int i = 0; i < runs; ++i) {
      VLOG_EVERY_N(0, std::max(1, runs / 5))
          << "inference running iter " << i << "...";
      elapsedTime = std::chrono::system_clock::now();
      duration = elapsedTime - currTime;
      if (duration.count() >= 900.0) {
        LOG(INFO) << "Keeping test alive for io_timeout...";
        currTime = std::chrono::system_clock::now();
      }
      CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
      std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));
    }
  }
};

TEST_F(InferenceStressTest, MobilenetV1) {
  Run("mobilenet_v1_1.0_224_quant_edgetpu.tflite");
}

TEST_F(InferenceStressTest, SsdMobileNetV1) {
  Run("ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite");
}

TEST_F(InferenceStressTest, InceptionV2) {
  Run("inception_v2_224_quant_edgetpu.tflite");
}

TEST_F(InferenceStressTest, InceptionV4) {
  Run("inception_v4_299_quant_edgetpu.tflite");
}

// Stress tests with sleep in-between inference runs.
// We cap the runs here as they will take a lot of time to finish.
TEST_F(InferenceStressTest, MobilenetV1_WithSleep) {
  Run("mobilenet_v1_1.0_224_quant_edgetpu.tflite",
      /*sleep=*/true);
}

TEST_F(InferenceStressTest, InceptionV2_WithSleep) {
  Run("inception_v2_224_quant_edgetpu.tflite",
      /*sleep=*/true);
}

}  // namespace
}  // namespace coral
