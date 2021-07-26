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

// Stress test with multiple Edge TPU devices.
//
// By default, it launches one thread per Edge TPU devices it can find on the
// host system. And each thread will run `FLAGS_num_inferences` inferences.  It
// also checks the result returned is correct.
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

ABSL_FLAG(int, num_inferences, 3000,
          "Number of inferences for each thread to run.");

namespace coral {
namespace {

class MultipleTpusStressTest : public MultipleEdgeTpuCacheTestBase {
 public:
  static void SetUpTestCase() {
    max_num_tpus_ =
        edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu().size();
  }

 protected:
  void StressTest(const std::string& model_name,
                  const std::string& image_name) {
    const int num_runs = absl::GetFlag(FLAGS_num_inferences);
    const auto tpu_contexts = GetTpuContextCache(max_num_tpus_);
    CHECK_GT(tpu_contexts.size(), 1);
    LOG(INFO) << "Testing with " << tpu_contexts.size() << " Edge TPUs...";
    LOG(INFO) << "Each thread will run " << num_runs << " inferences.";
    auto model = LoadModelOrDie(TestDataPath(model_name));
    std::vector<std::thread> workers;
    workers.reserve(tpu_contexts.size());
    for (int i = 0; i < tpu_contexts.size(); ++i) {
      workers.emplace_back([&, i]() {
        const auto tid = std::this_thread::get_id();
        LOG(INFO) << "thread: " << tid << " created.";

        auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_contexts[i]);
        CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
        CopyResizedImage(TestDataPath(image_name),
                         *interpreter->input_tensor(0));
        for (int i = 0; i < num_runs; ++i)
          ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

        LOG(INFO) << "thread: " << tid << " done stress run.";
      });
    }

    for (int i = 0; i < tpu_contexts.size(); ++i) workers[i].join();
    LOG(INFO) << "Stress test done for model: " << model_name;
  }

  static int max_num_tpus_;
};

int MultipleTpusStressTest::max_num_tpus_ = 0;

TEST_F(MultipleTpusStressTest, MobileNetV1) {
  StressTest("mobilenet_v1_1.0_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST_F(MultipleTpusStressTest, MobileNetV2) {
  StressTest("mobilenet_v2_1.0_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST_F(MultipleTpusStressTest, InceptionV1) {
  StressTest("inception_v1_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST_F(MultipleTpusStressTest, InceptionV2) {
  StressTest("inception_v2_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST_F(MultipleTpusStressTest, InceptionV3) {
  StressTest("inception_v3_299_quant_edgetpu.tflite", "cat.bmp");
}

TEST_F(MultipleTpusStressTest, InceptionV4) {
  StressTest("inception_v4_299_quant_edgetpu.tflite", "cat.bmp");
}

TEST_F(MultipleTpusStressTest, SsdMobileNetV1) {
  StressTest("ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
             "cat.bmp");
}

TEST_F(MultipleTpusStressTest, SsdMobileNetV2) {
  StressTest("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
             "cat.bmp");
}

}  // namespace
}  // namespace coral
