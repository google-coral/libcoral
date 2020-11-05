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
void InferenceStressTest(const std::string& model_name, bool sleep = false) {
  auto model = LoadModelOrDie(TestDataPath(model_name));
  auto tpu_context = GetEdgeTpuContextOrDie();
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  FillRandomInt(MutableTensorData<uint8_t>(*interpreter->input_tensor(0)));

  const int runs = sleep ? absl::GetFlag(FLAGS_stress_with_sleep_test_runs)
                         : absl::GetFlag(FLAGS_stress_test_runs);
  const int sleep_sec = sleep ? absl::GetFlag(FLAGS_stress_sleep_sec) : 0;
  for (int i = 0; i < runs; ++i) {
    VLOG_EVERY_N(0, std::max(1, runs / 5))
        << "inference running iter " << i << "...";
    CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
    std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));
  }
}

TEST(InferenceStressTest, MobilenetV1) {
  InferenceStressTest("mobilenet_v1_1.0_224_quant_edgetpu.tflite");
}

TEST(InferenceStressTest, SsdMobileNetV1) {
  InferenceStressTest("ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite");
}

TEST(InferenceStressTest, InceptionV2) {
  InferenceStressTest("inception_v2_224_quant_edgetpu.tflite");
}

TEST(InferenceStressTest, InceptionV4) {
  InferenceStressTest("inception_v4_299_quant_edgetpu.tflite");
}

// Stress tests with sleep in-between inference runs.
// We cap the runs here as they will take a lot of time to finish.
TEST(InferenceStressTest, MobilenetV1_WithSleep) {
  InferenceStressTest("mobilenet_v1_1.0_224_quant_edgetpu.tflite",
                      /*sleep=*/true);
}

TEST(InferenceStressTest, InceptionV2_WithSleep) {
  InferenceStressTest("inception_v2_224_quant_edgetpu.tflite",
                      /*sleep=*/true);
}

}  // namespace
}  // namespace coral
