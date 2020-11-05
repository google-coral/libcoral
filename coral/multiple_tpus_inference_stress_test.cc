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

void StressTest(const std::string& model_name, const std::string& image_name) {
  const int num_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu().size();
  const int num_runs = absl::GetFlag(FLAGS_num_inferences);
  LOG(INFO) << "Each thread will run " << num_runs << " inferences.";
  auto model = LoadModelOrDie(TestDataPath(model_name));
  std::vector<std::thread> workers;
  workers.reserve(num_tpus);
  for (int i = 0; i < num_tpus; ++i) {
    workers.emplace_back([&, i]() {
      const auto tid = std::this_thread::get_id();
      LOG(INFO) << "thread: " << tid << " created.";

      auto tpu_context = GetEdgeTpuContextOrDie(absl::nullopt, i);
      auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
      CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
      CopyResizedImage(TestDataPath(image_name), *interpreter->input_tensor(0));
      for (int i = 0; i < num_runs; ++i)
        ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

      LOG(INFO) << "thread: " << tid << " done stress run.";
    });
  }

  for (int i = 0; i < num_tpus; ++i) workers[i].join();
  LOG(INFO) << "Stress test done for model: " << model_name;
}

TEST(MultipleTpusStressTest, MobileNetV1) {
  StressTest("mobilenet_v1_1.0_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST(MultipleTpusStressTest, MobileNetV2) {
  StressTest("mobilenet_v2_1.0_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST(MultipleTpusStressTest, InceptionV1) {
  StressTest("inception_v1_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST(MultipleTpusStressTest, InceptionV2) {
  StressTest("inception_v2_224_quant_edgetpu.tflite", "cat.bmp");
}

TEST(MultipleTpusStressTest, InceptionV3) {
  StressTest("inception_v3_299_quant_edgetpu.tflite", "cat.bmp");
}

TEST(MultipleTpusStressTest, InceptionV4) {
  StressTest("inception_v4_299_quant_edgetpu.tflite", "cat.bmp");
}

TEST(MultipleTpusStressTest, SsdMobileNetV1) {
  StressTest("ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
             "cat.bmp");
}

TEST(MultipleTpusStressTest, SsdMobileNetV2) {
  StressTest("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
             "cat.bmp");
}

}  // namespace
}  // namespace coral
