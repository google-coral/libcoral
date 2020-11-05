#include <cmath>

#include "absl/flags/flag.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

ABSL_FLAG(int, stress_test_runs, 500, "Number of iterations for stress test.");

namespace coral {
namespace {

TEST(ModelLoadingStressTest, AlternateEdgeTpuModels) {
  const std::vector<std::string> model_names = {
      "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
      "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
      "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
      "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
      "inception_v1_224_quant_edgetpu.tflite",
      "inception_v2_224_quant_edgetpu.tflite",
      "inception_v3_299_quant_edgetpu.tflite",
      "inception_v4_299_quant_edgetpu.tflite",
  };

  const int num_runs = absl::GetFlag(FLAGS_stress_test_runs);
  for (int i = 0; i < num_runs; ++i) {
    VLOG_EVERY_N(0, 100) << "Stress test iter " << i << "...";
    for (int j = 0; j < model_names.size(); ++j) {
      auto model = LoadModelOrDie(TestDataPath(model_names[j]));
      auto tpu_context = GetEdgeTpuContextOrDie();
      MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
    }
  }
}

}  // namespace
}  // namespace coral
