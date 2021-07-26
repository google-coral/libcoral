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
  auto tpu_context = GetTestEdgeTpuContextOrDie();
  for (int i = 0; i < num_runs; ++i) {
    VLOG_EVERY_N(0, 100) << "Stress test iter " << i << "...";
    for (int j = 0; j < model_names.size(); ++j) {
      auto model = LoadModelOrDie(TestDataPath(model_names[j]));
      MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
    }
  }
}

}  // namespace
}  // namespace coral
