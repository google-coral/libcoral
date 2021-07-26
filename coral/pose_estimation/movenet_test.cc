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

#include <fstream>

#include "absl/flags/flag.h"
#include "coral/pose_estimation/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

ABSL_FLAG(bool, dump_results, false,
          "Whether or not to dump the test results to /tmp.");

namespace coral {
namespace {

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class MovenetModelTest : public ModelTestBase {};

void TestMovenet(const std::string& model_name,
                 edgetpu::EdgeTpuContext* tpu_context) {
  auto model = LoadModelOrDie(TestDataPath(model_name));
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  const auto input_shape = TensorShape(*interpreter->input_tensor(0));
  ASSERT_EQ(input_shape.size(), 4);

  const auto output_shape = TensorShape(*interpreter->output_tensor(0));
  ASSERT_EQ(output_shape.size(), 4);
  ASSERT_EQ(output_shape[2], kNumKeypointPerPose);
  ASSERT_EQ(output_shape[3], 3);

  CopyResizedImage(TestDataPath("squat.bmp"), *interpreter->input_tensor(0));
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

  // Parse Movenet results.
  const float* output_tensor_data = interpreter->typed_output_tensor<float>(0);
  std::vector<Pose> poses(1);
  poses[0].score = 1.0;  // Movenet doesn't produce single pose score.
  for (int k = 0; k < kNumKeypointPerPose; ++k) {
    poses[0].keypoints[k].y = output_tensor_data[k * 3];
    poses[0].keypoints[k].x = output_tensor_data[k * 3 + 1];
    poses[0].keypoints[k].score = output_tensor_data[k * 3 + 2];
  }

  if (absl::GetFlag(FLAGS_dump_results)) {
    DumpPoseEstimationResults(model_name, poses);
  }

  // Read inference results from test file.
  const auto& expected_poses =
      ParsePoseEstimationReference(TestDataPath(absl::StrCat(
          model_name.substr(0, model_name.find('.')), "_reference.csv")));
  CheckPoseEstimationResults(poses, expected_poses, 0.4);
}

TEST_P(MovenetModelTest, SinglePoseLightning) {
  TestMovenet("movenet_single_pose_lightning_ptq" + GetParam(),
              GetTpuContextIfNecessary());
}

TEST_P(MovenetModelTest, SinglePoseThunder) {
  TestMovenet("movenet_single_pose_thunder_ptq" + GetParam(),
              GetTpuContextIfNecessary());
}

INSTANTIATE_TEST_CASE_P(EdgeTpu, MovenetModelTest,
                        ::testing::Values("_edgetpu.tflite"));

INSTANTIATE_TEST_CASE_P(Cpu, MovenetModelTest, ::testing::Values(".tflite"));

}  // namespace
}  // namespace coral
