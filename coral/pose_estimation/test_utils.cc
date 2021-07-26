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

#include "coral/pose_estimation/test_utils.h"

#include <cmath>
#include <fstream>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "gtest/gtest.h"

namespace coral {

void DumpPoseEstimationResults(const std::string& model_name,
                               const std::vector<Pose>& poses) {
  const std::string file_name = absl::StrCat(
      "/tmp/", model_name.substr(0, model_name.find('.')), "_reference.csv");
  std::ofstream file(file_name, std::ios::trunc);
  CHECK(file) << "Cannot open " << file_name;
  LOG(INFO) << "Dumping pose estimation results to " << file_name;
  file << "pose_id,pose_score,keypoint_id,keypoint_score,keypoint_x,keypoint_"
          "y\n";
  for (int i = 0; i < poses.size(); i++)
    for (int k = 0; k < kNumKeypointPerPose; k++)
      file << absl::StrCat(
          i, ",", poses[i].score, ",", k, ",", poses[i].keypoints[k].score, ",",
          poses[i].keypoints[k].x, ",", poses[i].keypoints[k].y, "\n");
}

std::vector<Pose> ParsePoseEstimationReference(const std::string& file_path) {
  std::vector<Pose> keypoints;

  std::ifstream file{file_path};
  CHECK(file) << "Cannot open " << file_path;
  CHECK(file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'));

  std::vector<Pose> poses;
  int keypoint_counter = 0;
  int pose_id = -1;
  for (std::string line; std::getline(file, line);) {
    const std::vector<std::string> elements = absl::StrSplit(line, ',');
    CHECK_EQ(elements.size(), 6) << file_path << ": " << line;

    if (keypoint_counter == 0) {
      poses.push_back(Pose());
      CHECK(absl::SimpleAtof(elements[1], &poses.back().score));
    }

    CHECK(absl::SimpleAtoi(elements[0], &pose_id));
    CHECK_EQ(pose_id, poses.size() - 1);

    int32_t keypoint_id;
    CHECK(absl::SimpleAtoi(elements[2], &keypoint_id));
    CHECK_EQ(keypoint_id, keypoint_counter);
    CHECK(absl::SimpleAtof(elements[3],
                           &poses.back().keypoints[keypoint_id].score));
    CHECK(
        absl::SimpleAtof(elements[4], &poses.back().keypoints[keypoint_id].x));
    CHECK(
        absl::SimpleAtof(elements[5], &poses.back().keypoints[keypoint_id].y));

    keypoint_counter = (keypoint_counter + 1) % kNumKeypointPerPose;
  }
  CHECK_EQ(keypoint_counter, 0);
  return poses;
}

void CheckPoseEstimationResults(const std::vector<Pose>& results,
                                const std::vector<Pose>& expected,
                                float keypoint_score_threshold) {
  ASSERT_EQ(results.size(), expected.size());
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i].score, results[i].score, 0.1);
    int num_checked_keypoints = 0;
    for (int k = 0; k < kNumKeypointPerPose; k++) {
      // Only keppoints with high confidence scores are checked.
      if (expected[i].keypoints[k].score > keypoint_score_threshold &&
          results[i].keypoints[k].score > keypoint_score_threshold) {
        ++num_checked_keypoints;
        EXPECT_NEAR(expected[i].keypoints[k].y, results[i].keypoints[k].y,
                    0.04);
        EXPECT_NEAR(expected[i].keypoints[k].x, results[i].keypoints[k].x,
                    0.04);
      }
    }
    ASSERT_GE(num_checked_keypoints, kNumKeypointPerPose / 2);
  }
}

void TestDecoder(const std::string& model_name, const std::string& image_name,
                 const std::vector<std::vector<uint8_t>>& expected_masks,
                 bool dump_results, edgetpu::EdgeTpuContext* tpu_context) {
  auto model = LoadModelOrDie(TestDataPath("posenet/" + model_name));
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  const auto input_shape = TensorShape(*interpreter->input_tensor(0));
  ASSERT_EQ(input_shape.size(), 4);
  const float input_height = input_shape[1];
  const float input_width = input_shape[2];

  // test_image.bmp is royalty free from https://unsplash.com/photos/XuN44TajBGo
  // and shows two people standing. Expect keypoints to be roughly where
  // expected.
  CopyResizedImage(TestDataPath("posenet/" + image_name),
                   *interpreter->input_tensor(0));
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

  auto keypoint_yx = TensorData<float>(*interpreter->output_tensor(0));
  auto keypoint_scores = TensorData<float>(*interpreter->output_tensor(1));
  auto pose_scores = TensorData<float>(*interpreter->output_tensor(2));
  auto nums_poses = TensorData<float>(*interpreter->output_tensor(3));
  const size_t n_poses = static_cast<size_t>(nums_poses[0]);
  std::vector<Pose> poses(n_poses);
  for (int i = 0; i < n_poses; ++i) {
    for (int k = 0; k < kNumKeypointPerPose; ++k) {
      poses[i].score = pose_scores[i];
      poses[i].keypoints[k].score =
          keypoint_scores[i * kNumKeypointPerPose + k];
      poses[i].keypoints[k].y =
          keypoint_yx[i * kNumKeypointPerPose * 2 + 2 * k] / input_height;
      poses[i].keypoints[k].x =
          keypoint_yx[i * kNumKeypointPerPose * 2 + 2 * k + 1] / input_width;
    }
  }

  if (dump_results) {
    DumpPoseEstimationResults(model_name, poses);
  }

  const auto& expected_poses = ParsePoseEstimationReference(absl::StrCat(
      TestDataPath("posenet/"), model_name.substr(0, model_name.find('.')),
      "_reference.csv"));
  CheckPoseEstimationResults(poses, expected_poses, 0.7);

  // Check instance masks.
  if (expected_masks.empty()) return;

  auto instance_masks = TensorData<float>(*interpreter->output_tensor(4));
  ASSERT_EQ(expected_masks.size(), n_poses);
  ASSERT_GE(instance_masks.size(), n_poses * expected_masks[0].size());

  for (int i = 0; i < n_poses; ++i) {
    const auto& expected_mask = expected_masks[i];
    const auto mask_size = expected_mask.size();
    int sum_l1 = 0;
    for (int k = 0; k < mask_size; ++k)
      sum_l1 +=
          std::abs(expected_mask[k] / 255 - instance_masks[i * mask_size + k]);
    // Expect the sum of the L1 distances for each mask to be less than
    // a threshold, say 95% correct values.
    EXPECT_LT(sum_l1, mask_size * 0.05);
  }
}

}  // namespace coral
