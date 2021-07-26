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

#ifndef LIBCORAL_CORAL_POSE_ESTIMATION_TEST_UTILS_H_
#define LIBCORAL_CORAL_POSE_ESTIMATION_TEST_UTILS_H_

#include <array>
#include <string>
#include <vector>

#include "tflite/public/edgetpu.h"

namespace coral {

// All Coral pose estimation models assume there are 17 keypoints per pose.
constexpr size_t kNumKeypointPerPose = 17;

struct Keypoint {
  float y;
  float x;
  float score;
};

struct Pose {
  float score;
  std::array<Keypoint, kNumKeypointPerPose> keypoints;
};

std::vector<Pose> ParsePoseEstimationReference(const std::string& file_path);

// Dumps pose estimation results to files.
// The result file has filename like <model_name>_reference.csv, which
// can be used to update the unit test reference results.
void DumpPoseEstimationResults(const std::string& model_name,
                               const std::vector<Pose>& poses);

// Compares two set of pose estimation results. It only takes into account
// keypoints that have score above the given threshold.
void CheckPoseEstimationResults(const std::vector<Pose>& results,
                                const std::vector<Pose>& expected,
                                float keypoint_score_threshold);

// Test the decoder of posenet and bodypix models with the specified test image.
// If the testee model produces pixel level segmentation, it will check
// segmentation results when 'expected_masks' is not empty.
void TestDecoder(const std::string& model_name, const std::string& image_name,
                 const std::vector<std::vector<uint8_t>>& expected_masks,
                 bool dump_results, edgetpu::EdgeTpuContext* tpu_context);

}  // namespace coral

#endif  // LIBCORAL_CORAL_POSE_ESTIMATION_TEST_UTILS_H_
