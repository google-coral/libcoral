#include "coral/posenet/test_utils.h"

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
namespace {

void DumpPosenetResult(const std::string& model_name,
                       absl::Span<const float> poses,
                       absl::Span<const float> keypoint_scores,
                       size_t n_poses) {
  const std::string file_name = absl::StrCat(
      "/tmp/", model_name.substr(0, model_name.find(".")), "_results.csv");
  std::ofstream file(file_name, std::ios::trunc);
  CHECK(file) << "Cannot open " << file_name;
  file << "pose_id,keypoint_id,keypoint_score,keypoint_x,keypoint_y\n";
  for (int pose_id = 0; pose_id < n_poses; pose_id++)
    for (int keypoint_id = 0; keypoint_id < 17; keypoint_id++)
      file << absl::StrCat(pose_id, ",", keypoint_id, ",",
                           keypoint_scores[pose_id * 17 + keypoint_id], ",",
                           poses[pose_id * (17 * 2) + 2 * keypoint_id + 1], ",",
                           poses[pose_id * (17 * 2) + 2 * keypoint_id], "\n");
}

}  // namespace

std::vector<std::array<Keypoint, 17>> ParsePosenetReference(
    const std::string& file_path) {
  const size_t kNumKeypointPerPose = 17;
  std::vector<std::array<Keypoint, kNumKeypointPerPose>> keypoints;

  std::ifstream file{file_path};
  CHECK(file) << "Cannot open " << file_path;
  CHECK(file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'));

  std::vector<size_t> keypoints_found_per_pose;
  for (std::string line; std::getline(file, line);) {
    const std::vector<std::string> elements = absl::StrSplit(line, ',');
    CHECK_EQ(elements.size(), 5);
    int32_t pose_id;
    CHECK(absl::SimpleAtoi(elements[0], &pose_id));
    if (keypoints.size() < pose_id + 1) {
      keypoints.resize(pose_id + 1);
      keypoints_found_per_pose.resize(pose_id + 1);
    }
    int32_t keypoint_id;
    CHECK(absl::SimpleAtoi(elements[1], &keypoint_id));
    auto& keypoint = keypoints[pose_id][keypoint_id];
    CHECK(absl::SimpleAtof(elements[2], &keypoint.score));
    CHECK(absl::SimpleAtof(elements[3], &keypoint.x));
    CHECK(absl::SimpleAtof(elements[4], &keypoint.y));
    keypoints_found_per_pose[pose_id]++;
  }
  for (const auto num_keypoints : keypoints_found_per_pose)
    CHECK_EQ(num_keypoints, kNumKeypointPerPose);
  return keypoints;
}

void TestDecoder(const std::string& model_name,
                 const std::vector<std::array<Keypoint, 17>>& expected_poses,
                 float expected_pose_score,
                 const std::vector<std::vector<uint8_t>>& expected_masks,
                 bool dump_results) {
  auto model = LoadModelOrDie(TestDataPath("posenet/" + model_name));
  auto tpu_context =
      ContainsEdgeTpuCustomOp(*model) ? GetEdgeTpuContextOrDie() : nullptr;
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // test_image.bmp is royalty free from https://unsplash.com/photos/XuN44TajBGo
  // and shows two people standing. Expect keypoints to be roughly where
  // expected.
  CopyResizedImage(TestDataPath("posenet/test_image.bmp"),
                   *interpreter->input_tensor(0));
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

  auto poses = TensorData<float>(*interpreter->output_tensor(0));
  auto keypoint_scores = TensorData<float>(*interpreter->output_tensor(1));
  auto pose_scores = TensorData<float>(*interpreter->output_tensor(2));
  auto nums_poses = TensorData<float>(*interpreter->output_tensor(3));

  const auto n_poses = static_cast<size_t>(nums_poses[0]);
  ASSERT_EQ(n_poses, 2);

  for (int i = 0; i < n_poses; ++i) {
    EXPECT_GT(pose_scores[i], expected_pose_score);
    for (int k = 0; k < 17; k++) {
      if (expected_poses[i][k].score > 0.5) {
        EXPECT_NEAR(expected_poses[i][k].score, keypoint_scores[i * 17 + k],
                    0.1);
        EXPECT_NEAR(expected_poses[i][k].y, poses[i * 17 * 2 + 2 * k], 4);
        EXPECT_NEAR(expected_poses[i][k].x, poses[i * 17 * 2 + 2 * k + 1],
                    4 /*within 3 pixels*/);
      }
    }
  }

  if (dump_results)
    DumpPosenetResult(model_name, poses, keypoint_scores, n_poses);

  // Check instance masks.
  if (expected_masks.empty()) return;

  auto instance_masks = TensorData<float>(*interpreter->output_tensor(4));
  ASSERT_EQ(expected_masks.size(), n_poses);

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
