#ifndef EDGETPU_CPP_POSENET_TEST_UTILS_H_
#define EDGETPU_CPP_POSENET_TEST_UTILS_H_

#include <array>
#include <string>
#include <vector>

namespace coral {

struct Keypoint {
  float y;
  float x;
  float score;
};

std::vector<std::array<Keypoint, 17>> ParsePosenetReference(
    const std::string& file_path);

void TestDecoder(const std::string& model_name,
                 const std::vector<std::array<Keypoint, 17>>& expected_poses,
                 float expected_pose_score,
                 const std::vector<std::vector<uint8_t>>& expected_masks = {},
                 bool dump_results = false);

}  // namespace coral

#endif  // EDGETPU_CPP_POSENET_TEST_UTILS_H_
