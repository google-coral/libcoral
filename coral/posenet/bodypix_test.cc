// Tests correctness of models.
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "coral/posenet/posenet_decoder_op.h"
#include "coral/posenet/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {

TEST(BodypixModelCorrectnessTest, TestBodyPixWithDecoder_512_512) {
  // test_image.bmp is royalty free from https://unsplash.com/photos/XuN44TajBGo
  // and shows two people standing. Expect keypoints to be roughly match the
  // image. Expect two instance masks, one for each person.
  std::vector<std::array<Keypoint, 17>> expected_poses(
      {std::array<Keypoint, 17>(
           {{Keypoint{141, 221, 0.983}, Keypoint{134, 221, 0.369},
             Keypoint{135, 216, 0.982}, Keypoint{134, 271, 0.054},
             Keypoint{135, 199, 0.978}, Keypoint{175, 224, 0.993},
             Keypoint{184, 179, 0.995}, Keypoint{231, 241, 0.975},
             Keypoint{239, 176, 0.980}, Keypoint{280, 239, 0.993},
             Keypoint{288, 167, 0.990}, Keypoint{286, 224, 0.988},
             Keypoint{288, 198, 0.993}, Keypoint{351, 219, 0.987},
             Keypoint{350, 204, 0.980}, Keypoint{411, 210, 0.500},
             Keypoint{411, 203, 0.594}}}),
       std::array<Keypoint, 17>(
           {{Keypoint{151, 273, 0.986}, Keypoint{145, 278, 0.991},
             Keypoint{146, 272, 0.885}, Keypoint{148, 293, 0.990},
             Keypoint{104, 256, 0.022}, Keypoint{184, 308, 0.997},
             Keypoint{183, 273, 0.993}, Keypoint{223, 317, 0.988},
             Keypoint{230, 258, 0.955}, Keypoint{259, 333, 0.977},
             Keypoint{193, 233, 0.238}, Keypoint{264, 301, 0.989},
             Keypoint{265, 277, 0.984}, Keypoint{341, 298, 0.919},
             Keypoint{341, 287, 0.494}, Keypoint{314, 286, 0.008},
             Keypoint{303, 288, 0.023}}})});

  ImageDims image_dims;
  auto mask0 = ReadBmp(TestDataPath("posenet/mask0.bmp"), &image_dims);
  ASSERT_EQ(image_dims.height, 33);
  ASSERT_EQ(image_dims.width, 33);

  auto mask1 = ReadBmp(TestDataPath("posenet/mask1.bmp"), &image_dims);
  ASSERT_EQ(image_dims.height, 33);
  ASSERT_EQ(image_dims.width, 33);

  TestDecoder(
      "bodypix_mobilenet_v1_075_512_512_16_segments_quant_edgetpu_decoder."
      "tflite",
      expected_poses, /*expected_pose_score=*/0.70, {mask0, mask1});
}

}  // namespace coral
