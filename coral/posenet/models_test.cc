// Tests correctness of models.
#include <array>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/posenet/posenet_decoder_op.h"
#include "coral/posenet/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

ABSL_FLAG(bool, dump_results, false,
          "Whether or not to dump the test results to /tmp.");

namespace coral {
namespace {

void TestPosenetDecoder(const std::string& model_name,
                        float expected_pose_score) {
  const auto& expected_poses = ParsePosenetReference(absl::StrCat(
      TestDataPath("posenet/"), model_name.substr(0, model_name.find(".")),
      "_reference.csv"));
  TestDecoder(model_name, expected_poses, expected_pose_score,
              /*expected_masks=*/{}, absl::GetFlag(FLAGS_dump_results));
}

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class PosenetModelTest : public ::testing::TestWithParam<std::string> {};

TEST_P(PosenetModelTest, TestPoseNetWithDecoder_353_481) {
  TestPosenetDecoder(
      "posenet_mobilenet_v1_075_353_481_quant_decoder" + GetParam(), 0.7);
}

TEST_P(PosenetModelTest, TestPoseNetWithDecoder_481_641) {
  TestPosenetDecoder(
      "posenet_mobilenet_v1_075_481_641_quant_decoder" + GetParam(), 0.7);
}

TEST_P(PosenetModelTest, TestPoseNetWithDecoder_721_1281) {
  TestPosenetDecoder(
      "posenet_mobilenet_v1_075_721_1281_quant_decoder" + GetParam(), 0.7);
}

INSTANTIATE_TEST_CASE_P(PosenetModelCpuTest, PosenetModelTest,
                        ::testing::Values(".tflite"));
INSTANTIATE_TEST_CASE_P(PosenetModelEdgeTpuTest, PosenetModelTest,
                        ::testing::Values("_edgetpu.tflite"));

}  // namespace
}  // namespace coral
