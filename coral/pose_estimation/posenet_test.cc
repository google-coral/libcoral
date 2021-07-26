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

// Tests correctness of models.
#include <array>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/pose_estimation/posenet_decoder_op.h"
#include "coral/pose_estimation/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tflite/public/edgetpu.h"

ABSL_FLAG(bool, dump_results, false,
          "Whether or not to dump the test results to /tmp.");

namespace coral {
namespace {

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class PosenetModelTest : public ModelTestBase {};

TEST_P(PosenetModelTest, TestPoseNetWithDecoder_353_481) {
  TestDecoder("posenet_mobilenet_v1_075_353_481_16_quant_decoder" + GetParam(),
              "test_image.bmp",
              /*expected_masks=*/{}, absl::GetFlag(FLAGS_dump_results),
              GetTpuContextIfNecessary());
}

TEST_P(PosenetModelTest, TestPoseNetWithDecoder_481_641) {
  TestDecoder("posenet_mobilenet_v1_075_481_641_16_quant_decoder" + GetParam(),
              "test_image.bmp",
              /*expected_masks=*/{}, absl::GetFlag(FLAGS_dump_results),
              GetTpuContextIfNecessary());
}

TEST_P(PosenetModelTest, TestPoseNetWithDecoder_721_1281) {
  TestDecoder("posenet_mobilenet_v1_075_721_1281_16_quant_decoder" + GetParam(),
              "test_image.bmp",
              /*expected_masks=*/{}, absl::GetFlag(FLAGS_dump_results),
              GetTpuContextIfNecessary());
}

INSTANTIATE_TEST_CASE_P(Cpu, PosenetModelTest, ::testing::Values(".tflite"));
INSTANTIATE_TEST_CASE_P(EdgeTpu, PosenetModelTest,
                        ::testing::Values("_edgetpu.tflite"));

}  // namespace
}  // namespace coral
