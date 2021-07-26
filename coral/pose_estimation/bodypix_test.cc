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
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "coral/pose_estimation/posenet_decoder_op.h"
#include "coral/pose_estimation/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

ABSL_FLAG(bool, dump_results, false,
          "Whether or not to dump the test results to /tmp.");

namespace coral {

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class BodypixModelTest : public ModelTestBase {};

TEST_P(BodypixModelTest, TestBodyPixWithDecoder_512_512) {
  ImageDims image_dims;
  auto mask0 = ReadBmp(TestDataPath("posenet/mask0.bmp"), &image_dims);
  ASSERT_EQ(image_dims.height, 33);
  ASSERT_EQ(image_dims.width, 33);

  auto mask1 = ReadBmp(TestDataPath("posenet/mask1.bmp"), &image_dims);
  ASSERT_EQ(image_dims.height, 33);
  ASSERT_EQ(image_dims.width, 33);

  TestDecoder("bodypix_mobilenet_v1_075_512_512_16_quant_decoder" + GetParam(),
              "test_image.bmp", {mask0, mask1},
              absl::GetFlag(FLAGS_dump_results), GetTpuContextIfNecessary());
}

INSTANTIATE_TEST_CASE_P(Cpu, BodypixModelTest, ::testing::Values(".tflite"));
INSTANTIATE_TEST_CASE_P(EdgeTpu, BodypixModelTest,
                        ::testing::Values("_edgetpu.tflite"));

}  // namespace coral
