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

#include "coral/test_utils.h"

#include <cmath>

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {
using ::testing::Eq;

TEST(TestUtilsTest, TopKContains) {
  const std::vector<Class> top_k = {
      {121, 0.95}, {135, 0.90}, {71, 0.85}, {1, 0.55}};
  EXPECT_TRUE(TopKContains(top_k, 121));
  EXPECT_TRUE(TopKContains(top_k, 135));
  EXPECT_TRUE(TopKContains(top_k, 71));
  EXPECT_FALSE(TopKContains(top_k, 2));
}

TEST(TestUtilsTest, ImageResizeBMP) {
  //  expected_output_image is resized from input_image with TF2.1:
  //  tf.image.resize(images, size, method=ResizeMethod.BILINEAR,
  //  preserve_aspect_ratio=False, antialias=False, name=None)
  for (const auto& dims : std::vector<ImageDims>{{224, 224, 3},
                                                 {229, 229, 3},
                                                 {300, 300, 3},
                                                 {513, 513, 3},
                                                 {200, 300, 3}}) {
    ImageDims real_dims;
    auto expected = ReadBmp(TestDataPath(absl::StrFormat(
                                "checker%dX%d.bmp", dims.height, dims.width)),
                            &real_dims);
    ASSERT_EQ(real_dims, dims);
    auto resized = GetInputFromImage(TestDataPath("checker800X600.bmp"), dims);
    ASSERT_EQ(resized.size(), expected.size());
    for (int i = 0; i < expected.size(); ++i)
      EXPECT_LE(std::abs(expected[i] - resized[i]), 1);
  }
}

TEST(TestUtilsTest, ReadBMPImage) {
  ImageDims dims;
  EXPECT_THAT(ReadBmp(TestDataPath("rgb6X4.bmp"), &dims),
              Eq(std::vector<uint8_t>{
                  255, 0, 255, 255, 0, 255, 0,   255, 0,   0,   255, 0,
                  255, 0, 255, 255, 0, 255, 0,   255, 0,   0,   255, 0,
                  255, 0, 255, 255, 0, 255, 0,   255, 0,   0,   255, 0,
                  0,   0, 0,   0,   0, 0,   255, 255, 255, 255, 255, 255,
                  0,   0, 0,   0,   0, 0,   255, 255, 255, 255, 255, 255,
                  0,   0, 0,   0,   0, 0,   255, 255, 255, 255, 255, 255}));
  EXPECT_THAT(dims, Eq(ImageDims{6, 4, 3}));
}
}  // namespace
}  // namespace coral
