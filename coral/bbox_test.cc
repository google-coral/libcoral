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

#include "coral/bbox.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {

TEST(BBoxTest, IsBoxValid) {
  EXPECT_FALSE((BBox<float>{0.7, 0.1, 0.5, 0.3}).valid());
  EXPECT_FALSE((BBox<float>{0.5, 0.3, 0.7, 0.1}).valid());
  EXPECT_TRUE((BBox<float>{0.5, 0.1, 0.7, 0.3}).valid());
}

TEST(BBoxTest, ComputeBoxArea) {
  EXPECT_FLOAT_EQ((BBox<float>{0.5, 0.1, 0.7, 0.4}.area()), 0.06);
}

TEST(BBoxTest, IntersectionOverUnion) {
  EXPECT_FLOAT_EQ(IntersectionOverUnion(BBox<float>{0.1, 0.2, 0.5, 0.4},
                                        BBox<float>{0.1, 0.2, 0.3, 0.4}),
                  0.5);
  EXPECT_FLOAT_EQ(IntersectionOverUnion(BBox<float>{0.1, 0.2, 0.5, 0.4},
                                        BBox<float>{0.1, 0.2, 0.5, 0.3}),
                  0.5);
  EXPECT_FLOAT_EQ(IntersectionOverUnion(BBox<float>{0.1, 0.2, 0.5, 0.4},
                                        BBox<float>{0.2, 0.2, 0.6, 0.4}),
                  0.6);
  EXPECT_FLOAT_EQ(IntersectionOverUnion(BBox<float>{0.1, 0.2, 0.5, 0.4},
                                        BBox<float>{0.6, 0.2, 0.9, 0.4}),
                  0.0);
}

}  // namespace coral
