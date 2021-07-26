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

#include "coral/detection/adapter.h"

#include <sstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(DetectionAdapter, Object) {
  Object a{1, 0.2, BBox<float>{0.0, 0.0, 1.0, 1.0}};
  Object b{1, 0.5, BBox<float>{0.0, 0.0, 1.0, 1.0}};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_NE(a, b);

  Object tmp{5, 0.7, BBox<float>{0.0, 0.0, 0.0, 0.0}};
  EXPECT_NE(a, tmp);
  a = tmp;
  EXPECT_EQ(a, tmp);
}

TEST(DetectionAdapter, ToString) {
  const auto obj = Object{1, 0.2, BBox<float>{0.0, 0.0, 1.0, 1.0}};
  EXPECT_EQ(ToString(obj),
            "Object(id=1,score=0.2,bbox=BBox(ymin=0,xmin=0,ymax=1,xmax=1))");
  std::stringstream ss;
  ss << obj;
  EXPECT_EQ(ss.str(), ToString(obj));
}

TEST(DetectionAdapter, GetDetectionResults) {
  auto bboxes = {0.11f, 0.12f, 0.21f, 0.22f,  //
                 0.13f, 0.14f, 0.23f, 0.24f,  //
                 0.15f, 0.16f, 0.25f, 0.26f,  //
                 0.17f, 0.18f, 0.27f, 0.28f};
  auto ids = {0.0f, 1.0f, 2.0f, 3.0f};
  auto scores = {0.6f, 0.1f, 0.9f, 0.7f};
  const size_t count = 4;

  const auto o0 = Object{0, 0.6, BBox<float>{0.11, 0.12, 0.21, 0.22}};
  const auto o1 = Object{1, 0.1, BBox<float>{0.13, 0.14, 0.23, 0.24}};
  const auto o2 = Object{2, 0.9, BBox<float>{0.15, 0.16, 0.25, 0.26}};
  const auto o3 = Object{3, 0.7, BBox<float>{0.17, 0.18, 0.27, 0.28}};

  EXPECT_THAT(GetDetectionResults(bboxes, ids, scores, count),
              ElementsAre(o2, o3, o0, o1));

  EXPECT_THAT(GetDetectionResults(bboxes, ids, scores, count,
                                  /*threshold=*/0.0, /*top_k=*/0),
              IsEmpty());

  EXPECT_THAT(GetDetectionResults(bboxes, ids, scores, count,
                                  /*threshold=*/0.95, /*top_k=*/0),
              IsEmpty());

  EXPECT_THAT(GetDetectionResults(bboxes, ids, scores, count,
                                  /*threshold=*/0.0, /*top_k=*/10),
              ElementsAre(o2, o3, o0, o1));

  EXPECT_THAT(GetDetectionResults(bboxes, ids, scores, count,
                                  /*threshold=*/0.0, /*top_k=*/3),
              ElementsAre(o2, o3, o0));

  EXPECT_THAT(GetDetectionResults(bboxes, ids, scores, count,
                                  /*threshold=*/0.8, /*top_k=*/3),
              ElementsAre(o2));
}

}  // namespace
}  // namespace coral
