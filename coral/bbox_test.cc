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
