#include "coral/classification/adapter.h"

#include <sstream>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(ClassificationAdapter, Class) {
  const auto a = Class{1, 0.2};
  const auto b = Class{1, 0.3};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_NE(a, b);
}

TEST(ClassificationAdapter, ToString) {
  const auto c = Class{1, 0.2};
  EXPECT_EQ(ToString(c), "Class(id=1,score=0.2)");
  std::stringstream ss;
  ss << c;
  EXPECT_EQ(ss.str(), ToString(c));
}

TEST(ClassificationAdapter, GetClassificationResults) {
  const auto scores = std::vector<float>{0.1, 0.9, 0.5, 0.95};

  const auto c0 = Class{0, 0.1};
  const auto c1 = Class{1, 0.9};
  const auto c2 = Class{2, 0.5};
  const auto c3 = Class{3, 0.95};

  EXPECT_THAT(GetClassificationResults(scores), ElementsAre(c3, c1, c2, c0));

  EXPECT_THAT(GetClassificationResults(scores, /*threshold=*/0.8),
              ElementsAre(c3, c1));

  EXPECT_THAT(GetClassificationResults(scores, /*threshold=*/0.99), IsEmpty());

  EXPECT_THAT(GetClassificationResults(scores,
                                       /*threshold=*/0.0, /*top_k=*/3),
              ElementsAre(c3, c1, c2));

  EXPECT_THAT(GetClassificationResults(scores,
                                       /*threshold=*/0.0, /*top_k=*/1),
              ElementsAre(c3));

  EXPECT_THAT(GetClassificationResults(scores,
                                       /*threshold=*/0.0, /*top_k=*/300),
              ElementsAre(c3, c1, c2, c0));

  EXPECT_THAT(GetClassificationResults(scores,
                                       /*threshold=*/0.0, /*top_k=*/0),
              IsEmpty());

  EXPECT_THAT(GetClassificationResults(scores,
                                       /*threshold=*/0.7, /*top_k=*/3),
              ElementsAre(c3, c1));

  EXPECT_THAT(GetClassificationResults(scores,
                                       /*threshold=*/0.99, /*top_k=*/10),
              IsEmpty());
}

}  // namespace
}  // namespace coral
