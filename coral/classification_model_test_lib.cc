#include "coral/test_utils.h"
#include "gtest/gtest.h"

namespace coral {
namespace {
// All ARG_* values come from compiler defines, e.g. -DARG_TEST_NAME=my_test.
TEST(ARG_TEST_SUITE_NAME, ARG_TEST_NAME) {
  TestClassification(CORAL_TOSTRING(ARG_MODEL_PATH),
                     CORAL_TOSTRING(ARG_IMAGE_PATH), ARG_EFFECTIVE_SCALE,
                     {ARG_EFFECTIVE_MEANS}, ARG_RGB2BGR, ARG_SCORE_THRESHOLD,
                     ARG_K, ARG_EXPECTED_TOPK_LABEL);
}
}  // namespace
}  // namespace coral
