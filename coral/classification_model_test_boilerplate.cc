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
#include "gtest/gtest.h"

namespace coral {

class ARG_TEST_SUITE_NAME : public ClassificationModelTestBase {};

// All ARG_* values come from compiler defines, e.g. -DARG_TEST_NAME=my_test.
TEST_F(ARG_TEST_SUITE_NAME, ARG_TEST_NAME) {
  TestClassification(CORAL_TOSTRING(ARG_MODEL_PATH),
                     CORAL_TOSTRING(ARG_IMAGE_PATH), ARG_EFFECTIVE_SCALE,
                     {ARG_EFFECTIVE_MEANS}, ARG_RGB2BGR, ARG_SCORE_THRESHOLD,
                     ARG_K, ARG_EXPECTED_TOPK_LABEL);
}
}  // namespace coral
