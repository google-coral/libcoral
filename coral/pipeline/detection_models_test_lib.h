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

// Library to test SSD detection model pipeline.
#ifndef LIBCORAL_CORAL_PIPELINE_DETECTION_MODELS_TEST_LIB_H_
#define LIBCORAL_CORAL_PIPELINE_DETECTION_MODELS_TEST_LIB_H_

#include <vector>

#include "absl/types/span.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/internal/default_allocator.h"
#include "coral/test_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tflite/public/edgetpu.h"

namespace coral {

// Returns number of segments for testing.
std::vector<int> NumSegments();

template <typename T>
absl::Span<const T> TensorData(const PipelineTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<const T*>(tensor.buffer->ptr()),
                        tensor.bytes / sizeof(T));
}

class PipelinedSsdDetectionModelTest
    : public MultipleEdgeTpuCacheTestBase,
      public ::testing::WithParamInterface<int> {
 public:
  static void SetUpTestSuite();
  static void TearDownTestSuite();
  void TestCatMsCocoDetection(const std::string& model_segment_base_name,
                              int num_segments, float score_threshold,
                              float iou_threshold);

 protected:
  static internal::DefaultAllocator* input_allocator_;
};

}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_DETECTION_MODELS_TEST_LIB_H_
