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

// This test suite is for pipelined SSD models. SSD postprocessing operator
// doesn't output deterministic values for all tensor elements. The other model
// test suite under this folder, models_test, compares the pipeline results vs
// reference results, which do not work for SSD models.

#include "coral/pipeline/detection_models_test_lib.h"
#include "coral/pipeline/test_utils.h"

namespace coral {
namespace {

TEST_P(PipelinedSsdDetectionModelTest, SsdMobilenetV1) {
  TestCatMsCocoDetection("pipeline/ssd_mobilenet_v1_coco_quant_postprocess",
                         /*num_segments=*/GetParam(), /*score_threshold=*/0.7,
                         /*iou_threshold=*/0.8);
}

TEST_P(PipelinedSsdDetectionModelTest, SsdMobilenetV2) {
  TestCatMsCocoDetection("pipeline/ssd_mobilenet_v2_coco_quant_postprocess",
                         /*num_segments=*/GetParam(), /*score_threshold=*/0.95,
                         /*iou_threshold=*/0.86);
}

INSTANTIATE_TEST_CASE_P(PipelinedSsdDetectionModelTest,
                        PipelinedSsdDetectionModelTest,
                        ::testing::ValuesIn(NumSegments()));

}  // namespace
}  // namespace coral
