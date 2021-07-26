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

#include <string>

#include "coral/test_utils.h"
#include "gtest/gtest.h"
#include "tflite/public/edgetpu.h"

namespace coral {

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class DetectionModelTest : public ModelTestBase {};

TEST_P(DetectionModelTest, TestSsdModelsWithCat) {
  // SSD Mobilenet V1.
  // 4 tensors are returned after post processing operator.
  //
  // 1: detected bounding boxes;
  // 2: detected class label;
  // 3: detected score;
  // 4: number of detected objects;
  // SSD Mobilenet V1
  const auto suffix = GetParam();
  auto* tpu_context = GetTpuContextIfNecessary();
  TestCatMsCocoDetection("ssd_mobilenet_v1_coco_quant_postprocess" + suffix,
                         /*score_threshold=*/0.7, /*iou_threshold=*/0.8,
                         tpu_context);
  // SSD Mobilenet V2
  TestCatMsCocoDetection("ssd_mobilenet_v2_coco_quant_postprocess" + suffix,
                         /*score_threshold=*/0.95, /*iou_threshold=*/0.86,
                         tpu_context);
  // SSDLite Mobiledet
  TestCatMsCocoDetection("ssdlite_mobiledet_coco_qat_postprocess" + suffix,
                         /*score_threshold=*/0.7, /*iou_threshold=*/0.8,
                         tpu_context);

  // TF2 SSD Mobiledet V2
  TestCatMsCocoDetection("tf2_ssd_mobilenet_v2_coco17_ptq" + suffix,
                         /*score_threshold=*/0.7, /*iou_threshold=*/0.8,
                         tpu_context);

  // TF2 SSD Mobiledet V1 FPN 640x640
  TestCatMsCocoDetection("tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq" + suffix,
                         /*score_threshold=*/0.7, /*iou_threshold=*/0.65,
                         tpu_context);

  // EfficientDet lite models
  TestCatMsCocoDetection("efficientdet_lite0_320_ptq" + suffix,
                         /*score_threshold=*/0.55,
                         /*iou_threshold=*/0.80, tpu_context);

  TestCatMsCocoDetection("efficientdet_lite1_384_ptq" + suffix,
                         /*score_threshold=*/0.81,
                         /*iou_threshold=*/0.87, tpu_context);

  TestCatMsCocoDetection("efficientdet_lite2_448_ptq" + suffix,
                         /*score_threshold=*/0.67,
                         /*iou_threshold=*/0.88, tpu_context);

  TestCatMsCocoDetection("efficientdet_lite3_512_ptq" + suffix,
                         /*score_threshold=*/0.75,
                         /*iou_threshold=*/0.85, tpu_context);

  if (tpu_context == nullptr || tpu_context->GetDeviceEnumRecord().type ==
                                    edgetpu::DeviceType::kApexPci) {
    // This model only works with PCIe connected Edge TPU.
    TestCatMsCocoDetection("efficientdet_lite3x_640_ptq" + suffix,
                           /*score_threshold=*/0.6,
                           /*iou_threshold=*/0.85, tpu_context);
  }
}

TEST_P(DetectionModelTest, TestFaceModel) {
  TestDetection("ssd_mobilenet_v2_face_quant_postprocess" + GetParam(),
                "grace_hopper.bmp",
                /*expected_box=*/{0.21, 0.29, 0.57, 0.74}, /*expected_label=*/0,
                /*score_threshold=*/0.7, /*iou_threshold=*/0.62,
                GetTpuContextIfNecessary());
}

TEST_P(DetectionModelTest, TestFineTunedPetModel) {
  TestDetection("ssd_mobilenet_v1_fine_tuned_pet" + GetParam(), "cat.bmp",
                /*expected_box=*/{0.11, 0.35, 0.66, 0.7},
                /*expected_label=*/0, /*score_threshold=*/0.8,
                /*iou_threshold=*/0.81, GetTpuContextIfNecessary());
}

INSTANTIATE_TEST_CASE_P(Cpu, DetectionModelTest, ::testing::Values(".tflite"));

INSTANTIATE_TEST_CASE_P(EdgeTpu, DetectionModelTest,
                        ::testing::Values("_edgetpu.tflite"));
}  // namespace coral
