#include <string>

#include "coral/test_utils.h"
#include "gtest/gtest.h"

namespace coral {

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class DetectionModelTest : public ::testing::TestWithParam<std::string> {};

TEST_P(DetectionModelTest, TestSSDModelsWithCat) {
  // Mobilenet V1 SSD.
  // 4 tensors are returned after post processing operator.
  //
  // 1: detected bounding boxes;
  // 2: detected class label;
  // 3: detected score;
  // 4: number of detected objects;
  // Mobilenet V1 SSD
  const auto suffix = GetParam();
  TestCatMsCocoDetection("ssd_mobilenet_v1_coco_quant_postprocess" + suffix,
                         /*score_threshold=*/0.7, /*iou_threshold=*/0.8);
  // Mobilenet V2 SSD
  TestCatMsCocoDetection("ssd_mobilenet_v2_coco_quant_postprocess" + suffix,
                         /*score_threshold=*/0.95, /*iou_threshold=*/0.86);
  // Mobiledet SSDLite
  TestCatMsCocoDetection("ssdlite_mobiledet_coco_qat_postprocess" + suffix,
                         /*score_threshold=*/0.7, /*iou_threshold=*/0.8);
}

TEST_P(DetectionModelTest, TestFaceModel) {
  TestDetection("ssd_mobilenet_v2_face_quant_postprocess" + GetParam(),
                "grace_hopper.bmp",
                /*expected_box=*/{0.21, 0.29, 0.57, 0.74}, /*expected_label=*/0,
                /*score_threshold=*/0.7, /*iou_threshold=*/0.62);
}

TEST_P(DetectionModelTest, TestFineTunedPetModel) {
  TestDetection("ssd_mobilenet_v1_fine_tuned_pet" + GetParam(), "cat.bmp",
                /*expected_box=*/{0.11, 0.35, 0.66, 0.7},
                /*expected_label=*/0, /*score_threshold=*/0.8,
                /*iou_threshold=*/0.81);
}

INSTANTIATE_TEST_CASE_P(DetectionCpuModelTest, DetectionModelTest,
                        ::testing::Values(".tflite"));
INSTANTIATE_TEST_CASE_P(DetectionEdgeTpuModelTest, DetectionModelTest,
                        ::testing::Values("_edgetpu.tflite"));
}  // namespace coral
