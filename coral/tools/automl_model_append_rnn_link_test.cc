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

#include <cmath>
#include <numeric>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "coral/tools/automl_video_object_tracking_utils.h"
#include "coral/tools/tflite_graph_util.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {

// Gets the flatten results of the first `tensor_size` tensors of the
// interpreter.
std::vector<float> GetFlattenResults(const tflite::Interpreter& interpreter,
                                     int tensor_size) {
  std::vector<float> flatten_results;
  for (int i = 0; i < tensor_size; ++i) {
    auto results = TensorData<float>(*interpreter.output_tensor(i));
    flatten_results.insert(flatten_results.end(), results.begin(),
                           results.end());
  }
  return flatten_results;
}

// Runs inferences on several traffic frames, checks the detection results,
// returns the raw inference results of the first 4 output tensors.
void TestAutoMlVideoObjectTrackingLarge(
    std::function<void(tflite::Interpreter&)> run_inference,
    tflite::Interpreter& interpreter,
    std::vector<std::vector<float>>& flatten_raw_results) {
  // Label map is
  /*
   item {
     name: "sedan"
     id: 1
   }
   item {
     name: "large_veh_bus"
     id: 2
   }
   item {
     name: "pickup_suv_van"
     id: 3
   }
  */
  // Note the model prediction starts from 0.
  // Outputs are:
  //   TFLite_Detection_PostProcess,
  //   TFLite_Detection_PostProcess:1,
  //   TFLite_Detection_PostProcess:2,
  //   TFLite_Detection_PostProcess:3,
  //   raw_outputs/lstm_c, raw_outputs/lstm_h (for model w/o recurrent links)
  const float* boxes = interpreter.typed_output_tensor<float>(0);
  const float* confidences = interpreter.typed_output_tensor<float>(2);
  const float* classes = interpreter.typed_output_tensor<float>(1);
  const std::vector<BBox<float>> expected_boxes = {
      BBox<float>{0.58, 0.19, 0.78, 0.50},
      BBox<float>{0.58, 0.15, 0.78, 0.46},
      BBox<float>{0.58, 0.11, 0.78, 0.43},
  };
  const std::vector<float> expected_scores = {0.35, 0.35, 0.35};

  const std::vector<float> expected_ious = {0.55, 0.55, 0.55};
  const int kExpectedClassIndex = 0;

  const int num_frames = expected_boxes.size();
  flatten_raw_results.resize(num_frames);
  for (int frame_i = 0; frame_i < num_frames; ++frame_i) {
    CopyResizedImage(
        TestDataPath(absl::StrFormat(
            "automl_video_ondevice/traffic_frames/%04d.bmp", frame_i + 1)),
        *interpreter.input_tensor(0));
    run_inference(interpreter);
    const int num_detections =
        static_cast<int>(interpreter.typed_output_tensor<float>(3)[0]);
    ASSERT_GT(num_detections, 0);

    // Check the detection result that best matches the expectation.
    int best_match_class_index = -1;
    float best_match_score = 0.;
    float best_match_iou = 0.;
    for (int j = 0; j < num_detections; ++j) {
      const BBox<float> box{boxes[4 * j], boxes[4 * j + 1], boxes[4 * j + 2],
                            boxes[4 * j + 3]};
      const float iou = IntersectionOverUnion(expected_boxes[frame_i], box);
      if (iou > best_match_iou && confidences[j] > best_match_score) {
        best_match_class_index = classes[j];
        best_match_iou = iou;
        best_match_score = confidences[j];
      }
    }
    EXPECT_EQ(best_match_class_index, kExpectedClassIndex) << absl::StrFormat(
        "Detection result at frame %d has wrong label %d (expected %d)",
        frame_i, best_match_class_index, kExpectedClassIndex);
    EXPECT_GE(best_match_score, expected_scores[frame_i]);
    EXPECT_GE(best_match_iou, expected_ious[frame_i]);
    flatten_raw_results[frame_i] = GetFlattenResults(interpreter, 4);
  }
}

class AutoMlVotRnnModelEdgeTpuTest : public EdgeTpuCacheTestBase {};

TEST_F(AutoMlVotRnnModelEdgeTpuTest, AppendRnnLinks) {
  // automl_vot_large model is a LSTM model whose hidden states are maintained
  // by client codes.
  auto model_before = LoadModelOrDie(
      TestDataPath("automl_video_ondevice/traffic_model_edgetpu.tflite"));
  // The continuously generated test model must contain the EdgeTPU operator to
  // actually test the changes.
  ASSERT_TRUE(ContainsEdgeTpuCustomOp(*model_before));

  auto tpu_context = GetTpuContextCache();
  auto interpreter_before =
      BuildLstmEdgeTpuInterpreter(*model_before, tpu_context);
  std::vector<std::vector<float>> raw_results_before;
  TestAutoMlVideoObjectTrackingLarge(
      [](tflite::Interpreter& interpreter) { RunLstmInference(&interpreter); },
      *interpreter_before, raw_results_before);

  // After recurrent links are added, the client code is not needed anymore.
  std::string output_path = "/tmp/traffic_model_recurrent_links_edgetpu.tflite";
  ASSERT_EQ(
      AppendRecurrentLinks(
          TestDataPath("automl_video_ondevice/traffic_model_edgetpu.tflite"),
          /*input_tensor_names=*/
          {"raw_inputs/init_lstm_c", "raw_inputs/init_lstm_h"},
          /*output_tensor_names=*/{"raw_outputs/lstm_c", "raw_outputs/lstm_h"},
          output_path),
      absl::OkStatus());
  auto model_after = LoadModelOrDie(output_path);
  auto interpreter_after =
      MakeEdgeTpuInterpreterOrDie(*model_after, tpu_context);
  ASSERT_EQ(interpreter_after->AllocateTensors(), kTfLiteOk);
  std::vector<std::vector<float>> raw_results_after;
  TestAutoMlVideoObjectTrackingLarge(
      [](tflite::Interpreter& interpreter) {
        ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
      },
      *interpreter_after, raw_results_after);

  // Models w/ and w/o recurrent links should have exactly the same inference
  // results.
  EXPECT_EQ(raw_results_before, raw_results_after);
}

}  // namespace coral
