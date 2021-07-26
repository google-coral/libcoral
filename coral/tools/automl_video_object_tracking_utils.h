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

// Utility library that help run the AutoML video object tracking model, which
// is a modified LSTM model with multiple input tensors.
#ifndef LIBCORAL_CORAL_TOOLS_AUTOML_VIDEO_OBJECT_TRACKING_UTILS_H_
#define LIBCORAL_CORAL_TOOLS_AUTOML_VIDEO_OBJECT_TRACKING_UTILS_H_

#include <memory>
#include <vector>

#include "coral/tflite_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tflite/public/edgetpu.h"

namespace coral {

// Builds the interpreter and sets LSTM hidden states to zero.
// It assumes the model has 3 inputs: image, hidden state c, and hidden state h,
// and 6 outputs: 4 output tensors from SSD postprocessing, and hidden states
// c and h.
std::unique_ptr<tflite::Interpreter> BuildLstmEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context);

// Runs inference and updates the LSTM hidden states.
void RunLstmInference(tflite::Interpreter* interpreter);
}  // namespace coral

#endif  // LIBCORAL_CORAL_TOOLS_AUTOML_VIDEO_OBJECT_TRACKING_UTILS_H_
