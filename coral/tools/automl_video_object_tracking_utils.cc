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

#include "coral/tools/automl_video_object_tracking_utils.h"

#include <algorithm>

#include "glog/logging.h"

namespace coral {
namespace {
constexpr int kImageInputSize = 256;
constexpr int kHiddenStateSize = 20480;
}  // namespace

std::unique_ptr<tflite::Interpreter> BuildLstmEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context) {
  auto interpreter = MakeEdgeTpuInterpreterOrDie(model, edgetpu_context);
  interpreter->SetNumThreads(1);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  CHECK_EQ(interpreter->inputs().size(), 3);
  CHECK_EQ(interpreter->outputs().size(), 6);

  CHECK(MatchShape(TensorShape(*interpreter->input_tensor(0)),
                   {1, kImageInputSize, kImageInputSize, 3}));

  auto input_lstm_c = MutableTensorData<uint8_t>(*interpreter->input_tensor(1));
  CHECK_EQ(input_lstm_c.size(), kHiddenStateSize);

  auto input_lstm_h = MutableTensorData<uint8_t>(*interpreter->input_tensor(2));
  CHECK_EQ(input_lstm_h.size(), kHiddenStateSize);

  CHECK_EQ(TensorData<uint8_t>(*interpreter->output_tensor(4)).size(),
           kHiddenStateSize);
  CHECK_EQ(TensorData<uint8_t>(*interpreter->output_tensor(5)).size(),
           kHiddenStateSize);

  // Make sure input and output hidden state tensors are not the same.
  CHECK_NE(interpreter->typed_input_tensor<uint8_t>(1),
           interpreter->typed_output_tensor<uint8_t>(4));
  CHECK_NE(interpreter->typed_input_tensor<uint8_t>(2),
           interpreter->typed_output_tensor<uint8_t>(5));

  // Initialize hidden state with zeros.
  std::fill(input_lstm_c.begin(), input_lstm_c.end(), 0);
  std::fill(input_lstm_h.begin(), input_lstm_h.end(), 0);

  return interpreter;
}

void RunLstmInference(tflite::Interpreter* interpreter) {
  CHECK(interpreter);
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

  auto input_lstm_c = MutableTensorData<uint8_t>(*interpreter->input_tensor(1));
  auto output_lstm_c = TensorData<uint8_t>(*interpreter->output_tensor(4));
  std::copy(output_lstm_c.begin(), output_lstm_c.end(), input_lstm_c.begin());

  auto input_lstm_h = MutableTensorData<uint8_t>(*interpreter->input_tensor(2));
  auto output_lstm_h = TensorData<uint8_t>(*interpreter->output_tensor(5));
  std::copy(output_lstm_h.begin(), output_lstm_h.end(), input_lstm_h.begin());
}

}  // namespace coral
