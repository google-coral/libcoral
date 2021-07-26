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

// Tool to operate a surgery to tflite models to achieve state_saver
// functionality.
//
// A tflite model can be operated only when there are pairs of
// (input_state_tensor, output_state_tensor) whose tensor shapes are the same.
// The input_state_tensor and output_state_tensor will be pointing to the same
// tensor which becomes a variable tensor. They will also be removed from the
// inputs and outputs of the model subgraph.
//
// Note It is the caller's responsiblity to give the pairs of input/output
// state tensors. The given `input_tensor_names` list and `output_tensor_names`
// list must have an one-to-one mapping in order. Example command line:
// blaze run -c opt :append_recurrent_links -- \
// --input_graph test_data/tools/split_concat_edgetpu.tflite \
// --output_graph /tmp/output_edgetpu.tflite \
// --input_tensor_names inputs/rnn1,inputs/rnn2 \
// --output_tensor_names outputs/rnn1,outputs/rnn2

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"
#include "coral/tools/tflite_graph_util.h"
#include "glog/logging.h"

ABSL_FLAG(std::string, input_graph, "",
          "Path to the input graph. Must be in tflite format.");

ABSL_FLAG(std::string, output_graph, "",
          "Path to the output graph. Output graph will be in tflite format.");

ABSL_FLAG(std::vector<std::string>, input_tensor_names, {},
          "A comma-separated list of input tensor names.");

ABSL_FLAG(std::vector<std::string>, output_tensor_names, {},
          "A comma-separated list of output tensor names.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  auto status = coral::AppendRecurrentLinks(
      absl::GetFlag(FLAGS_input_graph), absl::GetFlag(FLAGS_input_tensor_names),
      absl::GetFlag(FLAGS_output_tensor_names),
      absl::GetFlag(FLAGS_output_graph));
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
  return 0;
}
