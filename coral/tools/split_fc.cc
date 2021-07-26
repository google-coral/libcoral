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

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"
#include "coral/tools/tflite_graph_util.h"
#include "glog/logging.h"

ABSL_FLAG(std::string, input_graph, "",
          "Path to the input graph. Must be in tflite format.");

ABSL_FLAG(std::string, output_graph, "",
          "Path to the output graph. Output graph will be in tflite format.");

ABSL_FLAG(std::string, fc_input_tensor_name, "", "FC input tensor name.");
ABSL_FLAG(std::string, fc_weights_tensor_name, "", "FC weights tensor name.");
ABSL_FLAG(std::string, fc_bias_tensor_name, "", "FC bias tensor name.");
ABSL_FLAG(std::string, fc_output_tensor_name, "", "FC output tensor name.");

ABSL_FLAG(int, feature_dim_index, 0,
          "Index of the element in tensor shape array that specifies the "
          "feature dimension.");

ABSL_FLAG(float, split_ratio, 0.5, "FC layer split ratio.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  auto status = coral::SplitFullyConnected(
      absl::GetFlag(FLAGS_input_graph),
      absl::GetFlag(FLAGS_fc_input_tensor_name),
      absl::GetFlag(FLAGS_fc_weights_tensor_name),
      absl::GetFlag(FLAGS_fc_bias_tensor_name),
      absl::GetFlag(FLAGS_fc_output_tensor_name),
      absl::GetFlag(FLAGS_output_graph), absl::GetFlag(FLAGS_feature_dim_index),
      absl::GetFlag(FLAGS_split_ratio));
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
  return 0;
}
