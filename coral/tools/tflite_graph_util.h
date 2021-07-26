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

#ifndef LIBCORAL_CORAL_TOOLS_TFLITE_GRAPH_UTIL_H_
#define LIBCORAL_CORAL_TOOLS_TFLITE_GRAPH_UTIL_H_

// Utility library for tflite graph tooling related functions.

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/model.h"

namespace coral {

// Concatenates two tflite models into one, assuming each input model has
// only one subgraph.
// Optional Args:
//   bypass_output_tensors: A list of output tensor names from model0, which
//                          should also become output tensors in the merged
//                          graph (i.e. skip model1). By default any output
//                          tensors of model0 which are not input tensors for
//                          model1 become dead ends.
void ConcatModels(const tflite::Model& model0, const tflite::Model& model1,
                  flatbuffers::FlatBufferBuilder* builder,
                  const std::vector<std::string>& bypass_output_tensors = {});

// Appends recurrent links between certain pairs of output tensor and input
// tensor to the input tflite model from `input_path`, saves the updated tflite
// model to `output_path`.
//
// The steps to each pair of tensors include:
// 1. Point the output tensor to the input tensor.
// 2. Mark the input tensor as Variable.
// 3. Remove both tensors from the model subgraph's inputs and outputs.
//
// This function requires valid `input_tensor_names` and `output_tensor_names`,
// and will return error when
// 1. Size of `input_tensor_names` and size of `output_tensor_names` mismatch.
// 2. Any tensor name in `input_tensor_names` is not in the model's input
//    tensors, or any tensor name in `output_tensor_names` is not in the model's
//    output tensors.
// 3. Any pair of `input_tensor` and `output_tensor` have different tensor
//    shapes or data types.
absl::Status AppendRecurrentLinks(
    const absl::string_view input_path,
    const std::vector<std::string>& input_tensor_names,
    const std::vector<std::string>& output_tensor_names,
    const absl::string_view output_path);

// Splits the specified FC layer into two and then merges the result back along
// the feature dimension. For example with a split-ratio of 0.5, a FC layer
// with output feature dimension of 1000 will be split into two FC layers with
// output feature dimension of 500, and then a concat layer will be appended
// after the two FC layers to merge their results back to 1000-d.
// `feature_dim_index` indicates which element in tensor shape array specifies
// the feature dimension. Dimensions other than the feature dimension, e.g.
// batch remains the same during the operation. In practice, the fully connected
// layer can be implemented as Conv 1x1.
absl::Status SplitFullyConnected(const std::string& input_model_path,
                                 const std::string& fc_input_tensor_name,
                                 const std::string& fc_weights_tensor_name,
                                 const std::string& fc_bias_tensor_name,
                                 const std::string& fc_output_tensor_name,
                                 const std::string& output_model_path,
                                 int feature_dim_index,
                                 float split_ratio = 0.5);

}  // namespace coral

#endif  // LIBCORAL_CORAL_TOOLS_TFLITE_GRAPH_UTIL_H_
