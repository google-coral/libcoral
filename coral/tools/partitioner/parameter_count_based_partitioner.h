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

#ifndef LIBCORAL_CORAL_TOOLS_PARTITIONER_PARAMETER_COUNT_BASED_PARTITIONER_H_
#define LIBCORAL_CORAL_TOOLS_PARTITIONER_PARAMETER_COUNT_BASED_PARTITIONER_H_

#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "coral/tools/partitioner/strategy.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {

// Parameter (size) count based partitioner
class ParameterCountBasedPartitioner {
 public:
  // Note that `input_model_content` is a serialized tflite flatbuffer and must
  // outlives this object.
  explicit ParameterCountBasedPartitioner(
      const std::vector<char>& input_model_content);

  PartitionStrategy GetStrategy(int num_segments) const;

 protected:
  std::vector<int> GetNumOpsPerSegment(int num_segments) const;

 private:
  const tflite::Model* model_;
  // A mapping between execution order -> node index.
  std::vector<int> exe_order_to_node_idx_;
};
}  // namespace coral

#endif  // LIBCORAL_CORAL_TOOLS_PARTITIONER_PARAMETER_COUNT_BASED_PARTITIONER_H_
