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

#ifndef LIBCORAL_CORAL_TOOLS_PARTITIONER_STRATEGY_H_
#define LIBCORAL_CORAL_TOOLS_PARTITIONER_STRATEGY_H_

#include <vector>

#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {

// Describes a segment of a tflite graph.
struct SegmentConfig {
  // Indices of operators that make up the segment of a graph, must be sorted
  // topologically.
  std::vector<int> target_nodes;
  // Indices of tensors that are considered as inputs to the segment.
  std::vector<int> target_inputs;
  // Indices of tensors that are considered as outputs to the segment.
  std::vector<int> target_outputs;
};

using PartitionStrategy = std::vector<SegmentConfig>;

PartitionStrategy GetStrategyFromNumOps(
    const tflite::Model* model, const std::vector<int>& exe_order_to_node_idx,
    const std::vector<int>& num_ops_per_segment);

// This is a convenient version of the above function. It calls
// coral::TopologicalSort to get the execution order of operators.
PartitionStrategy GetStrategyFromNumOps(
    const tflite::Model* model, const std::vector<int>& num_ops_per_segment);

}  // namespace coral
#endif  // LIBCORAL_CORAL_TOOLS_PARTITIONER_STRATEGY_H_
