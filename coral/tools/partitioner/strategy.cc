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

#include "coral/tools/partitioner/strategy.h"

#include "coral/tools/partitioner/utils.h"
#include "glog/logging.h"

namespace coral {

PartitionStrategy GetStrategyFromNumOps(
    const tflite::Model* model, const std::vector<int>& num_ops_per_segment) {
  return GetStrategyFromNumOps(model, TopologicalSort(*model),
                               num_ops_per_segment);
}

PartitionStrategy GetStrategyFromNumOps(
    const tflite::Model* model, const std::vector<int>& exe_order_to_node_idx,
    const std::vector<int>& num_ops_per_segment) {
  const int num_segments = num_ops_per_segment.size();
  CHECK_GT(num_segments, 1) << "num_segments must be >1.";
  for (int num_ops : num_ops_per_segment) {
    CHECK_GT(num_ops, 0) << "each segment needs at least one operator.";
  }

  // Construct graph.
  const auto& edges = BuildEdgeList(*model);

  // Locate segment nodes.
  const int num_nodes = model->subgraphs()->Get(0)->operators()->size();
  CHECK_LE(num_segments, num_nodes);
  CHECK_EQ(std::accumulate(num_ops_per_segment.begin(),
                           num_ops_per_segment.end(), 0),
           num_nodes)
      << "Sum of num_segment_ops must be equal to model's nodes number.";
  const auto& segment_nodes_list = LocateSubgraphNodes(
      edges, num_nodes, exe_order_to_node_idx, num_ops_per_segment);
  CHECK_EQ(segment_nodes_list.size(), num_segments);

  // Construct SegmentConfig
  std::vector<SegmentConfig> segment_configs(num_segments);
  const auto* ops = model->subgraphs()->Get(0)->operators();
  for (int i = 0; i < num_segments; ++i) {
    const auto& segment_nodes = segment_nodes_list[i];
    segment_configs[i].target_nodes = segment_nodes.all_nodes;
    // Set input tensors indices.
    for (const int op_index : segment_nodes.input_nodes) {
      const auto* tensor_indices = (op_index == kGraphInputGenNode)
                                       ? model->subgraphs()->Get(0)->inputs()
                                       : ops->Get(op_index)->outputs();
      if (op_index != kGraphInputGenNode) {
        CHECK_EQ(tensor_indices->size(), 1)
            << "Only one output tensor per operator. op_index: " << op_index
            << ", tensor_indices size: " << tensor_indices->size();
      }

      std::copy(tensor_indices->begin(), tensor_indices->end(),
                std::inserter(segment_configs[i].target_inputs,
                              segment_configs[i].target_inputs.end()));
    }
    // Set output tensors indices.
    for (const int op_index : segment_nodes.output_nodes) {
      const auto* op = ops->Get(op_index);
      if (op_index < num_nodes - 1) {
        // It assumes that all operators must have only one output tensor,
        // except for the last one. This is to accommodate models with custom
        // operator in the end, such as SSD models.
        CHECK_EQ(op->outputs()->size(), 1)
            << "Only one output tensor per operator. op_index: " << op_index
            << ", tensor_indices size: " << op->outputs()->size();
      }
      std::copy(op->outputs()->begin(), op->outputs()->end(),
                std::inserter(segment_configs[i].target_outputs,
                              segment_configs[i].target_outputs.end()));
    }
  }

  return segment_configs;
}

}  // namespace coral
