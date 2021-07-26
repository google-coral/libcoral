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

#include "coral/tools/partitioner/parameter_count_based_partitioner.h"

#include <iterator>
#include <numeric>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "coral/tools/partitioner/strategy.h"
#include "coral/tools/partitioner/utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {

ParameterCountBasedPartitioner::ParameterCountBasedPartitioner(
    const std::vector<char>& input_model_content)
    : model_(CHECK_NOTNULL(tflite::GetModel(input_model_content.data()))) {
  CHECK_EQ(model_->subgraphs()->size(), 1) << "Only 1 subgraph is supported.";
  exe_order_to_node_idx_ = TopologicalSort(*model_);
}

PartitionStrategy ParameterCountBasedPartitioner::GetStrategy(
    int num_segments) const {
  CHECK_GT(num_segments, 1) << "num_segments must be >1.";
  const auto& num_ops_per_segment = GetNumOpsPerSegment(num_segments);
  return GetStrategyFromNumOps(model_, exe_order_to_node_idx_,
                               num_ops_per_segment);
}

std::vector<int> ParameterCountBasedPartitioner::GetNumOpsPerSegment(
    int num_segments) const {
  // Calculate parameter sizes.
  const auto& parameter_sizes = CalculateParameterSizes(*model_);
  const int64_t total_size =
      std::accumulate(parameter_sizes.begin(), parameter_sizes.end(), 0LL);
  VLOG(1) << "total parameter size (bytes): " << total_size;
  const int64_t average_size = (total_size / num_segments) + 1;

  // Search for the first output node.
  const int num_nodes = parameter_sizes.size();
  const auto& edges = BuildEdgeList(*model_);
  const auto& graph = BuildGraph(edges, num_nodes);
  const auto out_degree = CalculateOutDegree(graph);
  int first_output_node_exe_idx = 0;
  while (out_degree[exe_order_to_node_idx_[first_output_node_exe_idx]] != 0) {
    ++first_output_node_exe_idx;
    CHECK_LT(first_output_node_exe_idx, exe_order_to_node_idx_.size());
  }

  // Build `num_ops_per_segment`.
  std::vector<int> num_ops_per_segment(num_segments, 0);
  std::vector<int64_t> segment_sizes(num_segments, 0);
  int allocated_parameter_size = 0;
  int node_i = 0;
  for (int segment_i = 0; segment_i < num_segments - 1; ++segment_i) {
    const int max_segment_end_exe_idx =
        first_output_node_exe_idx - (num_segments - segment_i - 1);
    CHECK_GE(max_segment_end_exe_idx, node_i);
    while (node_i <= max_segment_end_exe_idx) {
      const int64_t cur_size = parameter_sizes[exe_order_to_node_idx_[node_i]];
      // Cutting just before `average_size` is achieved seems to be a better
      // heuristic as later segment of a model tend to require less computation
      // given same amount of parameter size.
      if ((segment_sizes[segment_i] + cur_size) > average_size) break;
      allocated_parameter_size += cur_size;
      segment_sizes[segment_i] += cur_size;
      num_ops_per_segment[segment_i]++;
      ++node_i;
    }
  }
  CHECK_LE(node_i, first_output_node_exe_idx);
  num_ops_per_segment[num_segments - 1] = (num_nodes - node_i);
  segment_sizes[num_segments - 1] = (total_size - allocated_parameter_size);

  // Sanity check
  for (int i = 0; i < num_segments; ++i) {
    CHECK_GT(num_ops_per_segment[i], 0);
  }
  CHECK_EQ(num_nodes, std::accumulate(num_ops_per_segment.begin(),
                                      num_ops_per_segment.end(), 0));

  for (int i = 0; i < num_segments; ++i) {
    VLOG(1) << "Segment idx: " << i
            << " total weights size: " << segment_sizes[i]
            << " num of segment ops: " << num_ops_per_segment[i];
  }

  return num_ops_per_segment;
}

}  // namespace coral
