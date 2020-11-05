#ifndef EDGETPU_CPP_TOOLS_PARTITIONER_STRATEGY_H_
#define EDGETPU_CPP_TOOLS_PARTITIONER_STRATEGY_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {

// Describes a segment of a tflite graph.
struct SegmentConfig {
  // Indices of operators that make up the segment of a graph, must be sorted
  // topologically.
  std::vector<int> target_nodes;
  // Indices of tensors that are considered as inputs to the segment.
  absl::flat_hash_set<int> target_inputs;
  // Indices of tensors that are considered as outputs to the segment.
  absl::flat_hash_set<int> target_outputs;
};

using PartitionStrategy = std::vector<SegmentConfig>;

PartitionStrategy GetStrategyFromNumOps(
    const tflite::Model* model, const std::vector<int>& num_ops_per_segment);

}  // namespace coral
#endif  // EDGETPU_CPP_TOOLS_PARTITIONER_STRATEGY_H_
