#include "coral/tools/partitioner/partitioner.h"

#include <iterator>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "coral/tools/partitioner/strategy.h"
#include "coral/tools/partitioner/utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {

CountBasedPartitioner::CountBasedPartitioner(
    const std::vector<char>& input_model_content)
    : model_(CHECK_NOTNULL(tflite::GetModel(input_model_content.data()))) {
  CHECK_EQ(model_->subgraphs()->size(), 1) << "Only 1 subgraph is supported.";
}

PartitionStrategy CountBasedPartitioner::GetStrategy(int num_segments) const {
  CHECK_GT(num_segments, 1) << "num_segments must be >1.";
  const auto& num_ops_per_segment = GetNumOpsPerSegment(num_segments);
  return GetStrategyFromNumOps(model_, num_ops_per_segment);
}

ParameterCountBasedPartitioner::ParameterCountBasedPartitioner(
    const std::vector<char>& input_model_content)
    : CountBasedPartitioner(input_model_content) {}

std::vector<int> ParameterCountBasedPartitioner::GetNumOpsPerSegment(
    int num_segments) const {
  // Calculate parameter sizes.
  const auto& parameter_sizes = CalculateParameterSizes(*model_);
  int64_t total_size = 0;
  for (int i = 0; i < parameter_sizes.size(); ++i) {
    total_size += parameter_sizes[i];
  }

  VLOG(1) << "total parameter size (bytes): " << total_size;

  const int64_t average_size = (total_size / num_segments) + 1;

  // Construct graph.
  const auto& edges = BuildEdgeList(*model_);
  const int num_nodes = parameter_sizes.size();
  const auto& graph = BuildGraph(edges, num_nodes);
  // Find execution order based on topological sorted order. This is a mapping
  // between execution order -> node index.
  const auto& exe_order_to_node_idx = TopologicalSort(graph);

  // Build `num_ops_per_segment`.
  std::vector<int> num_ops_per_segment(num_segments, 0);
  std::vector<int64_t> segment_sizes(num_segments, 0);
  int segment_index = 0;
  int allocated_parameter_size = 0;
  for (int i = 0; i < num_nodes; ++i) {
    const int64_t cur_size = parameter_sizes[exe_order_to_node_idx[i]];
    // Cutting just before `average_size` is achieved seems to be a better
    // heuristic as later segment of a model tend to require less computation
    // given same amount of parameter size.
    if ((segment_sizes[segment_index] + cur_size) > average_size) {
      segment_index++;
    }
    // The last segment takes whatever is left.
    if (segment_index == (num_segments - 1)) {
      num_ops_per_segment[segment_index] += (num_nodes - i);
      segment_sizes[segment_index] += (total_size - allocated_parameter_size);
      break;
    }

    allocated_parameter_size += cur_size;
    segment_sizes[segment_index] += cur_size;
    num_ops_per_segment[segment_index]++;
  }

  // Sanity check
  for (int i = 0; i < num_segments; ++i) {
    CHECK_GT(num_ops_per_segment[i], 0);
  }

  for (int i = 0; i < num_segments; ++i) {
    VLOG(1) << "Segment idx: " << i
            << " total weights size: " << segment_sizes[i]
            << " num of segment ops: " << num_ops_per_segment[i];
  }

  return num_ops_per_segment;
}

}  // namespace coral
