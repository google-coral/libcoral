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

#ifndef LIBCORAL_CORAL_TOOLS_PARTITIONER_PROFILING_BASED_PARTITIONER_H_
#define LIBCORAL_CORAL_TOOLS_PARTITIONER_PROFILING_BASED_PARTITIONER_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "coral/tools/model_pipelining_benchmark_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
class ProfilingBasedPartitioner {
 public:
  ProfilingBasedPartitioner(const std::string& edgetpu_compiler_binary,
                            const std::string& model_path,
                            EdgeTpuType device_type, int num_segments,
                            const std::string& output_dir);

  virtual ~ProfilingBasedPartitioner() = default;

  // Gets the initial latency lower bound and upper bound in ns by running
  // ParameterCountBasedPartitioner.
  std::pair<int64_t, int64_t> GetBounds(int delegate_search_step);

  // Partitions the cpu tflite model to `num_segments` segments.
  //
  // It follows the rule that the first `num_segments-1` segments would be the
  // longest ones whose latencies are lower than the `target_latency`, and the
  // last segment takes whatever left.
  // `search_step` is the number of operators that are added to or removed from
  // a segment during each iteration. For large models, consider set the step
  // to a larger value. Otherwise, it could take long time to finish due to
  // large search space.
  // Returns the latency of the last segment. Return value can be positive
  // infinite if such partition doesn't exist.
  int64_t PartitionOnTargetLatency(int64_t target_latency,
                                   int partition_search_step = 1,
                                   int delegate_search_step = 1);

  const std::vector<int>& partition() const { return partition_; }
  const tflite::Model* model() const { return model_; }
  const std::vector<int>& exe_order_to_node_idx() const {
    return exe_order_to_node_idx_;
  }

 protected:
  // Partitions, compiles temporary segments, analyzes and returns the latency
  // of `segment_index`th segment in ns.
  //
  // If cache is used and the segment to analyze is in `segment_latency_cache_`,
  // returns the cached latency directly.
  virtual int64_t PartitionCompileAndAnalyze(const std::vector<int>& num_ops,
                                             int segment_index,
                                             int delegate_search_step);

  // Recursively searches model partition to meet the latency target.
  // Returns whether a valid candidate can be found.
  bool SearchPartition(std::vector<int>& candidate_partition,
                       int64_t target_latency, int segment_index,
                       int partition_search_step, int delegate_search_step);

 private:
  const std::string edgetpu_compiler_binary_;
  const std::string model_path_;
  const EdgeTpuType device_type_;
  const int num_segments_;
  // The tflite model content and instance of the input model.
  std::vector<char> model_content_;
  const tflite::Model* model_;

  // Key is the tuple of (start op index, num of ops) of a segment, value is the
  // latency of this segment in ns.
  absl::flat_hash_map<std::pair<int, int>, int64_t> segment_latency_cache_;

  // Segments that have failed compilation.
  std::set<std::pair<int, int>> compilation_failure_segments_;

  // The partition found for the target latency. Empty vector means there is no
  // valid partition.
  std::vector<int> partition_;
  // A mapping between execution order -> node index.
  std::vector<int> exe_order_to_node_idx_;
  int first_output_node_exe_idx_;

  std::vector<std::shared_ptr<EdgeTpuContext>> edgetpu_contexts_;

  // An accessor function that does lazy initialization of edgetpu_contexts_.
  const std::vector<std::shared_ptr<EdgeTpuContext>>& edgetpu_contexts();
};

// Partitions a cpu tflite model into given number of segments in the following
// bisecting way.
//
// 1. In the initialization stage, caller can specify search upper and
// lower bound with `initial_bounds`. If `initial_bounds` contains negative
// values, it will try to find the bounds using heuristic based partitioner,
// which may fail.
//
// 2. At each iteration of bisecting, ProfilingBasedPartitioner is applied to
// search for a partition with given number of segments that meet the target
// time. ProfilingBasedPartitioner finds a valid partition by greedily searching
// for longest segments whose latencies do not exceed the target time. Then
// upper bound or lower bound will be updated based on search result. Bisecting
// will end when the difference between the lower-bound and upper-bound is
// below a certain threshold.
//
// The partitioner tries to continue search when some model segments can not be
// compiled. However, there could be cases, where the partitioner can not find
// a valid partition candidate.
bool BisectPartition(const std::string& edgetpu_compiler_binary,
                     const std::string& model_path, EdgeTpuType device_type,
                     int num_segments, const std::string& output_dir,
                     int64_t diff_threshold_ns, int partition_search_step,
                     int delegate_search_step,
                     const std::pair<int64_t, int64_t>& initial_bounds);
}  // namespace coral

#endif  // LIBCORAL_CORAL_TOOLS_PARTITIONER_PROFILING_BASED_PARTITIONER_H_
