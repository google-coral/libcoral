#ifndef EDGETPU_CPP_TOOLS_PARTITIONER_PROFILING_PARTITION_UTIL_H_
#define EDGETPU_CPP_TOOLS_PARTITIONER_PROFILING_PARTITION_UTIL_H_

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

  void SaveTempSegments();

  void DeleteTempSegments();

  // Gets the initial latency lower bound and upper bound in ns by running
  // ParameterCountBasedPartitioner.
  std::pair<int64_t, int64_t> GetBounds();

  // Partitions the cpu tflite model to `num_segments` segments.
  //
  // It follows the rule that the first `num_segments-1` segments would be the
  // longest ones whose latencies are lower than the `target_latency`, and the
  // last segment takes whatever left.
  std::vector<int64_t> PartitionOnTargetLatency(int64_t target_latency);

 protected:
  // Partitions, compiles temporary segments, analyzes and returns the latency
  // of `segment_index`th segment in ns.
  //
  // If cache is used and the segment to analyze is in `segment_latency_cache_`,
  // returns the cached latency directly.
  virtual int64_t PartitionCompileAndAnalyze(const std::vector<int>& num_ops,
                                             int segment_index);

 private:
  const std::string edgetpu_compiler_binary_;
  const std::string model_path_;
  const EdgeTpuType device_type_;
  const int num_segments_;
  const std::string output_dir_;
  // The tflite model content and instance of the input model.
  std::vector<char> model_content_;
  const tflite::Model* model_;
  // Number of nodes of the input model.
  int num_nodes_;

  // Hard-coded values, subject to change.
  const int num_inferences_ = 10;
  const bool use_cache_ = true;
  // Key is the tuple of (start op index, num of ops) of a segment, value is the
  // latency of this segment in ns.
  absl::flat_hash_map<std::pair<int, int>, int64_t> segment_latency_cache_;

  std::vector<std::string> tmp_cpu_segment_paths_, tmp_edgetpu_segment_paths_,
      out_cpu_segment_paths_, out_edgetpu_segment_paths_,
      tmp_edgetpu_segment_logs_, out_edgetpu_segment_logs_;
};

// Partitions a cpu tflite model into given number of segments in the following
// bisecting way.

// 1. In the initialization stage, ParameterCountBasedPartitioner is applied to
// serve as the baseline, and the longest segment inference time and shortest
// segment inference time would be used as the upper-bound and lower-bound of
// bisecting.
// 2. At each iteration of bisecting, ProfilingBasedPartitioner is applied to
// search for a partition with given number of segments that meet the target
// time. ProfilingBasedPartitioner finds a valid partition by greedily searching
// for longest segments whose latencies do not exceed the target time.
// If such partition can be found, the upper-bound will be set to the target
// time at current iteration; otherwise, the lower-bound will be set to the
// target time. Bisecting will end when the difference between the lower-bound
// and upper-bound is below a certain threshold.
void BisectPartition(const std::string& edgetpu_compiler_binary,
                     const std::string& model_path, EdgeTpuType device_type,
                     int num_segments, const std::string& output_dir,
                     int diff_threshold_ns);
}  // namespace coral

#endif  // EDGETPU_CPP_TOOLS_PARTITIONER_PROFILING_PARTITION_UTIL_H_
