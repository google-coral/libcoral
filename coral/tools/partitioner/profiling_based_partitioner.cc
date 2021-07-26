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

#include "coral/tools/partitioner/profiling_based_partitioner.h"

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <future>  // NOLINT
#include <thread>  // NOLINT

#include "absl/strings/substitute.h"
#include "coral/pipeline/test_utils.h"
#include "coral/tools/model_pipelining_benchmark_util.h"
#include "coral/tools/partitioner/strategy.h"
#include "coral/tools/partitioner/utils.h"
#include "glog/logging.h"
namespace coral {
namespace {

constexpr int kNumInferences = 10;
constexpr bool kUseCache = true;

bool CompileModel(const std::string& compiler_binary,
                  const std::string& model_path, const std::string& output_dir,
                  int num_segments, bool search_delegate,
                  int delegate_search_step = 1) {
  const std::string cmd = absl::Substitute(
      "$0 $1 --out_dir=$2 --num_segments=$3 $4 --delegate_search_step=$5",
      compiler_binary, (search_delegate ? "--search_delegate" : ""), output_dir,
      num_segments, model_path, delegate_search_step);
  VLOG(1) << cmd;
  return system(cmd.c_str()) == 0;
}

inline void DeleteFolder(const std::string& dir_path) {
  const std::string cmd = absl::Substitute("rm -rf $0", dir_path);
  CHECK_EQ(system(cmd.c_str()), 0);
}

void PartitionWithNumOps(const tflite::Model* model,
                         const std::vector<int>& exe_order_to_node_idx,
                         const std::vector<int>& num_ops_per_segment,
                         const std::vector<std::string>& model_segments_paths) {
  CHECK_EQ(num_ops_per_segment.size(), model_segments_paths.size());
  coral::PartitionStrategy strategy = coral::GetStrategyFromNumOps(
      model, exe_order_to_node_idx, num_ops_per_segment);
  for (int i = 0; i < strategy.size(); ++i) {
    const auto& segment_info = strategy[i];
    const auto& segment_contents = coral::ExtractModelSegment(
        *model, segment_info.target_nodes,
        {segment_info.target_inputs.begin(), segment_info.target_inputs.end()},
        {segment_info.target_outputs.begin(),
         segment_info.target_outputs.end()});
    VLOG(1) << "Write segment content to: " << model_segments_paths[i];
    coral::WriteFileOrExit(model_segments_paths[i], segment_contents);
  }
}

inline std::string NaiveStem(const std::string path) {
  auto dot_pos = path.rfind('.');
  auto sep_pos = path.rfind('/');
  auto start_pos = sep_pos == std::string::npos ? 0 : sep_pos + 1;
  auto end_pos = dot_pos == std::string::npos ? path.size() : dot_pos;
  return path.substr(start_pos, end_pos - start_pos);
}

inline std::vector<std::string> GenerateSegmentsNames(
    const std::string& model_base_name, int num_segments,
    const std::string& suffix) {
  std::vector<std::string> result(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    result[i] = absl::StrCat(model_base_name, "_segment_", i, "_of_",
                             num_segments, suffix);
  }
  return result;
}

}  // namespace

ProfilingBasedPartitioner::ProfilingBasedPartitioner(
    const std::string& edgetpu_compiler_binary, const std::string& model_path,
    EdgeTpuType device_type, int num_segments, const std::string& output_dir)
    : edgetpu_compiler_binary_(edgetpu_compiler_binary),
      model_path_(model_path),
      device_type_(device_type),
      num_segments_(num_segments) {
  coral::ReadFileOrExit(model_path, &model_content_);
  model_ = tflite::GetModel(model_content_.data());
  exe_order_to_node_idx_ = TopologicalSort(*model_);

  const auto& edges = BuildEdgeList(*model_);
  const auto& graph = BuildGraph(edges, exe_order_to_node_idx_.size());
  const auto out_degree = CalculateOutDegree(graph);
  first_output_node_exe_idx_ = 0;
  for (; first_output_node_exe_idx_ < exe_order_to_node_idx_.size();
       ++first_output_node_exe_idx_) {
    if (out_degree[exe_order_to_node_idx_[first_output_node_exe_idx_]] == 0)
      break;
  }
  LOG(INFO) << "Total number of operators: " << exe_order_to_node_idx_.size();
  LOG(INFO) << "First output node execution order index: "
            << first_output_node_exe_idx_;
}

const std::vector<std::shared_ptr<EdgeTpuContext>>&
ProfilingBasedPartitioner::edgetpu_contexts() {
  if (edgetpu_contexts_.empty()) {
    edgetpu_contexts_ =
        coral::PrepareEdgeTpuContexts(num_segments_, device_type_);
  }
  return edgetpu_contexts_;
}

int64_t ProfilingBasedPartitioner::PartitionCompileAndAnalyze(
    const std::vector<int>& num_ops, int segment_index,
    int delegate_search_step) {
  LOG(INFO) << "Analyzing partition candidate, num_ops:"
            << absl::StrJoin(num_ops, ",");

  std::vector<int> segment_starts(num_ops.size(), 0);
  for (int i = 1; i < num_ops.size(); ++i) {
    segment_starts[i] = segment_starts[i - 1] + num_ops[i - 1];
  }
  if (kUseCache) {
    if (segment_latency_cache_.find(
            {segment_starts[segment_index], num_ops[segment_index]}) !=
        segment_latency_cache_.end()) {
      return segment_latency_cache_[{segment_starts[segment_index],
                                     num_ops[segment_index]}];
    }
  }

  // Check whether the known failure set contains any segment of the candidate
  // partition. If yes, return immediately.
  for (int i = 0; i < num_ops.size(); ++i) {
    if (compilation_failure_segments_.find({segment_starts[i], num_ops[i]}) !=
        compilation_failure_segments_.end()) {
      LOG(WARNING) << "Can not compile model segment " << i;
      return -1;
    }
  }

  char pattern[] = "/tmp/tmpdir.XXXXXX";
  const std::string tmp_dir = std::string(mkdtemp(pattern)) + "/";
  const std::string out_prefix = NaiveStem(model_path_);
  const auto tmp_cpu_segment_paths =
      GenerateSegmentsNames(tmp_dir + out_prefix, num_segments_, ".tflite");

  PartitionWithNumOps(model_, exe_order_to_node_idx_, num_ops,
                      tmp_cpu_segment_paths);

  VLOG(1) << "Compile model segments asynchronously...";
  std::vector<std::future<bool>> future_results(tmp_cpu_segment_paths.size());
  for (int i = 0; i < tmp_cpu_segment_paths.size(); ++i) {
    future_results[i] =
        std::async(std::launch::async, CompileModel, edgetpu_compiler_binary_,
                   tmp_cpu_segment_paths[i], tmp_dir, /*num_segments=*/1,
                   /*search_delegate=*/true, delegate_search_step);
  }
  bool success = true;
  for (int i = 0; i < future_results.size(); ++i) {
    // Need to wait for all compilation threads to avoid racing condition
    // from compiling different partition candidates.
    if (!future_results[i].get()) {
      LOG(WARNING) << "Can not compile " << tmp_cpu_segment_paths[i];
      // This is a newly discovered compilation failure case. Record it.
      compilation_failure_segments_.insert({segment_starts[i], num_ops[i]});
      success = false;
    }
  }

  std::vector<int64_t> latencies;
  if (success) {
    const auto tmp_edgetpu_segment_paths = GenerateSegmentsNames(
        tmp_dir + out_prefix, num_segments_, "_edgetpu.tflite");
    latencies = std::get<2>(coral::BenchmarkPartitionedModel(
        tmp_edgetpu_segment_paths, &edgetpu_contexts(), kNumInferences));

    if (kUseCache) {
      for (int i = 0; i < num_segments_; ++i) {
        segment_latency_cache_[{segment_starts[i], num_ops[i]}] = latencies[i];
      }
    }
  }

  DeleteFolder(tmp_dir);
  return success ? latencies[segment_index] : -1;
}

std::pair<int64_t, int64_t> ProfilingBasedPartitioner::GetBounds(
    int delegate_search_step) {
  char pattern[] = "/tmp/tmpdir.XXXXXX";
  const std::string tmp_dir = std::string(mkdtemp(pattern)) + "/";
  const auto tmp_edgetpu_segment_paths = GenerateSegmentsNames(
      tmp_dir + NaiveStem(model_path_), num_segments_, "_edgetpu.tflite");
  CHECK(CompileModel(edgetpu_compiler_binary_, model_path_, tmp_dir,
                     num_segments_, /*search_delegate=*/true,
                     delegate_search_step))
      << "Can not compile initial partition.";
  const auto latencies = std::get<2>(coral::BenchmarkPartitionedModel(
      tmp_edgetpu_segment_paths, &edgetpu_contexts(), kNumInferences));

  DeleteFolder(tmp_dir);

  int64_t lower_bound = std::numeric_limits<int64_t>::max(), upper_bound = 0;
  for (auto latency : latencies) {
    lower_bound = std::min(lower_bound, latency);
    upper_bound = std::max(upper_bound, latency);
  }
  return {lower_bound, upper_bound};
}

bool ProfilingBasedPartitioner::SearchPartition(
    std::vector<int>& candidate_partition, int64_t target_latency,
    int segment_index, int partition_search_step, int delegate_search_step) {
  if (segment_index == candidate_partition.size() - 1) {
    // If this is the last segment, it means it already found the valid
    // candidate.
    return true;
  }

  const int searchable_total_num_ops =
      std::accumulate(candidate_partition.begin() + segment_index,
                      candidate_partition.end(), 0);

  // The last segment must contain all output nodes.
  candidate_partition[candidate_partition.size() - 1] =
      std::max(static_cast<int>(exe_order_to_node_idx_.size()) -
                   first_output_node_exe_idx_,
               partition_search_step);
  for (int i = partition_search_step + 1; i < candidate_partition.size() - 1;
       ++i) {
    candidate_partition[i] = partition_search_step;
  }
  candidate_partition[segment_index] =
      searchable_total_num_ops -
      (candidate_partition.size() - 2 - segment_index) * partition_search_step -
      candidate_partition[candidate_partition.size() - 1];
  CHECK_GT(candidate_partition[segment_index], 0);
  CHECK_EQ(std::accumulate(candidate_partition.begin() + segment_index,
                           candidate_partition.end(), 0),
           searchable_total_num_ops)
      << absl::StrJoin(candidate_partition, ",");

  std::vector<int> tmp_partition = candidate_partition;
  while (tmp_partition[segment_index] > 0) {
    const int64_t segment_latency = PartitionCompileAndAnalyze(
        tmp_partition, segment_index, delegate_search_step);
    LOG(INFO) << "Search segment " << segment_index
              << ", latency: " << segment_latency
              << ", target: " << target_latency;

    // If this segment meets the latency target, search for the next segment.
    // If search succeeds, return true, otherwise continue search for the
    // current segment.
    if (segment_latency > 0 && segment_latency <= target_latency) {
      if (SearchPartition(tmp_partition, target_latency, segment_index + 1,
                          partition_search_step, delegate_search_step)) {
        candidate_partition = tmp_partition;
        return true;
      } else {
        LOG(WARNING)
            << "This segment meets latency target. But failed to find valid "
               "candidate for remaining segments. Continue search.";
      }
    }

    // If segment latency is larger than the target or the current partition
    // failed compilation, shorten the segment, and try again.
    const int delta =
        std::min(tmp_partition[segment_index], partition_search_step);
    tmp_partition[segment_index] -= delta;
    tmp_partition[segment_index + 1] += delta;
  }

  return false;
}

int64_t ProfilingBasedPartitioner::PartitionOnTargetLatency(
    int64_t target_latency, int partition_search_step,
    int delegate_search_step) {
  CHECK_GT(partition_search_step, 0);
  // Initially, the first segment takes most operators and each other segment
  // takes one operator.
  const int total_num_ops = exe_order_to_node_idx_.size();
  std::vector<int> num_ops(num_segments_, partition_search_step);
  num_ops[0] = total_num_ops - (num_segments_ - 1) * partition_search_step;
  CHECK_GT(num_ops[0], 0);
  if (!SearchPartition(num_ops, target_latency, 0, partition_search_step,
                       delegate_search_step)) {
    LOG(WARNING) << "Can not find valid partition candidate for target latency "
                 << target_latency;
    return std::numeric_limits<int64_t>::max();
  }

  CHECK_EQ(std::accumulate(num_ops.begin(), num_ops.end(), 0), total_num_ops);
  const auto last_segment_latency = PartitionCompileAndAnalyze(
      num_ops, num_segments_ - 1, delegate_search_step);
  CHECK_GT(last_segment_latency, 0);
  LOG(INFO) << "target_latency: " << target_latency
            << ", num_ops: " << absl::StrJoin(num_ops, ",")
            << ", last segment latency: " << last_segment_latency;
  partition_ = num_ops;
  return last_segment_latency;
}

bool BisectPartition(const std::string& edgetpu_compiler_binary,
                     const std::string& model_path, EdgeTpuType device_type,
                     int num_segments, const std::string& output_dir,
                     int64_t diff_threshold_ns, int partition_search_step,
                     int delegate_search_step,
                     const std::pair<int64_t, int64_t>& initial_bounds) {
  ProfilingBasedPartitioner partitioner =
      ProfilingBasedPartitioner(edgetpu_compiler_binary, model_path,
                                device_type, num_segments, output_dir);
  const std::pair<int64_t, int64_t> bounds =
      (initial_bounds.first >= 0 && initial_bounds.second >= 0)
          ? initial_bounds
          : partitioner.GetBounds(delegate_search_step);
  int64_t lower_bound = bounds.first;
  int64_t upper_bound = bounds.second;
  CHECK_GT(upper_bound, lower_bound);
  std::vector<int> best_partition;
  if (std::abs(upper_bound - lower_bound) <= diff_threshold_ns) {
    LOG(INFO) << "The initial upper and lower bound are close enough. "
                 "Profiling based partitioner is not run.";
    return true;
  }
  LOG(INFO) << "Initial upper_bound: " << upper_bound
            << ", lower_bound: " << lower_bound;

  while (std::abs(upper_bound - lower_bound) > diff_threshold_ns) {
    const int64_t target_latency = (lower_bound + upper_bound) / 2;
    const int64_t last_segment_latency = partitioner.PartitionOnTargetLatency(
        target_latency, partition_search_step, delegate_search_step);
    // If latency of the last segment is lower than `target_latency`, it
    // means that `target_latency` gives a valid partition.
    if (last_segment_latency < target_latency) {
      best_partition = partitioner.partition();
      CHECK(!best_partition.empty());
      upper_bound = target_latency;
    } else {
      // If latency of the last segment is higher than `target_latency` but
      // lower than `upper_bound`, `upper_bound` could be updated as well.
      if (last_segment_latency < upper_bound) {
        best_partition = partitioner.partition();
        CHECK(!best_partition.empty());
        upper_bound = last_segment_latency;
      }
      lower_bound = target_latency;
    }
    LOG(INFO) << "Updated upper_bound: " << upper_bound
              << ", lower_bound: " << lower_bound;
  }

  if (best_partition.empty()) return false;

  // Create and compile final model segments, and save to output directory.
  const auto out_cpu_segment_paths = GenerateSegmentsNames(
      output_dir + "/" + NaiveStem(model_path), num_segments, ".tflite");
  PartitionWithNumOps(partitioner.model(), partitioner.exe_order_to_node_idx(),
                      best_partition, out_cpu_segment_paths);
  for (const auto& path : out_cpu_segment_paths) {
    CHECK(CompileModel(edgetpu_compiler_binary, path, output_dir,
                       /*num_segments=*/1,
                       /*search_delegate=*/true, delegate_search_step));
  }
  return true;
}

}  // namespace coral
