#include "coral/tools/partitioner/profiling_partition_util.h"

#include <cmath>
#include <cstdio>

#include "absl/strings/substitute.h"
#include "coral/pipeline/test_utils.h"
#include "coral/tools/model_pipelining_benchmark_util.h"
#include "coral/tools/partitioner/strategy.h"
#include "coral/tools/partitioner/utils.h"
#include "glog/logging.h"
namespace coral {
namespace {
void CompileModel(const std::string& compiler_binary,
                  const std::string& model_path, const std::string& output_dir,
                  int num_segments = 1) {
  const std::string cmd =
      absl::Substitute("$0 --out_dir=$1 --num_segments=$2 $3", compiler_binary,
                       output_dir, num_segments, model_path);
  LOG(INFO) << cmd;
  CHECK_EQ(system(cmd.c_str()), 0);
}

// Analyzes the pipelining performance of given segments. Returns the latencies
// of all segments.
std::vector<int64_t> AnalyzePipelining(
    const std::vector<std::string>& model_segments_paths, int num_segments,
    EdgeTpuType device_type, int num_inferences) {
  auto edgetpu_contexts =
      coral::PrepareEdgeTpuContexts(num_segments, device_type);

  coral::PerfStats stats = coral::BenchmarkPartitionedModel(
      model_segments_paths, &edgetpu_contexts, num_inferences);
  return std::get<2>(stats);
}

void PartitionWithNumOps(const tflite::Model* model,
                         const std::vector<int>& num_ops_per_segment,
                         const std::vector<std::string>& model_segments_paths) {
  CHECK_EQ(num_ops_per_segment.size(), model_segments_paths.size());
  coral::PartitionStrategy strategy =
      coral::GetStrategyFromNumOps(model, num_ops_per_segment);
  for (int i = 0; i < strategy.size(); ++i) {
    const auto& segment_info = strategy[i];
    const auto& segment_contents = coral::ExtractModelSegment(
        *model, segment_info.target_nodes,
        {segment_info.target_inputs.begin(), segment_info.target_inputs.end()},
        {segment_info.target_outputs.begin(),
         segment_info.target_outputs.end()});
    LOG(INFO) << "Write segment content to: " << model_segments_paths[i];
    coral::WriteFileOrExit(model_segments_paths[i], segment_contents);
  }
}

std::string NaiveStem(const std::string path) {
  auto dot_pos = path.rfind('.');
  auto sep_pos = path.rfind('/');
  auto start_pos = sep_pos == std::string::npos ? 0 : sep_pos + 1;
  auto end_pos = dot_pos == std::string::npos ? path.size() : dot_pos;
  return path.substr(start_pos, end_pos - start_pos);
}

std::vector<std::string> GenerateSegmentsNames(
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
      num_segments_(num_segments),
      output_dir_(output_dir) {
  coral::ReadFileOrExit(model_path, &model_content_);
  model_ = tflite::GetModel(model_content_.data());
  num_nodes_ = model_->subgraphs()->Get(0)->operators()->size();

  const std::string tmp_prefix = "tmp";
  const std::string out_prefix = NaiveStem(model_path);
  tmp_cpu_segment_paths_ = GenerateSegmentsNames(output_dir + "/" + tmp_prefix,
                                                 num_segments, ".tflite");
  tmp_edgetpu_segment_paths_ = GenerateSegmentsNames(
      output_dir + "/" + tmp_prefix, num_segments, "_edgetpu.tflite");
  tmp_edgetpu_segment_logs_ = GenerateSegmentsNames(
      output_dir + "/" + tmp_prefix, num_segments, "_edgetpu.log");
  out_cpu_segment_paths_ = GenerateSegmentsNames(output_dir + "/" + out_prefix,
                                                 num_segments, ".tflite");
  out_edgetpu_segment_paths_ = GenerateSegmentsNames(
      output_dir + "/" + out_prefix, num_segments, "_edgetpu.tflite");
  out_edgetpu_segment_logs_ = GenerateSegmentsNames(
      output_dir + "/" + out_prefix, num_segments, "_edgetpu.log");
}

void ProfilingBasedPartitioner::SaveTempSegments() {
  for (int i = 0; i < num_segments_; ++i) {
    const std::string cmd = absl::Substitute(
        "cp $0 $1 && cp $2 $3 && cp $4 $5", tmp_cpu_segment_paths_[i],
        out_cpu_segment_paths_[i], tmp_edgetpu_segment_paths_[i],
        out_edgetpu_segment_paths_[i], tmp_edgetpu_segment_logs_[i],
        out_edgetpu_segment_logs_[i]);
    LOG(INFO) << cmd;
    CHECK_EQ(system(cmd.c_str()), 0);
  }
}

void ProfilingBasedPartitioner::DeleteTempSegments() {
  for (int i = 0; i < num_segments_; ++i) {
    CHECK_EQ(std::remove(tmp_cpu_segment_paths_[i].c_str()), 0);
    CHECK_EQ(std::remove(tmp_edgetpu_segment_paths_[i].c_str()), 0);
    CHECK_EQ(std::remove(tmp_edgetpu_segment_logs_[i].c_str()), 0);
  }
}

int64_t ProfilingBasedPartitioner::PartitionCompileAndAnalyze(
    const std::vector<int>& num_ops, int segment_index) {
  if (use_cache_) {
    int start_op_index =
        std::accumulate(num_ops.begin(), num_ops.begin() + segment_index, 0);
    if (segment_latency_cache_.find({start_op_index, num_ops[segment_index]}) !=
        segment_latency_cache_.end()) {
      return segment_latency_cache_[{start_op_index, num_ops[segment_index]}];
    }
  }

  PartitionWithNumOps(model_, num_ops, tmp_cpu_segment_paths_);
  for (const std::string& segment_path : tmp_cpu_segment_paths_) {
    CompileModel(edgetpu_compiler_binary_, segment_path, output_dir_);
  }
  std::vector<int64_t> latencies = AnalyzePipelining(
      tmp_edgetpu_segment_paths_, num_segments_, device_type_, num_inferences_);

  if (use_cache_) {
    int start_op_index = 0;
    for (int i = 0; i < num_segments_; ++i) {
      segment_latency_cache_[{start_op_index, num_ops[i]}] = latencies[i];
      start_op_index += num_ops[i];
    }
  }
  return latencies[segment_index];
}

std::pair<int64_t, int64_t> ProfilingBasedPartitioner::GetBounds() {
  CompileModel(edgetpu_compiler_binary_, model_path_, output_dir_,
               num_segments_);
  std::vector<int64_t> latencies = AnalyzePipelining(
      out_edgetpu_segment_paths_, num_segments_, device_type_, num_inferences_);

  int64_t lower_bound = std::numeric_limits<int64_t>::max(), upper_bound = 0;
  for (auto latency : latencies) {
    lower_bound = std::min(lower_bound, latency);
    upper_bound = std::max(upper_bound, latency);
  }
  return {lower_bound, upper_bound};
}

std::vector<int64_t> ProfilingBasedPartitioner::PartitionOnTargetLatency(
    int64_t target_latency) {
  std::vector<int64_t> segment_latencies(num_segments_,
                                         std::numeric_limits<int64_t>::max());
  // Initially, the first segment takes most operators and each other segment
  // takes one operator.
  std::vector<int> num_ops(num_segments_, 1);
  num_ops[0] = num_nodes_ - (num_segments_ - 1);

  bool early_stop = false;
  for (int segment_index = 0; segment_index < num_segments_ - 1;
       ++segment_index) {
    // Find the longest segment whose latency is no higher than
    // target_latency.
    int64_t segment_latency;
    while (true) {
      segment_latency = PartitionCompileAndAnalyze(num_ops, segment_index);
      if (segment_latency > target_latency) {
        // If the latency of one-op segment exceeds the time limit, set the
        // last segment latency to infinity and early stop.
        if (num_ops[segment_index] == 1) {
          segment_latencies[num_segments_ - 1] =
              std::numeric_limits<int64_t>::max();
          early_stop = true;
          break;
        }
        num_ops[segment_index]--;
        num_ops[segment_index + 1]++;
      } else {
        break;
      }
    }
    if (early_stop) {
      break;
    }
    segment_latencies[segment_index] = segment_latency;
  }
  // Last segment takes whatever is left.
  if (!early_stop) {
    segment_latencies[num_segments_ - 1] =
        PartitionCompileAndAnalyze(num_ops, num_segments_ - 1);
  }
  LOG(INFO) << "target_latency: " << target_latency
            << ", num_ops: " << absl::StrJoin(num_ops, ",")
            << ", latencies: " << absl::StrJoin(segment_latencies, ",");
  return segment_latencies;
}

void BisectPartition(const std::string& edgetpu_compiler_binary,
                     const std::string& model_path, EdgeTpuType device_type,
                     int num_segments, const std::string& output_dir,
                     int diff_threshold_ns) {
  ProfilingBasedPartitioner partitioner =
      ProfilingBasedPartitioner(edgetpu_compiler_binary, model_path,
                                device_type, num_segments, output_dir);
  std::pair<int64_t, int64_t> bounds = partitioner.GetBounds();
  int64_t lower_bound = bounds.first;
  int64_t upper_bound = bounds.second;
  while (std::abs(upper_bound - lower_bound) > diff_threshold_ns) {
    LOG(INFO) << "upper_bound: " << upper_bound
              << ", lower_bound: " << lower_bound;
    int64_t target_latency = (lower_bound + upper_bound) / 2;
    std::vector<int64_t> latencies =
        partitioner.PartitionOnTargetLatency(target_latency);
    // If latency of the last segment is lower than `target_latency`, it
    // means that `target_latency` gives a valid partition.
    if (latencies.back() < target_latency) {
      partitioner.SaveTempSegments();
      upper_bound = target_latency;
    } else {
      // If latency of the last segment is higher than `target_latency` but
      // lower than `upper_bound`, `upper_bound` could be updated as well.
      if (latencies.back() < upper_bound) {
        partitioner.SaveTempSegments();
        upper_bound = latencies.back();
      }
      lower_bound = target_latency;
    }
  }
  LOG(INFO) << "Bisecting ends. Deleting temp segments.";
  partitioner.DeleteTempSegments();
}

}  // namespace coral
