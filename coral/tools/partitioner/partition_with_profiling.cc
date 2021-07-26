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

// Commandline tool that partitions a cpu tflite model to segments via
// ProfilingBasedPartitioner.
#include <sys/stat.h>
#include <sys/types.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "coral/tools/model_pipelining_benchmark_util.h"
#include "coral/tools/partitioner/profiling_based_partitioner.h"
#include "coral/tools/partitioner/strategy.h"
#include "coral/tools/partitioner/utils.h"
#include "glog/logging.h"

ABSL_FLAG(std::string, edgetpu_compiler_binary, "",
          "Path to the edgetpu compiler binary.");
ABSL_FLAG(std::string, model_path, "", "Path to the model to be partitioned.");
ABSL_FLAG(std::string, output_dir, "/tmp/models", "Output directory.");
ABSL_FLAG(int, num_segments, -1, "Number of output segment models.");
ABSL_FLAG(
    EdgeTpuType, device_type, EdgeTpuType::kAny,
    "Type of Edge TPU device to use, values: `pcionly`, `usbonly`, `any`.");
ABSL_FLAG(int64_t, diff_threshold_ns, 1000000,
          "The target difference (in ns) between the slowest segment "
          "(upper bound) and the fastest segment (lower bound).");
ABSL_FLAG(int32_t, partition_search_step, 1,
          "The number of operators that are added to or removed from a segment "
          "during each iteration of the partition search. "
          "Default is 1.");
ABSL_FLAG(int32_t, delegate_search_step, 1,
          "Step size for the delegate search when compiling the model. "
          "Default is 1.");
ABSL_FLAG(int64_t, initial_lower_bound_ns, -1,
          "Initial lower bound of the segment latency. "
          "(Latency of the fastest segment.)");
ABSL_FLAG(int64_t, initial_upper_bound_ns, -1,
          "Initial upper bound of the segment latency. "
          "(Latency of the slowest segment.)");

bool FileExists(const std::string& file_name) {
  struct stat info;
  return stat(file_name.c_str(), &info) == 0;
}

bool CanAccess(const std::string& file_name) {
  return access(file_name.c_str(), R_OK | W_OK | X_OK) == 0;
}

bool IsDir(const std::string& dir_name) {
  struct stat info;
  CHECK_EQ(stat(dir_name.c_str(), &info), 0) << dir_name << " does not exist";
  return info.st_mode & S_IFDIR;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const auto& edgetpu_compiler_binary =
      absl::GetFlag(FLAGS_edgetpu_compiler_binary);
  const auto& model_path = absl::GetFlag(FLAGS_model_path);
  const auto& output_dir = absl::GetFlag(FLAGS_output_dir);
  const int num_segments = absl::GetFlag(FLAGS_num_segments);
  const auto& device_type = absl::GetFlag(FLAGS_device_type);
  const int64_t diff_threshold_ns = absl::GetFlag(FLAGS_diff_threshold_ns);
  const int partition_search_step = absl::GetFlag(FLAGS_partition_search_step);
  const int delegate_search_step = absl::GetFlag(FLAGS_delegate_search_step);
  const std::pair<int64_t, int64_t> initial_bounds{
      absl::GetFlag(FLAGS_initial_lower_bound_ns),
      absl::GetFlag(FLAGS_initial_upper_bound_ns)};

  LOG_IF(FATAL, !FileExists(edgetpu_compiler_binary))
      << "edgetpu_compiler_binary " << edgetpu_compiler_binary
      << " does not exist or cannot be opened.";
  LOG_IF(FATAL, !FileExists(model_path))
      << "model_path " << model_path << " does not exist or cannot be opened.";
  if (!FileExists(output_dir)) {
    LOG(INFO) << "output_dir " << output_dir
              << " does not exist, creating it...";
    CHECK_EQ(mkdir(output_dir.c_str(), 0755), 0);
  } else {
    LOG_IF(FATAL, !IsDir(output_dir))
        << "output_dir " << output_dir << " is not a valid directory.";
    LOG_IF(FATAL, !CanAccess(output_dir))
        << "permission denied to output_dir " << output_dir;
  }
  LOG_IF(FATAL, num_segments < 2) << "num_segments must be at least 2.";
  LOG(INFO) << "edgetpu_compiler_binary: " << edgetpu_compiler_binary;
  LOG(INFO) << "model_path: " << model_path;
  LOG(INFO) << "output_dir: " << output_dir;
  LOG(INFO) << "num_segments: " << num_segments;
  LOG(INFO) << "device_type: " << AbslUnparseFlag(device_type);
  LOG(INFO) << "diff_threshold_ns: " << diff_threshold_ns;
  LOG(INFO) << "partition_search_step: " << partition_search_step;
  LOG(INFO) << "delegate_search_step: " << delegate_search_step;
  LOG(INFO) << "initial bounds: (" << initial_bounds.first << ", "
            << initial_bounds.second << ")";

  const auto& start_time = std::chrono::steady_clock::now();
  CHECK(coral::BisectPartition(edgetpu_compiler_binary, model_path, device_type,
                               num_segments, output_dir, diff_threshold_ns,
                               partition_search_step, delegate_search_step,
                               initial_bounds))
      << "Failed to find valid partition that meets the latency target.";

  std::chrono::duration<double> time_span =
      std::chrono::steady_clock::now() - start_time;
  LOG(INFO) << "Segments have been saved under " << output_dir;
  LOG(INFO) << "Total time: " << time_span.count() << "s.";
  return 0;
}
