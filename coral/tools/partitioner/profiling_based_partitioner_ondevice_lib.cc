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

#include "coral/tools/partitioner/profiling_based_partitioner_ondevice_lib.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/pipeline/test_utils.h"
#include "coral/test_utils.h"
#include "coral/tools/model_pipelining_benchmark_util.h"
#include "coral/tools/partitioner/profiling_based_partitioner.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

ABSL_FLAG(std::string, edgetpu_compiler_binary, "",
          "Path to the edgetpu compiler binary.");

namespace coral {
namespace {

constexpr int kNumInferences = 10;

// We only test with PCIe connection, because USB connection can have larger
// variance in latency.
const EdgeTpuType kDeviceType = EdgeTpuType::kPciOnly;

}  // namespace

void TestProfilingBasedPartitioner(
    const std::string& model_name, int num_segments, float expected_speedup,
    int partition_search_step, int delegate_search_step,
    int64_t diff_threshold_ns,
    const std::pair<int64_t, int64_t>& initial_bounds) {
  LOG(INFO) << "Testing model: " << model_name
            << ", segment num: " << num_segments;
  const auto cpu_model_path = TestDataPath(model_name + ".tflite");
  const auto edgetpu_model_path = TestDataPath(model_name + "_edgetpu.tflite");

  char pattern[] = "/tmp/tmpdir.XXXXXX";
  const std::string tmp_dir = std::string(mkdtemp(pattern)) + "/";
  ASSERT_TRUE(BisectPartition(absl::GetFlag(FLAGS_edgetpu_compiler_binary),
                              cpu_model_path, kDeviceType, num_segments,
                              tmp_dir, diff_threshold_ns, partition_search_step,
                              delegate_search_step, initial_bounds));

  // Benchmark holistic Edge TPU model.
  auto edgetpu_contexts = PrepareEdgeTpuContexts(num_segments, kDeviceType);
  const int64_t baseline_latency_ns = std::get<1>(BenchmarkPartitionedModel(
      {edgetpu_model_path}, &edgetpu_contexts, kNumInferences));

  // Benchmark pipelined Edge TPU model.
  const std::size_t found = model_name.find_last_of("/\\");
  const std::string basename =
      found == std::string::npos ? model_name : model_name.substr(found + 1);
  const int64_t pipeline_latency_ns = std::get<1>(
      BenchmarkPartitionedModel(SegmentsNames(tmp_dir + basename, num_segments),
                                &edgetpu_contexts, kNumInferences));

  LOG(INFO) << "Single model latency (ns): " << baseline_latency_ns;
  LOG(INFO) << "Pipeline latency (ns): " << pipeline_latency_ns;
  EXPECT_GE(static_cast<float>(baseline_latency_ns) / pipeline_latency_ns,
            expected_speedup);
}

}  // namespace coral
