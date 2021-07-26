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

// Tool to analyze model pipelining performance.
//
// Run ./model_pipelining_performance_analysis --help to see details on flags.
#include <fstream>

#include "absl/container/node_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/pipeline/test_utils.h"
#include "coral/tools/model_pipelining_benchmark_util.h"
#include "glog/logging.h"

ABSL_FLAG(std::string, data_dir, "/tmp/models/",
          "Models location prefix, this tool assumes data_dir has a flat "
          "layout, i.e. there is no subfolders.");

ABSL_FLAG(std::vector<std::string>, model_list,
          std::vector<std::string>({"inception_v3_299_quant",
                                    "inception_v4_299_quant"}),
          "Comma separated list of model names (without _edgetpu.tflite "
          "suffix) to get performance metric for.");

ABSL_FLAG(IntList, num_segments_list, {std::vector<int>({1, 2, 3, 4})},
          "Comma separated list that specifies number of segments to check for "
          "performance.");

ABSL_FLAG(int, num_inferences, 100, "Number of inferences to run each model.");

ABSL_FLAG(
    EdgeTpuType, device_type, EdgeTpuType::kAny,
    "Type of Edge TPU device to use, values: `pcionly`, `usbonly`, `any`.");

ABSL_FLAG(std::string, segment_latencies_path, "",
          "Path to the file to write all segments latencies. If set, analysis "
          "tool could only be allowed to run one model.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto& data_dir = absl::GetFlag(FLAGS_data_dir) + "/";
  const auto& model_list = absl::GetFlag(FLAGS_model_list);
  const auto& num_segments_list = absl::GetFlag(FLAGS_num_segments_list);
  const auto& num_inferences = absl::GetFlag(FLAGS_num_inferences);
  const auto& device_type = absl::GetFlag(FLAGS_device_type);
  const auto& segment_latencies_path =
      absl::GetFlag(FLAGS_segment_latencies_path);
  if (!segment_latencies_path.empty()) {
    CHECK_EQ(model_list.size(), 1) << "Only one model is allowed when all "
                                      "segment latencies are required to "
                                      "be written to file.";
  }

  LOG(INFO) << "data_dir: " << data_dir;
  LOG(INFO) << "list of models: " << absl::StrJoin(model_list, "\n");
  LOG(INFO) << "num_segments_list: " << AbslUnparseFlag(num_segments_list);
  LOG(INFO) << "num_inferences: " << num_inferences;
  LOG(INFO) << "device_type: " << AbslUnparseFlag(device_type);

  const int max_num_segments = *std::max_element(
      num_segments_list.elements.begin(), num_segments_list.elements.end());
  auto edgetpu_contexts =
      coral::PrepareEdgeTpuContexts(max_num_segments, device_type);

  // Benchmark all model_list and num_segments_list combinations.
  absl::node_hash_map<std::string, std::vector<coral::PerfStats>> stats_map;
  for (const auto& model_name : model_list) {
    for (const auto& num_segments : num_segments_list.elements) {
      std::vector<std::string> model_segments_paths;
      if (num_segments == 1) {
        model_segments_paths = {data_dir + model_name + "_edgetpu.tflite"};
      } else {
        model_segments_paths =
            coral::SegmentsNames(data_dir + model_name, num_segments);
      }

      const auto& stats = coral::BenchmarkPartitionedModel(
          model_segments_paths, &edgetpu_contexts, num_inferences);
      LOG(INFO) << "Model name: " << model_name
                << " num_segments: " << std::get<0>(stats)
                << " latency (in ns): " << std::get<1>(stats);

      stats_map[model_name].push_back(stats);
    }
  }

  LOG(INFO) << "========Summary=========";
  for (const auto& model_name : model_list) {
    LOG(INFO) << "Model: " << model_name;
    const auto& baseline = stats_map[model_name][0];
    for (const auto& stats : stats_map[model_name]) {
      LOG(INFO) << "    num_segments: " << std::get<0>(stats)
                << " latency (in ns): " << std::get<1>(stats) << " speedup: "
                << static_cast<float>(std::get<1>(baseline)) /
                       std::get<1>(stats);
    }
  }

  // Write latencies of all segments if the output file is defined.
  // The file would look like:
  //   `num_segments`
  //   latency_segment_0
  //   latency_segment_1
  //   ...
  //   latency_segment_`num_segments-1`
  if (!segment_latencies_path.empty()) {
    std::ofstream out_f;
    out_f.open(segment_latencies_path);
    for (const auto& model_name : model_list) {
      for (const auto& stats : stats_map[model_name]) {
        out_f << std::get<0>(stats) << "\n";
        auto segment_latencies = std::get<2>(stats);
        for (int segment_latency : segment_latencies) {
          out_f << segment_latency << "\n";
        }
      }
    }
    out_f.close();
  }
  return 0;
}
