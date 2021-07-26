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

// Commandline tool that partitions a given TFLite model with given op numbers
// for all segments.

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "coral/tools/partitioner/strategy.h"
#include "coral/tools/partitioner/utils.h"
#include "glog/logging.h"

struct IntList {
  std::vector<int> elements;
};

// Returns a textual flag value corresponding to the IntList `list`.
std::string AbslUnparseFlag(const IntList& list) {
  // Let flag module unparse the element type for us.
  return absl::StrJoin(list.elements, ",", [](std::string* out, int element) {
    out->append(absl::UnparseFlag(element));
  });
}

// Parses an IntList from the command line flag value `text`.
// Returns true and sets `*list` on success; returns false and sets `*error` on
// failure.
bool AbslParseFlag(absl::string_view text, IntList* list, std::string* error) {
  // We have to clear the list to overwrite any existing value.
  list->elements.clear();
  // absl::StrSplit("") produces {""}, but we need {} on empty input.
  if (text.empty()) {
    return true;
  }
  for (const auto& part : absl::StrSplit(text, ',')) {
    // Let the flag module parse each element value for us.
    int element;
    if (!absl::ParseFlag(part, &element, error)) {
      return false;
    }
    list->elements.push_back(element);
  }
  return true;
}

ABSL_FLAG(std::string, model_path, "", "Path to the model to be partitioned.");
ABSL_FLAG(std::string, output_dir, "", "Output directory.");
ABSL_FLAG(std::string, segment_prefix, "tmp",
          "Prefix of the output segment paths.");
ABSL_FLAG(IntList, num_ops, {}, "Given list of ops numbers per segment.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto& model_path = absl::GetFlag(FLAGS_model_path);
  const auto& output_dir = absl::GetFlag(FLAGS_output_dir);
  const auto& segment_prefix = absl::GetFlag(FLAGS_segment_prefix);
  const std::vector<int> num_ops_per_segment =
      absl::GetFlag(FLAGS_num_ops).elements;
  const int num_segments = num_ops_per_segment.size();
  LOG(INFO) << "model_path: " << model_path;
  LOG(INFO) << "output_dir: " << output_dir;
  LOG(INFO) << "num_ops: " << AbslUnparseFlag(absl::GetFlag(FLAGS_num_ops));

  std::vector<char> model_content;
  coral::ReadFileOrExit(model_path, &model_content);
  const auto* model = tflite::GetModel(model_content.data());
  coral::PartitionStrategy strategy =
      coral::GetStrategyFromNumOps(model, num_ops_per_segment);

  for (int i = 0; i < strategy.size(); ++i) {
    const auto& segment_info = strategy[i];
    const auto& segment_contents = coral::ExtractModelSegment(
        *model, segment_info.target_nodes,
        {segment_info.target_inputs.begin(), segment_info.target_inputs.end()},
        {segment_info.target_outputs.begin(),
         segment_info.target_outputs.end()});
    std::string segment_cpu_filepath =
        absl::StrCat(output_dir, "/", segment_prefix, "_segment_", i, "_of_",
                     num_segments, ".tflite");
    LOG(INFO) << "Write segment content to: " << segment_cpu_filepath;
    coral::WriteFileOrExit(segment_cpu_filepath, segment_contents);
  }
  return 0;
}
