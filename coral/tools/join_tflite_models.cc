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

// Tool to join two tflite models. The models may contain custom operator,
// which can not be imported / exported properly by tflite/toco yet.
//
// Two models can be joined together only if all the input tensors of
// input_graph_head are present as output tensors of input_graph_base.
// Any additional output tensors of input_graph_base will become dead ends,
// unless specified with --bypass_tensors, in which case they will be routed
// to the end as output_tensors of the final graph.
// Note also that there should be no name collisions between other tensors of
// the two input graphs.

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"
#include "coral/tools/tflite_graph_util.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"
#include "glog/logging.h"

ABSL_FLAG(std::string, input_graph_base, "",
          "Path to the base input graph. Must be in tflite format.");

ABSL_FLAG(std::string, input_graph_head, "",
          "Path to the head input graph. Must be in tflite format.");

ABSL_FLAG(std::string, output_graph, "",
          "Path to the output graph. Output graph will be in tflite format.");

ABSL_FLAG(std::string, bypass_output_tensors, "",
          "A list of output tensor names from base input graph, which "
          "should also become output tensors in the merged output graph.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  auto base = CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(
      absl::GetFlag(FLAGS_input_graph_base).c_str()));
  auto head = CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(
      absl::GetFlag(FLAGS_input_graph_head).c_str()));

  flatbuffers::FlatBufferBuilder fbb;
  coral::ConcatModels(*base->GetModel(), *head->GetModel(), &fbb,
                      absl::StrSplit(absl::GetFlag(FLAGS_bypass_output_tensors),
                                     ',', absl::SkipEmpty()));

  CHECK(flatbuffers::SaveFile(absl::GetFlag(FLAGS_output_graph).c_str(),
                              reinterpret_cast<char*>(fbb.GetBufferPointer()),
                              fbb.GetSize(),
                              /*binary=*/true));
}
