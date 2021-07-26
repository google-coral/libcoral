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

#include "coral/tools/partitioner/parameter_count_based_partitioner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "coral/test_utils.h"
#include "coral/tools/partitioner/utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace {

std::vector<std::string> ModelNames() {
  return {
      "inception_v3_299_quant",
      "inception_v4_299_quant",
      "ssd_mobilenet_v1_coco_quant_postprocess",
      "ssd_mobilenet_v2_coco_quant_postprocess",
  };
}

// Tests all supported models with different number of segments.
//
// The first test parameter is tflite model base name (without .tflite
// suffix); The second test parameter is number of segments a model is
// partitioned into.
class ParameterCountBasedPartitionerTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<std::string, int>> {
 protected:
  void SetUp() override {
    model_name_ = std::get<0>(GetParam()) + ".tflite";
    num_segments_ = std::get<1>(GetParam());

    const auto& model_filepath = TestDataPath(model_name_);
    ReadFileOrExit(model_filepath, &model_content_);
    std::unique_ptr<coral::ParameterCountBasedPartitioner> partitioner =
        absl::make_unique<coral::ParameterCountBasedPartitioner>(
            model_content_);
    strategy_ = partitioner->GetStrategy(num_segments_);
  }

  std::string model_name_;
  std::vector<char> model_content_;
  int num_segments_;
  PartitionStrategy strategy_;
};

// A valid partition of the graph should satisfy:
//  1) there's no intersection between segments.
//  2) union of all segments should cover the whole graph.
//
// Additionally, all intermediate tensors (`target_inputs` and `target_outputs`)
// should be consumed internally.
TEST_P(ParameterCountBasedPartitionerTest, CheckValidPartition) {
  // Check no intersection between segments.
  for (int i = 0; i < num_segments_; ++i) {
    for (int j = i + 1; j < num_segments_; ++j) {
      const auto& segment_a = strategy_[i].target_nodes;
      const auto& segment_b = strategy_[j].target_nodes;
      std::vector<int> segments_intersection(
          std::max(segment_a.size(), segment_b.size()));
      const auto& it = std::set_intersection(segment_a.begin(), segment_a.end(),
                                             segment_b.begin(), segment_b.end(),
                                             segments_intersection.begin());
      segments_intersection.resize(it - segments_intersection.begin());
      EXPECT_TRUE(segments_intersection.empty());
    }
  }

  // Check union covers the whole graph.
  absl::flat_hash_set<int> segments_union;
  for (int i = 0; i < num_segments_; ++i) {
    std::copy(strategy_[i].target_nodes.begin(),
              strategy_[i].target_nodes.end(),
              std::inserter(segments_union, segments_union.end()));
  }
  const int num_nodes = tflite::GetModel(model_content_.data())
                            ->subgraphs()
                            ->Get(0)
                            ->operators()
                            ->size();
  std::vector<int> expected_union(num_nodes);
  std::generate(expected_union.begin(), expected_union.end(),
                [n = 0]() mutable { return n++; });
  EXPECT_THAT(segments_union,
              testing::UnorderedElementsAreArray(expected_union));

  // Check intermediate `target_inputs` and `target_outputs` are consumed
  // internally.
  absl::flat_hash_set<int> intermediate_inputs, intermediate_outputs;
  for (int i = 0; i < num_segments_ - 1; ++i) {
    std::copy(strategy_[i].target_outputs.begin(),
              strategy_[i].target_outputs.end(),
              std::inserter(intermediate_outputs, intermediate_outputs.end()));
    std::copy(strategy_[i + 1].target_inputs.begin(),
              strategy_[i + 1].target_inputs.end(),
              std::inserter(intermediate_inputs, intermediate_inputs.end()));
  }
  EXPECT_THAT(intermediate_inputs,
              testing::UnorderedElementsAreArray(intermediate_outputs));
}

// Check that each segment has roughly the same parameter sizes.
TEST_P(ParameterCountBasedPartitionerTest, CheckPartitionQuality) {
  const auto* model = tflite::GetModel(model_content_.data());
  const auto& parameter_sizes = CalculateParameterSizes(*model);

  // Compare size of each segment vs the average size of the segments.
  // Note that the last segment is ignored intentionally because it takes all
  // the remaining operators and can be much larger than average size.
  // This is OK heuristic as later segments tend to require less computation
  // given the same parameter size.
  std::vector<int64_t> segment_sizes(num_segments_, 0);
  int64_t total_size = 0;
  for (int i = 0; i < num_segments_ - 1; ++i) {
    for (const auto& node : strategy_[i].target_nodes) {
      segment_sizes[i] += parameter_sizes[node];
    }
    total_size += segment_sizes[i];
  }
  int64_t average_size = total_size / (num_segments_ - 1);

  for (int i = 0; i < num_segments_ - 1; ++i) {
    float diff_percent =
        (static_cast<float>(segment_sizes[i]) / average_size - 1) * 100;
    EXPECT_LT(std::abs(diff_percent), 35);
  }
}

// Check that the the last segment has the same outputs as the original model.
// Note the tensor order must be preserved as well.
TEST_P(ParameterCountBasedPartitionerTest, CheckOutputs) {
  const auto* model = tflite::GetModel(model_content_.data());
  const auto* model_subgraph = model->subgraphs()->Get(0);
  const int num_outputs = model_subgraph->outputs()->size();

  const auto& segment_info = strategy_.back();
  const auto& last_segment_content = coral::ExtractModelSegment(
      *model, segment_info.target_nodes,
      {segment_info.target_inputs.begin(), segment_info.target_inputs.end()},
      {segment_info.target_outputs.begin(), segment_info.target_outputs.end()});
  const auto* last_segment_model =
      tflite::GetModel(last_segment_content.data());
  const auto* last_segment_subgraph = last_segment_model->subgraphs()->Get(0);
  ASSERT_EQ(last_segment_subgraph->outputs()->size(), num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    EXPECT_EQ(last_segment_subgraph->tensors()
                  ->Get(last_segment_subgraph->outputs()->Get(i))
                  ->name()
                  ->str(),
              model_subgraph->tensors()
                  ->Get(model_subgraph->outputs()->Get(i))
                  ->name()
                  ->str());
  }
}

INSTANTIATE_TEST_CASE_P(ParameterCountBasedPartitionerTest,
                        ParameterCountBasedPartitionerTest,
                        ::testing::Combine(::testing::ValuesIn(ModelNames()),
                                           ::testing::Values(2, 3, 4)));
}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
