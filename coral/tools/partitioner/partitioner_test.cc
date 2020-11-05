#include "coral/tools/partitioner/partitioner.h"

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

namespace coral {
namespace {

std::vector<std::string> ModelNames() {
  return {
      "inception_v3_299_quant",
      "inception_v4_299_quant",
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
  }

  std::string model_name_;
  int num_segments_;
};

// A valid partition of the graph should satisfy:
//  1) there's no intersection between segments.
//  2) union of all segments should cover the whole graph.
//
// Additionally, all intermediate tensors (`target_inputs` and `target_outputs`)
// should be consumed internally.
TEST_P(ParameterCountBasedPartitionerTest, CheckValidPartition) {
  const auto& model_filepath = TestDataPath(model_name_);
  std::vector<char> model_content;
  ReadFileOrExit(model_filepath, &model_content);
  std::unique_ptr<coral::Partitioner> partitioner =
      absl::make_unique<coral::ParameterCountBasedPartitioner>(model_content);

  const auto& strategy = partitioner->GetStrategy(num_segments_);

  // Check no intersection between segments.
  for (int i = 0; i < num_segments_; ++i) {
    for (int j = i + 1; j < num_segments_; ++j) {
      const auto& segment_a = strategy[i].target_nodes;
      const auto& segment_b = strategy[j].target_nodes;
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
    std::copy(strategy[i].target_nodes.begin(), strategy[i].target_nodes.end(),
              std::inserter(segments_union, segments_union.end()));
  }
  const int num_nodes = tflite::GetModel(model_content.data())
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
    std::copy(strategy[i].target_outputs.begin(),
              strategy[i].target_outputs.end(),
              std::inserter(intermediate_outputs, intermediate_outputs.end()));
    std::copy(strategy[i + 1].target_inputs.begin(),
              strategy[i + 1].target_inputs.end(),
              std::inserter(intermediate_inputs, intermediate_inputs.end()));
  }
  EXPECT_THAT(intermediate_inputs,
              testing::UnorderedElementsAreArray(intermediate_outputs));
}

// Check that each segment has roughly the same parameter sizes.
TEST_P(ParameterCountBasedPartitionerTest, CheckPartitionQuality) {
  const auto& model_filepath = TestDataPath(model_name_);
  std::vector<char> model_content;
  ReadFileOrExit(model_filepath, &model_content);
  const auto* model = tflite::GetModel(model_content.data());

  const auto& parameter_sizes = CalculateParameterSizes(*model);

  std::unique_ptr<coral::Partitioner> partitioner =
      absl::make_unique<coral::ParameterCountBasedPartitioner>(model_content);

  const auto& strategy = partitioner->GetStrategy(num_segments_);
  std::vector<int64_t> segment_sizes(num_segments_, 0);
  int64_t total_size = 0;
  for (int i = 0; i < num_segments_; ++i) {
    for (const auto& node : strategy[i].target_nodes) {
      segment_sizes[i] += parameter_sizes[node];
    }
    total_size += segment_sizes[i];
  }
  int64_t average_size = total_size / num_segments_;

  // Note that the last segment is not checked because it takes all
  // the remaining operators and can be much larger than average size. This is
  // OK heuristic as later segments tend to require less computation given the
  // same parameter size.
  for (int i = 0; i < num_segments_ - 1; ++i) {
    float diff_percent =
        (static_cast<float>(segment_sizes[i]) / average_size - 1) * 100;
    EXPECT_LT(std::abs(diff_percent), 35);
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
