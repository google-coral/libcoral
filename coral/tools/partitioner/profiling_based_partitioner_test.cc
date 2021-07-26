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

#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "coral/test_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {
class MockProfilingBasedPartitioner : public ProfilingBasedPartitioner {
 public:
  MockProfilingBasedPartitioner(const std::string& edgetpu_compiler_binary,
                                const std::string& model_path,
                                EdgeTpuType device_type, int num_segments,
                                const std::string& output_dir)
      : ProfilingBasedPartitioner(edgetpu_compiler_binary, model_path,
                                  device_type, num_segments, output_dir) {}
  MOCK_METHOD(int64_t, PartitionCompileAndAnalyze,
              (const std::vector<int>&, int, int), (override));
};

class ProfilingBasedPartitionerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_path_ = TestDataPath("inception_v3_299_quant.tflite");
  }

  std::string model_path_;
};

int64_t NumOpsBy1000(const std::vector<int>& num_ops, int segment_index,
                     int delegate_search_step) {
  return num_ops[segment_index] * 1000;
}

TEST_F(ProfilingBasedPartitionerTest, OnTargetLatencyTwoSegments) {
  auto partitioner = absl::make_unique<MockProfilingBasedPartitioner>(
      /*edgetpu_compiler_binary=*/"", model_path_, EdgeTpuType::kAny,
      /*num_segments=*/2, /*output_dir=*/"/tmp");
  EXPECT_CALL(*partitioner,
              PartitionCompileAndAnalyze(testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Invoke(NumOpsBy1000));
  EXPECT_EQ(partitioner->PartitionOnTargetLatency(50000), 82000);
  EXPECT_THAT(partitioner->partition(), testing::ElementsAre(50, 82));
}

TEST_F(ProfilingBasedPartitionerTest, OnTargetLatencyFourSegments) {
  auto partitioner = absl::make_unique<MockProfilingBasedPartitioner>(
      /*edgetpu_compiler_binary=*/"", model_path_, EdgeTpuType::kAny,
      /*num_segments=*/4, /*output_dir=*/"/tmp");
  EXPECT_CALL(*partitioner,
              PartitionCompileAndAnalyze(testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Invoke(NumOpsBy1000));
  EXPECT_EQ(partitioner->PartitionOnTargetLatency(30000), 42000);
  EXPECT_THAT(partitioner->partition(), testing::ElementsAre(30, 30, 30, 42));
}

TEST_F(ProfilingBasedPartitionerTest, OnTargetLatencyEarlyStop) {
  auto partitioner = absl::make_unique<MockProfilingBasedPartitioner>(
      /*edgetpu_compiler_binary=*/"", model_path_, EdgeTpuType::kAny,
      /*num_segments=*/3, /*output_dir=*/"/tmp");
  EXPECT_CALL(*partitioner,
              PartitionCompileAndAnalyze(testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Invoke(NumOpsBy1000));
  int target_latency = 500;
  EXPECT_EQ(partitioner->PartitionOnTargetLatency(target_latency),
            std::numeric_limits<int64_t>::max());
  EXPECT_TRUE(partitioner->partition().empty());
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
