#include "coral/tools/partitioner/profiling_partition_util.h"

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
              (const std::vector<int>&, int), (override));
};

class ProfilingBasedPartitionerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_path_ = TestDataPath("inception_v3_299_quant.tflite");
  }

  std::string model_path_;
};

int64_t NumOpsBy1000(const std::vector<int>& num_ops, int segment_index) {
  return num_ops[segment_index] * 1000;
}

TEST_F(ProfilingBasedPartitionerTest, OnTargetLatencyTwoSegments) {
  auto partitioner = absl::make_unique<MockProfilingBasedPartitioner>(
      /*edgetpu_compiler_binary=*/"", model_path_, EdgeTpuType::kAny,
      /*num_segments=*/2, /*output_dir=*/"/tmp");
  EXPECT_CALL(*partitioner, PartitionCompileAndAnalyze(testing::_, testing::_))
      .WillRepeatedly(testing::Invoke(NumOpsBy1000));
  std::vector<int64_t> segment_latencies =
      partitioner->PartitionOnTargetLatency(50000);
  EXPECT_THAT(std::vector<int64_t>({50000, 82000}),
              testing::ContainerEq(segment_latencies));
}

TEST_F(ProfilingBasedPartitionerTest, OnTargetLatencyFourSegments) {
  auto partitioner = absl::make_unique<MockProfilingBasedPartitioner>(
      /*edgetpu_compiler_binary=*/"", model_path_, EdgeTpuType::kAny,
      /*num_segments=*/4, /*output_dir=*/"/tmp");
  EXPECT_CALL(*partitioner, PartitionCompileAndAnalyze(testing::_, testing::_))
      .WillRepeatedly(testing::Invoke(NumOpsBy1000));
  std::vector<int64_t> segment_latencies =
      partitioner->PartitionOnTargetLatency(30000);
  EXPECT_THAT(std::vector<int64_t>({30000, 30000, 30000, 42000}),
              testing::ContainerEq(segment_latencies));
}

TEST_F(ProfilingBasedPartitionerTest, OnTargetLatencyEarlyStop) {
  auto partitioner = absl::make_unique<MockProfilingBasedPartitioner>(
      /*edgetpu_compiler_binary=*/"", model_path_, EdgeTpuType::kAny,
      /*num_segments=*/3, /*output_dir=*/"/tmp");
  EXPECT_CALL(*partitioner, PartitionCompileAndAnalyze(testing::_, testing::_))
      .WillRepeatedly(testing::Invoke(NumOpsBy1000));
  int target_latency = 500;
  std::vector<int64_t> segment_latencies =
      partitioner->PartitionOnTargetLatency(target_latency);
  CHECK_GT(segment_latencies.back(), target_latency);
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
