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

#include "coral/pipeline/internal/segment_runner.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "coral/error_reporter.h"
#include "coral/pipeline/allocator.h"
#include "coral/pipeline/internal/default_allocator.h"
#include "coral/pipeline/test_utils.h"
#include "coral/test_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/interpreter.h"
#include "tflite/public/edgetpu.h"

namespace coral {
namespace internal {
namespace {

TensorMap GenerateRandomInputs(
    const tflite::Interpreter& interpreter,
    const std::unordered_map<std::string, int>& tensor_consumers_count,
    Allocator* allocator) {
  TensorMap input_tensors;
  for (int input_index : interpreter.inputs()) {
    const auto* tflite_tensor = interpreter.tensor(input_index);
    PipelineTensor input_tensor;
    input_tensor.buffer = allocator->Alloc(tflite_tensor->bytes);
    input_tensor.bytes = tflite_tensor->bytes;
    input_tensor.type = tflite_tensor->type;
    auto* ptr = reinterpret_cast<uint8_t*>(input_tensor.buffer->ptr());
    FillRandomInt(ptr, ptr + input_tensor.bytes);

    auto it = tensor_consumers_count.find(tflite_tensor->name);
    CHECK(it != tensor_consumers_count.end());
    input_tensors.insert(
        {tflite_tensor->name, {input_tensor, /*num_consumers=*/it->second}});
  }
  return input_tensors;
}

std::unordered_map<std::string, int> BuildTensorConsumersCountMap(
    const std::unordered_set<std::string>& input_tensor_names) {
  std::unordered_map<std::string, int> tensor_consumers_count(
      input_tensor_names.size());
  for (const auto& tensor_name : input_tensor_names) {
    tensor_consumers_count[tensor_name] = 1;
  }
  return tensor_consumers_count;
}

void FreeTensors(const TensorMap& tensors, Allocator* allocator) {
  for (const auto& pair : tensors) {
    const auto& tensor = pair.second;
    allocator->Free(tensor.tensor.buffer);
  }
}

void CheckResults(const tflite::Interpreter& interpreter,
                  const TensorMap& tensors) {
  ASSERT_LE(interpreter.outputs().size(), tensors.size());
  for (int i = 0; i < interpreter.outputs().size(); ++i) {
    const auto* tflite_tensor = interpreter.output_tensor(i);
    const auto& it = tensors.find(tflite_tensor->name);
    ASSERT_NE(it, tensors.end());
    ASSERT_EQ(it->second.tensor.bytes, tflite_tensor->bytes);
    const auto* actual = CHECK_NOTNULL(
        reinterpret_cast<const uint8_t*>(it->second.tensor.buffer->ptr()));
    const auto* expected = CHECK_NOTNULL(
        reinterpret_cast<const uint8_t*>(tflite_tensor->data.data));
    for (int j = 0; j < tflite_tensor->bytes; ++j) {
      EXPECT_EQ(actual[j], expected[j]);
    }
  }
}

class SegmentRunnerTest : public EdgeTpuCacheTestBase {};

// Tests that SegmentRunner returns same output tensors as using
// tflite::Interpreter when feeding the same inputs.
TEST_F(SegmentRunnerTest, SameResultAsTfliteInterpreter) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  auto* tpu_context = GetTpuContextCache();

  EdgeTpuErrorReporter error_reporter;
  auto pipeline_interpreter =
      CreateInterpreter(*model, tpu_context, &error_reporter);
  auto tflite_interpreter =
      CreateInterpreter(*model, tpu_context, &error_reporter);

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*pipeline_interpreter);
  auto tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  // Add a fake tensor to make a harder case.
  tensor_consumers_count.insert({"fake_tensor_name", 20});
  auto default_allocator = absl::make_unique<DefaultAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count,
      &input_tensor_names,
      pipeline_interpreter.get(),
      &input_queue,
      &output_queue,
      default_allocator.get(),
      default_allocator.get(),
  };

  // Set up input for SegmentRunner.
  const auto& input_tensors = GenerateRandomInputs(
      *pipeline_interpreter, tensor_consumers_count, default_allocator.get());
  input_queue.push(input_tensors);
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Let `tflite_interpreter` uses the same random inputs.
  for (int i = 0; i < pipeline_interpreter->inputs().size(); ++i) {
    auto* tflite_tensor = tflite_interpreter->input_tensor(i);
    auto it = input_tensors.find(tflite_tensor->name);
    CHECK(it != input_tensors.end());
    std::memcpy(tflite_tensor->data.data,
                CHECK_NOTNULL(it->second.tensor.buffer->ptr()),
                it->second.tensor.bytes);
  }

  // Run inference with SegmentRunner.
  runner.RunInference();
  ASSERT_TRUE(runner.runner_status().ok());
  EXPECT_EQ(runner.stats().num_inferences, 1);
  EXPECT_GT(runner.stats().total_time_ns, 0);

  // Run inference with tflite::Interpreter.
  CHECK(tflite_interpreter->Invoke() == kTfLiteOk);

  // Check that results are exactly the same.
  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  CheckResults(*tflite_interpreter, output_tensors);
  FreeTensors(output_tensors, default_allocator.get());
}

TEST_F(SegmentRunnerTest, InterpreterInferenceError) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  auto* tpu_context = GetTpuContextCache();

  EdgeTpuErrorReporter error_reporter;
  // Bad interpreter. Tensors are not allocated.
  auto bad_interpreter = CreateInterpreter(*model, tpu_context, &error_reporter,
                                           /*allocate_tensors=*/false);

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*bad_interpreter);
  auto tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  auto default_allocator = absl::make_unique<DefaultAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count, &input_tensor_names,
      bad_interpreter.get(),   &input_queue,
      &output_queue,           default_allocator.get(),
      default_allocator.get(),
  };

  // Set up input for SegmentRunner.
  const auto& input_tensors = GenerateRandomInputs(
      *bad_interpreter, tensor_consumers_count, default_allocator.get());
  input_queue.push(input_tensors);
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Run inference with SegmentRunner.
  runner.RunInference();
  ASSERT_EQ(runner.runner_status(),
            absl::InternalError("Invoke called on model that is not ready."));
  EXPECT_EQ(runner.stats().num_inferences, 0);
  EXPECT_EQ(runner.stats().total_time_ns, 0);

  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  EXPECT_TRUE(output_tensors.empty());
}

TEST_F(SegmentRunnerTest, EmptyInputMapError) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  auto* tpu_context = GetTpuContextCache();

  EdgeTpuErrorReporter error_reporter;
  auto bad_interpreter =
      CreateInterpreter(*model, tpu_context, &error_reporter);

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*bad_interpreter);
  auto tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  auto default_allocator = absl::make_unique<DefaultAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count, &input_tensor_names,
      bad_interpreter.get(),   &input_queue,
      &output_queue,           default_allocator.get(),
      default_allocator.get(),
  };

  // Push an empty input map, which should trigger error.
  input_queue.push({});
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Run inference with SegmentRunner.
  runner.RunInference();
  ASSERT_EQ(runner.runner_status(),
            absl::InternalError("Empty input tensor map."));
  EXPECT_EQ(runner.stats().num_inferences, 0);
  EXPECT_EQ(runner.stats().total_time_ns, 0);

  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  EXPECT_TRUE(output_tensors.empty());
}

class MockAllocator : public Allocator {
 public:
  MOCK_METHOD(Buffer*, Alloc, (size_t), (override));
  MOCK_METHOD(void, Free, (Buffer*), (override));
};

// Class that issues expected (valid) memory address for MockAllocator based on
// tflite::Interpreter.
class AddressCalculator {
 public:
  explicit AddressCalculator(const tflite::Interpreter* interpreter)
      : interpreter_(interpreter) {}

  ~AddressCalculator() {
    for (auto* addr : allocated_memory_list_) {
      std::free(addr);
    }
  }

  // Allocates buffer for input tensor whose index is `i`.
  void* alloc_input(int i) {
    CHECK_LT(i, interpreter_->inputs().size());
    auto* addr = std::malloc(interpreter_->input_tensor(i)->bytes);
    allocated_memory_list_.push_back(addr);
    return addr;
  }

  // Returns size (in bytes) for input tensor whose index is `i`.
  size_t input_size(int i) { return interpreter_->input_tensor(i)->bytes; }

  // Allocates buffer for output tensor whose index is `i`.
  void* alloc_output(int i) {
    CHECK_LT(i, interpreter_->outputs().size());
    auto* addr = std::malloc(interpreter_->output_tensor(i)->bytes);
    allocated_memory_list_.push_back(addr);
    return addr;
  }

  // Returns size (in bytes) for output tensor whose index is `i`.
  size_t output_size(int i) { return interpreter_->output_tensor(i)->bytes; }

 private:
  const tflite::Interpreter* interpreter_ = nullptr;
  std::vector<void*> allocated_memory_list_;
};

TEST_F(SegmentRunnerTest, InputTensorsFreedByRunner) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  EdgeTpuErrorReporter error_reporter;
  auto interpreter =
      CreateInterpreter(*model, GetTpuContextCache(), &error_reporter);

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*interpreter);
  const auto& tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  auto mock_allocator = absl::make_unique<MockAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count, &input_tensor_names,
      interpreter.get(),       &input_queue,
      &output_queue,           mock_allocator.get(),
      mock_allocator.get(),
  };

  // Set expectation for `mock_allocator`
  auto addr_calculator = AddressCalculator(interpreter.get());
  auto* input_buffer = new HeapBuffer(addr_calculator.alloc_input(0));
  auto* output_buffer = new HeapBuffer(addr_calculator.alloc_output(0));
  EXPECT_CALL(*mock_allocator, Alloc)
      .Times(2)
      .WillOnce(testing::Return(input_buffer))
      .WillOnce(testing::Return(output_buffer));
  EXPECT_CALL(*mock_allocator, Free(input_buffer)).Times(1);

  // Set up input for SegmentRunner.
  const auto& input_tensors = GenerateRandomInputs(
      *interpreter, tensor_consumers_count, mock_allocator.get());
  input_queue.push(input_tensors);
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Run inference with SegmentRunner.
  runner.RunInference();
  ASSERT_TRUE(runner.runner_status().ok());
  EXPECT_EQ(runner.stats().num_inferences, 1);
  EXPECT_GT(runner.stats().total_time_ns, 0);

  // Check output tensors.
  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  const auto& output_tensor = output_tensors.begin()->second;
  EXPECT_EQ(output_tensor.num_consumers, 0);
  EXPECT_EQ(output_tensor.tensor.buffer->ptr(), output_buffer->ptr());

  // Ideally, one should free `output_tensors` after consuming them, e.g.,
  // `FreeTensors(output_tensors, mock_allocator.get()). This step is skipped
  // here on purpose to make sure only `input_tensor_addr` was freed insided
  // SegmentRunner::RunInference().
}

// Set input tensors' num_consumer count >1 on purpose to see it is kept alive
// after call to RunInference().
TEST_F(SegmentRunnerTest, InputTensorsKeptAliveByRunner) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  EdgeTpuErrorReporter error_reporter;
  auto interpreter =
      CreateInterpreter(*model, GetTpuContextCache(), &error_reporter);

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*interpreter);
  const auto& tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  auto mock_allocator = absl::make_unique<MockAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count, &input_tensor_names,
      interpreter.get(),       &input_queue,
      &output_queue,           mock_allocator.get(),
      mock_allocator.get(),
  };

  // Set expectation for `mock_allocator`
  auto addr_calculator = AddressCalculator(interpreter.get());
  auto* input_buffer = new HeapBuffer(addr_calculator.alloc_input(0));
  auto* output_buffer = new HeapBuffer(addr_calculator.alloc_output(0));
  EXPECT_CALL(*mock_allocator, Alloc)
      .Times(2)
      .WillOnce(testing::Return(input_buffer))
      .WillOnce(testing::Return(output_buffer));
  EXPECT_CALL(*mock_allocator, Free(testing::Ne(nullptr))).Times(0);

  // Set up input for SegmentRunner.
  auto input_tensors = GenerateRandomInputs(
      *interpreter, tensor_consumers_count, mock_allocator.get());
  for (auto& pair : input_tensors) {
    pair.second.num_consumers = 2;
  }
  input_queue.push(input_tensors);
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Run inference with SegmentRunner.
  runner.RunInference();
  ASSERT_TRUE(runner.runner_status().ok());
  EXPECT_EQ(runner.stats().num_inferences, 1);
  EXPECT_GT(runner.stats().total_time_ns, 0);

  // Check output tensors.
  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  ASSERT_EQ(output_tensors.size(), 2);

  ASSERT_EQ(input_tensors.size(), 1);
  const auto& unconsumed_input_tensor =
      output_tensors.at(interpreter->input_tensor(0)->name);
  EXPECT_EQ(unconsumed_input_tensor.num_consumers, 1);
  EXPECT_EQ(unconsumed_input_tensor.tensor.buffer->ptr(), input_buffer->ptr());

  const auto& real_output_tensor =
      output_tensors.at(interpreter->output_tensor(0)->name);
  EXPECT_EQ(real_output_tensor.num_consumers, 0);
  EXPECT_EQ(real_output_tensor.tensor.buffer->ptr(), output_buffer->ptr());

  // Ideally, one should free `input_tensors` and `output_tensors` after
  // consuming them, e.g., `FreeTensors(output_tensors, mock_allocator.get()).
  // This step is skipped here on purpose to make sure Allocator::free() was not
  // called insided SegmentRunner::RunInference().
}
}  // namespace
}  // namespace internal
}  // namespace coral
