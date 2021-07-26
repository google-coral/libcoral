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

#ifndef LIBCORAL_CORAL_PIPELINE_PIPELINED_MODEL_RUNNER_H_
#define LIBCORAL_CORAL_PIPELINE_PIPELINED_MODEL_RUNNER_H_

#include <thread>  // NOLINT

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "coral/pipeline/allocator.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/internal/segment_runner.h"
#include "coral/pipeline/internal/thread_safe_queue.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

// Runs inferencing for a segmented model, using a pipeline of Edge TPUs.
// This class assumes each segment has a dedicated Edge TPU, which allows all
// segments to run in parallel and improves throughput.
//
// For example, if you have a pool of requests to process:
//
//    ```
//    auto model_segments_interpreters =
//        ModelSegmentsInterpreters(model_segments_paths);
//    // Caller can set custom allocators for input and output tensors with
//    // `input_tensor_allocator` and `output_tensor_allocator` arguments.
//    auto runner = PipelinedModelRunner(model_segments_interpreters);
//    auto* input_tensor_allocator = runner.GetInputTensorAllocator();
//    auto* output_tensor_allocator = runner.GetOutputTensorAllocator();
//
//    const int total_num_requests = 1000;
//
//    auto request_producer = [&runner, &total_num_requests]() {
//      for (int i = 0; i < total_num_requests; ++i) {
//        // Caller is responsible for allocating input tensors.
//        CHECK(runner.Push(CreateInputTensors(input_tensor_allocator)).ok());
//      }
//    };
//
//    auto result_consumer = [&runner, &total_num_requests]() {
//      for (int i = 0; i < total_num_requests; ++i) {
//        std::vector<Tensor> output_tensors;
//        CHECK(runner.Pop(&output_tensors).ok());
//        ConsumeOutputTensors(output_tensors);
//        // Caller is responsible for deallocating output tensors.
//        FreeTensors(output_tensor_allocator, output_tensors);
//      }
//    };
//
//    auto producer_thread = std::thread(request_producer);
//    auto consumer_thread = std::thread(result_consumer);
//
//    ```
//
// Or, if you have a stream of requests to process:
//
//    ```
//    auto model_segments_interpreters =
//        ModelSegmentsInterpreters(model_segments_paths);
//    // Caller can set custom allocators for input and output tensors with
//    // `input_tensor_allocator` and `output_tensor_allocator` arguments.
//    auto runner = PipelinedModelRunner(model_segments_interpreters);
//    auto* input_tensor_allocator = runner.GetInputTensorAllocator();
//    auto* output_tensor_allocator = runner.GetOutputTensorAllocator();
//
//    auto request_producer = [&runner]() {
//      while (true) {
//        // Caller is responsible for allocating input tensors.
//        CHECK(runner.Push(CreateInputTensors(input_tensor_allocator)).ok());
//        if (ShouldStop()) {
//          // Pushing special inputs to signal no more inputs will be pushed.
//          CHECK(runner.Push({}).ok());
//          break;
//        }
//      }
//    };
//
//    auto result_consumer = [&runner]() {
//      std::vector<Tensor> output_tensors;
//      while (runner.Pop(&output_tensors).ok() && !output_tensors.empty()) {
//        ConsumeOutputTensors(output_tensors);
//        // Caller is responsible for deallocating output tensors.
//        FreeTensors(output_tensor_allocator, output_tensors);
//      }
//    };
//
//    auto producer_thread = std::thread(request_producer);
//    auto consumer_thread = std::thread(result_consumer);
//    ```
//
// This class is thread-safe.
class PipelinedModelRunner {
 public:
  // Initializes the PipelinedModelRunner with model segments.
  //
  // @param model_segments_interpreters
  // A vector of pointers to tflite::Interpreter
  // objects, each representing a model segment and unique Edge TPU context.
  // `model_segments_interpreters[0]` should be the first segment interpreter of
  // the model, `model_segments_interpreters[1]` is the second segment, and so
  // on.
  // @param input_tensor_allocator A custom Allocator for input tensors. By
  // default (`nullptr`), it uses an allocator provided by this class.
  // @param output_tensor_allocator A custom Allocator for output tensors. By
  // default (`nullptr`), it uses an allocator provided by this class.
  //
  // **Note:**
  //  * `input_tensor_allocator` is only used to free the input tensors, as
  //     this class assumes that input tensors are allocated by caller.
  //  * `output_tensor_allocator` is only used to allocate output tensors,
  //      as this class assumes that output tensors are freed by caller
  //      after consuming them.
  explicit PipelinedModelRunner(
      const std::vector<tflite::Interpreter*>& model_segments_interpreters,
      Allocator* input_tensor_allocator = nullptr,
      Allocator* output_tensor_allocator = nullptr);

  ~PipelinedModelRunner();

  // Returns the default input tensor allocator (or the allocator given to the
  // constructor).
  Allocator* GetInputTensorAllocator() const { return input_tensor_allocator_; }

  // Returns the default output tensor allocator (or the allocator given to the
  // constructor).
  Allocator* GetOutputTensorAllocator() const {
    return output_tensor_allocator_;
  }

  // Sets input queue size. By default, input queue size is unlimited.
  //
  // @param size Input queue size.
  //
  // Note: It is OK to change queue size threshold when PipelinedModelRunner is
  // active. If new threshold is smaller than current queue size, push to the
  // queue will be blocking until the current queue size drops below the new
  // threshold.
  void SetInputQueueSize(size_t size) { queues_[0].set_max_queue_size(size); }

  // Sets output queue size. By default, output queue size is unlimited.
  //
  // @param size Output queue size.
  //
  // Note: It is OK to change queue size threshold when PipelinedModelRunner is
  // active. If new threshold is smaller than current queue size, push to the
  // queue will be blocking until the current queue size drops below the new
  // threshold.
  void SetOutputQueueSize(size_t size) {
    queues_[num_segments_].set_max_queue_size(size);
  }

  // Pushes input tensors to be processed by the pipeline.
  //
  // @param input_tensors A vector of input tensors, each wrapped as a
  // PipelineTensor. The order must match Interpreter::inputs() from the
  // first model segment.
  // @return absl::OkStatus if successful; absl::InternalError otherwise.
  //
  // **Note:**
  //   *  Caller is responsible for allocating memory for input tensors. By
  //      default, this class will free those tensors when they are consumed.
  //      Caller can set a custom allocator for input tensors if needed.
  //
  //   *  Pushing an empty vector `{}` is allowed, which signals the class that
  //      no more inputs will be added (the function will return false if inputs
  //      were pushed after this special push). This special push allows
  //      Pop()'s consumer to properly drain unconsumed output tensors. See
  //      above example for details.
  //
  //   *  Caller will get blocked if current input queue size is greater than
  //      input queue size threshold. By default, input queue size threshold is
  //      unlimited, i.e., call to Push() is non-blocking.
  absl::Status Push(const std::vector<PipelineTensor>& input_tensors);

  // Gets output tensors from the pipeline.
  //
  // @param output_tensors A pointer to a vector of PipelineTensor objects
  // where outputs should be stored. Returned output tensors order matches
  // Interpreter::outputs() of the last model segment.
  //
  // @return absl::OkStatus when output is received, or the pipeline input queue
  // has already been stopped, and is empty, in which case `output_tensors` will
  // be empty. Otherwise absl::InternalError.
  //
  // **Note:**
  //   *  Caller is responsible for deallocating memory for output tensors after
  //      consuming the tensors. By default, the output tensors are allocated
  //      using default tensor allocator. Caller can set a custom allocator for
  //      output tensors if needed.
  //
  //   *  Caller will get blocked if there is no output tensors available and no
  //      empty push is received.
  absl::Status Pop(std::vector<PipelineTensor>* output_tensors);

  // Returns performance stats for each segment.
  std::vector<SegmentStats> GetSegmentStats() const;

 private:
  // Returns ok status or the first error of runners.
  absl::Status GetRunnerStatus() const;

  // Returns true if pipeline was shutdown successfully, false if pipeline was
  // shutdown before.
  absl::Status ShutdownPipeline() ABSL_LOCKS_EXCLUDED(mu_);

  std::vector<tflite::Interpreter*> segments_interpreters_;

  const int num_segments_;

  // Queues for input, output, and intermediate tensors.
  // `segments_interpreters_[i]` consumes elements from `queues_[i]` and
  // produces elements to `queues_[i+1]`.
  //
  // size = num_segments_ + 1
  std::vector<internal::WaitQueue<internal::TensorMap>> queues_;

  // Each thread works with one model segment. size = num_segments_.
  std::vector<std::thread> threads_;

  // Records how many consumers each input/intermediate tensor has. This is
  // needed for each segment to decide when to release underlying memory for
  // each input/intermediate tensor.
  std::unordered_map<std::string, int> tensor_consumers_count_;

  // Segment runner is a convenient wrapper that gathers everything that is
  // needed to run one model segment.
  std::vector<std::unique_ptr<internal::SegmentRunner>> segments_runners_;

  // `input_tensor_names_per_segment_[i]` stores input tensors names for the
  // i-th model segment.
  std::vector<std::unordered_set<std::string>> input_tensor_names_per_segment_;

  // Default tensor allocator for input and output tensors if caller does not
  // provide one.
  std::unique_ptr<Allocator> default_allocator_;
  // Tensor allocator for intermediate tensors.
  std::unique_ptr<Allocator> intermediate_tensor_allocator_;
  // Memory allocator for input tensors (of the first model segment).
  Allocator* input_tensor_allocator_ = nullptr;
  // Memory allocator for output tensors (of the last model segment).
  Allocator* output_tensor_allocator_ = nullptr;

  absl::Mutex mu_;
  bool pipeline_on_ ABSL_GUARDED_BY(mu_) = true;
};

}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_PIPELINED_MODEL_RUNNER_H_
