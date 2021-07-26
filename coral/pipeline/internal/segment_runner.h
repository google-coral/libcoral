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

#ifndef LIBCORAL_CORAL_PIPELINE_INTERNAL_SEGMENT_RUNNER_H_
#define LIBCORAL_CORAL_PIPELINE_INTERNAL_SEGMENT_RUNNER_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "coral/pipeline/allocator.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/internal/thread_safe_queue.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/stateful_error_reporter.h"
#include "tflite/public/edgetpu.h"

namespace coral {
namespace internal {

// A wrapper on top of PipelineTensor, it keeps track of how many consumers a
// tensor has. This is critical for managing the lifetime of intermediate
// tensors between model segments.
struct ManagedPipelineTensor {
  ManagedPipelineTensor() = default;
  ManagedPipelineTensor(const PipelineTensor& tensor, int num_consumers)
      : tensor(tensor), num_consumers(num_consumers) {}
  PipelineTensor tensor;
  int num_consumers = 0;
};

using TensorMap = std::unordered_map<std::string, ManagedPipelineTensor>;

// Wrapper class that provides API to run inference with a model segment.
//
// Segment runner does not use internal input and output tensor buffers
// allocated by tflite::Interpreter. Instead, it uses input tensor buffers
// allocated by the caller (which will be released by this class using
// `input_tensor_allocator` if applicable) and output tensor buffers allocated
// using `output_tensor_allocator` (by this class).
//
// Note:
//  *) This class assumes interpreter->AllocateTensors() has been called;
class SegmentRunner {
 public:
  SegmentRunner() = default;

  // The 'interpeter' must be created with a stateful error reporter such that
  // error messages can be properly caught.
  SegmentRunner(
      const std::unordered_map<std::string, int>* tensor_consumers_count,
      const std::unordered_set<std::string>* segment_input_tensor_names,
      tflite::Interpreter* interpreter,
      WaitQueue<internal::TensorMap>* input_queue,
      WaitQueue<internal::TensorMap>* output_queue,
      Allocator* input_tensor_allocator, Allocator* output_tensor_allocator)
      : tensor_consumers_count_(CHECK_NOTNULL(tensor_consumers_count)),
        segment_input_tensor_names_(CHECK_NOTNULL(segment_input_tensor_names)),
        interpreter_(CHECK_NOTNULL(interpreter)),
        input_queue_(CHECK_NOTNULL(input_queue)),
        output_queue_(CHECK_NOTNULL(output_queue)),
        input_tensor_allocator_(CHECK_NOTNULL(input_tensor_allocator)),
        output_tensor_allocator_(CHECK_NOTNULL(output_tensor_allocator)) {
    auto* context = interpreter_->primary_subgraph().context();
    auto* edgetpu_context = static_cast<edgetpu::EdgeTpuContext*>(
        context->GetExternalContext(context, kTfLiteEdgeTpuContext));
    error_reporter_ = static_cast<tflite::StatefulErrorReporter*>(
        interpreter_->error_reporter());
    support_dma_ = (edgetpu_context->GetDeviceEnumRecord().type ==
                    edgetpu::DeviceType::kApexPci);
  }

  // Runs inference until `input_queue_` is stopped and there's no pending
  // requests in the queue, or there is an error during inference.
  void RunInference();

  SegmentStats stats() const {
    absl::ReaderMutexLock lock(&mu_);
    return stats_;
  }

  absl::Status runner_status() const {
    absl::ReaderMutexLock lock(&mu_);
    return runner_status_;
  }

 private:
  // Runs inference once.
  //
  // `input_tensors` are allocated by caller and will be deallocated using
  // `input_tensor_allocator_` if `num_consumers` reaches 0.
  //
  // Returned tensors are allocated by this function using
  // `output_tensor_allocator_`, and it is caller's responsibility to free the
  // memory.
  absl::StatusOr<TensorMap> RunInferenceOnce(const TensorMap& input_tensors);

  // Forces tflite::Interpreter to use external buffer for particular tensor.
  absl::Status SetExternalTensorBuffer(const char* buffer,
                                       std::size_t size_bytes,
                                       int tensor_index);

  // Key is tensor name, value is number of consumers for the tensor.
  const std::unordered_map<std::string, int>* tensor_consumers_count_ = nullptr;
  //
  // Note that one can get the same information from `interpreter_`, however,
  // input tensors names are byproducts when caller constructs
  // `tensor_consumers_count_`.
  const std::unordered_set<std::string>* segment_input_tensor_names_ = nullptr;
  tflite::Interpreter* interpreter_ = nullptr;
  WaitQueue<internal::TensorMap>* input_queue_ = nullptr;
  WaitQueue<internal::TensorMap>* output_queue_ = nullptr;
  Allocator* input_tensor_allocator_ = nullptr;
  Allocator* output_tensor_allocator_ = nullptr;

  tflite::StatefulErrorReporter* error_reporter_ = nullptr;

  mutable absl::Mutex mu_;
  SegmentStats stats_ ABSL_GUARDED_BY(mu_);

  absl::Status runner_status_ ABSL_GUARDED_BY(mu_);

  bool support_dma_ = false;
};
}  // namespace internal
}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_INTERNAL_SEGMENT_RUNNER_H_
