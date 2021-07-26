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

#include <thread>  // NOLINT

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "glog/logging.h"

namespace coral {
namespace internal {
namespace {
TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
  if (!src) {
    return nullptr;
  }
  TfLiteFloatArray* ret = static_cast<TfLiteFloatArray*>(
      std::malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
  if (!ret) {
    return nullptr;
  }
  ret->size = src->size;
  std::memcpy(ret->data, src->data, src->size * sizeof(float));
  return ret;
}

void DmaDelegateFreeBufferHandle(TfLiteContext* context,
                                 struct TfLiteDelegate* delegate,
                                 TfLiteBufferHandle* handle) {
  // Caller owns dma_fd so we don't close it here.
}

// No-op TfLiteDelegate for dma-buf input. Does not support CPU access.
TfLiteDelegate DmaDelegate = {.data_ = nullptr,
                              .Prepare = nullptr,
                              .CopyFromBufferHandle = nullptr,
                              .CopyToBufferHandle = nullptr,
                              .FreeBufferHandle = DmaDelegateFreeBufferHandle,
                              .flags = kTfLiteDelegateFlagsNone};
}  // namespace

absl::Status SegmentRunner::SetExternalTensorBuffer(const char* buffer,
                                                    std::size_t size_bytes,
                                                    int tensor_index) {
  const TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
  const TfLiteType& type = tensor->type;
  const char* name = tensor->name;
  std::vector<int> dims(tensor->dims->data,
                        tensor->dims->data + tensor->dims->size);
  if (tensor->quantization.type == kTfLiteNoQuantization) {
    // Deal with legacy model with old quantization parameters.
    if (interpreter_->SetTensorParametersReadOnly(tensor_index, type, name,
                                                  dims, tensor->params, buffer,
                                                  size_bytes) != kTfLiteOk) {
      LOG(ERROR) << "Can not set external tensor buffer: "
                 << error_reporter_->message();
      return absl::InternalError(error_reporter_->message());
    }
  } else {
    // For models with new quantization parameters, deep copy the parameters.
    CHECK(tensor->quantization.type == kTfLiteAffineQuantization);
    CHECK(tensor->quantization.params);
    TfLiteQuantization quant_clone = tensor->quantization;
    const auto* quant_params = reinterpret_cast<TfLiteAffineQuantization*>(
        tensor->quantization.params);
    // |quant_params_clone| will be owned by |quant_clone|, and will be
    // deallocated by std::free(). Therefore std::malloc() is used to allocate
    // its memory here.
    auto* quant_params_clone = reinterpret_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    quant_params_clone->scale = TfLiteFloatArrayCopy(quant_params->scale);
    CHECK(quant_params_clone->scale);
    quant_params_clone->zero_point =
        TfLiteIntArrayCopy(quant_params->zero_point);
    CHECK(quant_params_clone->zero_point);
    quant_params_clone->quantized_dimension = quant_params->quantized_dimension;
    quant_clone.params = quant_params_clone;
    if (interpreter_->SetTensorParametersReadOnly(tensor_index, type, name,
                                                  dims, quant_clone, buffer,
                                                  size_bytes) != kTfLiteOk) {
      LOG(ERROR) << "Can not set external tensor buffer: "
                 << error_reporter_->message();
      return absl::InternalError(error_reporter_->message());
    }
  }

  // Sanity check.
  const auto* tflite_tensor = interpreter_->tensor(tensor_index);
  CHECK(tflite_tensor->data.data == buffer)
      << "Tensor is not using the given buffer!";
  return absl::OkStatus();
}

absl::StatusOr<TensorMap> SegmentRunner::RunInferenceOnce(
    const TensorMap& input_tensors) {
  const auto& start_time = std::chrono::steady_clock::now();

  // Allocate output tensors.
  TensorMap output_tensors;
  for (const auto& output_tensor_index : interpreter_->outputs()) {
    const auto* tflite_tensor = interpreter_->tensor(output_tensor_index);
    PipelineTensor output_tensor;
    output_tensor.name = tflite_tensor->name;
    output_tensor.type = tflite_tensor->type;
    output_tensor.buffer =
        output_tensor_allocator_->Alloc(tflite_tensor->bytes);
    output_tensor.bytes = tflite_tensor->bytes;
    output_tensors.insert(
        {tflite_tensor->name, {output_tensor, /*num_consumers=*/0}});
  }

  // Force tflite interpreter to use external buffers for input tensors.
  for (const auto& tensor_index : interpreter_->inputs()) {
    const auto it =
        input_tensors.find(interpreter_->tensor(tensor_index)->name);
    CHECK(it != input_tensors.end());
    auto* buffer = it->second.tensor.buffer;
    if (buffer->fd() > -1) {
      if (support_dma_) {
        if (interpreter_->SetBufferHandle(tensor_index, buffer->fd(),
                                          &DmaDelegate) != kTfLiteOk) {
          LOG(ERROR) << "Can not set DMA buffer input: "
                     << error_reporter_->message();
          return absl::InternalError(error_reporter_->message());
        }
      } else {
        const auto status = SetExternalTensorBuffer(
            static_cast<const char*>(CHECK_NOTNULL(buffer->MapToHost())),
            it->second.tensor.bytes, tensor_index);
        if (!status.ok()) return status;
      }
    } else {
      const auto status = SetExternalTensorBuffer(
          static_cast<const char*>(CHECK_NOTNULL(buffer->ptr())),
          it->second.tensor.bytes, tensor_index);
      if (!status.ok()) return status;
    }
  }

  // Force tflite interpreter to use external buffers for output tensors.
  for (const auto& tensor_index : interpreter_->outputs()) {
    const auto it =
        output_tensors.find(interpreter_->tensor(tensor_index)->name);
    CHECK(it != output_tensors.end());
    const auto status = SetExternalTensorBuffer(
        CHECK_NOTNULL(
            reinterpret_cast<const char*>(it->second.tensor.buffer->ptr())),
        it->second.tensor.bytes, tensor_index);
    if (!status.ok()) return status;
  }

  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "Inference failed: " << error_reporter_->message();
    return absl::InternalError(error_reporter_->message());
  }

  // Unmap buffer if it was mapped before.
  for (const auto& tensor_index : interpreter_->inputs()) {
    const auto it =
        input_tensors.find(interpreter_->tensor(tensor_index)->name);
    CHECK(it != input_tensors.end());
    auto* buffer = it->second.tensor.buffer;
    if (buffer->fd() > -1 && !support_dma_) {
      CHECK(buffer->UnmapFromHost());
    }
  }

  std::chrono::duration<int64_t, std::nano> time_span =
      std::chrono::steady_clock::now() - start_time;

  {
    absl::MutexLock lock(&mu_);
    stats_.total_time_ns += time_span.count();
    stats_.num_inferences++;
  }

  return output_tensors;
}

void SegmentRunner::RunInference() {
  TensorMap input_tensors;
  const auto tid = std::this_thread::get_id();
  VLOG(1) << "Thread: " << tid << ". Runner loop started";
  while (input_queue_->Wait(&input_tensors)) {
    VLOG(1) << "Thread: " << tid << ". Run inference.";
    absl::StatusOr<TensorMap> status_or_output_tensors;
    if (input_tensors.empty()) {
      LOG(ERROR) << "Thread: " << tid
                 << ". Empty input tensor map. Stop runner.";
      status_or_output_tensors = absl::InternalError("Empty input tensor map.");
    } else {
      status_or_output_tensors = RunInferenceOnce(input_tensors);
    }

    // Reduce consumers count for used input tensors. For tensors that do not
    // have consumers, release the memory if caller provides a valid allocator.
    for (auto& pair : input_tensors) {
      auto& name = pair.first;
      auto& tensor = pair.second;
      // `input_tensors` is unconsumed tensors from previous segments, it can
      // contain tensors that will not be used by this segment.
      if (segment_input_tensor_names_->find(name) !=
          segment_input_tensor_names_->end()) {
        tensor.num_consumers--;
      }
      if (tensor.num_consumers == 0) {
        // Clean up input tensor.
        VLOG(1) << "Thread: " << tid << ". Releasing " << name << " at addr: "
                << static_cast<void*>(tensor.tensor.buffer->ptr());
        input_tensor_allocator_->Free(tensor.tensor.buffer);
      }
    }

    if (!status_or_output_tensors.ok()) {
      LOG(ERROR) << "Thread: " << tid
                 << ". Segment runner has error. Stop waiting loop.";
      {
        absl::MutexLock lock(&mu_);
        runner_status_ = status_or_output_tensors.status();
      }
      // Push an empty tensor map so that downstream runners could fail
      // gracefully.
      output_queue_->push({});
      break;
    }

    // Set output tensors' consumers count
    for (auto& pair : *status_or_output_tensors) {
      auto& name = pair.first;
      auto& tensor = pair.second;
      const auto it = tensor_consumers_count_->find(name);
      tensor.num_consumers =
          (it != tensor_consumers_count_->end()) ? it->second : 0;
    }

    // For input tensors that still have consumers, let them flow to the next
    // segment.
    for (auto& pair : input_tensors) {
      auto& name = pair.first;
      auto& tensor = pair.second;
      if (tensor.num_consumers > 0) {
        // Flow to the next segment.
        status_or_output_tensors->insert({name, tensor});
      }
    }

    output_queue_->push(*status_or_output_tensors);
  }
  VLOG(1) << "Thread: " << tid << ". Runner loop stopped";
}

}  // namespace internal
}  // namespace coral
