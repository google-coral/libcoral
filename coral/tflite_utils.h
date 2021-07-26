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

#ifndef LIBCORAL_CORAL_TFLITE_UTILS_H_
#define LIBCORAL_CORAL_TFLITE_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stateful_error_reporter.h"
#include "tflite/public/edgetpu.h"

namespace coral {

// Checks whether a vector/tensor shape matches a dimensional pattern. Negative
// numbers in the pattern indicate the corresponding shape dimension can be
// anything. Use -1 in the pattern for consistency.
//
// @param shape The shape you want to evaluate.
// @param pattern The pattern to compare against.
// @return True if the shape matches, False if not.
inline bool MatchShape(absl::Span<const int> shape,
                       const std::vector<int>& pattern) {
  if (shape.size() != pattern.size()) return false;
  for (size_t i = 0; i < shape.size(); ++i)
    if (pattern[i] >= 0 && pattern[i] != shape[i]) return false;
  return true;
}

// Gets the tensor shape.
inline absl::Span<const int> TensorShape(const TfLiteTensor& tensor) {
  return absl::Span<const int>(tensor.dims->data, tensor.dims->size);
}

// Gets the tensor size.
inline int TensorSize(const TfLiteTensor& tensor) {
  auto shape = TensorShape(tensor);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// Gets the immutable data from the given tensor.
template <typename T>
absl::Span<const T> TensorData(const TfLiteTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<const T*>(tensor.data.data),
                        tensor.bytes / sizeof(T));
}

// Gets the mutable data from the given tensor.
template <typename T>
absl::Span<T> MutableTensorData(const TfLiteTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<T*>(tensor.data.data),
                        tensor.bytes / sizeof(T));
}

// Dequantizes the specified vector space.
template <typename InputIt, typename OutputIt>
OutputIt Dequantize(InputIt first, InputIt last, OutputIt d_first, float scale,
                    int32_t zero_point) {
  while (first != last) *d_first++ = scale * (*first++ - zero_point);
  return d_first;
}

// Returns a dequantized version of the given vector span.
template <typename T, typename OutputIt>
OutputIt Dequantize(absl::Span<const T> span, OutputIt d_first, float scale,
                    int32_t zero_point) {
  return Dequantize(span.begin(), span.end(), d_first, scale, zero_point);
}

// Returns a dequantized version of the given tensor.
template <typename T>
std::vector<T> DequantizeTensor(const TfLiteTensor& tensor) {
  const auto scale = tensor.params.scale;
  const auto zero_point = tensor.params.zero_point;
  std::vector<T> result(TensorSize(tensor));

  if (tensor.type == kTfLiteUInt8)
    Dequantize(TensorData<uint8_t>(tensor), result.begin(), scale, zero_point);
  else if (tensor.type == kTfLiteInt8)
    Dequantize(TensorData<int8_t>(tensor), result.begin(), scale, zero_point);
  else
    LOG(FATAL) << "Unsupported tensor type: " << tensor.type;

  return result;
}

// Quantizes the specified vector space.
template <typename InputIt, typename OutputIt>
OutputIt Quantize(InputIt first, InputIt last, OutputIt d_first, float scale,
                  int32_t zero_point) {
  using InT = typename std::iterator_traits<InputIt>::value_type;
  using OutT = typename std::iterator_traits<OutputIt>::value_type;
  while (first != last) {
    *d_first++ = static_cast<OutT>(std::max<InT>(
        std::numeric_limits<OutT>::min(),
        std::min<InT>(std::numeric_limits<OutT>::max(),
                      std::round(zero_point + (*first++ / scale)))));
  }
  return d_first;
}

// Creates a new interpreter instance for an Edge TPU model.
//
// Also consider using `MakeEdgeTpuInterpreterOrDie()`.
//
// @param model The tflite model.
// @param tpu_context The Edge TPU context, from `coral::GetEdgeTpuContext()`.
//   If left null, the given interpreter will not resolve an Edge TPU delegate.
//   PoseNet custom op is always supported.
// @param resolver Optional. May be null to use a default resolver.
// @param error_reporter Optional. May be null to use default error reporter,
//   but beware that if null, tflite runtime error messages will not return.
// @param interpreter The pointer to receive the new interpreter.
absl::Status MakeEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* tpu_context,
    tflite::ops::builtin::BuiltinOpResolver* resolver,
    tflite::StatefulErrorReporter* error_reporter,
    std::unique_ptr<tflite::Interpreter>* interpreter);

// Returns a new interpreter instance for an Edge TPU model, crashing if it
// cannot be created.
//
// For example:
//
//  ```
//  const auto model = coral::LoadModelOrDie(model_path);
//  auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
//                             ? coral::GetEdgeTpuContextOrDie()
//                             : nullptr;
//  auto interpreter =
//      coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
//  ```
//
// @param model The tflite model.
// @param tpu_context The Edge TPU context, from `coral::GetEdgeTpuContext()`.
//   If left null, the given interpreter will not resolve an Edge TPU delegate.
//   PoseNet custom op is always supported.
// @param resolver Optional. May be null to use a default resolver.
// @param error_reporter Optional. May be null to use default error reporter,
//   but beware that if null, tflite runtime error messages will not return.
// @returns The new interpreter instance.
std::unique_ptr<tflite::Interpreter> MakeEdgeTpuInterpreterOrDie(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* tpu_context = nullptr,
    tflite::ops::builtin::BuiltinOpResolver* resolver = nullptr,
    tflite::StatefulErrorReporter* error_reporter = nullptr);

// Replaces existing tensor buffer with the provided one. Caller owns provided
// buffer. Tensor quantization parameters are preserved. This function is a
// required 'hack' for performance reasons until this functionality would
// become a part of TensorFlow Lite API.
absl::Status SetTensorBuffer(tflite::Interpreter* interpreter, int tensor_index,
                             const void* buffer, size_t buffer_size);

// Returns TPU context or nullptr if requested TPU context is not available.
//
// Parameter `device`:
//   - ""      -- any TPU device
//   - "usb"   -- any TPU device on USB bus
//   - "pci"   -- any TPU device on PCIe bus
//   - ":N"    -- N-th TPU device, e.g. ":0"
//   - "usb:N" -- N-th TPU device on USB bus, e.g. "usb:0"
//   - "pci:N" -- N-th TPU device on PCIe bus, e.g. "pci:0"
//
// Parameter `options`:
//   See edgetpu.h for details.
//
// All TPUs are always enumerated in the same order assuming hardware
// configuration doesn't change (no added/removed devices between enumerations).
// Under the assumption above, the same index N will always point to the same
// device.
//
// Consider 2 USB devices and 4 PCIe devices connected to the host. The way to
// reference specifically USB devices:
//   "usb:0", "usb:1".
// The way to reference specifically PCIe devices:
//   "pci:0", "pci:1", "pci:2", "pci:3".
// The generic way to reference all devices (no assumption about device type):
//   ":0", ":1", ":2", ":3", ":4", ":5".
std::shared_ptr<edgetpu::EdgeTpuContext> GetEdgeTpuContext(
    const std::string& device,
    const edgetpu::EdgeTpuManager::DeviceOptions& options = {});

// The same as above but crashes if requested TPU context is not available.
inline std::shared_ptr<edgetpu::EdgeTpuContext> GetEdgeTpuContextOrDie(
    const std::string& device,
    const edgetpu::EdgeTpuManager::DeviceOptions& options = {}) {
  return CHECK_NOTNULL(GetEdgeTpuContext(device, options));
}

// The same as previously defined `GetEdgeTpuContext` except `device` parameter
// is replaced with two separate ones: `device_type` and `device_index`.
//
// Custom options would only be passed when `device_type` and `device_index` are
// non-empty.
std::shared_ptr<edgetpu::EdgeTpuContext> GetEdgeTpuContext(
    absl::optional<edgetpu::DeviceType> device_type = absl::nullopt,
    absl::optional<int> device_index = absl::nullopt,
    const edgetpu::EdgeTpuManager::DeviceOptions& options = {});

// The same as above but crashes if requested TPU context is not available.
inline std::shared_ptr<edgetpu::EdgeTpuContext> GetEdgeTpuContextOrDie(
    absl::optional<edgetpu::DeviceType> device_type = absl::nullopt,
    absl::optional<int> device_index = absl::nullopt,
    const edgetpu::EdgeTpuManager::DeviceOptions& options = {}) {
  return CHECK_NOTNULL(GetEdgeTpuContext(device_type, device_index, options));
}

// Load a tflite model at the given file path or die trying.
inline std::unique_ptr<tflite::FlatBufferModel> LoadModelOrDie(
    const std::string& path) {
  return CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(path.c_str()));
}

// Load a tflite model via flatbuffer or die trying.
inline std::unique_ptr<tflite::FlatBufferModel> LoadModelOrDie(
    const flatbuffers::FlatBufferBuilder& fbb) {
  return CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize()));
}

// Invokes `tflite::Interpreter` using a given buffer as an input tensor.
//
// @param interpreter An initialized interpreter.
// @param buffer The interpreter input. We assume there is only one tensor.
// @param in_size The number of elements in the input buffer, which can have
//   padding elements at the end. `in_size` can be larger than the input tensor
//   size, denoted by n, and only the first n elements of the input buffer will
//   be used. `in_size` can not be smaller than n.
// @param reporter Optional. If left null, tflite runtime error messages will
//   not be returned. To get tflite runtime error messages,
//   `reporter` must be set to the one that is used to create interpreter.
absl::Status InvokeWithMemBuffer(
    tflite::Interpreter* interpreter, const void* buffer, size_t in_size,
    tflite::StatefulErrorReporter* reporter = nullptr);

// Invokes `tflite::Interpreter` using a given DMA file descriptor as an input
// tensor. Works only for Edge TPU models running on PCIe Edge TPU devices.
//
// @param interpreter An initialized interpreter.
// @param dma_fd The DMA file descriptor to use as input.
// @param in_size The number of elements in the input buffer, which can have
//   padding elements at the end. `in_size` can be larger than the input tensor
//   size, denoted by n, and only the first n elements of the input buffer will
//   be used. `in_size` can not be smaller than n.
// @param reporter Optional. If left null, tflite runtime error messages will
//   not be returned. To get tflite runtime error messages,
//   `reporter` must be set to the one that is used to create interpreter.
absl::Status InvokeWithDmaBuffer(
    tflite::Interpreter* interpreter, int dma_fd, size_t in_size,
    tflite::StatefulErrorReporter* reporter = nullptr);

// Checks whether a tflite model contains any Edge TPU custom operator.
bool ContainsEdgeTpuCustomOp(const tflite::FlatBufferModel& model);

// Returns all input tensor names for the given tflite::Interpreter.
std::unordered_set<std::string> GetInputTensorNames(
    const tflite::Interpreter& interpreter);

// Returns the input tensor matching `name` in the given tflite::Interpreter.
// Returns nullptr if such tensor does not exist.
const TfLiteTensor* GetInputTensor(const tflite::Interpreter& interpreter,
                                   const char* name);

}  // namespace coral

#endif  // LIBCORAL_CORAL_TFLITE_UTILS_H_
