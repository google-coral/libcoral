#ifndef EDGETPU_CPP_TFLITE_UTILS_H_
#define EDGETPU_CPP_TFLITE_UTILS_H_

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

// Returns whether the shape matches the pattern. Negative numbers in the
// pattern indicate the corresponding shape dimension can be anything. Use -1
// in the pattern for cosnsitency.
inline bool MatchShape(absl::Span<const int> shape,
                       const std::vector<int>& pattern) {
  if (shape.size() != pattern.size()) return false;
  for (size_t i = 0; i < shape.size(); ++i)
    if (pattern[i] >= 0 && pattern[i] != shape[i]) return false;
  return true;
}

inline absl::Span<const int> TensorShape(const TfLiteTensor& tensor) {
  return absl::Span<const int>(tensor.dims->data, tensor.dims->size);
}

inline int TensorSize(const TfLiteTensor& tensor) {
  auto shape = TensorShape(tensor);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template <typename T>
absl::Span<const T> TensorData(const TfLiteTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<const T*>(tensor.data.data),
                        tensor.bytes / sizeof(T));
}

template <typename T>
absl::Span<T> MutableTensorData(const TfLiteTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<T*>(tensor.data.data),
                        tensor.bytes / sizeof(T));
}

template <typename InputIt, typename OutputIt>
OutputIt Dequantize(InputIt first, InputIt last, OutputIt d_first, float scale,
                    int32_t zero_point) {
  while (first != last) *d_first++ = scale * (*first++ - zero_point);
  return d_first;
}

template <typename T, typename OutputIt>
OutputIt Dequantize(absl::Span<const T> span, OutputIt d_first, float scale,
                    int32_t zero_point) {
  return Dequantize(span.begin(), span.end(), d_first, scale, zero_point);
}

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

// Returns interpreter which can run Edge TPU models if tpu_context is not null,
// otherwise returns regular interpreter. PoseNet custom op is always supported.
// `resolver` and `error_reporter` can be null, in which case the default
// resolver and error reporter objects will be used.
// Note: when `error_reporter` is null, tflite runtime error message
// will not be returned.
absl::Status MakeEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* tpu_context,
    tflite::ops::builtin::BuiltinOpResolver* resolver,
    tflite::StatefulErrorReporter* error_reporter,
    std::unique_ptr<tflite::Interpreter>* interpreter);

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

inline std::unique_ptr<tflite::FlatBufferModel> LoadModelOrDie(
    const std::string& path) {
  return CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(path.c_str()));
}

inline std::unique_ptr<tflite::FlatBufferModel> LoadModelOrDie(
    const flatbuffers::FlatBufferBuilder& fbb) {
  return CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize()));
}

// Invoke tflite::Interpreter by using given buffer as an input tensor.
// For input, we assume there is only one tensor. Input buffer contains
// |in_size| elements and could have padding elements at the end. |in_size|
// could be larger than the input tensor size, denoted by n, and only the first
// n elements of the input buffer will be used. |in_size| can not be smaller
// than n.
//
// Note: null `reporter` is allowed, however, tflite runtime error message will
// not be returned in this case. To get tflite runtime error message, `reporter`
// must be set to the one that is used to create interpreter.
absl::Status InvokeWithMemBuffer(
    tflite::Interpreter* interpreter, const void* buffer, size_t in_size,
    tflite::StatefulErrorReporter* reporter = nullptr);

// Invoke tflite::Interpreter by using given DMA file descriptor as an input
// tensor. Works only for Edge TPU models running on PCIe TPU devices.
//
// Note: null `reporter` is allowed, however, tflite runtime error message will
// not be returned in this case. To get tflite runtime error message, `reporter`
// must be set to the one that is used to create interpreter.
absl::Status InvokeWithDmaBuffer(
    tflite::Interpreter* interpreter, int dma_fd, size_t in_size,
    tflite::StatefulErrorReporter* reporter = nullptr);

// Returns whether a tflite model contains any Edge TPU custom operator.
bool ContainsEdgeTpuCustomOp(const tflite::FlatBufferModel& model);

}  // namespace coral

#endif  // EDGETPU_CPP_TFLITE_UTILS_H_
