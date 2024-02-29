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

#include "coral/tflite_utils.h"

#include <cstdlib>
#include <cstring>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"
#include "coral/pose_estimation/posenet_decoder_op.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace coral {
namespace {
TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
  if (!src) return nullptr;

  auto* copy = static_cast<TfLiteFloatArray*>(
      malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
  CHECK(copy);
  copy->size = src->size;
  std::memcpy(copy->data, src->data, src->size * sizeof(float));
  return copy;
}

TfLiteAffineQuantization* TfLiteAffineQuantizationCopy(
    const TfLiteAffineQuantization* src) {
  if (!src) return nullptr;

  auto* copy = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  CHECK(copy);
  copy->scale = coral::TfLiteFloatArrayCopy(src->scale);
  copy->zero_point = TfLiteIntArrayCopy(src->zero_point);
  copy->quantized_dimension = src->quantized_dimension;
  return copy;
}

constexpr char kUsb[] = "usb";
constexpr char kPci[] = "pci";

using edgetpu::DeviceType;

bool MatchDevice(const std::string& s, const std::string& type, int* index) {
  const auto prefix(type + ":");
  if (!absl::StartsWith(s, prefix)) return false;
  if (!absl::SimpleAtoi(s.substr(prefix.size()), index)) return false;
  if (*index < 0) return false;
  return true;
}

absl::Status CheckInputSize(const TfLiteTensor& tensor, size_t size) {
  const size_t tensor_size = TensorSize(tensor);
  if (size < tensor_size)
    return absl::InternalError(absl::Substitute(
        "Input buffer ($0) has fewer entries than model input tensor ($1).",
        size, tensor_size));
  return absl::OkStatus();
}
}  // namespace

absl::Status MakeEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* tpu_context,
    tflite::ops::builtin::BuiltinOpResolver* resolver,
    tflite::StatefulErrorReporter* error_reporter,
    std::unique_ptr<tflite::Interpreter>* interpreter) {
  CHECK(interpreter);

  tflite::ops::builtin::BuiltinOpResolver builtin_resolver;
  if (resolver == nullptr) resolver = &builtin_resolver;
  resolver->AddCustom(kPosenetDecoderOp, RegisterPosenetDecoderOp());

  if (tpu_context)
    resolver->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  if (tflite::InterpreterBuilder(model.GetModel(), *resolver,
                                 error_reporter)(interpreter) == kTfLiteOk) {
    if (tpu_context)
      (*interpreter)->SetExternalContext(kTfLiteEdgeTpuContext, tpu_context);
    return absl::OkStatus();
  } else if (error_reporter) {
    return absl::InternalError(error_reporter->message());
  } else {
    return absl::InternalError(
        "Error in interpreter initialization. Lost tflite error messages due "
        "to null error reporter.");
  }
}

std::unique_ptr<tflite::Interpreter> MakeEdgeTpuInterpreterOrDie(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* tpu_context,
    tflite::ops::builtin::BuiltinOpResolver* resolver,
    tflite::StatefulErrorReporter* error_reporter) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  CHECK_EQ(MakeEdgeTpuInterpreter(model, tpu_context, resolver, error_reporter,
                                  &interpreter),
           absl::OkStatus());
  return interpreter;
}

absl::Status SetTensorBuffer(tflite::Interpreter* interpreter, int tensor_index,
                             const void* buffer, size_t buffer_size) {
  const auto* tensor = interpreter->tensor(tensor_index);
  CHECK(tensor);

  auto quantization = tensor->quantization;
  if (quantization.type != kTfLiteNoQuantization) {
    // Deep copy quantization parameters.
    if (quantization.type != kTfLiteAffineQuantization)
      return absl::InternalError("Invalid quantization type.");
    quantization.params = TfLiteAffineQuantizationCopy(
        reinterpret_cast<TfLiteAffineQuantization*>(quantization.params));
  }

  const auto shape = TensorShape(*tensor);
  if (interpreter->SetTensorParametersReadOnly(
          tensor_index, tensor->type, tensor->name,
          std::vector<int>(shape.begin(), shape.end()), quantization,
          reinterpret_cast<const char*>(buffer), buffer_size) != kTfLiteOk)
    return absl::InternalError("Cannot set tensor parameters.");
  CHECK_EQ(tensor->data.raw, buffer);
  return absl::OkStatus();
}

std::shared_ptr<edgetpu::EdgeTpuContext> GetEdgeTpuContext(
    absl::optional<DeviceType> device_type, absl::optional<int> device_index,
    const edgetpu::EdgeTpuManager::DeviceOptions& options) {
  auto* manager = edgetpu::EdgeTpuManager::GetSingleton();
  if (!device_index.has_value()) {
    return device_type.has_value() ? manager->OpenDevice(device_type.value())
                                   : manager->OpenDevice();
  } else {
    const int index = device_index.value();
    CHECK_GE(index, 0);
    auto tpus = manager->EnumerateEdgeTpu();
    if (device_type.has_value()) {
      int i = 0;
      for (auto& record : tpus)
        if (record.type == device_type.value() && i++ == index)
          return manager->OpenDevice(record.type, record.path, options);
    } else {
      if (index < tpus.size())
        return manager->OpenDevice(tpus[index].type, tpus[index].path, options);
    }
    return nullptr;
  }
}

std::shared_ptr<edgetpu::EdgeTpuContext> GetEdgeTpuContext(
    const std::string& device,
    const edgetpu::EdgeTpuManager::DeviceOptions& options) {
  if (device.empty()) {
    return GetEdgeTpuContext(absl::nullopt, absl::nullopt, options);
  } else if (device == kUsb) {
    return GetEdgeTpuContext(DeviceType::kApexUsb, absl::nullopt, options);
  } else if (device == kPci) {
    return GetEdgeTpuContext(DeviceType::kApexPci, absl::nullopt, options);
  } else {
    int index;
    if (MatchDevice(device, "", &index)) {
      return GetEdgeTpuContext(absl::nullopt, index, options);
    } else if (MatchDevice(device, kUsb, &index)) {
      return GetEdgeTpuContext(DeviceType::kApexUsb, index, options);
    } else if (MatchDevice(device, kPci, &index)) {
      return GetEdgeTpuContext(DeviceType::kApexPci, index, options);
    } else {
      return nullptr;
    }
  }
}

absl::Status InvokeWithMemBuffer(tflite::Interpreter* interpreter,
                                 const void* buffer, size_t in_size,
                                 tflite::StatefulErrorReporter* reporter) {
  CHECK(buffer);

  const int input_tensor_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);

  auto status = CheckInputSize(*input_tensor, in_size);
  if (!status.ok()) return status;

  status = SetTensorBuffer(interpreter, input_tensor_index, buffer,
                           input_tensor->bytes);
  if (!status.ok()) return status;

  if (interpreter->Invoke() != kTfLiteOk)
    return absl::InternalError("InvokeWithMemBuffer failed" +
                               (reporter ? ": " + reporter->message() : ""));

  return absl::OkStatus();
}

absl::Status InvokeWithDmaBuffer(tflite::Interpreter* interpreter, int dma_fd,
                                 size_t in_size,
                                 tflite::StatefulErrorReporter* reporter) {
  CHECK_GE(dma_fd, 0);

  const int input_tensor_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);

  auto status = CheckInputSize(*input_tensor, in_size);
  if (!status.ok()) return status;

  const auto old_buffer_handle = input_tensor->buffer_handle;
  input_tensor->buffer_handle = dma_fd;
  const bool success = interpreter->Invoke() == kTfLiteOk;
  input_tensor->buffer_handle = old_buffer_handle;

  if (!success)
    return absl::InternalError("InvokeWithDmaBuffer failed" +
                               (reporter ? ": " + reporter->message() : ""));

  return absl::OkStatus();
}

bool ContainsEdgeTpuCustomOp(const tflite::FlatBufferModel& model) {
  const auto* opcodes = model.GetModel()->operator_codes();
  for (const auto* subgraph : *model.GetModel()->subgraphs()) {
    for (const auto* op : *subgraph->operators()) {
      const auto* opcode = opcodes->Get(op->opcode_index());
      if (opcode->custom_code() &&
          opcode->custom_code()->str() == edgetpu::kCustomOp) {
        return true;
      }
    }
  }
  return false;
}

std::unordered_set<std::string> GetInputTensorNames(
    const tflite::Interpreter& interpreter) {
  std::unordered_set<std::string> names(interpreter.inputs().size());
  for (int i = 0; i < interpreter.inputs().size(); ++i) {
    names.insert(interpreter.input_tensor(i)->name);
  }
  return names;
}

const TfLiteTensor* GetInputTensor(const tflite::Interpreter& interpreter,
                                   const char* name) {
  for (const int input_index : interpreter.inputs()) {
    const auto* input_tensor = interpreter.tensor(input_index);
    if (std::strcmp(input_tensor->name, name) == 0) {
      return input_tensor;
    }
  }
  return nullptr;
}
}  // namespace coral
