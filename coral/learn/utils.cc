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

#include "coral/learn/utils.h"

#include <algorithm>
#include <cstring>
#include <tuple>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace coral {
namespace {

tflite::QuantizationParametersT* NewQuantParam(
    const std::vector<float>& min, const std::vector<float>& max,
    const std::vector<float>& scale, const std::vector<int64_t>& zero_point) {
  auto* result = new tflite::QuantizationParametersT;
  result->min = min;
  result->max = max;
  result->scale = scale;
  result->zero_point = zero_point;
  return result;
}

// Gets total number of elements represented by `shape`.
int GetTotalElements(const std::vector<int>& shape) {
  int result = 1;
  for (const auto& s : shape) {
    result *= s;
  }
  return result;
}

// Finds the index of `target_op`'s opcode, if it does not exist, add
// corresponding opcode to `model_t`. Returns index of `target_op`'s opcode.
int FindOrCreateOpcode(tflite::BuiltinOperator target_op,
                       const std::string& custom_code,
                       tflite::ModelT* model_t) {
  int target_opcode_index =
      FindOpcodeIndex(model_t->operator_codes, target_op, custom_code);
  VLOG(1) << "Target " << EnumNameBuiltinOperator(target_op)
          << "'s opcode index: " << target_opcode_index;

  if (target_opcode_index == -1) {
    // Add opcode.
    auto target_opcode = absl::make_unique<tflite::OperatorCodeT>();
    target_opcode->deprecated_builtin_code =
        tflite::ConvertBuiltinCodeToDeprecatedBuiltinCode(target_op);
    target_opcode->builtin_code = target_op;
    if (target_op == tflite::BuiltinOperator_CUSTOM) {
      target_opcode->custom_code = custom_code;
    }
    model_t->operator_codes.push_back(std::move(target_opcode));
    target_opcode_index = model_t->operator_codes.size() - 1;
    VLOG(1) << "Opcode is added with index: " << target_opcode_index;
  }
  return target_opcode_index;
}

// Gets operator's builtin options. NOTE that These options are tuned only for
// last layer backprop method. Please modify them if use for other purposes.
tflite::BuiltinOptionsUnion GetOpBuiltinOptions(
    tflite::BuiltinOperator op_type,
    const std::vector<internal::TensorConfig>& tensor_configs) {
  tflite::BuiltinOptionsUnion result;
  switch (op_type) {
    case tflite::BuiltinOperator_L2_NORMALIZATION:
      result.type = tflite::BuiltinOptions_L2NormOptions;
      result.value = new tflite::L2NormOptionsT;
      break;

    case tflite::BuiltinOperator_CONV_2D: {
      auto* options = new tflite::Conv2DOptionsT;
      options->padding = tflite::Padding_SAME;
      options->stride_h = 1;
      options->stride_w = 1;
      result.type = tflite::BuiltinOptions_Conv2DOptions;
      result.value = options;
      break;
    }
    case tflite::BuiltinOperator_FULLY_CONNECTED:
      result.type = tflite::BuiltinOptions_FullyConnectedOptions;
      result.value = new tflite::FullyConnectedOptionsT;
      break;

    case tflite::BuiltinOperator_RESHAPE: {
      auto* options = new tflite::ReshapeOptionsT;
      options->new_shape = tensor_configs[0].shape;
      result.type = tflite::BuiltinOptions_ReshapeOptions;
      result.value = options;
      break;
    }
    case tflite::BuiltinOperator_SOFTMAX: {
      auto options = new tflite::SoftmaxOptionsT;
      // Can NOT leave `beta` as default 0. Otherwise, it will trigger check
      // failure in `QuantizeMultiplierGreaterThanOne`.
      // More info: tensorflow/lite/kernels/internal/quantization_util.cc
      options->beta = 1.0f;
      result.type = tflite::BuiltinOptions_SoftmaxOptions;
      result.value = options;
      break;
    }
    default:
      LOG(FATAL) << "Unsupported operator type: " << op_type;
      break;
  }
  return result;
}

// Returns size in bytes for tensor buffer, based on location, type and shape.
//
// NOTE: intermediate tensors (i.e., non-parameter tensors) must be refer to
// empty buffer; otherwise, the tensors' buffer will be treated as read-only,
// which at least causes Conv2d to fail because it always resizes the output
// tensor. More info: tensorflow/lite/kernels/conv.cc.
size_t GetBufferSizeBytes(const std::vector<int>& shape,
                          internal::TensorLocation location,
                          tflite::TensorType type) {
  size_t result = 0;
  if (location == internal::TensorLocation::kParameter) {
    if (type == tflite::TensorType_UINT8) {
      result = GetTotalElements(shape);
    } else if (type == tflite::TensorType_INT32) {
      result = GetTotalElements(shape) * sizeof(int32_t);
    } else if (type == tflite::TensorType_FLOAT32) {
      result = GetTotalElements(shape) * sizeof(float);
    } else {
      LOG(FATAL) << "Unsupported tensor type: " << type;
    }
  }
  VLOG(1) << "Buffer size in bytes: " << result;
  return result;
}

absl::Status ValidateClassificationModel(const tflite::ModelT* model_t) {
  CHECK(model_t);

  if (model_t->subgraphs.size() != 1)
    return absl::InternalError(absl::Substitute(
        "Model must have one and only one subgraph. Actual: $0.",
        model_t->subgraphs.size()));

  if (model_t->subgraphs[0]->outputs.size() != 1)
    return absl::InternalError(
        absl::Substitute("Model must have one and only one output. Actual: $0.",
                         model_t->subgraphs[0]->outputs.size()));

  return absl::OkStatus();
}

absl::Status ValidateOperatorInputs(
    const std::vector<internal::TensorConfig>& tensor_configs,
    tflite::BuiltinOperator op_type, const tflite::ModelT* model_t) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);

  switch (op_type) {
    case tflite::BuiltinOperator_CONV_2D:
    case tflite::BuiltinOperator_FULLY_CONNECTED:
      if (tensor_configs.size() != 3)
        return absl::InternalError(
            absl::Substitute("Conv, FullyConnected operator must have three "
                             "input tensors. Actual: $0",
                             tensor_configs.size()));
      return absl::OkStatus();
    case tflite::BuiltinOperator_L2_NORMALIZATION:
    case tflite::BuiltinOperator_RESHAPE:
    case tflite::BuiltinOperator_SOFTMAX:
      if (tensor_configs.size() != 1)
        return absl::InternalError(absl::Substitute(
            "L2-Norm, Reshape, and Softmax operators must have "
            "one input tensor. Actual: $0",
            tensor_configs.size()));
      return absl::OkStatus();
    case tflite::BuiltinOperator_CUSTOM:
      return absl::OkStatus();
    default:
      return absl::InternalError(
          absl::Substitute("Unsupported operator type: $0", op_type));
  }
}

// Calculates quantization parameters for Conv2d / FullyConnected operator.
// Returns quantization parameters for kernel, bias, and output tensor.
std::unique_ptr<tflite::QuantizationParametersT> WeightsQuantParam(
    absl::Span<const float> weights) {
  float min_weights_value =
      std::min(0.0f, *std::min_element(weights.begin(), weights.end()));
  float max_weights_value =
      std::max(0.0f, *std::max_element(weights.begin(), weights.end()));
  VLOG(1) << absl::Substitute("Weights range: ($0, $1).", min_weights_value,
                              max_weights_value);
  float weights_scale;
  int32_t weights_zero_point;
  std::tie(weights_scale, weights_zero_point) =
      QuantizationParams<uint8_t>(min_weights_value, max_weights_value);
  VLOG(1) << absl::Substitute("Weights (scale, zero_point): ($0, $1) ",
                              weights_scale, weights_zero_point);
  return CreateQuantParam(/*min=*/{min_weights_value},
                          /*max=*/{max_weights_value},
                          /*scale=*/{weights_scale},
                          /*zero_point=*/{weights_zero_point});
}

std::unique_ptr<tflite::QuantizationParametersT> BiasesQuantParam(
    absl::Span<const float> biases,
    const tflite::QuantizationParametersT& input_tensor_quant,
    float weights_scale) {
  auto min_biases_value =
      std::min(0.0f, *std::min_element(biases.begin(), biases.end()));
  auto max_biases_value =
      std::max(0.0f, *std::max_element(biases.begin(), biases.end()));
  VLOG(1) << absl::Substitute("Biases range: ($0, $1).", min_biases_value,
                              max_biases_value);
  // TFLite's conv2d implementation is very picky about quantization parameter
  // of bias. See `scale` computation in `GetQuantizedConvolutionMultipler` of
  // tensorflow/lite/kernels/kernel_util.cc
  //
  // Basically, it asks for biases_scale = input_tensor_scale * weights_scale
  float biases_scale = input_tensor_quant.scale[0] * weights_scale;
  // TFLite requires biases's zero point be 0.
  int32_t biases_zero_point = 0;
  VLOG(1) << absl::Substitute("Biases (scale, zero_point): ($0, $1) ",
                              biases_scale, biases_zero_point);
  return CreateQuantParam(/*min=*/{min_biases_value},
                          /*max=*/{max_biases_value},
                          /*scale=*/{biases_scale},
                          /*zero_point=*/{biases_zero_point});
}

std::unique_ptr<tflite::QuantizationParametersT> OutputTensorQuantParam(
    float out_tensor_min, float out_tensor_max) {
  out_tensor_min = std::min(0.0f, out_tensor_min);
  out_tensor_max = std::max(0.0f, out_tensor_max);
  VLOG(1) << absl::Substitute("Output range: ($0, $1).", out_tensor_min,
                              out_tensor_max);
  float output_scale;
  int32_t output_zero_point;
  std::tie(output_scale, output_zero_point) =
      QuantizationParams<uint8_t>(out_tensor_min, out_tensor_max);
  VLOG(1) << absl::Substitute("Output (scale, zero_point): ($0, $1) ",
                              output_scale, output_zero_point);
  return CreateQuantParam(/*min=*/{out_tensor_min},
                          /*max=*/{out_tensor_max},
                          /*scale=*/{output_scale},
                          /*zero_point=*/{output_zero_point});
}
}  // namespace

namespace internal {

int AppendL2Norm(tflite::ModelT* model_t) {
  const tflite::TensorT* output_tensor = GetGraphOutputTensors(model_t)[0];
  return AppendOperator({{"Imprinting/L2Norm/Output", tflite::TensorType_UINT8,
                          TensorLocation::kOutput, output_tensor->shape,
                          NewQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f},
                                        /*scale=*/{1.0f / 128},
                                        /*zero_point=*/{128})}},
                        tflite::BuiltinOperator_L2_NORMALIZATION, model_t);
}

int AppendLinearLayer(
    const std::vector<int>& kernel_shape,
    std::unique_ptr<tflite::QuantizationParametersT> kernel_quant,
    std::unique_ptr<tflite::QuantizationParametersT> bias_quant,
    std::unique_ptr<tflite::QuantizationParametersT> output_quant,
    tflite::ModelT* model_t) {
  const tflite::TensorT* output_tensor = GetGraphOutputTensors(model_t)[0];

  tflite::BuiltinOperator op_type =
      kernel_shape.size() == 2 ? tflite::BuiltinOperator_FULLY_CONNECTED
                               : tflite::BuiltinOperator_CONV_2D;
  return AppendOperator(
      {{"Appended/FC/Weights", tflite::TensorType_UINT8,
        TensorLocation::kParameter, kernel_shape, kernel_quant.release()},
       {"Appended/FC/Bias",
        tflite::TensorType_INT32,
        TensorLocation::kParameter,
        {kernel_shape[0]},
        bias_quant.release()},
       {"Appended/FC/Output", tflite::TensorType_UINT8, TensorLocation::kOutput,
        CalculateLinearLayerOutputShape(output_tensor->shape, kernel_shape),
        output_quant.release()}},
      op_type, model_t);
}

int AppendReshape(tflite::ModelT* model_t) {
  // Output tensor of reshape should have the same quantization parameters as
  // its input, which is current graph's output tensor.
  const tflite::TensorT* output_tensor = GetGraphOutputTensors(model_t)[0];
  return AppendOperator(
      {{"Imprinting/Reshape/Output",
        tflite::TensorType_UINT8,
        TensorLocation::kOutput,
        {output_tensor->shape.front(), output_tensor->shape.back()},
        NewQuantParam(output_tensor->quantization->min,
                      output_tensor->quantization->max,
                      output_tensor->quantization->scale,
                      output_tensor->quantization->zero_point)}},
      tflite::BuiltinOperator_RESHAPE, model_t);
}

int AppendSoftmax(tflite::ModelT* model_t) {
  // Softmax's output is always within [0,1]
  const tflite::TensorT* output_tensor = GetGraphOutputTensors(model_t)[0];
  return AppendOperator(
      {{"Imprinting/Softmax/Output", tflite::TensorType_UINT8,
        TensorLocation::kOutput, output_tensor->shape,
        NewQuantParam(/*min=*/{0.0f}, /*max=*/{1.0f}, /*scale=*/{1.0f / 256},
                      /*zero_point=*/{0})}},
      tflite::BuiltinOperator_SOFTMAX, model_t);
}

int AppendBuffer(size_t buffer_size_bytes, tflite::ModelT* model_t) {
  CHECK(model_t);
  model_t->buffers.emplace_back(absl::make_unique<tflite::BufferT>());
  auto& buffer = model_t->buffers.back();
  buffer->data.resize(buffer_size_bytes);

  const auto buffer_index = model_t->buffers.size() - 1;
  VLOG(1) << "New buffer index is: " << buffer_index;
  return buffer_index;
}

int AppendTensor(const std::vector<int>& shape, const std::string& name,
                 int buffer_index, tflite::TensorType type,
                 std::unique_ptr<tflite::QuantizationParametersT> q_param,
                 tflite::SubGraphT* subgraph) {
  subgraph->tensors.emplace_back(absl::make_unique<tflite::TensorT>());
  auto& tensor = subgraph->tensors.back();
  tensor->type = type;
  tensor->shape = shape;
  tensor->buffer = buffer_index;
  tensor->name = name;
  if (q_param) tensor->quantization = std::move(q_param);

  const auto tensor_index = subgraph->tensors.size() - 1;
  VLOG(1) << "New tensor index: " << tensor_index;
  return tensor_index;
}

int AppendOperator(const std::vector<TensorConfig>& tensor_configs,
                   tflite::BuiltinOperator op_type, tflite::ModelT* model_t) {
  CHECK_EQ(ValidateOperatorInputs(tensor_configs, op_type, model_t),
           absl::OkStatus());

  auto& subgraph = model_t->subgraphs[0];
  subgraph->operators.emplace_back(absl::make_unique<tflite::OperatorT>());
  auto& op = subgraph->operators.back();
  op->opcode_index = FindOrCreateOpcode(op_type, /*custom_code=*/"", model_t);
  // Current graph's output will become input tensor.
  op->inputs.push_back(subgraph->outputs[0]);
  // Be careful about the ownership transfer here. Check BuiltinOptionUnion
  // class's API to understand better.
  op->builtin_options = GetOpBuiltinOptions(op_type, tensor_configs);

  // Add tensor to subgraph.
  for (const auto& config : tensor_configs) {
    VLOG(1) << "-----";
    VLOG(1) << "Tensor name: " << config.name;
    const int buffer_index = AppendBuffer(
        GetBufferSizeBytes(config.shape, config.location, config.type),
        model_t);
    const int tensor_index =
        AppendTensor(config.shape, config.name, buffer_index, config.type,
                     absl::WrapUnique(config.quant), subgraph.get());
    if (config.location == TensorLocation::kOutput) {
      op->outputs.push_back(tensor_index);
    } else {
      op->inputs.push_back(tensor_index);
    }
  }

  subgraph->outputs[0] = op->outputs[0];
  return subgraph->operators.size() - 1;
}

void SetLinearParams(const std::vector<uint8_t>& kernel,
                     const std::vector<int32_t>& bias, int op_index,
                     tflite::ModelT* model_t) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);
  auto& subgraph = model_t->subgraphs[0];
  CHECK_GT(subgraph->operators.size(), op_index);
  auto& conv2d_op = subgraph->operators[op_index];

  // Conv2d in TFLite has 3 inputs and uses the following convention:
  //  - input 1 is input tensor;
  //  - input 2 is kernel tensor;
  //  - input 3 is bias tensor;
  const int kernel_tensor_index = conv2d_op->inputs[1];
  const int bias_tensor_index = conv2d_op->inputs[2];

  auto& kernel_tensor = subgraph->tensors[kernel_tensor_index];
  auto& kernel_buffer = model_t->buffers[kernel_tensor->buffer];

  // Resize buffer if necessary.
  if (kernel_buffer->data.size() < kernel.size()) {
    kernel_buffer->data.resize(kernel.size());
  }
  std::memcpy(kernel_buffer->data.data(), kernel.data(), kernel.size());

  auto& bias_tensor = subgraph->tensors[bias_tensor_index];
  auto& bias_buffer = model_t->buffers[bias_tensor->buffer];
  if (!bias.empty()) {
    CHECK_EQ(bias.size() * sizeof(bias.data()[0]), bias_buffer->data.size());
    std::memcpy(bias_buffer->data.data(), bias.data(),
                bias_buffer->data.size());
  } else {
    std::fill(bias_buffer->data.begin(), bias_buffer->data.end(), 0);
  }
}

std::vector<int> CalculateLinearLayerOutputShape(
    const std::vector<int>& input_shape, const std::vector<int>& kernel_shape) {
  std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(kernel_shape.front());
  return output_shape;
}

}  // namespace internal

int FindOpcodeIndex(
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& opcodes,
    tflite::BuiltinOperator target_op, const std::string& custom_code) {
  for (int i = 0; i < opcodes.size(); ++i) {
    auto builtin_code = tflite::GetBuiltinCode(opcodes[i].get());
    if (builtin_code == target_op &&
        (builtin_code != tflite::BuiltinOperator_CUSTOM ||
         opcodes[i]->custom_code == custom_code)) {
      return i;
    }
  }
  return -1;
}

std::vector<int> FindOperatorsWithInput(tflite::BuiltinOperator target_op,
                                        int input_tensor_index,
                                        const tflite::ModelT* model_t,
                                        int base_op_index) {
  CHECK(model_t);
  CHECK_GE(base_op_index, 0);
  const auto& ops = model_t->subgraphs[0]->operators;
  const auto& opcodes = model_t->operator_codes;

  std::vector<int> operator_indices;
  for (int op_index = base_op_index; op_index < ops.size(); ++op_index) {
    const auto& op = ops[op_index];
    auto builtin_code = tflite::GetBuiltinCode(opcodes[op->opcode_index].get());
    if (std::find(op->inputs.begin(), op->inputs.end(), input_tensor_index) !=
            op->inputs.end() &&
        builtin_code == target_op) {
      operator_indices.push_back(op_index);
    }
  }
  return operator_indices;
}

int FindSingleOperatorWithInput(tflite::BuiltinOperator target_op,
                                int input_tensor_index,
                                const tflite::ModelT* model_t,
                                int base_op_index) {
  const auto& op_indices = FindOperatorsWithInput(target_op, input_tensor_index,
                                                  model_t, base_op_index);
  return op_indices.size() == 1 ? op_indices[0] : -1;
}

std::vector<int> FindOperators(tflite::BuiltinOperator target_op,
                               const tflite::ModelT* model_t) {
  CHECK(model_t);
  const auto& ops = model_t->subgraphs[0]->operators;
  const auto& opcodes = model_t->operator_codes;
  std::vector<int> operator_indices;
  for (int i = 0; i < ops.size(); ++i) {
    if (tflite::GetBuiltinCode(opcodes[ops[i]->opcode_index].get()) ==
        target_op) {
      operator_indices.push_back(i);
    }
  }
  return operator_indices;
}

int FindSingleOperator(tflite::BuiltinOperator target_op,
                       const tflite::ModelT* model_t) {
  const auto& op_indices = FindOperators(target_op, model_t);
  return op_indices.size() == 1 ? op_indices[0] : -1;
}

std::unique_ptr<tflite::QuantizationParametersT> CreateQuantParam(
    const std::vector<float>& min, const std::vector<float>& max,
    const std::vector<float>& scale, const std::vector<int64_t>& zero_point) {
  return absl::WrapUnique(NewQuantParam(min, max, scale, zero_point));
}

std::vector<tflite::TensorT*> GetGraphOutputTensors(
    const tflite::ModelT* model_t) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);
  const auto& subgraph = model_t->subgraphs[0];

  std::vector<tflite::TensorT*> result;
  result.reserve(subgraph->outputs.size());
  for (auto tensor_index : subgraph->outputs)
    result.push_back(subgraph->tensors[tensor_index].get());
  return result;
}

absl::Status AppendFullyConnectedAndSoftmaxLayerToModel(
    const tflite::Model& model, flatbuffers::FlatBufferBuilder* fbb,
    absl::Span<const float> weights, absl::Span<const float> biases,
    float out_tensor_min, float out_tensor_max) {
  auto model_t = absl::WrapUnique(model.UnPack());

  auto status = ValidateClassificationModel(model_t.get());
  if (!status.ok()) return status;

  // Get last tensor of input model.
  auto* embedding_output_tensor = GetGraphOutputTensors(model_t.get())[0];
  const auto& output_tensor_shape = embedding_output_tensor->shape;

  auto embedding_vector_dim = output_tensor_shape.back();

  // Quantize weights and biases.
  auto weights_q = WeightsQuantParam(weights);
  std::vector<uint8_t> weights_quant(weights.size());
  Quantize(weights.begin(), weights.end(), weights_quant.begin(),
           weights_q->scale[0], weights_q->zero_point[0]);

  auto biases_q = BiasesQuantParam(
      biases, *embedding_output_tensor->quantization, weights_q->scale[0]);
  std::vector<int32_t> biases_quant(biases.size());
  Quantize(biases.begin(), biases.end(), biases_quant.begin(),
           biases_q->scale[0], biases_q->zero_point[0]);
  // Append operators.
  auto output_q = OutputTensorQuantParam(out_tensor_min, out_tensor_max);
  const int fc_op_index = internal::AppendLinearLayer(
      /*kernel_shape=*/{static_cast<int>(weights.size()) / embedding_vector_dim,
                        embedding_vector_dim},
      std::move(weights_q), std::move(biases_q), std::move(output_q),
      model_t.get());
  if (output_tensor_shape.size() == 4) internal::AppendReshape(model_t.get());
  internal::AppendSoftmax(model_t.get());

  // Fill weights.
  internal::SetLinearParams(weights_quant, biases_quant, fc_op_index,
                            model_t.get());

  // Convert from tflite::ModelT format to FlatBufferBuilder.
  tflite::FinishModelBuffer(*fbb, tflite::Model::Pack(*fbb, model_t.get()));
  return absl::OkStatus();
}
}  // namespace coral
