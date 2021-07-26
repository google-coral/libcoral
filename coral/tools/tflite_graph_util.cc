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

#include "coral/tools/tflite_graph_util.h"

#include <fstream>
#include <map>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "coral/learn/utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace coral {
namespace {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

// Appends clones of model buffers to vector.
void CloneBuffers(const tflite::Model& model,
                  std::vector<Offset<tflite::Buffer>>* buffer_vector,
                  FlatBufferBuilder* builder) {
  CHECK(buffer_vector);
  CHECK(builder);
  for (int i = 0; i < model.buffers()->size(); ++i) {
    auto* buffer = model.buffers()->Get(i);
    if (buffer->data() == nullptr) {
      // Many transient tensors don't have data in the flatbuffer. Their
      // buffers will be allocated by the interpreter at run-time.
      VLOG(1) << "Buffer " << i << " is empty.";
      buffer_vector->push_back(tflite::CreateBuffer(*builder));
    } else {
      tflite::BufferT buffer_t;
      buffer->UnPackTo(&buffer_t);
      VLOG(1) << "Buffer " << i << " size in bytes: " << buffer_t.data.size();
      Offset<Vector<uint8_t>> data_buffer =
          builder->CreateVector(buffer_t.data.data(), buffer_t.data.size());
      buffer_vector->push_back(tflite::CreateBuffer(*builder, data_buffer));
    }
  }
}

// Appends clones of model tensors to vector, assuming the model has only one
// subgraph.
// |tensor_buffer_start_offset| specifies the offset to be added to the buffer
// index of all tensors of this model.
// |tensor_name_to_buffer_index_map| is the tensor to buffer index map, and
// |tensor_name_to_tensor_index_map| is the tensor name to tensor index map.
// Both maps will be updated by this function.
void CloneTensors(
    const tflite::Model& model, uint32_t tensor_buffer_start_offset,
    std::vector<Offset<tflite::Tensor>>* tensor_vector,
    FlatBufferBuilder* builder,
    std::map<std::string, uint32_t>* tensor_name_to_buffer_index_map,
    std::map<std::string, int32_t>* tensor_name_to_tensor_index_map) {
  CHECK(tensor_vector);
  CHECK(builder);
  CHECK(tensor_name_to_buffer_index_map);
  CHECK(tensor_name_to_tensor_index_map);

  const auto* subgraphs = model.subgraphs();
  const auto* tensors = subgraphs->Get(0)->tensors();
  for (int i = 0; i < tensors->size(); ++i) {
    const auto* tensor = tensors->Get(i);
    CHECK(tensor);
    tflite::TensorT tensor_t;
    tensor->UnPackTo(&tensor_t);
    if (tensor_name_to_buffer_index_map->count(tensor_t.name) > 0) {
      VLOG(1) << "Tensor " << tensor_t.name << " already exists.";
      continue;
    }

    const auto* q_param = tensor->quantization();
    const auto& q_param_t = tensor_t.quantization;

    // Do not support quantization details for now.
    CHECK_EQ(q_param->details_type(), tflite::QuantizationDetails_NONE);
    auto new_q_param =
        q_param == nullptr
            ? 0
            : tflite::CreateQuantizationParameters(
                  *builder,
                  q_param->min() ? builder->CreateVector(q_param_t->min) : 0,
                  q_param->max() ? builder->CreateVector(q_param_t->max) : 0,
                  q_param->scale() ? builder->CreateVector(q_param_t->scale)
                                   : 0,
                  q_param->zero_point()
                      ? builder->CreateVector(q_param_t->zero_point)
                      : 0,
                  q_param->details_type(), 0, q_param->quantized_dimension());

    // Update tensor name to buffer index map. Note that buffer index must be
    // recalcualted.
    const uint32_t buffer_index = tensor_buffer_start_offset + tensor_t.buffer;
    (*tensor_name_to_buffer_index_map)[tensor_t.name] = buffer_index;
    VLOG(1) << "Tensor " << tensor_vector->size() << " name: " << tensor_t.name
            << ", buffer index from " << tensor_t.buffer << " to "
            << buffer_index;

    // Update tensor name to tensor index map.
    CHECK_EQ(tensor_name_to_tensor_index_map->count(tensor_t.name), 0);
    (*tensor_name_to_tensor_index_map)[tensor_t.name] = tensor_vector->size();

    tensor_vector->push_back(tflite::CreateTensor(
        *builder, builder->CreateVector(tensor_t.shape), tensor_t.type,
        /*buffer=*/buffer_index, builder->CreateString(tensor_t.name),
        new_q_param));
  }
}

// Recalcuates tensor indices given a new tensor name to tensor index map.
std::vector<int32_t> RecalculateTensorIndices(
    const std::vector<int32_t>& old_tensor_indices, const tflite::Model& model,
    const std::map<std::string, int32_t>& new_tensor_name_to_tensor_index_map) {
  const auto* subgraphs = model.subgraphs();
  const auto* old_tensor_vector = subgraphs->Get(0)->tensors();
  std::vector<int32_t> result;
  result.reserve(old_tensor_indices.size());
  for (uint32_t i : old_tensor_indices) {
    const auto* tensor = old_tensor_vector->Get(i);
    const std::string tensor_name = tensor->name()->str();
    CHECK_GT(new_tensor_name_to_tensor_index_map.count(tensor_name), 0);
    const uint32_t new_index =
        new_tensor_name_to_tensor_index_map.at(tensor_name);
    VLOG(1) << "Change tensor " << tensor_name << " index from " << i << " to "
            << new_index;
    result.push_back(new_index);
  }
  return result;
}

// Appends clones of model operator codes to vector. No deduping.
void CloneOperatorCodes(
    const tflite::Model& model,
    std::vector<Offset<tflite::OperatorCode>>* opcode_vector,
    FlatBufferBuilder* builder) {
  CHECK(opcode_vector);
  CHECK(builder);
  for (int i = 0; i < model.operator_codes()->size(); ++i) {
    const auto* opcode = model.operator_codes()->Get(i);
    tflite::OperatorCodeT opcode_t;
    opcode->UnPackTo(&opcode_t);
    opcode_vector->push_back(tflite::CreateOperatorCode(
        *builder, tflite::GetBuiltinCode(&opcode_t),
        opcode->custom_code() ? builder->CreateString(opcode_t.custom_code) : 0,
        opcode_t.version));
  }
}

// Appends clones of model operators to vector.
// |opcode_index_start_offset| specifies the offset to be added to the opcode
// index of all operators.
void CloneOperators(
    const tflite::Model& model, uint32_t opcode_index_start_offset,
    const std::map<std::string, int32_t>& tensor_name_to_tensor_index_map,
    std::vector<Offset<tflite::Operator>>* op_vector,
    FlatBufferBuilder* builder) {
  CHECK(op_vector);
  CHECK(builder);
  const auto* ops = model.subgraphs()->Get(0)->operators();
  for (int i = 0; i < ops->size(); ++i) {
    const auto* op = ops->Get(i);
    CHECK(op->inputs());
    CHECK(op->outputs());

    tflite::OperatorT op_t;
    op->UnPackTo(&op_t);
    uint32_t new_opcode_index = op_t.opcode_index + opcode_index_start_offset;
    VLOG(1) << "Change operator " << i << " opcode index from "
            << op_t.opcode_index << " to " << new_opcode_index;

    // Recalculate input and output indices of this operator.
    Offset<Vector<int32_t>> new_input_index_vector =
        builder->CreateVector<int32_t>(RecalculateTensorIndices(
            op_t.inputs, model, tensor_name_to_tensor_index_map));
    Offset<Vector<int32_t>> new_output_index_vector =
        builder->CreateVector<int32_t>(RecalculateTensorIndices(
            op_t.outputs, model, tensor_name_to_tensor_index_map));

    const auto builtin_options_type = op_t.builtin_options.type;

    const auto custom_options_format = op_t.custom_options_format;

    op_vector->push_back(tflite::CreateOperator(
        *builder, new_opcode_index, new_input_index_vector,
        new_output_index_vector, builtin_options_type,
        op->builtin_options() ? op_t.builtin_options.Pack(*builder) : 0,
        op->custom_options() ? builder->CreateVector(op_t.custom_options.data(),
                                                     op_t.custom_options.size())
                             : 0,
        custom_options_format));
  }
}

// Returns index of a tensor specified by name. If non-found, return -1;
int FindOutputTensor(absl::string_view name,
                     const tflite::SubGraphT& subgraph_t) {
  for (const auto& i : subgraph_t.outputs) {
    if (subgraph_t.tensors[i]->name == name) {
      return i;
    }
  }
  return -1;
}

absl::Status CreateMutableModelFromFile(const absl::string_view model_filepath,
                                        tflite::ModelT& model) {
  auto fb_model = tflite::FlatBufferModel::BuildFromFile(model_filepath.data());
  if (!fb_model)
    return absl::InvalidArgumentError(
        absl::StrFormat("Error reading tflite model from %s.", model_filepath));
  auto* tflite_model = fb_model->GetModel();
  tflite_model->UnPackTo(&model, /*_resolver=*/nullptr);
  return absl::OkStatus();
}

absl::Status WriteFile(const absl::string_view out_file, const uint8_t* bytes,
                       size_t num_bytes) {
  std::fstream stream(out_file.data(), std::ios::binary | std::ios::out);
  for (size_t i = 0; i < num_bytes; i++) {
    stream << bytes[i];
  }
  if (stream.bad() || stream.fail())
    return absl::InternalError(
        absl::StrFormat("Fail writing output model to %s.", out_file));
  return absl::OkStatus();
}

std::vector<tflite::TensorT*> BuildTensorsByIndexes(
    const tflite::ModelT& model, const std::vector<int>& indexes) {
  const auto& tensors = model.subgraphs[0]->tensors;
  std::vector<tflite::TensorT*> selected_tensors(indexes.size());
  for (int i = 0; i < indexes.size(); ++i) {
    selected_tensors[i] = tensors[indexes[i]].get();
  }
  return selected_tensors;
}

// Returns the index of the tensor with `tensor_name` in the given list of
// tensors.
absl::Status FindTensorByname(const std::vector<tflite::TensorT*>& tensors,
                              const absl::string_view tensor_name, int& index) {
  auto iter =
      absl::c_find_if(tensors, [&tensor_name](const tflite::TensorT* tensor) {
        return tensor->name == tensor_name;
      });
  if (iter != tensors.end()) {
    index = iter - tensors.begin();
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s is not found in the tensors list.", tensor_name));
  }
}

absl::Status FindTensorByname(
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors,
    const absl::string_view tensor_name, int& index) {
  auto iter = absl::c_find_if(
      tensors, [&tensor_name](const std::unique_ptr<tflite::TensorT>& tensor) {
        return tensor->name == tensor_name;
      });
  if (iter != tensors.end()) {
    index = iter - tensors.begin();
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s is not found in the tensors list.", tensor_name));
  }
}

// Points the operators which have the `output_tensor` as one of the
// outputs/inputs to the `input_tensor`. Returns an error if none of the ops has
// the `output_tensor` as an output.
absl::Status RedirectTensor(int output_tensor_index, int input_tensor_index,
                            tflite::ModelT& model) {
  const auto& ops = model.subgraphs[0]->operators;
  const auto& tensors = model.subgraphs[0]->tensors;
  CHECK_LT(output_tensor_index, tensors.size());
  CHECK_LT(input_tensor_index, tensors.size());
  bool found_as_output = false;
  for (const auto& op : ops) {
    // Points the operator which has the `output_tensor` as output to the
    // `input_tensor`.
    auto output_iter =
        absl::c_find_if(op->outputs, [&output_tensor_index](int tensor_index) {
          return tensor_index == output_tensor_index;
        });
    if (output_iter != op->outputs.end()) {
      *output_iter = input_tensor_index;
      found_as_output = true;
    }
    // Points the operator which has the `output_tensor` as input to the
    // `input_tensor`.
    auto input_iter =
        absl::c_find_if(op->inputs, [&output_tensor_index](int tensor_index) {
          return tensor_index == output_tensor_index;
        });
    if (input_iter != op->inputs.end()) {
      *input_iter = input_tensor_index;
    }
  }
  if (found_as_output) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrFormat(
      "output_tensor_index %d is not found as an output of any operator.",
      output_tensor_index));
}

absl::Status AppendRecurrentLinks(
    const std::vector<std::string>& input_tensor_names,
    const std::vector<std::string>& output_tensor_names, tflite::ModelT& model,
    flatbuffers::FlatBufferBuilder& builder) {
  if (input_tensor_names.size() != output_tensor_names.size())
    return absl::InvalidArgumentError(
        absl::StrFormat("Sizes of input_tensor_names and output_tensor_names "
                        "mismatch, %d vs %d.",
                        input_tensor_names.size(), output_tensor_names.size()));

  auto* subgraph = model.subgraphs[0].get();
  const auto& tensors = model.subgraphs[0]->tensors;
  const std::vector<tflite::TensorT*> input_tensors =
      BuildTensorsByIndexes(model, subgraph->inputs);
  const std::vector<tflite::TensorT*> output_tensors =
      BuildTensorsByIndexes(model, subgraph->outputs);

  std::vector<int> input_indexes(input_tensor_names.size());
  std::vector<int> output_indexes(output_tensor_names.size());
  // For each pair of input_tensor and output_tensor, find the input index of
  // the input_tensor and the output index of the output_tensor, then redirect
  // the operator which used to have output_tensor as an output to the
  // input_tensor.
  for (int i = 0; i < input_tensor_names.size(); ++i) {
    auto status = FindTensorByname(input_tensors, input_tensor_names[i],
                                   input_indexes[i]);
    if (!status.ok()) return status;
    int input_tensor_index = subgraph->inputs[input_indexes[i]];

    status = FindTensorByname(output_tensors, output_tensor_names[i],
                              output_indexes[i]);
    if (!status.ok()) return status;
    int output_tensor_index = subgraph->outputs[output_indexes[i]];

    if (tensors[input_tensor_index]->type != tensors[output_tensor_index]->type)
      return absl::InvalidArgumentError(absl::StrFormat(
          "input_tensor %s and output_tensor %s have different data types.",
          input_tensor_names[i], output_tensor_names[i]));

    // Check the shapes of the two tensors.
    std::string input_shape_str =
        absl::StrJoin(tensors[input_tensor_index]->shape, " ");
    std::string output_shape_str =
        absl::StrJoin(tensors[output_tensor_index]->shape, " ");
    if (input_shape_str != output_shape_str)
      return absl::InvalidArgumentError(
          absl::StrFormat("input_tensor %s and output tensor %s have different "
                          "tensor shapes.",
                          input_tensor_names[i], output_tensor_names[i]));

    status = RedirectTensor(output_tensor_index, input_tensor_index, model);
    if (!status.ok()) return status;

    // Mark the tensor as Variable.
    tensors[input_tensor_index]->is_variable = true;
  }

  // Remove the tensors from subgraph inputs and outputs.
  absl::c_sort(input_indexes, std::greater<int>());
  for (const int input_index : input_indexes) {
    subgraph->inputs.erase(subgraph->inputs.begin() + input_index);
  }
  absl::c_sort(output_indexes, std::greater<int>());
  for (const int output_index : output_indexes) {
    subgraph->outputs.erase(subgraph->outputs.begin() + output_index);
  }

  // Write to builder.
  flatbuffers::Offset<tflite::Model> output_model_location =
      tflite::Model::Pack(builder, &model);
  FinishModelBuffer(builder, output_model_location);
  return absl::OkStatus();
}

// Returns tensor size in bytes required by its shape and type.
inline int GetRequiredTensorSizeBytes(const tflite::TensorT& tensor_t) {
  const int num_elements = std::accumulate(
      tensor_t.shape.begin(), tensor_t.shape.end(), 1, std::multiplies<int>());
  switch (tensor_t.type) {
    case tflite::TensorType::TensorType_FLOAT32:
    case tflite::TensorType::TensorType_INT32:
    case tflite::TensorType::TensorType_UINT32:
      return 4 * num_elements;
    case tflite::TensorType::TensorType_UINT8:
    case tflite::TensorType::TensorType_INT8:
      return num_elements;
    default:
      LOG(FATAL) << "Unsupported TensorType: " << tensor_t.type;
  }
  return 0;
}

// Creates underlying buffer of a tensor based on its shape and type.
// If `src_data` is not empty, it will populate the new tensor with data
// of `src_data`, otherwise it will leave the buffer uninitialized.
inline std::unique_ptr<tflite::BufferT> CreateTensorBuffer(
    tflite::TensorT& tensor, absl::Span<const uint8_t> src_data,
    int buffer_index) {
  tensor.buffer = buffer_index;
  auto buffer = absl::make_unique<tflite::BufferT>();
  if (!src_data.empty()) {
    const int buffer_size = GetRequiredTensorSizeBytes(tensor);
    VLOG(1) << "New buffer size " << buffer_size;
    buffer->data.resize(buffer_size);
    CHECK_LE(buffer_size, src_data.size());
    std::memcpy(buffer->data.data(), src_data.data(), buffer_size);
  }
  return buffer;
}

// Appends a new tensor, which is a part being split from an existing tensor, to
// the model. The new tensor should have the same properties as the original
// tensor except for name and shape.
// Returns the size of the buffer of the new tensor.
int AppendSplitTensorToModel(const tflite::TensorT& original_tensor,
                             const std::string& new_name,
                             const std::vector<int32_t>& new_shape,
                             absl::Span<const uint8_t> src_data,
                             tflite::ModelT& model) {
  VLOG(1) << "split tensor " << new_name;
  auto split_tensor = absl::make_unique<tflite::TensorT>();
  split_tensor->name = new_name;
  split_tensor->shape = new_shape;
  split_tensor->type = original_tensor.type;
  split_tensor->quantization =
      absl::make_unique<tflite::QuantizationParametersT>();
  *split_tensor->quantization = *original_tensor.quantization;
  auto split_tensor_buffer =
      CreateTensorBuffer(*split_tensor, src_data, model.buffers.size());
  const int new_buffer_size = split_tensor_buffer->data.size();
  model.buffers.push_back(std::move(split_tensor_buffer));
  model.subgraphs[0]->tensors.push_back(std::move(split_tensor));
  return new_buffer_size;
}

void SplitFullyConnected(int input_tensor_index, int weights_tensor_index,
                         int bias_tensor_index, int output_tensor_index,
                         int op_index, int feature_dim_index, float split_ratio,
                         tflite::ModelT& model,
                         flatbuffers::FlatBufferBuilder& builder) {
  auto& ops = model.subgraphs[0]->operators;
  const auto& tensors = model.subgraphs[0]->tensors;
  const auto& op_codes = model.operator_codes;
  const int fc_opcode_index = ops[op_index]->opcode_index;
  const auto opcode = tflite::GetBuiltinCode(op_codes[fc_opcode_index].get());
  bool is_conv1x1 = false;
  if (opcode != tflite::BuiltinOperator_FULLY_CONNECTED) {
    CHECK_EQ(opcode, tflite::BuiltinOperator_CONV_2D);
    is_conv1x1 = true;
  }

  const auto weights_tensor = tensors[weights_tensor_index].get();
  const auto output_tensor = tensors[output_tensor_index].get();
  const auto bias_tensor =
      bias_tensor_index >= 0 ? tensors[bias_tensor_index].get() : nullptr;

  const int original_out_feature_dim = output_tensor->shape[feature_dim_index];
  CHECK_EQ(original_out_feature_dim, weights_tensor->shape[0]);
  if (bias_tensor) CHECK_EQ(original_out_feature_dim, bias_tensor->shape[0]);

  const int split_fc1_out_feature_dim =
      static_cast<int>(original_out_feature_dim * split_ratio + 0.5);
  CHECK_GT(split_fc1_out_feature_dim, 0);
  const int split_fc2_out_feature_dim =
      original_out_feature_dim - split_fc1_out_feature_dim;
  CHECK_GT(split_fc2_out_feature_dim, 0);
  const std::vector<int> channels(
      {split_fc1_out_feature_dim, split_fc2_out_feature_dim});
  std::vector<int> split_output_tensors(channels.size());

  int weight_buffer_data_offset = 0;
  int bias_buffer_data_offset = 0;

  for (int i = 0; i < channels.size(); ++i) {
    // Split FC weights tensor.
    std::vector<int32_t> new_shape = weights_tensor->shape;
    new_shape[0] = channels[i];
    const int split_weight_buffer_size = AppendSplitTensorToModel(
        *weights_tensor, weights_tensor->name + "/fc_" + std::to_string(i),
        new_shape,
        absl::MakeSpan(model.buffers[weights_tensor->buffer]->data)
            .subspan(weight_buffer_data_offset),
        model);
    weight_buffer_data_offset += split_weight_buffer_size;
    const int split_weights_tensor_index =
        model.subgraphs[0]->tensors.size() - 1;
    VLOG(1) << "New FC weight tensor appened.";

    // Split FC bias tensor if available.
    int split_bias_tensor_index = -1;
    if (bias_tensor) {
      new_shape = bias_tensor->shape;
      new_shape[0] = channels[i];
      const int split_bias_buffer_size = AppendSplitTensorToModel(
          *bias_tensor, bias_tensor->name + "/fc_" + std::to_string(i),
          new_shape,
          absl::MakeSpan(model.buffers[bias_tensor->buffer]->data)
              .subspan(bias_buffer_data_offset),
          model);
      bias_buffer_data_offset += split_bias_buffer_size;
      split_bias_tensor_index = model.subgraphs[0]->tensors.size() - 1;
      VLOG(1) << "New FC bias tensor appened.";
    }

    // Split FC output tensor.
    new_shape = output_tensor->shape;
    new_shape[feature_dim_index] = channels[i];
    AppendSplitTensorToModel(*output_tensor,
                             output_tensor->name + "/fc_" + std::to_string(i),
                             new_shape, {}, model);
    const int split_output_tensor_index =
        model.subgraphs[0]->tensors.size() - 1;
    VLOG(1) << "New FC output tensor appened.";

    // Add a new FC op.
    auto split_op = absl::make_unique<tflite::OperatorT>();
    split_op->opcode_index = fc_opcode_index;
    split_op->inputs = {input_tensor_index, split_weights_tensor_index};
    if (split_bias_tensor_index >= 0)
      split_op->inputs.push_back(split_bias_tensor_index);
    split_op->outputs = {split_output_tensor_index};
    if (is_conv1x1) {
      split_op->builtin_options.type = tflite::BuiltinOptions_Conv2DOptions;
      tflite::Conv2DOptionsT* options = new tflite::Conv2DOptionsT;
      *options = *reinterpret_cast<tflite::Conv2DOptionsT*>(
          ops[op_index]->builtin_options.value);
      split_op->builtin_options.value = options;
    } else {
      split_op->builtin_options.type =
          tflite::BuiltinOptions_FullyConnectedOptions;
      tflite::FullyConnectedOptionsT* options =
          new tflite::FullyConnectedOptionsT;
      *options = *reinterpret_cast<tflite::FullyConnectedOptionsT*>(
          ops[op_index]->builtin_options.value);
      split_op->builtin_options.value = options;
    }
    ops.insert(ops.begin() + op_index + i + 1, std::move(split_op));
    split_output_tensors[i] = split_output_tensor_index;
  }

  // Add concat op.
  auto concat_op_code = absl::make_unique<tflite::OperatorCodeT>();
  concat_op_code->deprecated_builtin_code =
      tflite::ConvertBuiltinCodeToDeprecatedBuiltinCode(
          tflite::BuiltinOperator_CONCATENATION);
  concat_op_code->builtin_code = tflite::BuiltinOperator_CONCATENATION;
  model.operator_codes.push_back(std::move(concat_op_code));
  VLOG(1) << "Add concat op code to operator_codes.";
  auto concat_op = absl::make_unique<tflite::OperatorT>();
  concat_op->opcode_index = model.operator_codes.size() - 1;
  concat_op->inputs = split_output_tensors;
  concat_op->outputs = {output_tensor_index};
  tflite::ConcatenationOptionsT* concat_options =
      new tflite::ConcatenationOptionsT;
  concat_options->axis = -1;
  concat_op->builtin_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  concat_op->builtin_options.value = new tflite::ConcatenationOptionsT;
  reinterpret_cast<tflite::ConcatenationOptionsT*>(
      concat_op->builtin_options.value)
      ->axis = -1;
  ops.insert(ops.begin() + op_index + 1 + channels.size(),
             std::move(concat_op));
  VLOG(1) << "Add concat op.";

  // Delete the original fc op.
  ops.erase(ops.begin() + op_index);
}

absl::Status SplitFullyConnected(const std::string& fc_input_tensor_name,
                                 const std::string& fc_weight_tensor_name,
                                 const std::string& fc_bias_tensor_name,
                                 const std::string& fc_output_tensor_name,
                                 int feature_dim_index, float split_ratio,
                                 tflite::ModelT& model,
                                 flatbuffers::FlatBufferBuilder& builder) {
  const auto& ops = model.subgraphs[0]->operators;
  const auto& tensors = model.subgraphs[0]->tensors;

  int fc_input_tensor_index = -1;
  CHECK(FindTensorByname(tensors, fc_input_tensor_name, fc_input_tensor_index)
            .ok());
  int fc_op_index = FindSingleOperatorWithInput(
      tflite::BuiltinOperator_FULLY_CONNECTED, fc_input_tensor_index, &model,
      /*base_op_index=*/0);
  if (fc_op_index < 0) {
    fc_op_index = FindSingleOperatorWithInput(tflite::BuiltinOperator_CONV_2D,
                                              fc_input_tensor_index, &model,
                                              /*base_op_index=*/0);
  }
  CHECK_GE(fc_op_index, 0);
  const auto fc_op = ops[fc_op_index].get();

  // Search for weight, and bias tensor index. Note that tflite models
  // may have different order of the three tensors.
  int fc_weights_tensor_index = -1;
  CHECK(
      FindTensorByname(tensors, fc_weight_tensor_name, fc_weights_tensor_index)
          .ok());
  CHECK(std::find(fc_op->inputs.begin(), fc_op->inputs.end(),
                  fc_weights_tensor_index) != fc_op->inputs.end());
  int fc_bias_tensor_index = -1;
  if (!fc_bias_tensor_name.empty()) {
    CHECK(FindTensorByname(tensors, fc_bias_tensor_name, fc_bias_tensor_index)
              .ok());
    CHECK(std::find(fc_op->inputs.begin(), fc_op->inputs.end(),
                    fc_bias_tensor_index) != fc_op->inputs.end());
  }
  int fc_output_tensor_index = fc_op->outputs[0];
  SplitFullyConnected(fc_input_tensor_index, fc_weights_tensor_index,
                      fc_bias_tensor_index, fc_output_tensor_index, fc_op_index,
                      feature_dim_index, split_ratio, model, builder);

  // Write to builder.
  flatbuffers::Offset<tflite::Model> output_model_location =
      tflite::Model::Pack(builder, &model);
  FinishModelBuffer(builder, output_model_location);
  return absl::OkStatus();
}
}  // namespace

// Concanetate two tflite models, assuming each model has only one subgraph.
// |builder| contains the result model.
void ConcatModels(const tflite::Model& model0, const tflite::Model& model1,
                  flatbuffers::FlatBufferBuilder* builder,
                  const std::vector<std::string>& bypass_output_tensors) {
  CHECK(builder);

  CHECK(model0.subgraphs());
  CHECK_EQ(model0.subgraphs()->size(), 1);
  CHECK(model1.subgraphs());
  CHECK_EQ(model1.subgraphs()->size(), 1);

  // Merge all buffers.
  const int num_model0_buffers = model0.buffers()->size();
  const int num_model1_buffers = model1.buffers()->size();
  VLOG(1) << "model0 # buffers: " << num_model0_buffers
          << ", model1 # buffers: " << num_model1_buffers;
  std::vector<Offset<tflite::Buffer>> buffer_vector;
  CloneBuffers(model0, &buffer_vector, builder);
  CloneBuffers(model1, &buffer_vector, builder);
  VLOG(1) << "merged # buffers: " << buffer_vector.size();

  // Merge all tensors.
  const tflite::SubGraph& subgraph0 = *(*model0.subgraphs())[0];
  const tflite::SubGraph& subgraph1 = *(*model1.subgraphs())[0];
  const int num_model0_tensors = subgraph0.tensors()->size();
  const int num_model1_tensors = subgraph1.tensors()->size();
  VLOG(1) << "model0 # tensors: " << num_model0_tensors
          << ", model1 # tensors: " << num_model1_tensors;
  std::vector<Offset<tflite::Tensor>> tensor_vector;
  std::map<std::string, uint32_t> tensor_name_to_buffer_index_map;
  std::map<std::string, int32_t> tensor_name_to_tensor_index_map;
  CloneTensors(model0, /*tensor_buffer_start_offset=*/0, &tensor_vector,
               builder, &tensor_name_to_buffer_index_map,
               &tensor_name_to_tensor_index_map);
  CloneTensors(model1, /*tensor_buffer_start_offset=*/num_model0_buffers,
               &tensor_vector, builder, &tensor_name_to_buffer_index_map,
               &tensor_name_to_tensor_index_map);
  VLOG(1) << "merged # tensors: " << tensor_vector.size();
  CHECK_EQ(tensor_name_to_buffer_index_map.size(), tensor_vector.size());

  // Create vectors of input and output tensors indices.
  tflite::SubGraphT subgraph0_t, subgraph1_t;
  subgraph0.UnPackTo(&subgraph0_t);
  subgraph1.UnPackTo(&subgraph1_t);
  std::vector<int32_t> inputs = RecalculateTensorIndices(
      subgraph0_t.inputs, model0, tensor_name_to_tensor_index_map);
  std::vector<int32_t> outputs = RecalculateTensorIndices(
      subgraph1_t.outputs, model1, tensor_name_to_tensor_index_map);

  std::vector<int32_t> bypasses;
  for (const auto& bypass_name : bypass_output_tensors) {
    auto index = FindOutputTensor(bypass_name, subgraph0_t);
    CHECK_GE(index, 0) << "Unable to find bypass output tensor " << bypass_name;
    bypasses.push_back(index);
  }
  std::vector<int32_t> bypasses_recalc = RecalculateTensorIndices(
      bypasses, model0, tensor_name_to_tensor_index_map);
  // Add the bypass indices to the output list.
  for (const auto o : bypasses_recalc) {
    outputs.push_back(o);
  }

  // Merge operator codes.
  const int num_model0_opcodes = model0.operator_codes()->size();
  const int num_model1_opcodes = model1.operator_codes()->size();
  VLOG(1) << "model0 # opcodes: " << num_model0_opcodes
          << ", model1 # opcodes: " << num_model1_opcodes;
  std::vector<Offset<tflite::OperatorCode>> opcode_vector;
  CloneOperatorCodes(model0, &opcode_vector, builder);
  CloneOperatorCodes(model1, &opcode_vector, builder);
  CHECK_EQ(num_model0_opcodes + num_model1_opcodes, opcode_vector.size());

  // Merge operators.
  const int num_model0_ops = subgraph0.operators()->size();
  const int num_model1_ops = subgraph1.operators()->size();
  VLOG(1) << "model0 # ops: " << num_model0_ops
          << ", model1 # ops: " << num_model1_ops;
  std::vector<Offset<tflite::Operator>> op_vector;
  CloneOperators(model0, /*opcode_index_start_offset=*/0,
                 tensor_name_to_tensor_index_map, &op_vector, builder);
  CloneOperators(model1, /*opcode_index_start_offset=*/num_model0_opcodes,
                 tensor_name_to_tensor_index_map, &op_vector, builder);
  CHECK_EQ(num_model0_ops + num_model1_ops, op_vector.size());

  Offset<Vector<int32_t>> merged_inputs =
      builder->CreateVector<int32_t>(inputs);
  Offset<Vector<int32_t>> merged_outputs =
      builder->CreateVector<int32_t>(outputs);
  Offset<Vector<Offset<tflite::Tensor>>> merged_tensors =
      builder->CreateVector(tensor_vector);
  Offset<Vector<Offset<tflite::Operator>>> merged_ops =
      builder->CreateVector(op_vector);
  Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      *builder, merged_tensors, merged_inputs, merged_outputs, merged_ops,
      (subgraph0.name() ? builder->CreateString(subgraph0.name()->str()) : 0));
  Offset<Vector<Offset<tflite::SubGraph>>> merged_subgraphs =
      builder->CreateVector<Offset<tflite::SubGraph>>({subgraph});

  Offset<Vector<Offset<tflite::Buffer>>> merged_buffers =
      builder->CreateVector(buffer_vector);
  Offset<Vector<Offset<tflite::OperatorCode>>> merged_opcodes =
      builder->CreateVector(opcode_vector);
  auto merged_model = tflite::CreateModel(
      *builder, model0.version(), merged_opcodes, merged_subgraphs,
      (model0.description() ? builder->CreateString(model0.description()->str())
                            : 0),
      merged_buffers);

  tflite::FinishModelBuffer(*builder, merged_model);
}

absl::Status AppendRecurrentLinks(
    const absl::string_view input_path,
    const std::vector<std::string>& input_tensor_names,
    const std::vector<std::string>& output_tensor_names,
    const absl::string_view output_path) {
  // Create model.
  auto model = absl::make_unique<tflite::ModelT>();
  auto status = CreateMutableModelFromFile(input_path, *model);
  if (!status.ok()) return status;
  flatbuffers::FlatBufferBuilder builder;
  status = AppendRecurrentLinks(input_tensor_names, output_tensor_names, *model,
                                builder);
  if (!status.ok()) return status;

  return WriteFile(output_path, builder.GetBufferPointer(), builder.GetSize());
}

absl::Status SplitFullyConnected(const std::string& input_model_path,
                                 const std::string& fc_input_tensor_name,
                                 const std::string& fc_weights_tensor_name,
                                 const std::string& fc_bias_tensor_name,
                                 const std::string& fc_output_tensor_name,
                                 const std::string& output_model_path,
                                 int feature_dim_index, float split_ratio) {
  // Create model.
  auto model = absl::make_unique<tflite::ModelT>();
  auto status = CreateMutableModelFromFile(input_model_path, *model);
  if (!status.ok()) return status;
  flatbuffers::FlatBufferBuilder builder;
  status = SplitFullyConnected(fc_input_tensor_name, fc_weights_tensor_name,
                               fc_bias_tensor_name, fc_output_tensor_name,
                               feature_dim_index, split_ratio, *model, builder);
  if (!status.ok()) return status;

  return WriteFile(output_model_path, builder.GetBufferPointer(),
                   builder.GetSize());
}

}  // namespace coral
