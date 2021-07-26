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

#include "coral/learn/imprinting_engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "coral/learn/utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace coral {
namespace {
// Model metadata, storing the numbers of training images for each class.
// Training metadata is a map from class label index to the weight of seen
// average embedding.
//
// To understand it better, e.g., we have two seen normalized embeddings `f_0`
// and `f_1`. Their average embedding is `normalized_f_01`. We can know
// sqrt(`f_0`+`f_1`) is `normalized_f_01` multiplied by some scalar weight
// `sqrt_sum_01`. With this `normalized_f_01` and this weight, we can resume
// (`f_0`+`f_1`), which is needed to calculate (`f_0`+`f_1`+`f_2)
// when we want to train a new sample with embedding `f_2`.
//
// This map is used for online learning and is stored in model description
// field with the following format:
// 0 5.4
// 1 6.5
// 2 4.1
//
// Converts string (e.g. "0 5.4\n1 6.5\n2 4.1") to training metadata map.
std::map<int, float> ParseTrainingMetadata(const std::string& description) {
  VLOG(1) << "Model description: " << description;
  if (description.empty()) return {};

  std::map<int, float> metadata;
  const std::vector<std::string> v =
      absl::StrSplit(description, absl::ByAnyChar(" \n"));
  for (int i = 0; i < v.size() - 1; i += 2) {
    int label;
    float sqrt_sum;
    if (absl::SimpleAtoi(v[i], &label) &&
        absl::SimpleAtof(v[i + 1], &sqrt_sum)) {
      metadata.insert({label, sqrt_sum});
    } else {
      return {};
    }
  }
  return metadata;
}

// Converts training metadata map to srting.
std::string SerializeTrainingMetadata(const std::map<int, float>& metadata) {
  std::string description;
  for (const auto& entry : metadata) {
    absl::StrAppend(&description,
                    absl::StrFormat("%d %f\n", entry.first, entry.second));
  }
  return description;
}

const std::array<tflite::BuiltinOperator, 4> kModelTail = {
    tflite::BuiltinOperator_CONV_2D,
    tflite::BuiltinOperator_MUL,
    tflite::BuiltinOperator_RESHAPE,
    tflite::BuiltinOperator_SOFTMAX,
};

std::vector<tflite::BuiltinOperator> OpCodes(const tflite::ModelT& model_t) {
  const auto& ops = model_t.subgraphs[0]->operators;
  std::vector<tflite::BuiltinOperator> result(ops.size());
  for (size_t i = 0; i < ops.size(); ++i)
    result[i] =
        GetBuiltinCode(model_t.operator_codes[ops[i]->opcode_index].get());
  return result;
}

absl::Status ValidateImprintingModel(const tflite::ModelT& model_t) {
  const auto opcodes = OpCodes(model_t);
  if (!absl::c_linear_search(opcodes, tflite::BuiltinOperator_L2_NORMALIZATION))
    return absl::InternalError(
        "Unsupported model architecture. Input model must have an L2Norm "
        "layer.");
  const int l2norm_op_index =
      FindSingleOperator(tflite::BuiltinOperator_L2_NORMALIZATION, &model_t);

  if (opcodes.size() < kModelTail.size() ||
      !std::equal(kModelTail.rbegin(), kModelTail.rend(), opcodes.rbegin()))
    return absl::InternalError(
        "The last 4 operators must be Conv2d, Mul, Reshape, Softmax");

  const auto& ops = model_t.subgraphs[0]->operators;
  const auto& tensors = model_t.subgraphs[0]->tensors;

  const int fc_op_index = ops.size() - 4;
  // The assumption for weight imprinting is a L2Norm operator followed by the
  // last 1x1 Conv operator. However, MLIR generated model has a Quantize
  // operator between L2Norm and Conv operators, which theoretically does not
  // affect the performance of imprinting.
  if (fc_op_index - l2norm_op_index > 2 ||
      (fc_op_index - l2norm_op_index == 2 &&
       opcodes[l2norm_op_index + 1] != tflite::BuiltinOperator_QUANTIZE)) {
    return absl::InternalError(
        "Unsupported model architecture. L2Norm operator must be followed with "
        "the last Conv2d operator.");
  }

  const auto& l2norm_op = ops[l2norm_op_index];

  const auto& embedding_output_tensor = tensors[l2norm_op->outputs[0]];
  if (!MatchShape(embedding_output_tensor->shape, {1, 1, 1, -1}))
    return absl::InternalError(
        "Embedding extractor's output tensor should be [1, 1, 1, x]");

  const auto& fc_op = ops[fc_op_index];
  const auto& bias_tensor = tensors[fc_op->inputs[2]];
  const auto bias_zero_point = bias_tensor->quantization->zero_point[0];
  if (bias_zero_point < 0 || bias_zero_point > 255)
    return absl::InternalError("bias_zero_point is out of [0, 255] range!");

  const auto& softmax_op = ops[ops.size() - 1];
  const auto& logit_output_tensor = tensors[softmax_op->outputs[0]];
  if (!MatchShape(logit_output_tensor->shape, {1, -1}))
    return absl::InternalError("Logit output tensor should be [1, x]");

  if (model_t.subgraphs[0]->outputs != std::vector<int>{softmax_op->outputs[0]})
    return absl::InternalError("Graph must have output from Softmax");

  return absl::OkStatus();
}

std::vector<uint8_t>& FCKernelTensorData(const tflite::ModelT& model) {
  const auto& ops = model.subgraphs[0]->operators;
  const auto& fc_op = ops[ops.size() - 4];
  const auto& tensors = model.subgraphs[0]->tensors;
  return model.buffers[tensors[fc_op->inputs[1]]->buffer]->data;
}

void UpdateImprintingModel(tflite::ModelT* model_t, int num_classes,
                           const std::vector<uint8_t>& weights) {
  const auto& ops = model_t->subgraphs[0]->operators;
  const auto& tensors = model_t->subgraphs[0]->tensors;
  const auto& buffers = model_t->buffers;

  // Modify FC shape and value.
  const auto& fc_op = ops[ops.size() - 4];
  auto& kernel_tensor = tensors[fc_op->inputs[1]];
  buffers[kernel_tensor->buffer]->data = weights;
  kernel_tensor->shape[0] = num_classes;

  auto& bias_tensor = tensors[fc_op->inputs[2]];
  const auto bias_zero_point = bias_tensor->quantization->zero_point[0];
  buffers[bias_tensor->buffer]->data =
      std::vector<uint8_t>(num_classes * sizeof(int32_t), bias_zero_point);
  bias_tensor->shape[0] = num_classes;

  auto& output_tensor = tensors[fc_op->outputs[0]];
  output_tensor->shape[3] = num_classes;
  // Modify quantization parameters to work for a new range, especailly for a
  // better one.
  output_tensor->quantization =
      CreateQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f},
                       /*scale=*/{1.0f / 128},
                       /*zero_point=*/{128});

  // Modify Mul shape.
  const auto& mul_op = ops[ops.size() - 3];
  auto& mul_tensor = tensors[mul_op->outputs[0]];
  mul_tensor->shape[3] = num_classes;

  const auto& scale_factor_tensor = tensors[mul_op->inputs[1]];
  float quantized_scale_factor = buffers[scale_factor_tensor->buffer]->data[0];
  float scale_factor;
  Dequantize(&quantized_scale_factor, &quantized_scale_factor + 1,
             &scale_factor,
             /*scale=*/scale_factor_tensor->quantization->scale[0],
             /*zero_point=*/scale_factor_tensor->quantization->zero_point[0]);

  mul_tensor->quantization =
      CreateQuantParam(/*min=*/{-scale_factor}, /*max=*/{scale_factor},
                       /*scale=*/{1.0f / 128 * scale_factor},
                       /*zero_point=*/{128});

  // Modify Reshape value.
  const auto& reshape_op = ops[ops.size() - 2];
  auto& reshape_tensor = tensors[reshape_op->outputs[0]];
  reshape_tensor->shape.back() = num_classes;

  // There can be two types of reshape op for tflite.
  // - If there is a second input tensor, it indicates the reshape's shape.
  // - If there is only one input tensor, the operator will compare the input
  // tensor shape and the output tensor shape.
  if (reshape_op->inputs.size() == 2) {
    auto& reshape_shape_tensor = tensors[reshape_op->inputs[1]];
    auto& reshape_shape_buffer = buffers[reshape_shape_tensor->buffer];
    reinterpret_cast<int32_t*>(reshape_shape_buffer->data.data())[1] =
        num_classes;
  }
  auto* reshape_option_t = reinterpret_cast<tflite::ReshapeOptionsT*>(
      reshape_op->builtin_options.value);
  if (reshape_option_t && !reshape_option_t->new_shape.empty()) {
    reshape_option_t->new_shape.back() = num_classes;
  }
  reshape_tensor->quantization =
      CreateQuantParam(/*min=*/{-scale_factor}, /*max=*/{scale_factor},
                       /*scale=*/{1.0f / 128 * scale_factor},
                       /*zero_point=*/{128});

  // Modify Softmax shape.
  const auto& softmax_op = ops[ops.size() - 1];
  auto& softmax_tensor = tensors[softmax_op->outputs[0]];
  softmax_tensor->shape[1] = num_classes;

  // Set proper output tensor.
  model_t->subgraphs[0]->outputs = {softmax_op->outputs[0]};
}
}  // namespace

absl::Status ImprintingModel::Create(
    const tflite::Model& prototype,
    std::unique_ptr<ImprintingModel>* out_model) {
  auto model = absl::WrapUnique(new ImprintingModel());
  model->model_t_ = absl::WrapUnique(prototype.UnPack());

  auto status = ValidateImprintingModel(*model->model_t_);
  if (!status.ok()) return status;

  const auto& ops = model->model_t_->subgraphs[0]->operators;
  const auto& tensors = model->model_t_->subgraphs[0]->tensors;

  const int l2norm_op_index = FindSingleOperator(
      tflite::BuiltinOperator_L2_NORMALIZATION, model->model_t_.get());
  const auto& l2norm_op = ops[l2norm_op_index];
  const auto& embedding_output_tensor = tensors[l2norm_op->outputs[0]];

  const auto& fc_op = ops[ops.size() - 4];
  const auto& kernel_tensor = tensors[fc_op->inputs[1]];

  model->embedding_dim_ = embedding_output_tensor->shape[3];
  model->fc_quant_scale_ = kernel_tensor->quantization->scale[0];
  model->fc_quant_zero_point_ = kernel_tensor->quantization->zero_point[0];

  model->model_t_->subgraphs[0]->outputs = {l2norm_op->outputs[0]};

  tflite::FinishModelBuffer(
      model->extractor_fbb_,
      tflite::Model::Pack(model->extractor_fbb_, model->model_t_.get()));

  *out_model = std::move(model);
  return absl::OkStatus();
}

std::unique_ptr<ImprintingModel> ImprintingModel::CreateOrDie(
    const tflite::Model& prototype) {
  std::unique_ptr<ImprintingModel> model;
  CHECK_EQ(Create(std::move(prototype), &model), absl::OkStatus());
  return model;
}

std::vector<ImprintingClass> ImprintingModel::LoadExistingClasses() const {
  auto metadata = ParseTrainingMetadata(model_t_->description);
  auto& data = FCKernelTensorData(*model_t_);

  CHECK_EQ(data.size() % embedding_dim_, 0);
  const auto num_classes = data.size() / embedding_dim_;
  std::vector<ImprintingClass> classes;
  classes.reserve(num_classes);

  for (int i = 0; i < num_classes; ++i) {
    classes.push_back(ImprintingClass{std::vector<float>(embedding_dim_),
                                      metadata.find(i) != metadata.end()});
    auto& added = classes.back();

    auto class_data = data.begin() + i * embedding_dim_;
    Dequantize(class_data, class_data + embedding_dim_, added.weights.begin(),
               /*scale=*/fc_quant_scale_, /*zero_point=*/fc_quant_zero_point_);

    if (added.trainable) {
      const auto l2norm = metadata[i];
      for (auto& weight : added.weights) weight *= l2norm;
    }
  }
  return classes;
}

absl::Status ImprintingModel::SerializeModel(
    const std::vector<ImprintingClass>& classes,
    flatbuffers::FlatBufferBuilder* fbb) {
  std::vector<uint8_t> data(classes.size() * embedding_dim_);

  std::map<int, float> metadata;
  for (int i = 0; i < classes.size(); ++i) {
    auto weights = classes[i].weights;
    const float l2norm = L2Normalize(weights);
    Quantize(weights.begin(), weights.end(), data.begin() + i * embedding_dim_,
             fc_quant_scale_, fc_quant_zero_point_);
    if (classes[i].trainable) metadata[i] = l2norm;
  }

  UpdateImprintingModel(model_t_.get(), data.size() / embedding_dim_, data);
  model_t_->description = SerializeTrainingMetadata(metadata);

  tflite::FinishModelBuffer(*fbb, tflite::Model::Pack(*fbb, model_t_.get()));
  return absl::OkStatus();
}

std::unique_ptr<ImprintingEngine> ImprintingEngine::Create(
    std::unique_ptr<ImprintingModel> model, bool keep_classes) {
  auto engine = absl::WrapUnique(new ImprintingEngine(std::move(model)));
  if (keep_classes) engine->classes_ = engine->model_->LoadExistingClasses();
  return engine;
}

absl::Status ImprintingEngine::Train(absl::Span<const float> embedding,
                                     int class_id) {
  if (embedding.size() != model_->embedding_dim())
    return absl::InternalError("Invalid number of weights.");

  const int num_classes = classes_.size();
  // Get previous trained image number of class |class_id|.
  // Or add a new category if |class_id| is not trained.
  if (class_id < num_classes && !classes_[class_id].trainable) {
    return absl::InternalError(
        "Cannot change the base model classes not trained with imprinting "
        "method!");
  }

  if (class_id > num_classes) {
    return absl::InternalError(
        "The class index of a new category is too large!");
  }

  const auto embedding_dim = model_->embedding_dim();
  if (class_id == num_classes)
    classes_.push_back({std::vector<float>(embedding_dim), true});

  auto& weights = classes_[class_id].weights;

  for (int i = 0; i < embedding_dim; ++i) weights[i] += embedding[i];

  return absl::OkStatus();
}

absl::Status ImprintingEngine::SerializeModel(
    flatbuffers::FlatBufferBuilder* fbb) {
  if (classes_.empty()) return absl::InternalError("Model is not trained.");
  return model_->SerializeModel(classes_, fbb);
}

}  // namespace coral
