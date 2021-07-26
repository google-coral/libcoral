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

#ifndef LIBCORAL_CORAL_LEARN_UTILS_H_
#define LIBCORAL_CORAL_LEARN_UTILS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"
#include "glog/logging.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace internal {

// NOTE: all of the following AppendXXX functions are tuned for last layer
// backprop method, especially quantization parameters. You should adapt the
// implementation accordingly if used in other cases.

// Appends L2Normalization. Returns index of the L2Norm operator in subgraph.
int AppendL2Norm(tflite::ModelT* model_t);

// Appends Conv2d / FC layer. Returns index of the Conv2d / FC operator in
// subgraph. It distinguishes the two operators using the size of kernel_shape,
// where size of 4 indicates a Conv while size of 2 indicates a FC.
//
// |quant_params| contains the quantization parameters for kernel weights,
// biases and output tensor.
int AppendLinearLayer(
    const std::vector<int>& kernel_shape,
    std::unique_ptr<tflite::QuantizationParametersT> kernel_quant,
    std::unique_ptr<tflite::QuantizationParametersT> bias_quant,
    std::unique_ptr<tflite::QuantizationParametersT> output_quant,
    tflite::ModelT* model_t);

// Appends Reshape. Returns index of the Reshape operator in subgraph.
int AppendReshape(tflite::ModelT* model_t);

// Appends Softmax. Returns index of the Softmax operator in subgraph.
int AppendSoftmax(tflite::ModelT* model_t);

// Creates and appends buffer to model. Returns new buffer index.
int AppendBuffer(size_t buffer_size_bytes, tflite::ModelT* model_t);

// Appends tensor to subgraph and returns new tensor's index.
int AppendTensor(const std::vector<int>& shape, const std::string& name,
                 int buffer_index, tflite::TensorType type,
                 std::unique_ptr<tflite::QuantizationParametersT> q_param,
                 tflite::SubGraphT* subgraph);

enum class TensorLocation {
  // Refers intermediate tensor, input of an operator.
  kInput,
  // Refers intermediate tensor, output of an operator.
  kOutput,
  // Refers parameter tensor, e.g., kernel of convolution.
  kParameter,
};

struct TensorConfig {
  std::string name;
  tflite::TensorType type;
  TensorLocation location;
  std::vector<int> shape;
  tflite::QuantizationParametersT* quant;
};

// Appends an operator to model. Returns index of the new operator in subgraph.
// |tensor_configs| should only contains parameter tensors and output tensors
// for the new operator. It assumes the input of the new operator is the first
// output of the graph, and the output of the new operator is the new first
// output tensor of the graph. Does not support custom operator.
int AppendOperator(const std::vector<TensorConfig>& tensor_configs,
                   tflite::BuiltinOperator op_type, tflite::ModelT* model_t);

// Sets Conv2d / FC parameters, i.e., kernel and bias.
// Bias will be set to zeros if `bias` is set to empty.
//
// Note on weights data ordering.
// "Typical TF Lite weights are [filter_count, filter_height, filter_width,
// input_depth]". See comments inside `AllocateTemporaryTensorsIfRequired` in
// More info: tensorflow/lite/kernels/conv.cc.
void SetLinearParams(const std::vector<uint8_t>& kernel,
                     const std::vector<int32_t>& bias, int op_index,
                     tflite::ModelT* model_t);

// Calculates the shape of conv/fc's output, given shape of input tensor and
// kernel tensor.
std::vector<int> CalculateLinearLayerOutputShape(
    const std::vector<int>& input_shape, const std::vector<int>& kernel_shape);

}  // namespace internal

// Calculates scale and zero point, given min, max range and target data type T.
template <typename T>
std::pair<float, int32_t> QuantizationParams(float f_min, float f_max) {
  int32_t zero_point = 0;
  float scale = 0;
  const T qmin = std::numeric_limits<T>::min();
  const T qmax = std::numeric_limits<T>::max();
  const float qmin_double = qmin;
  const float qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  CHECK_LE(f_min, 0);
  CHECK_GE(f_max, 0);
  if (f_min == f_max) {
    // Special case where the min,max range is a point. Should be {0}.
    CHECK_EQ(f_min, 0);
    CHECK_EQ(f_max, 0);
    return {scale, zero_point};
  }

  // General case.
  //
  // First determine the scale.
  scale = (f_max - f_min) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const float zero_point_from_min = qmin_double - f_min / scale;
  const float zero_point_from_max = qmax_double - f_max / scale;

  const float zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(f_min / scale);

  const float zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(f_max / scale);

  const float zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  //  padding).

  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(std::round(zero_point_double));
  }

  // The zero point should always be in the range of quantized value,
  // // [qmin, qmax].
  CHECK_GE(nudged_zero_point, qmin);
  CHECK_LE(nudged_zero_point, qmax);

  zero_point = nudged_zero_point;
  // finally, return the values
  return {scale, zero_point};
}

// L2-normalizes a vector, returns L2-norm.
template <typename T>
T L2Normalize(std::vector<T>& v) {
  const float l2_norm =
      std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), T(0)));

  if (std::abs(l2_norm) > 1e-5)
    for (auto& e : v) e /= l2_norm;

  return l2_norm;
}

template <typename T>
std::vector<T> L2NormalizedVector(const std::vector<T>& v) {
  std::vector<T> copy(v);
  L2Normalize(copy);
  return copy;
}

// Finds the opcode index of target operator. Returns -1 if `target_op` does not
// exist in `opcodes`. For custom operator, custom code must match as well.
int FindOpcodeIndex(
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& opcodes,
    tflite::BuiltinOperator target_op, const std::string& custom_code);

// Returns the indices of operators specified by operator code with given tensor
// as their inputs[0]. It is counted from base_op_index.
std::vector<int> FindOperatorsWithInput(tflite::BuiltinOperator target_op,
                                        int input_tensor_index,
                                        const tflite::ModelT* model_t,
                                        int base_op_index = 0);

// Returns the index of the single operator specified by operator code with
// given tensor as their inputs[0]. It is counted from base_op_index.
// Returns -1 if the operator can not be found or there are multiple matches.
int FindSingleOperatorWithInput(tflite::BuiltinOperator target_op,
                                int input_tensor_index,
                                const tflite::ModelT* model_t,
                                int base_op_index);

// Returns indices of operators specified by operator code.
std::vector<int> FindOperators(tflite::BuiltinOperator target_op,
                               const tflite::ModelT* model_t);

// Returns the index of the single operator specified by operator code.
// Returns -1 if the operator can not be found or there are multiple matches.
int FindSingleOperator(tflite::BuiltinOperator target_op,
                       const tflite::ModelT* model_t);

// Creates quantization parameters.
std::unique_ptr<tflite::QuantizationParametersT> CreateQuantParam(
    const std::vector<float>& min, const std::vector<float>& max,
    const std::vector<float>& scale, const std::vector<int64_t>& zero_point);

// Returns vector of pointers to graph output tensors.
std::vector<tflite::TensorT*> GetGraphOutputTensors(
    const tflite::ModelT* model_t);

// Appends Fully-Connected (FC) layer and softmax layer to tflite model.
//
// This function does the following:
//   1) Read tflite model from |in_model_path| as input;
//        input model is assumed to be an embedding extractor, e.g., a
//        classification model without the last FC+Softmax layer.
//   2) Append (learned) weights and biases as a FC/Conv layer to input modell
//        the appended operator is decided by the output shape of tflite model.
//        If the output is a 2D tensor, FC would be appended, otherwise Conv
//        would be appended.
//   3) Append softmax layer after the FC/Conv layer;
//   4) Save tflite model to |out_model_path|;
absl::Status AppendFullyConnectedAndSoftmaxLayerToModel(
    const tflite::Model& model, flatbuffers::FlatBufferBuilder* fbb,
    absl::Span<const float> weights, absl::Span<const float> biases,
    float out_tensor_min, float out_tensor_max);
}  // namespace coral

#endif  // LIBCORAL_CORAL_LEARN_UTILS_H_
