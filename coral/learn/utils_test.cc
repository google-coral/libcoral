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
#include <cmath>
#include <numeric>
#include <tuple>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "flatbuffers/flatbuffers.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace {
using ::testing::ElementsAre;
using ::testing::Pointwise;

MATCHER_P(Near, tolerance, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) < tolerance;
}

tflite::QuantizationParametersT* GetKernelQuant(const tflite::ModelT* model_t,
                                                int op_index) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);
  auto& subgraph = model_t->subgraphs[0];
  CHECK_GT(subgraph->operators.size(), op_index);
  auto& conv2d_op = subgraph->operators[op_index];
  auto& kernel_tensor = subgraph->tensors[conv2d_op->inputs[1]];
  return kernel_tensor->quantization.get();
}

std::vector<uint8_t> QuantizeVector(
    const std::vector<float>& values,
    const tflite::QuantizationParametersT& params) {
  std::vector<uint8_t> quant_values(values.size());
  Quantize(values.begin(), values.end(), quant_values.begin(), params.scale[0],
           params.zero_point[0]);
  return quant_values;
}

// Generates dummy quantization parameters for conv2d operator.
// It assumes input tensor of conv2d operator has value within range [-1.0, 1.0]
int AppendTestLinearLayer(const std::vector<int>& kernel_shape,
                          tflite::ModelT* model_t) {
  return internal::AppendLinearLayer(
      kernel_shape,
      /*kernel_quant=*/
      CreateQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f}, /*scale=*/{1.0f / 128},
                       /*zero_point=*/{128}),
      /*bias_quant=*/
      CreateQuantParam(/*min=*/{}, /*max=*/{}, /*scale=*/{1.0f / (128 * 128)},
                       /*zero_point=*/{0}),
      /*output_quant=*/
      CreateQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f}, /*scale=*/{1.0f / 128},
                       /*zero_point=*/{128}),
      model_t);
}

// Builds a test graph that consists of
//    input_tensor
//        |
//        v
//     Conv2d/FC
//        |
//        v
//    output_tensor
std::unique_ptr<tflite::ModelT> BuildTestGraph(
    const std::vector<int>& input_shape, const std::vector<int>& kernel_shape,
    const std::vector<float>& kernel) {
  auto model_t = absl::make_unique<tflite::ModelT>();
  model_t->description = "Hand-crafted tflite graph for testing";
  model_t->version = 3;  // Must specify, current version is 3.

  // Create sentinel buffer.
  internal::AppendBuffer(/*buffer_size_bytes=*/0, model_t.get());

  // Create a subgraph with only input tensor.
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  const int input_buffer_index =
      internal::AppendBuffer(/*buffer_size_bytes=*/0, model_t.get());
  auto input_tensor_quant =
      CreateQuantParam(/*min=*/{-128.0f}, /*max=*/{128.0f}, /*scale=*/{1.0f},
                       /*zero_point=*/{128});
  const int input_tensor_index = internal::AppendTensor(
      input_shape, /*name=*/"TestGraph/input", input_buffer_index,
      tflite::TensorType_UINT8, std::move(input_tensor_quant), subgraph.get());
  subgraph->inputs = {input_tensor_index};
  // Current graph output is input tensor itself.
  subgraph->outputs = {input_tensor_index};
  model_t->subgraphs.push_back(std::move(subgraph));

  // Add Conv2d Operator.
  const std::vector<tflite::TensorT*> output_tensors =
      GetGraphOutputTensors(model_t.get());
  CHECK_EQ(output_tensors.size(), 1);
  const tflite::TensorT* current_output_tensor = output_tensors[0];
  std::vector<int> output_shape = internal::CalculateLinearLayerOutputShape(
      current_output_tensor->shape, kernel_shape);

  const auto op_type = kernel_shape.size() == 2
                           ? tflite::BuiltinOperator_FULLY_CONNECTED
                           : tflite::BuiltinOperator_CONV_2D;
  const int conv2d_op_index = internal::AppendOperator(
      {
          {"TestGraph/Conv2d/Kernel", tflite::TensorType_UINT8,
           internal::TensorLocation::kParameter, kernel_shape,
           CreateQuantParam(/*min=*/{-128.0f}, /*max=*/{128.0f},
                            /*scale=*/{1.0f},
                            /*zero_point=*/{128})
               .release()},
          {"TestGraph/Conv2d/Bias",
           tflite::TensorType_INT32,
           internal::TensorLocation::kParameter,
           {kernel_shape[0]},
           CreateQuantParam(/*min=*/{}, /*max=*/{}, /*scale=*/{1.0f},
                            /*zero_point=*/{0})
               .release()},
          {"TestGraph/Conv2d/Output", tflite::TensorType_UINT8,
           internal::TensorLocation::kOutput, output_shape,
           CreateQuantParam(/*min=*/{-128.0f}, /*max=*/{128.0f},
                            /*scale=*/{1.0f},
                            /*zero_point=*/{128})
               .release()},
      },
      op_type, model_t.get());

  // Set kernel value.
  auto* kernel_quant = GetKernelQuant(model_t.get(), conv2d_op_index);
  internal::SetLinearParams(QuantizeVector(kernel, *kernel_quant), /*bias=*/{},
                            conv2d_op_index, model_t.get());
  return model_t;
}  // namespace

// Runs inference with ModelT as input type.
std::vector<float> RunInference(const tflite::ModelT* model_t,
                                const std::vector<float>& input) {
  flatbuffers::FlatBufferBuilder fbb;
  tflite::FinishModelBuffer(fbb, tflite::Model::Pack(fbb, model_t));
  auto model = LoadModelOrDie(fbb);
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  const auto* input_tensor = interpreter->input_tensor(0);
  CHECK_EQ(input_tensor->quantization.type, kTfLiteAffineQuantization);
  auto* quantization_params =
      static_cast<TfLiteAffineQuantization*>(input_tensor->quantization.params);
  Quantize(input.begin(), input.end(),
           MutableTensorData<uint8_t>(*input_tensor).begin(),
           quantization_params->scale->data[0],
           quantization_params->zero_point->data[0]);
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
  return DequantizeTensor<float>(*interpreter->output_tensor(0));
}

std::unique_ptr<tflite::ModelT> LoadTestModel(const std::string& model_name) {
  return absl::WrapUnique(CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(
                                            TestDataPath(model_name).c_str()))
                              ->GetModel()
                              ->UnPack());
}

TEST(UtilsCpuTest, BuildConvTestGrapAndRunInference) {
  auto model_t = BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                                /*kernel_shape=*/{2, 1, 1, 5}, /*kernel=*/
                                {
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // kernel 1
                                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f,  // kernel 2
                                });
  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 1);
  EXPECT_THAT(RunInference(model_t.get(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
              Pointwise(Near(0.01), {15.0f, 55.0f}));
}

TEST(UtilsCpuTest, BuildFCTestGrapAndRunInference) {
  auto model_t = BuildTestGraph(/*input_shape=*/{1, 5},
                                /*kernel_shape=*/{2, 5}, /*kernel=*/
                                {
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // kernel 1
                                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f,  // kernel 2
                                });
  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 1);
  EXPECT_THAT(RunInference(model_t.get(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
              Pointwise(Near(0.01), {15.0f, 55.0f}));
}

TEST(UtilsCpuTest, AppendL2Norm) {
  auto model_t =
      BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                     /*kernel_shape=*/{3, 1, 1, 5}, /*kernel=*/
                     {
                         1.0f, 1.0f, 1.0f, 1.0f, 1.0f,       // kernel 1
                         -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
                         3.0f, 3.0f, 3.0f, 3.0f, 3.0f,       // kernel 3
                     });
  ASSERT_EQ(internal::AppendL2Norm(model_t.get()), 1);
  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 2);
  EXPECT_THAT(RunInference(model_t.get(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
              Pointwise(Near(0.01),
                        {1 / std::sqrt(11.0f),   //
                         -1 / std::sqrt(11.0f),  //
                         3 / std::sqrt(11.0f)}));
}

TEST(UtilsCpuTest, AppendConv2dLayer) {
  auto model_t =
      BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                     /*kernel_shape=*/{2, 1, 1, 5}, /*kernel=*/
                     {
                         1.0f, 1.0f, 1.0f, 1.0f, 1.0f,       // kernel 1
                         -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
                     });

  internal::AppendL2Norm(model_t.get());
  const int op_index = AppendTestLinearLayer(
      /*kernel_shape=*/{4, 1, 1, 2}, model_t.get());
  ASSERT_EQ(op_index, 2);
  // Set weights for fully-connected layer.
  const std::vector<float>& fc_weights = {
      std::sqrt(2.0f) / 2, -std::sqrt(2.0f) / 2,  // kernel 1
      std::sqrt(2.0f) / 2, std::sqrt(2.0f) / 2,   // kernel 2
      std::sqrt(7.0f) / 4, 3.0f / 4,              // kernel 3
      std::sqrt(5.0f) / 3, 2.0f / 3,              // kernel 4
  };

  auto* conv_weights_quant = GetKernelQuant(model_t.get(), op_index);
  internal::SetLinearParams(QuantizeVector(fc_weights, *conv_weights_quant),
                            /*bias=*/{}, op_index, model_t.get());
  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 3);

  // output tensor of L2Norm layer is [sqrt(2)/2, -sqrt(2)/2], with above
  // `fc_weights`, result is expected to be:
  // [1, 0, (sqrt(14)-sqrt(18))/8, (sqrt(10)-sqrt(8))/6]
  EXPECT_THAT(RunInference(model_t.get(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
              Pointwise(Near(0.01),
                        {1.0f,                                       //
                         0.0f,                                       //
                         (std::sqrt(14.0f) - std::sqrt(18.0f)) / 8,  //
                         (std::sqrt(10.0f) - std::sqrt(8.0f)) / 6}));
}

TEST(UtilsCpuTest, AppendFullyConnectedLayer) {
  auto model_t =
      BuildTestGraph(/*input_shape=*/{1, 5},
                     /*kernel_shape=*/{2, 5}, /*kernel=*/
                     {
                         1.0f, 1.0f, 1.0f, 1.0f, 1.0f,       // kernel 1
                         -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
                     });

  internal::AppendL2Norm(model_t.get());
  const int op_index = AppendTestLinearLayer(
      /*kernel_shape=*/{4, 2}, model_t.get());
  ASSERT_EQ(op_index, 2);

  // Set weights for fully-connected layer.
  const std::vector<float>& fc_weights = {
      std::sqrt(2.0f) / 2, -std::sqrt(2.0f) / 2,  // kernel 1
      std::sqrt(2.0f) / 2, std::sqrt(2.0f) / 2,   // kernel 2
      std::sqrt(7.0f) / 4, 3.0f / 4,              // kernel 3
      std::sqrt(5.0f) / 3, 2.0f / 3,              // kernel 4
  };

  auto* fc_weights_quant = GetKernelQuant(model_t.get(), op_index);
  internal::SetLinearParams(QuantizeVector(fc_weights, *fc_weights_quant),
                            /*bias=*/{}, op_index, model_t.get());
  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 3);
  // output tensor of L2Norm layer is [sqrt(2)/2, -sqrt(2)/2], with above
  // `fc_weights`, result is expected to be:
  // [1, 0, (sqrt(14)-sqrt(18))/8, (sqrt(10)-sqrt(8))/6]
  EXPECT_THAT(RunInference(model_t.get(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
              Pointwise(Near(0.01),
                        {
                            1.0f,                                       //
                            0.0f,                                       //
                            (std::sqrt(14.0f) - std::sqrt(18.0f)) / 8,  //
                            (std::sqrt(10.0f) - std::sqrt(8.0f)) / 6,
                        }));
}

TEST(UtilsCpuTest, AppendReshape) {
  auto model_t =
      BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                     /*kernel_shape=*/{2, 1, 1, 5}, /*kernel=*/
                     {
                         1.0f, 1.0f, 1.0f, 1.0f, 1.0f,       // kernel 1
                         -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
                     });

  internal::AppendL2Norm(model_t.get());
  const int conv_op_index = AppendTestLinearLayer(
      /*kernel_shape=*/{4, 1, 1, 2}, model_t.get());

  ASSERT_EQ(internal::AppendReshape(model_t.get()), 3);

  // Set weights for fully-connected layer.
  const std::vector<float>& conv_weights = {
      std::sqrt(2.0f) / 2, -std::sqrt(2.0f) / 2,  // kernel 1
      std::sqrt(2.0f) / 2, std::sqrt(2.0f) / 2,   // kernel 2
      std::sqrt(7.0f) / 4, 3.0f / 4,              // kernel 3
      std::sqrt(5.0f) / 3, 2.0f / 3,              // kernel 4
  };

  auto* conv_weights_quant = GetKernelQuant(model_t.get(), conv_op_index);
  internal::SetLinearParams(QuantizeVector(conv_weights, *conv_weights_quant),
                            /*bias=*/{}, conv_op_index, model_t.get());

  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 4);
  // Check graph's output tensor's shape.
  const std::vector<tflite::TensorT*> output_tensors =
      GetGraphOutputTensors(model_t.get());
  ASSERT_EQ(output_tensors.size(), 1);
  ASSERT_THAT(output_tensors[0]->shape, ElementsAre(1, 4));

  // output tensor of L2Norm layer is [sqrt(2)/2, -sqrt(2)/2], with above
  // `fc_weights`, result is expected to be:
  // [1, 0, (sqrt(14)-sqrt(18))/8, (sqrt(10)-sqrt(8))/6]
  EXPECT_THAT(RunInference(model_t.get(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
              Pointwise(Near(0.01),
                        {
                            1.0f,                                       //
                            0.0f,                                       //
                            (std::sqrt(14.0f) - std::sqrt(18.0f)) / 8,  //
                            (std::sqrt(10.0f) - std::sqrt(8.0f)) / 6,
                        }));
}

TEST(UtilsCpuTest, AppendSoftmax) {
  auto model_t =
      BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                     /*kernel_shape=*/{2, 1, 1, 5}, /*kernel=*/
                     {
                         1.0f, 1.0f, 1.0f, 1.0f, 1.0f,       // kernel 1
                         -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
                     });

  internal::AppendL2Norm(model_t.get());
  const int fc_op_index = AppendTestLinearLayer(
      /*kernel_shape=*/{4, 1, 1, 2}, model_t.get());

  internal::AppendReshape(model_t.get());
  ASSERT_EQ(internal::AppendSoftmax(model_t.get()), 4);

  // Set weights for fully-connected layer.
  const std::vector<float>& fc_weights = {
      std::sqrt(2.0f) / 2, -std::sqrt(2.0f) / 2,  // kernel 1
      std::sqrt(2.0f) / 2, std::sqrt(2.0f) / 2,   // kernel 2
      std::sqrt(7.0f) / 4, 3.0f / 4,              // kernel 3
      std::sqrt(5.0f) / 3, 2.0f / 3,              // kernel 4
  };

  auto* fc_weights_quant = GetKernelQuant(model_t.get(), fc_op_index);
  internal::SetLinearParams(QuantizeVector(fc_weights, *fc_weights_quant),
                            /*bias=*/{}, fc_op_index, model_t.get());

  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 5);

  // Result after Fully-connect layer is:
  // [1, 0, (sqrt(14)-sqrt(18))/8, (sqrt(10)-sqrt(8))/6]
  std::vector<float> expected = {
      std::exp(1.0f), std::exp(0.0f),
      std::exp((std::sqrt(14.0f) - std::sqrt(18.0f)) / 8),
      std::exp((std::sqrt(10.0f) - std::sqrt(8.0f)) / 6)};
  auto sum = std::accumulate(expected.begin(), expected.end(), 0.0f);
  for (auto& e : expected) e /= sum;

  EXPECT_THAT(RunInference(model_t.get(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
              Pointwise(Near(0.01), expected));
}

TEST(UtilsCpuTest, FindOperators) {
  const auto model_t = LoadTestModel("mobilenet_v1_1.0_224_quant.tflite");
  EXPECT_THAT(
      FindOperators(tflite::BuiltinOperator_CONV_2D, model_t.get()),
      ElementsAre(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28));
}

TEST(UtilsCpuTest, FindSingleOperator) {
  const auto model_t = LoadTestModel("mobilenet_v1_1.0_224_quant.tflite");
  EXPECT_EQ(FindSingleOperator(tflite::BuiltinOperator_SOFTMAX, model_t.get()),
            30);
  EXPECT_EQ(FindSingleOperator(tflite::BuiltinOperator_LSTM, model_t.get()),
            -1);
  EXPECT_EQ(FindSingleOperator(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                               model_t.get()),
            -1);
}

TEST(UtilsCpuTest, FindOperatorsWithInput) {
  const auto model_t = LoadTestModel("mobilenet_v1_1.0_224_quant.tflite");
  // Use MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6 as input tensor.
  EXPECT_THAT(FindOperatorsWithInput(tflite::BuiltinOperator_CONV_2D,
                                     /*input_tensor_index=*/61, model_t.get(),
                                     /*base_op_index=*/0),
              ElementsAre(16));
}

// Test finding a single operator given input tensor with a real model.
TEST(UtilsCpuTest, FindSingleOperatorWithInput) {
  const auto model_t = LoadTestModel("mobilenet_v1_1.0_224_quant.tflite");
  // Use MobilenetV1/Logits/SpatialSqueeze as input tensor.
  const int input_tensor_index = 87;
  const int base_op_index = 0;

  EXPECT_EQ(FindSingleOperatorWithInput(tflite::BuiltinOperator_SOFTMAX,
                                        input_tensor_index, model_t.get(),
                                        base_op_index),
            30);

  EXPECT_EQ(FindSingleOperatorWithInput(tflite::BuiltinOperator_LSTM,
                                        input_tensor_index, model_t.get(),
                                        base_op_index),
            -1);
}

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class UtilsRealModelTest : public ModelTestBase {
 protected:
  std::string GenerateModelPath(const std::string& file_name) {
    return file_name + GetParam();
  }

  void TestAppendFullyConnectedAndSoftmaxLayerToModel(
      const std::string& in_model_path) {
    auto in_model = LoadModelOrDie(in_model_path);
    auto in_interpreter =
        MakeEdgeTpuInterpreterOrDie(*in_model, GetTpuContextIfNecessary());
    ASSERT_EQ(in_interpreter->AllocateTensors(), kTfLiteOk);
    auto in_input =
        MutableTensorData<uint8_t>(*in_interpreter->input_tensor(0));
    FillRandomInt(in_input);
    ASSERT_EQ(in_interpreter->Invoke(), kTfLiteOk);
    auto in_result = DequantizeTensor<float>(*in_interpreter->output_tensor(0));
    const int embedding_vector_dim = in_result.size();
    const float embedding_vector_sum =
        std::accumulate(in_result.begin(), in_result.end(), 0.0f);
    // Generates dummy weights, of dimension embedding_vector_dim x 3. Each
    // kernel has the following pattern (times a scalar to make max logits score
    // = 1) : Kernel 1: 1, 1, 1, ... Kernel 2: 2, 2, 2, ... kernel 3: 3, 3, 3,
    // ...
    std::vector<float> weights(embedding_vector_dim * 3);
    const float scalar = 1 / (embedding_vector_sum * 3);
    for (int i = 1; i <= 3; ++i)
      std::fill(weights.begin() + embedding_vector_dim * (i - 1),
                weights.begin() + embedding_vector_dim * i, scalar * i);

    std::vector<float> biases(3, 10.0f);
    std::vector<float> expected_fc_output = {
        embedding_vector_sum * scalar + biases[0],
        embedding_vector_sum * scalar * 2 + biases[1],
        embedding_vector_sum * scalar * 3 + biases[2]};
    const float out_tensor_min =
        *std::min_element(expected_fc_output.begin(), expected_fc_output.end());
    const float out_tensor_max =
        *std::max_element(expected_fc_output.begin(), expected_fc_output.end());

    flatbuffers::FlatBufferBuilder fbb;
    ASSERT_EQ(AppendFullyConnectedAndSoftmaxLayerToModel(
                  *in_model->GetModel(), &fbb, weights, biases, out_tensor_min,
                  out_tensor_max),
              absl::OkStatus());

    // Calculate expected value.
    std::vector<float> expected = expected_fc_output;
    float max_score = *std::max_element(expected.begin(), expected.end());
    // Subtract max_score to avoid overflow.
    for (auto& e : expected) e = std::exp(e - max_score);
    float exp_sum = std::accumulate(expected.begin(), expected.end(), 0.0f);
    for (auto& e : expected) e /= exp_sum;

    auto out_model = LoadModelOrDie(fbb);
    auto out_interpreter =
        MakeEdgeTpuInterpreterOrDie(*out_model, GetTpuContextIfNecessary());
    ASSERT_EQ(out_interpreter->AllocateTensors(), kTfLiteOk);
    auto out_input =
        MutableTensorData<uint8_t>(*out_interpreter->input_tensor(0));
    std::copy(in_input.begin(), in_input.end(), out_input.begin());
    ASSERT_EQ(out_interpreter->Invoke(), kTfLiteOk);
    EXPECT_THAT(DequantizeTensor<float>(*out_interpreter->output_tensor(0)),
                Pointwise(Near(5e-3), expected));
  }
};

TEST_P(UtilsRealModelTest, AppendConv2dAndSoftmaxLayerToModel) {
  TestAppendFullyConnectedAndSoftmaxLayerToModel(TestDataPath(
      GenerateModelPath("mobilenet_v1_1.0_224_quant_embedding_extractor")));
}

TEST_P(UtilsRealModelTest, AppendFullyConnectedAndSoftmaxLayerToModel) {
  TestAppendFullyConnectedAndSoftmaxLayerToModel(TestDataPath(
      GenerateModelPath("efficientnet-edgetpu-S_quant_embedding_extractor")));
}

INSTANTIATE_TEST_CASE_P(UtilsRealCpuModelTest, UtilsRealModelTest,
                        ::testing::Values(".tflite"));
INSTANTIATE_TEST_CASE_P(UtilsRealEdgeTpuModelTest, UtilsRealModelTest,
                        ::testing::Values("_edgetpu.tflite"));

}  // namespace
}  // namespace coral
