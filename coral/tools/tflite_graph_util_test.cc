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

#include "absl/strings/str_format.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {
namespace {
using Image = std::vector<uint8_t>;

class TfliteGraphUtilSplitFcCpuTest : public ModelEquivalenceTestBase {
 protected:
  void CheckTensorShape(const tflite::FlatBufferModel& fb_model,
                        const std::string& tensor_name,
                        const std::vector<int>& expected_shape) {
    tflite::ModelT model;
    fb_model.GetModel()->UnPackTo(&model);
    bool found = false;
    for (const auto& tensor : model.subgraphs[0]->tensors) {
      if (tensor->name != tensor_name) continue;
      EXPECT_THAT(tensor->shape, ::testing::ContainerEq(expected_shape));
      found = true;
      break;
    }
    ASSERT_TRUE(found) << "Can not find tensor " << tensor_name;
  }
};

TEST_F(TfliteGraphUtilSplitFcCpuTest, UsPopularProductModel) {
  const std::string input_model_path =
      TestDataPath("tfhub_tf1_popular_us_products_ptq.tflite");
  auto input_model =
      tflite::FlatBufferModel::BuildFromFile(input_model_path.data());
  ASSERT_TRUE(input_model);
  CheckTensorShape(*input_model, "MatMul1", {100000, 1});

  const char* kTmpOutputModelPath = "/tmp/split_result.tflite";
  ASSERT_TRUE(
      SplitFullyConnected(
          input_model_path,
          /*fc_input_tensor_name=*/
          "module_apply_default/normalized_embedding;module_apply_default/"
          "normalized_embedding/Square;module_apply_default/"
          "normalized_embedding/Sum/reduction_indices;module_apply_default/"
          "normalized_embedding/Sum;module_apply_default/normalized_embedding/"
          "Maximum/y;module_apply_default/normalized_embedding/"
          "Maximum;module_apply_default/normalized_embedding/Rsqrt",
          /*fc_weights_tensor_name=*/"Const",
          /*fc_bias_tensor_name=*/"",
          /*fc_output_tensor_name=*/"MatMul1", kTmpOutputModelPath,
          /*feature_dim_index=*/0,
          /*split_ratio=*/0.4)
          .ok());

  auto output_model =
      tflite::FlatBufferModel::BuildFromFile(kTmpOutputModelPath);
  ASSERT_TRUE(output_model);
  CheckTensorShape(*output_model, "MatMul1/fc_0", {40000, 1});
  CheckTensorShape(*output_model, "MatMul1/fc_1", {60000, 1});
  CheckTensorShape(*output_model, "MatMul1", {100000, 1});

  TestModelEquivalence(TestDataPath("missvickie_potato_chips.bmp"),
                       TestDataPath("tfhub_tf1_popular_us_products_ptq.tflite"),
                       kTmpOutputModelPath, /*tolerance=*/1);
  TestModelEquivalence("",  // random input
                       TestDataPath("tfhub_tf1_popular_us_products_ptq.tflite"),
                       kTmpOutputModelPath, /*tolerance=*/1);
}

TEST_F(TfliteGraphUtilSplitFcCpuTest, INatPlantModel) {
  const std::string input_model_path =
      TestDataPath("mobilenet_v2_1.0_224_inat_plant_quant.tflite");
  auto input_model =
      tflite::FlatBufferModel::BuildFromFile(input_model_path.data());
  ASSERT_TRUE(input_model);
  CheckTensorShape(*input_model, "Logits/MatMul;Logits/BiasAdd", {1, 2102});

  const char* kTmpOutputModelPath = "/tmp/split_result.tflite";
  ASSERT_TRUE(SplitFullyConnected(
                  input_model_path,
                  /*fc_input_tensor_name=*/"AvgPool/AvgPool",
                  /*fc_weights_tensor_name=*/
                  "Logits/MatMul;Logits/weights_quant/FakeQuantWithMinMaxVars",
                  /*fc_bias_tensor_name=*/"Logits/biases",
                  /*fc_output_tensor_name=*/"Logits/MatMul;Logits/BiasAdd",
                  kTmpOutputModelPath,
                  /*feature_dim_index=*/1,
                  /*split_ratio=*/0.8)
                  .ok());

  auto output_model =
      tflite::FlatBufferModel::BuildFromFile(kTmpOutputModelPath);
  ASSERT_TRUE(output_model);
  CheckTensorShape(*output_model, "Logits/MatMul;Logits/BiasAdd/fc_0",
                   {1, 1682});
  CheckTensorShape(*output_model, "Logits/MatMul;Logits/BiasAdd/fc_1",
                   {1, 420});
  CheckTensorShape(*output_model, "Logits/MatMul;Logits/BiasAdd", {1, 2102});

  TestModelEquivalence(
      TestDataPath("sunflower.bmp"),
      TestDataPath("mobilenet_v2_1.0_224_inat_plant_quant.tflite"),
      kTmpOutputModelPath, /*tolerance=*/1);
  TestModelEquivalence(
      "",  // random input
      TestDataPath("mobilenet_v2_1.0_224_inat_plant_quant.tflite"),
      kTmpOutputModelPath, /*tolerance=*/1);
}

// Inception V1 has conv1x1 instead of the fully connected layer.
TEST_F(TfliteGraphUtilSplitFcCpuTest, InceptionV1) {
  const std::string input_model_path =
      TestDataPath("inception_v1_224_quant.tflite");
  auto input_model =
      tflite::FlatBufferModel::BuildFromFile(input_model_path.data());
  ASSERT_TRUE(input_model);
  CheckTensorShape(
      *input_model,
      "InceptionV1/Logits/Conv2d_0c_1x1/BiasAdd;InceptionV1/Logits/"
      "Conv2d_0c_1x1/Conv2D;InceptionV1/Logits/Conv2d_0c_1x1/biases1",
      {1, 1, 1, 1001});

  const char* kTmpOutputModelPath = "/tmp/split_result.tflite";
  ASSERT_TRUE(
      SplitFullyConnected(
          input_model_path,
          /*fc_input_tensor_name=*/"InceptionV1/Logits/AvgPool_0a_7x7/AvgPool",
          /*fc_weights_tensor_name=*/
          "InceptionV1/Logits/Conv2d_0c_1x1/Conv2D;InceptionV1/Logits/"
          "Conv2d_0c_1x1/weights_quant/FakeQuantWithMinMaxVars",
          /*fc_bias_tensor_name=*/
          "InceptionV1/Logits/Conv2d_0c_1x1/BiasAdd;InceptionV1/Logits/"
          "Conv2d_0c_1x1/Conv2D;InceptionV1/Logits/Conv2d_0c_1x1/biases",
          /*fc_output_tensor_name=*/
          "InceptionV1/Logits/Conv2d_0c_1x1/BiasAdd;InceptionV1/Logits/"
          "Conv2d_0c_1x1/Conv2D;InceptionV1/Logits/Conv2d_0c_1x1/biases1",
          kTmpOutputModelPath,
          /*feature_dim_index=*/3,
          /*split_ratio=*/0.6)
          .ok());

  auto output_model =
      tflite::FlatBufferModel::BuildFromFile(kTmpOutputModelPath);
  ASSERT_TRUE(output_model);
  CheckTensorShape(
      *output_model,
      "InceptionV1/Logits/Conv2d_0c_1x1/BiasAdd;InceptionV1/Logits/"
      "Conv2d_0c_1x1/Conv2D;InceptionV1/Logits/Conv2d_0c_1x1/biases1/fc_0",
      {1, 1, 1, 601});
  CheckTensorShape(
      *output_model,
      "InceptionV1/Logits/Conv2d_0c_1x1/BiasAdd;InceptionV1/Logits/"
      "Conv2d_0c_1x1/Conv2D;InceptionV1/Logits/Conv2d_0c_1x1/biases1/fc_1",
      {1, 1, 1, 400});
  CheckTensorShape(
      *output_model,
      "InceptionV1/Logits/Conv2d_0c_1x1/BiasAdd;InceptionV1/Logits/"
      "Conv2d_0c_1x1/Conv2D;InceptionV1/Logits/Conv2d_0c_1x1/biases1",
      {1, 1, 1, 1001});

  TestModelEquivalence(TestDataPath("sunflower.bmp"),
                       TestDataPath("inception_v1_224_quant.tflite"),
                       kTmpOutputModelPath, /*tolerance=*/1);
  TestModelEquivalence("",  // random input
                       TestDataPath("inception_v1_224_quant.tflite"),
                       kTmpOutputModelPath, /*tolerance=*/1);
}

// Builds the interpreter and sets RNN hidden state input tensors to zero.
// It is for RNN models with the recurrent link cuts after converted to TFLite
// format.
std::unique_ptr<tflite::Interpreter> BuildRnnEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    const std::vector<int>& input_rnn_indexes,
    const std::vector<int>& output_rnn_indexes,
    edgetpu::EdgeTpuContext* edgetpu_context) {
  auto interpreter = MakeEdgeTpuInterpreterOrDie(model, edgetpu_context);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  CHECK_EQ(input_rnn_indexes.size(), output_rnn_indexes.size());
  for (int i = 0; i < output_rnn_indexes.size(); ++i) {
    // Make sure input and output hidden state tensors are not the same.
    CHECK_NE(interpreter->typed_input_tensor<uint8_t>(input_rnn_indexes[i]),
             interpreter->typed_output_tensor<uint8_t>(input_rnn_indexes[i]));
    // Initialize hidden state with zeros.
    auto input_rnn_tensor = MutableTensorData<uint8_t>(
        *interpreter->input_tensor(input_rnn_indexes[i]));
    std::fill(input_rnn_tensor.begin(), input_rnn_tensor.end(), 0);
  }
  return interpreter;
}

// Runs inference and updates the RNN hidden state input tensors with the
// corresponding output tensors. It is for RNN models with the recurrent link
// cuts after converted to TFLite format.
void RunRnnInference(tflite::Interpreter* interpreter,
                     const std::vector<int>& input_rnn_indexes,
                     const std::vector<int>& output_rnn_indexes) {
  CHECK(interpreter);
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
  CHECK_EQ(input_rnn_indexes.size(), output_rnn_indexes.size());
  for (int i = 0; i < output_rnn_indexes.size(); ++i) {
    auto input_rnn_tensor = MutableTensorData<uint8_t>(
        *interpreter->input_tensor(input_rnn_indexes[i]));
    auto output_rnn_tensor =
        TensorData<uint8_t>(*interpreter->output_tensor(output_rnn_indexes[i]));
    std::copy(output_rnn_tensor.begin(), output_rnn_tensor.end(),
              input_rnn_tensor.begin());
  }
}

// Gets the flatten results of the first `tensor_size` tensors of the
// interpreter.
std::vector<float> GetFlattenResults(const tflite::Interpreter& interpreter,
                                     int tensor_size) {
  std::vector<float> flatten_results;
  for (int i = 0; i < tensor_size; ++i) {
    auto results = TensorData<float>(*interpreter.output_tensor(i));
    flatten_results.insert(flatten_results.end(), results.begin(),
                           results.end());
  }
  return flatten_results;
}

// Run inference with random generated input data.
// It assumes that all RNN input tensors are after all non-RNN input tensors,
// and all RNN output tensors are after all non-RNN output tensors.
std::vector<std::vector<float>> RunInferenceWithModel(
    const std::vector<std::vector<Image>>& input_data,
    std::function<void(tflite::Interpreter&)> run_inference,
    tflite::Interpreter& interpreter, int num_output_tensors) {
  const int num_frames = input_data.size();
  std::vector<std::vector<float>> flatten_raw_results(num_frames);
  for (int frame_i = 0; frame_i < num_frames; ++frame_i) {
    for (int i = 0; i < input_data[frame_i].size(); ++i) {
      auto input_rnn_tensor =
          MutableTensorData<uint8_t>(*interpreter.input_tensor(i));
      std::copy(input_data[frame_i][i].begin(), input_data[frame_i][i].end(),
                input_rnn_tensor.begin());
    }
    run_inference(interpreter);
    flatten_raw_results[frame_i] =
        GetFlattenResults(interpreter, num_output_tensors);
  }
  return flatten_raw_results;
}

class TfliteGraphUtilConcatModelsEdgeTpuTest : public EdgeTpuCacheTestBase {
 protected:
  // This test compares the results of model concatenation with recorded golden
  // results. The test models are real production models.
  void TestConcatModels(
      const std::string& graph0_path, const std::string& graph1_path,
      const std::string& expected_graph,
      const std::vector<std::string>& bypass_output_tensors = {}) {
    auto model0 = tflite::FlatBufferModel::BuildFromFile(
        TestDataPath(graph0_path).c_str());
    ASSERT_TRUE(model0);
    auto model1 = tflite::FlatBufferModel::BuildFromFile(
        TestDataPath(graph1_path).c_str());
    ASSERT_TRUE(model1);

    flatbuffers::FlatBufferBuilder fbb;
    ConcatModels(*model0->GetModel(), *model1->GetModel(), &fbb,
                 bypass_output_tensors);
    auto result_model = LoadModelOrDie(fbb);
    auto result_interpreter =
        MakeEdgeTpuInterpreterOrDie(*result_model, GetTpuContextCache());
    ASSERT_EQ(result_interpreter->inputs().size(), 1);
    ASSERT_EQ(result_interpreter->AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(result_interpreter->input_tensor(0)->type, kTfLiteUInt8);

    auto expected_model = tflite::FlatBufferModel::BuildFromFile(
        TestDataPath(expected_graph).c_str());
    auto expected_interpreter =
        MakeEdgeTpuInterpreterOrDie(*expected_model, GetTpuContextCache());
    ASSERT_EQ(expected_interpreter->inputs().size(), 1);
    ASSERT_EQ(expected_interpreter->AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(expected_interpreter->input_tensor(0)->type, kTfLiteUInt8);

    // We check the model equivalence by running inferences on random input and
    // comparing results. Note that we can not compare the flatbuffer of the
    // result model with the golden model byte-by-byte as flatbuffer
    // serialization may change and/or platform dependent.
    constexpr int kRandomSeed = 12345;
    for (int i = 0; i < 10; ++i) {
      FillRandomInt(
          MutableTensorData<uint8_t>(*result_interpreter->input_tensor(0)),
          kRandomSeed);
      FillRandomInt(
          MutableTensorData<uint8_t>(*expected_interpreter->input_tensor(0)),
          kRandomSeed);
      ASSERT_EQ(result_interpreter->Invoke(), kTfLiteOk);
      ASSERT_EQ(expected_interpreter->Invoke(), kTfLiteOk);
      for (int k = 0; k < result_interpreter->outputs().size(); ++k) {
        ASSERT_EQ(result_interpreter->output_tensor(k)->type,
                  expected_interpreter->output_tensor(k)->type);
        if (result_interpreter->output_tensor(k)->type == kTfLiteUInt8) {
          auto tensor0 =
              TensorData<uint8_t>(*result_interpreter->output_tensor(k));
          auto tensor1 =
              TensorData<uint8_t>(*expected_interpreter->output_tensor(k));
          EXPECT_THAT(tensor0, testing::ContainerEq(tensor1));
        } else if (result_interpreter->output_tensor(k)->type ==
                   kTfLiteFloat32) {
          auto tensor0 =
              TensorData<float>(*result_interpreter->output_tensor(k));
          auto tensor1 =
              TensorData<float>(*expected_interpreter->output_tensor(k));
          EXPECT_THAT(tensor0, testing::ContainerEq(tensor1));
        } else {
          LOG(FATAL) << "Unsupported output tensor type: "
                     << result_interpreter->output_tensor(k)->type;
        }
      }
    }
  }
};

TEST_F(TfliteGraphUtilConcatModelsEdgeTpuTest, ConcatMobilenetClassification) {
  TestConcatModels(
      "tools/mobilenet_quant_v1_224_feature_layers_edgetpu.tflite",
      "tools/mobilenet_quant_v1_224_head_layers.tflite",
      "tools/mobilenet_quant_v1_1.0_224_partial_delegation.tflite");
}

TEST_F(TfliteGraphUtilConcatModelsEdgeTpuTest,
       ConcatMobilenetClassificationWithBypass) {
  // Like ConcatMobilenetClassification but also bypass the AvgPool tensor
  // out to the output.
  TestConcatModels(
      "tools/mobilenet_quant_v1_224_feature_layers_edgetpu.tflite",
      "tools/mobilenet_quant_v1_224_head_layers.tflite",
      "tools/mobilenet_quant_v1_1.0_224_partial_delegation_with_bypass.tflite",
      {"MobilenetV1/Logits/AvgPool_1a/AvgPool"});
}

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class TfliteGraphUtilAppendRecurrentLinksTest : public ModelTestBase {};

TEST_P(TfliteGraphUtilAppendRecurrentLinksTest, EmptyTensorNames) {
  const auto suffix = GetParam();
  // This test flatbuffer model is converted from manually written json file
  // using
  // `flatc`(https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html).
  //
  // From flatbuffer to json:
  // flatc -t --strict-json -o "${output_folder}" \
  //   tensorflow/lite/schema/schema.fbs -- "${tflite_input}"
  // From json to flatbuffer:
  // flatc -o "${output_folder}" --raw-binary --unknown-json \
  //   --allow-non-utf8 -b tensorflow/lite/schema/schema.fbs "${json_input}"
  const std::string input_path(TestDataPath("tools/split_concat" + suffix));
  const std::string output_path("/tmp/tmp_empty_names" + suffix);
  ASSERT_EQ(AppendRecurrentLinks(input_path, /*input_tensor_names=*/{},
                                 /*output_tensor_names=*/{}, output_path),
            absl::OkStatus());
  std::unique_ptr<tflite::FlatBufferModel> model = LoadModelOrDie(output_path);
  std::unique_ptr<tflite::Interpreter> interpreter =
      MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextIfNecessary());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(interpreter->inputs().size(), 3);
  EXPECT_EQ(interpreter->outputs().size(), 5);
}

TEST_P(TfliteGraphUtilAppendRecurrentLinksTest, OnePair) {
  const auto suffix = GetParam();
  const std::string input_path(TestDataPath("tools/split_concat" + suffix));
  const std::string output_path("/tmp/tmp_one_pair" + suffix);
  ASSERT_EQ(AppendRecurrentLinks(input_path,
                                 /*input_tensor_names=*/{"inputs/rnn2"},
                                 /*output_tensor_names=*/{"outputs/rnn2"},
                                 output_path),
            absl::OkStatus());
  std::unique_ptr<tflite::FlatBufferModel> model = LoadModelOrDie(output_path);
  std::unique_ptr<tflite::Interpreter> interpreter =
      MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextIfNecessary());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(interpreter->inputs().size(), 2);
  EXPECT_EQ(interpreter->outputs().size(), 4);
}

TEST_P(TfliteGraphUtilAppendRecurrentLinksTest, SequenceInferenceCorrectness) {
  const auto suffix = GetParam();
  const std::string input_path(TestDataPath("tools/split_concat" + suffix));
  const std::string output_path("/tmp/tmp_two_pairs" + suffix);
  ASSERT_EQ(AppendRecurrentLinks(
                input_path,
                /*input_tensor_names=*/
                {"inputs/rnn1", "inputs/rnn2"},
                /*output_tensor_names=*/{"outputs/rnn1", "outputs/rnn2"},
                output_path),
            absl::OkStatus());

  const int num_input_tensors = 1;
  const int num_output_tensors = 3;
  const std::vector<int> input_rnn_indexes = {1, 2};
  const std::vector<int> output_rnn_indexes = {3, 4};

  // Build the interpreter before surgery.
  auto model_before = LoadModelOrDie(input_path);
  auto interpreter_before = BuildRnnEdgeTpuInterpreter(
      *model_before, input_rnn_indexes, output_rnn_indexes,
      GetTpuContextIfNecessary());

  // Generate random input.
  constexpr int num_frames = 5;
  std::vector<std::vector<Image>> input_data(num_frames);
  for (int frame_i = 0; frame_i < num_frames; ++frame_i) {
    for (int i = 0; i < num_input_tensors; ++i) {
      Image input(TensorSize(*interpreter_before->input_tensor(i)));
      FillRandomInt(input.begin(), input.end());
      input_data[frame_i].push_back(input);
    }
  }

  // Run inference with the interpreter before surgey.
  std::vector<std::vector<float>> flatten_raw_results_before =
      RunInferenceWithModel(
          input_data,
          [&input_rnn_indexes,
           &output_rnn_indexes](tflite::Interpreter& interpreter) {
            RunRnnInference(&interpreter, input_rnn_indexes,
                            output_rnn_indexes);
          },
          *interpreter_before, num_output_tensors);

  // Build the interpreter after surgery.
  auto model_after = LoadModelOrDie(output_path);
  auto interpreter_after =
      MakeEdgeTpuInterpreterOrDie(*model_after, GetTpuContextIfNecessary());
  ASSERT_EQ(interpreter_after->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(interpreter_after->inputs().size(), num_input_tensors);
  ASSERT_EQ(interpreter_after->outputs().size(), num_output_tensors);

  // Run inference with the interpreter after surgey.
  std::vector<std::vector<float>> flatten_raw_results_after =
      RunInferenceWithModel(
          input_data,
          [](tflite::Interpreter& interpreter) { interpreter.Invoke(); },
          *interpreter_after, num_output_tensors);

  EXPECT_EQ(flatten_raw_results_before, flatten_raw_results_after);
}

INSTANTIATE_TEST_CASE_P(EdgeTpu, TfliteGraphUtilAppendRecurrentLinksTest,
                        ::testing::Values("_edgetpu.tflite"));

INSTANTIATE_TEST_CASE_P(Cpu, TfliteGraphUtilAppendRecurrentLinksTest,
                        ::testing::Values(".tflite"));

}  // namespace
}  // namespace coral
