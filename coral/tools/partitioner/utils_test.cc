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

#include "coral/tools/partitioner/utils.h"

#include <memory>
#include <random>

#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "coral/test_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace {

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel* model) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(model->GetModel(), resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk) {
    LOG(FATAL) << "Error in interpreter initialization.";
    return nullptr;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors.";
    return nullptr;
  }

  return interpreter;
}

void ExtractModelSegmentAndVerify(
    const std::string& src_model_filepath, const std::vector<int>& target_ops,
    const std::vector<std::string>& target_inputs_str,
    const std::vector<std::string>& target_outputs_str) {
  std::vector<char> src_model_content;
  ReadFileOrExit(src_model_filepath, &src_model_content);
  const tflite::Model* src_model = tflite::GetModel(src_model_content.data());

  LOG(INFO) << "Converting tensor names to tensor indices...";
  const auto& name_to_index_map = BuildTensorNameToIndexMap(*src_model);
  std::vector<int> target_inputs(target_inputs_str.size());
  for (int i = 0; i < target_inputs_str.size(); ++i) {
    target_inputs[i] = name_to_index_map.at(target_inputs_str[i]);
  }
  std::vector<int> target_outputs(target_outputs_str.size());
  for (int i = 0; i < target_outputs_str.size(); ++i) {
    target_outputs[i] = name_to_index_map.at(target_outputs_str[i]);
  }

  LOG(INFO) << "Extracting subgraph...";
  const auto& segment_content = ExtractModelSegment(
      *src_model, target_ops, target_inputs, target_outputs);

  // Modify src_model's output tensor fields to let it keep some tensors.
  auto extended_model_t = absl::WrapUnique(src_model->UnPack());
  for (const auto& target_input : target_inputs) {
    extended_model_t->subgraphs[0]->outputs.push_back(target_input);
  }
  for (const auto& target_output : target_outputs) {
    extended_model_t->subgraphs[0]->outputs.push_back(target_output);
  }
  auto fbb = absl::make_unique<flatbuffers::FlatBufferBuilder>();
  tflite::FinishModelBuffer(*fbb,
                            tflite::Model::Pack(*fbb, extended_model_t.get()));

  LOG(INFO) << "Run inference with src_model...";
  auto flatbuffer_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(fbb->GetBufferPointer()), fbb->GetSize());
  auto interpreter = CreateInterpreter(flatbuffer_model.get());
  // Fill input tensor with random value.
  std::default_random_engine generator(12345);
  std::uniform_int_distribution<> distribution(0, UINT8_MAX);
  for (int i = 0; i < interpreter->inputs().size(); ++i) {
    auto* tensor = interpreter->input_tensor(i);
    auto* tensor_buffer = reinterpret_cast<uint8_t*>(tensor->data.data);
    std::generate(
        tensor_buffer, tensor_buffer + tensor->bytes,
        [&generator, &distribution]() { return distribution(generator); });
  }
  CHECK(interpreter->Invoke() == kTfLiteOk);

  LOG(INFO) << "Run inference with segment...";
  auto segment_flatbuffer_model = tflite::FlatBufferModel::BuildFromBuffer(
      segment_content.data(), segment_content.size());
  auto segment_interpreter = CreateInterpreter(segment_flatbuffer_model.get());
  // Set `subgraph_interpreter` input based on `interpreter` tensors.
  for (int i = 0; i < target_inputs.size(); ++i) {
    const auto* src_tensor = interpreter->tensor(target_inputs[i]);
    auto* segment_tensor = segment_interpreter->input_tensor(i);
    CHECK_EQ(std::strcmp(segment_tensor->name, src_tensor->name), 0);
    std::memcpy(segment_tensor->data.data, src_tensor->data.data,
                src_tensor->bytes);
  }
  CHECK(segment_interpreter->Invoke() == kTfLiteOk);

  LOG(INFO) << "Checking result...";
  for (int i = 0; i < target_outputs.size(); ++i) {
    const auto* src_tensor = interpreter->tensor(target_outputs[i]);
    const auto* segment_tensor = segment_interpreter->output_tensor(i);
    CHECK_EQ(std::strcmp(segment_tensor->name, src_tensor->name), 0);
    ASSERT_EQ(segment_tensor->bytes, src_tensor->bytes);
    const auto* src_data =
        reinterpret_cast<const uint8_t*>(src_tensor->data.data);
    const auto* segment_data =
        reinterpret_cast<const uint8_t*>(segment_tensor->data.data);
    for (int j = 0; j < src_tensor->bytes; ++j) {
      EXPECT_EQ(segment_data[j], src_data[j]);
    }
  }
}

TEST(Utils, BuildEdgeList) {
  //       | t0
  //       0
  // t1 /  |  \ t3
  //   v   |t2 v
  //   1   |   2
  // t4 \  |  / t5
  //     v v v
  //       3
  //       | t6
  const int num_tensors = 7;
  std::vector<std::unique_ptr<tflite::TensorT>> tensors;
  tensors.reserve(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    auto tmp_tensor = absl::make_unique<tflite::TensorT>();
    tmp_tensor->name = absl::StrCat("tensor-", i);
    tensors.push_back(std::move(tmp_tensor));
  }

  const int num_ops = 4;
  std::vector<std::unique_ptr<tflite::OperatorT>> ops;
  ops.reserve(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    auto tmp_op = absl::make_unique<tflite::OperatorT>();
    ops.push_back(std::move(tmp_op));
  }

  // Specify connections
  ops[0]->inputs.push_back(0);
  ops[0]->outputs.push_back(1);
  ops[0]->outputs.push_back(2);
  ops[0]->outputs.push_back(3);

  ops[1]->inputs.push_back(1);
  ops[1]->outputs.push_back(4);

  ops[2]->inputs.push_back(3);
  ops[2]->outputs.push_back(5);

  ops[3]->inputs.push_back(2);
  ops[3]->inputs.push_back(4);
  ops[3]->inputs.push_back(5);
  ops[3]->outputs.push_back(6);

  auto subgraph_t = absl::make_unique<tflite::SubGraphT>();
  subgraph_t->operators = std::move(ops);
  subgraph_t->tensors = std::move(tensors);

  auto model_t = absl::make_unique<tflite::ModelT>();
  model_t->subgraphs.push_back(std::move(subgraph_t));

  // Serialize into new flatbuffer.
  auto fbb = absl::make_unique<flatbuffers::FlatBufferBuilder>();
  tflite::FinishModelBuffer(*fbb, tflite::Model::Pack(*fbb, model_t.get()));
  auto* model = tflite::GetModel(fbb->GetBufferPointer());

  const auto& edges = BuildEdgeList(*model);
  EXPECT_EQ(edges.size(), 5);
  const std::vector<Edge> expected_edges = {
      {0, 1}, {0, 2}, {0, 3}, {1, 3}, {2, 3},
  };
  EXPECT_THAT(edges, testing::UnorderedElementsAreArray(expected_edges));
}

TEST(Utils, BuildGraph) {
  //      0
  //    / | \
  //   v  |  v
  //   1  |  2
  //    \ | /
  //     vvv
  //      3
  std::vector<Edge> edges = {
      {0, 1}, {0, 2}, {0, 3}, {1, 3}, {2, 3},
  };
  constexpr int num_nodes = 4;
  const auto& graph = BuildGraph(edges, num_nodes);
  const Graph expected_graph = {
      {2, 1, 3},
      {3},
      {3},
      {},
  };
  ASSERT_EQ(graph.size(), num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    EXPECT_THAT(graph[i],
                testing::UnorderedElementsAreArray(expected_graph[i]));
  }
}

TEST(Utils, BuildReverseGraph) {
  //      0
  //    / | \
  //   v  |  v
  //   1  |  2
  //    \ | /
  //     vvv
  //      3
  std::vector<Edge> edges = {
      {0, 1}, {0, 2}, {0, 3}, {1, 3}, {2, 3},
  };
  constexpr int num_nodes = 4;
  const auto& graph = BuildGraph(edges, num_nodes);
  const auto& reverse_graph = BuildReverseGraph(graph);
  const Graph expected_graph = {
      {},
      {0},
      {0},
      {1, 2, 0},
  };
  ASSERT_EQ(reverse_graph.size(), num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    EXPECT_THAT(reverse_graph[i],
                testing::UnorderedElementsAreArray(expected_graph[i]));
  }
}

TEST(Utils, CalculateInAndOutDegree) {
  //      0
  //    /   \
  //   v     v
  //   1     2
  //   |     |
  //   v     v
  //   4     5
  //    \   /
  //     v v
  //      3
  const Graph graph = {
      {2, 1}, {4}, {5}, {}, {3}, {3},
  };
  EXPECT_THAT(CalculateInDegree(graph), testing::ElementsAre(0, 1, 1, 2, 1, 1));
  EXPECT_THAT(CalculateOutDegree(graph),
              testing::ElementsAre(2, 1, 1, 0, 1, 1));
}

TEST(Utils, TopologicalSort) {
  //      0
  //    / | \
  //   v  |  v
  //   1  |  2   4
  //    \ |   \  /
  //     vv    vv
  //      3     5
  const Graph graph = {
      {2, 1, 3}, {3}, {5}, {}, {5}, {},
  };
  const auto& result = TopologicalSort(graph);
  const std::vector<int> expected = {4, 0, 1, 2, 3, 5};
  EXPECT_THAT(result, testing::ContainerEq(expected));
}

TEST(Utils, BuildTensorNameToIndexMap) {
  const std::string model_filepath =
      TestDataPath("inception_v3_299_quant.tflite");
  std::vector<char> model_content;
  ReadFileOrExit(model_filepath, &model_content);
  const tflite::Model* model = tflite::GetModel(model_content.data());
  const auto& result = BuildTensorNameToIndexMap(*model);

  const auto* tensors = model->subgraphs()->Get(0)->tensors();
  ASSERT_EQ(result.size(), tensors->size());
  for (const auto& pair : result) {
    const auto& name = pair.first;
    const auto& index = pair.second;
    EXPECT_EQ(tensors->Get(index)->name()->str(), name);
  }
}

TEST(Utils, ExtractModelSegment_TopSegment) {
  const std::string src_model_filepath =
      TestDataPath("inception_v3_299_quant.tflite");

  // These are hand-picked parameters to make sure the extracted segment is a
  // valid subgraph (i.e., nodes are all connected, with inputs/outputs tensors
  // well-defined).
  const std::vector<int> target_ops = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11};
  const std::vector<std::string> target_inputs_str = {
      "input",
  };
  const std::vector<std::string> target_outputs_str = {
      "InceptionV3/InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/"
      "Relu;InceptionV3/InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/"
      "add_fold;InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/"
      "Conv2D_Fold;InceptionV3/InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/"
      "Conv2D_Fold;InceptionV3/InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/"
      "BatchNorm_Fold/bias",
      "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/"
      "Relu;InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/"
      "add_fold;InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/"
      "Conv2D_Fold;InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/"
      "Conv2D_Fold;InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/"
      "BatchNorm_Fold/bias",
  };

  ExtractModelSegmentAndVerify(src_model_filepath, target_ops,
                               target_inputs_str, target_outputs_str);
}

TEST(Utils, ExtractModelSegment_MiddleSegment) {
  const std::string src_model_filepath =
      TestDataPath("inception_v4_299_quant.tflite");

  // These are hand-picked parameters to make sure the extracted segment is a
  // valid subgraph (i.e., nodes are all connected, with inputs/outputs tensors
  // well-defined).
  const std::vector<int> target_ops = {5, 6, 7, 8, 9};
  const std::vector<std::string> target_inputs_str = {
      "InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu;InceptionV4/InceptionV4/"
      "Conv2d_2b_3x3/add_fold;InceptionV4/InceptionV4/Mixed_5e/Branch_2/"
      "Conv2d_0a_1x1/Conv2D_Fold;InceptionV4/InceptionV4/Conv2d_2b_3x3/"
      "Conv2D_Fold;InceptionV4/InceptionV4/Conv2d_2b_3x3/BatchNorm_Fold/bias",
      "InceptionV4/InceptionV4/Mixed_3a/Branch_0/MaxPool_0a_3x3/MaxPool1",
  };
  const std::vector<std::string> target_outputs_str = {
      "InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_1a_3x3/"
      "Relu;InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_1a_3x3/"
      "add_fold;InceptionV4/InceptionV4/Mixed_5e/Branch_3/Conv2d_0b_1x1/"
      "Conv2D_Fold;InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_1a_3x3/"
      "Conv2D_Fold;InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_1a_3x3/"
      "BatchNorm_Fold/bias",
      "InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0a_1x1/"
      "Relu;InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0a_1x1/"
      "add_fold;InceptionV4/InceptionV4/Mixed_5e/Branch_2/Conv2d_0a_1x1/"
      "Conv2D_Fold;InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0a_1x1/"
      "Conv2D_Fold;InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0a_1x1/"
      "BatchNorm_Fold/bias",
  };

  ExtractModelSegmentAndVerify(src_model_filepath, target_ops,
                               target_inputs_str, target_outputs_str);
}

TEST(Utils, ExtractModelSegment_BottomSegment) {
  const std::string src_model_filepath =
      TestDataPath("ssd_mobilenet_v2_coco_quant_postprocess.tflite");

  // These are hand-picked parameters to make sure the extracted segment is a
  // valid subgraph (i.e., nodes are all connected, with inputs/outputs tensors
  // well-defined).
  const std::vector<int> target_ops = {101, 102, 103, 107, 108, 109, 110};
  const std::vector<std::string> target_inputs_str = {
      "BoxPredictor_0/Reshape;BoxPredictor_0/BoxEncodingPredictor/act_quant/"
      "FakeQuantWithMinMaxVars",
      "BoxPredictor_1/Reshape;BoxPredictor_1/BoxEncodingPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_2/Reshape;BoxPredictor_2/BoxEncodingPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_3/Reshape;BoxPredictor_3/BoxEncodingPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_4/Reshape;BoxPredictor_4/BoxEncodingPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_5/Reshape;BoxPredictor_5/BoxEncodingPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_0/Reshape_1;BoxPredictor_0/ClassPredictor/act_quant/"
      "FakeQuantWithMinMaxVars",
      "BoxPredictor_1/Reshape_1;BoxPredictor_1/ClassPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_2/Reshape_1;BoxPredictor_2/ClassPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_3/Reshape_1;BoxPredictor_3/ClassPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_4/Reshape_1;BoxPredictor_4/ClassPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
      "BoxPredictor_5/Reshape_1;BoxPredictor_5/ClassPredictor/act_quant/"
      "FakeQuantWithMinMaxVars1",
  };
  const std::vector<std::string> target_outputs_str = {
      "TFLite_Detection_PostProcess",
      "TFLite_Detection_PostProcess:1",
      "TFLite_Detection_PostProcess:2",
      "TFLite_Detection_PostProcess:3",
  };

  ExtractModelSegmentAndVerify(src_model_filepath, target_ops,
                               target_inputs_str, target_outputs_str);
}

//          0
//      /        \
//      1        2
//      |        |
//      3        4
//      \        /
//          5
// With num_nodes_per_subgraph = {2, 1, 3}
//
// This is a hard case, usually all of subgraph[i]'s output tensors can be
// consumed by subgraph[i+1]. But with this setting:
//
// The first subgraph will have 2 output tensors, while the second subgraph will
// only have 1 input tensors regardless the implementation of topological sort.
TEST(Utils, LocateSubgraphNodes_HardCase_SingleOpSubgraph) {
  const std::vector<Edge> edge_list = {
      {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 5},
  };
  const int num_nodes = 6;
  const std::vector<int> num_nodes_per_subgraph = {2, 1, 3};

  const auto& subgraph_nodes_list = LocateSubgraphNodes(
      edge_list, num_nodes, {0, 2, 4, 1, 3, 5}, num_nodes_per_subgraph);

  std::vector<std::vector<int>> expected_all_nodes = {
      {0, 2},
      {4},
      {1, 3, 5},
  };

  ASSERT_EQ(subgraph_nodes_list.size(), 3);
  EXPECT_EQ(subgraph_nodes_list[0].input_nodes.size(), 1);
  EXPECT_EQ(*subgraph_nodes_list[0].input_nodes.begin(), kGraphInputGenNode);
  EXPECT_EQ(subgraph_nodes_list[0].output_nodes.size(), 2);
  EXPECT_THAT(subgraph_nodes_list[0].all_nodes,
              testing::ContainerEq(expected_all_nodes[0]));

  EXPECT_EQ(subgraph_nodes_list[1].input_nodes.size(), 1);
  EXPECT_EQ(subgraph_nodes_list[1].output_nodes.size(), 1);
  EXPECT_THAT(subgraph_nodes_list[1].all_nodes,
              testing::ContainerEq(expected_all_nodes[1]));

  EXPECT_EQ(subgraph_nodes_list[2].input_nodes.size(), 2);
  EXPECT_EQ(subgraph_nodes_list[2].output_nodes.size(), 1);
  EXPECT_THAT(subgraph_nodes_list[2].all_nodes,
              testing::ContainerEq(expected_all_nodes[2]));
}

//      0
//      |
//      1
//    / | \
//   2  3  4
//    \ | /
//      5
// With num_nodes_per_subgraph = {2, 4}
TEST(Utils, LocateSubgraphNodes_MultipleChildOps) {
  const std::vector<Edge> edge_list = {
      {0, 1}, {1, 2}, {1, 3}, {1, 4}, {2, 5}, {3, 5}, {4, 5},
  };
  const int num_nodes = 6;
  const std::vector<int> num_nodes_per_subgraph = {2, 4};

  const auto& subgraph_nodes_list = LocateSubgraphNodes(
      edge_list, num_nodes, {0, 1, 4, 3, 2, 5}, num_nodes_per_subgraph);

  std::vector<std::vector<int>> expected_all_nodes = {
      {0, 1},
      {4, 3, 2, 5},
  };

  ASSERT_EQ(subgraph_nodes_list.size(), 2);
  EXPECT_EQ(subgraph_nodes_list[0].input_nodes.size(), 1);
  EXPECT_EQ(*subgraph_nodes_list[0].input_nodes.begin(), kGraphInputGenNode);
  EXPECT_EQ(subgraph_nodes_list[0].output_nodes.size(), 1);
  EXPECT_THAT(subgraph_nodes_list[0].all_nodes,
              testing::ContainerEq(expected_all_nodes[0]));

  EXPECT_EQ(subgraph_nodes_list[1].input_nodes.size(), 1);
  EXPECT_EQ(subgraph_nodes_list[1].output_nodes.size(), 1);
  EXPECT_THAT(subgraph_nodes_list[1].all_nodes,
              testing::ContainerEq(expected_all_nodes[1]));
}
}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
