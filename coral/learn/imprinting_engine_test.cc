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

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "coral/classification/adapter.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "flatbuffers/flatbuffers.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace coral {

std::unique_ptr<ImprintingEngine> CreateTestEngineFromBuffer(
    const flatbuffers::FlatBufferBuilder& fbb, bool keep_classes) {
  auto model = LoadModelOrDie(fbb);
  return ImprintingEngine::Create(
      ImprintingModel::CreateOrDie(*model->GetModel()), keep_classes);
}

// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class ImprintingEngineTest : public ModelTestBase {
 public:
  static std::unique_ptr<ImprintingEngine> CreateTestEngine(
      const std::string& model_name, bool keep_classes) {
    auto model = tflite::FlatBufferModel::BuildFromFile(
        GenerateInputModelPath(model_name).c_str());
    CHECK(model);
    return ImprintingEngine::Create(
        ImprintingModel::CreateOrDie(*model->GetModel()), keep_classes);
  }

  struct TestDatapoint {
    std::string image;
    const int predicted_class_id;
    const float classification_score;
  };

  struct TrainingDatapoint {
    std::vector<std::string> images;
    const int groundtruth_class_id;
  };

  static std::string ImagePath(const std::string& file_name) {
    return TestDataPath(absl::StrCat("imprinting/", file_name));
  }

  static std::string GenerateInputModelPath(const std::string& file_name) {
    return TestDataPath(file_name + GetParam());
  }

  // Checks that last 4 operators are Conv2d, Mul, Reshape, Softmax.
  void CheckRetrainedLayers(const std::string& output_file_path) {
    auto model =
        tflite::FlatBufferModel::BuildFromFile(output_file_path.c_str());
    ASSERT_TRUE(model != nullptr);
    const auto model_t = absl::WrapUnique(model->GetModel()->UnPack());

    auto get_builtin_opcode = [](const tflite::ModelT* model_t, int op_index) {
      auto& op = model_t->subgraphs[0]->operators[op_index];
      auto& opcodes = model_t->operator_codes;
      return tflite::GetBuiltinCode(opcodes[op->opcode_index].get());
    };

    VLOG(1) << "# of operators in graph: "
            << model_t->subgraphs[0]->operators.size();

    CHECK_GE(model_t->subgraphs[0]->operators.size(), 5);
    const int last_op_index = model_t->subgraphs[0]->operators.size() - 1;
    CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index),
             tflite::BuiltinOperator_SOFTMAX);
    CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index - 1),
             tflite::BuiltinOperator_RESHAPE);
    CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index - 2),
             tflite::BuiltinOperator_MUL);
    CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index - 3),
             tflite::BuiltinOperator_CONV_2D);
  }

  void TestTrainedModel(const flatbuffers::FlatBufferBuilder& fbb,
                        const std::vector<TestDatapoint>& test_datapoints) {
    auto model = LoadModelOrDie(fbb);
    auto classifier =
        MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextIfNecessary());
    CHECK_EQ(classifier->AllocateTensors(), kTfLiteOk);

    for (const auto& test_datapoint : test_datapoints) {
      CopyResizedImage(ImagePath(test_datapoint.image),
                       *classifier->input_tensor(0));
      CHECK_EQ(classifier->Invoke(), kTfLiteOk);
      auto top = GetTopClassificationResult(*classifier);
      EXPECT_EQ(top.id, test_datapoint.predicted_class_id);
      EXPECT_GT(top.score, test_datapoint.classification_score);
    }
  }

  static absl::Status Train(ImprintingEngine* engine,
                            const std::vector<TrainingDatapoint>& points) {
    auto buffer = engine->ExtractorModelBuffer();

    auto model = tflite::FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(buffer.data()), buffer.size());
    auto extractor =
        MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextIfNecessary());
    CHECK_EQ(extractor->AllocateTensors(), kTfLiteOk);

    for (auto& point : points) {
      for (auto& image : point.images) {
        CopyResizedImage(ImagePath(image), *extractor->input_tensor(0));
        CHECK_EQ(extractor->Invoke(), kTfLiteOk);
        auto embedding = DequantizeTensor<float>(*extractor->output_tensor(0));
        auto status = engine->Train(embedding, point.groundtruth_class_id);
        if (!status.ok()) return status;
      }
    }
    return absl::OkStatus();
  }
};

TEST_P(ImprintingEngineTest, TestKeepClasses) {
  EXPECT_EQ(CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                             /*keep_classes=*/false)
                ->GetClasses()
                .size(),
            0);

  EXPECT_EQ(CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                             /*keep_classes=*/true)
                ->GetClasses()
                .size(),
            1001);
}

TEST_P(ImprintingEngineTest, TestKeepClassesForTrainedModel) {
  flatbuffers::FlatBufferBuilder fbb;
  {
    auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                   /*keep_classes=*/false);
    EXPECT_TRUE(engine->GetClasses().empty());
    EXPECT_EQ(Train(engine.get(),
                    {{{"cat_train_0.bmp"}, 0},
                     {{"hotdog_train_0.bmp", "hotdog_train_0.bmp"}, 1}}),
              absl::OkStatus());
    ASSERT_EQ(engine->SerializeModel(&fbb), absl::OkStatus());
  }

  EXPECT_EQ(CreateTestEngineFromBuffer(fbb, /*keep_classes=*/false)
                ->GetClasses()
                .size(),
            0);

  EXPECT_EQ(CreateTestEngineFromBuffer(fbb, /*keep_classes=*/true)
                ->GetClasses()
                .size(),
            2);
}

TEST_P(ImprintingEngineTest, TestModelWithoutL2NormLayer) {
  auto model = tflite::FlatBufferModel::BuildFromFile(
      GenerateInputModelPath("mobilenet_v1_1.0_224_quant").c_str());
  ASSERT_TRUE(model);

  std::unique_ptr<ImprintingModel> imprinting_model;
  EXPECT_EQ(
      ImprintingModel::Create(*model->GetModel(), &imprinting_model),
      absl::InternalError("Unsupported model architecture. Input model must "
                          "have an L2Norm layer."));
}

TEST_P(ImprintingEngineTest, TestModelWithoutTraining) {
  auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                 /*keep_classes=*/false);

  flatbuffers::FlatBufferBuilder fbb;
  EXPECT_EQ(engine->SerializeModel(&fbb),
            absl::InternalError("Model is not trained."));
}

TEST_P(ImprintingEngineTest, TestTrainingIndexTooLarge) {
  auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                 /*keep_classes=*/true);

  EXPECT_EQ(
      Train(engine.get(), {{{"cat_train_0.bmp"}, 1002}}),
      absl::InternalError("The class index of a new category is too large!"));
}

TEST_P(ImprintingEngineTest, TestTrainingChangeBaseModelClasses) {
  auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                 /*keep_classes=*/true);

  EXPECT_EQ(Train(engine.get(), {{{"cat_train_0.bmp"}, 100}}),
            absl::InternalError("Cannot change the base model classes not "
                                "trained with imprinting method!"));
}

TEST_P(ImprintingEngineTest,
       TrainWithMobileNetV1L2NormAndRealImagesNotKeepClasses) {
  flatbuffers::FlatBufferBuilder fbb;
  {
    auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                   /*keep_classes=*/false);

    ASSERT_EQ(
        Train(engine.get(), {{{"cat_train_0.bmp"}, 0},
                             {{"hotdog_train_0.bmp", "hotdog_train_1.bmp"}, 1},
                             {{"dog_train_0.bmp"}, 2}}),
        absl::OkStatus());
    ASSERT_EQ(engine->SerializeModel(&fbb), absl::OkStatus());
  }

  TestTrainedModel(fbb, {{"cat_train_0.bmp", 0, 0.99f},
                         {"hotdog_train_0.bmp", 1, 0.99f},
                         {"dog_train_0.bmp", 2, 0.99f},
                         {"cat_test_0.bmp", 0, 0.99f},
                         {"hotdog_test_0.bmp", 1, 0.99f},
                         {"dog_test_0.bmp", 2, 0.99f}});
}

// This test should perform almost the same with
// TrainWithMobileNetV1L2NormAndRealImagesNotKeepClasses.
TEST_P(ImprintingEngineTest, TrainWithMobileNetV1L2NormAndRealImagesTraining) {
  flatbuffers::FlatBufferBuilder fbb1;
  {
    auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                   /*keep_classes=*/false);
    ASSERT_EQ(Train(engine.get(), {{{"cat_train_0.bmp"}, 0},  //
                                   {{"hotdog_train_0.bmp"}, 1}}),
              absl::OkStatus());
    ASSERT_EQ(engine->SerializeModel(&fbb1), absl::OkStatus());
  }

  flatbuffers::FlatBufferBuilder fbb2;
  {
    auto engine = CreateTestEngineFromBuffer(fbb1, /*keep_classes=*/true);
    ASSERT_EQ(Train(engine.get(),
                    {{{"hotdog_train_1.bmp"}, 1}, {{"dog_train_0.bmp"}, 2}}),
              absl::OkStatus());
    ASSERT_EQ(engine->SerializeModel(&fbb2), absl::OkStatus());
  }

  TestTrainedModel(fbb2, {{"cat_train_0.bmp", 0, 0.99f},
                          {"hotdog_train_0.bmp", 1, 0.99f},
                          {"dog_train_0.bmp", 2, 0.99f},
                          {"cat_test_0.bmp", 0, 0.99f},
                          {"hotdog_test_0.bmp", 1, 0.99f},
                          {"dog_test_0.bmp", 2, 0.99f}});
}

TEST_P(ImprintingEngineTest,
       TrainWithMobileNetV1L2NormAndRealImagesKeepClasses) {
  flatbuffers::FlatBufferBuilder fbb;
  {
    auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                   /*keep_classes=*/true);

    ASSERT_EQ(Train(engine.get(),
                    {{{"cat_train_0.bmp"}, 1001},
                     {{"hotdog_train_0.bmp", "hotdog_train_1.bmp"}, 1002},
                     {{"dog_train_0.bmp"}, 1003}}),
              absl::OkStatus());
    ASSERT_EQ(engine->SerializeModel(&fbb), absl::OkStatus());
  }

  TestTrainedModel(fbb, {{"cat_train_0.bmp", 1001, 0.99f},
                         {"hotdog_train_0.bmp", 1002, 0.99f},
                         {"dog_train_0.bmp", 1003, 0.99f},
                         {"cat_test_0.bmp", 1001, 0.99f},
                         {"hotdog_test_0.bmp", 1002, 0.91f},
                         // 203 soft-coated wheaten terrier
                         {"dog_test_0.bmp", 203, 0.6f}});
}

TEST_P(ImprintingEngineTest,
       TrainImprintedRetrainedMobileNetV1L2NormWithRealImagesNotKeepClasses) {
  flatbuffers::FlatBufferBuilder fbb1;
  {
    auto engine = CreateTestEngine("mobilenet_v1_1.0_224_l2norm_quant",
                                   /*keep_classes=*/false);
    ASSERT_EQ(Train(engine.get(), {{{"cat_train_0.bmp"}, 0},  //
                                   {{"hotdog_train_0.bmp"}, 1}}),
              absl::OkStatus());
    ASSERT_EQ(engine->SerializeModel(&fbb1), absl::OkStatus());
  }

  flatbuffers::FlatBufferBuilder fbb2;
  {
    auto engine = CreateTestEngineFromBuffer(fbb1,
                                             /*keep_classes=*/false);
    ASSERT_EQ(Train(engine.get(), {{{"hotdog_train_0.bmp"}, 0},  //
                                   {{"dog_train_0.bmp"}, 1}}),
              absl::OkStatus());
    ASSERT_EQ(engine->SerializeModel(&fbb2), absl::OkStatus());
  }

  TestTrainedModel(fbb2, {{"hotdog_train_0.bmp", 0, 0.99f},
                          {"dog_train_0.bmp", 1, 0.99f},
                          {"hotdog_test_0.bmp", 0, 0.99f},
                          {"dog_test_0.bmp", 1, 0.99f}});
}

INSTANTIATE_TEST_CASE_P(Cpu, ImprintingEngineTest,
                        ::testing::Values(".tflite"));

INSTANTIATE_TEST_CASE_P(EdgeTpu, ImprintingEngineTest,
                        ::testing::Values("_edgetpu.tflite"));

}  // namespace coral
