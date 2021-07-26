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

#include <chrono>  // NOLINT
#include <random>
#include <set>
#include <thread>  // NOLINT

#include "absl/status/status.h"
#include "coral/classification/adapter.h"
#include "coral/error_reporter.h"
#include "coral/test_utils.h"
#include "gtest/gtest.h"

namespace coral {
namespace {
constexpr int kMobileNet_EgyptianCat = 286;
constexpr int kMobileNet_Chickadee = 20;

constexpr int kInatBird_BlackCappedChickadee = 659;
constexpr int kInatInsect_ThornbushDasher = 912;

void* FakeOpInit(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void FakeOpFree(TfLiteContext* context, void* buffer) {}

TfLiteStatus FakeOpPrepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(input->dims->size);
  for (int i = 0; i < output_size->size; ++i)
    output_size->data[i] = input->dims->data[i];
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus FakeOpEval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteError;
}

constexpr char kFakeOp[] = "fake-op-double";

const TfLiteRegistration kFakeOpRegistration = {FakeOpInit, FakeOpFree,
                                                FakeOpPrepare, FakeOpEval};

TEST(TfLiteUtilsCpuTest, TestRunInferenceFailure_ModelInvokeError) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(kFakeOp, &kFakeOpRegistration);
  EdgeTpuErrorReporter reporter;
  auto model = LoadModelOrDie(
      TestDataPath("invalid_models/model_invoking_error.tflite"));
  auto interpreter =
      MakeEdgeTpuInterpreterOrDie(*model, /*tpu_context=*/nullptr,
                                  /*resolver=*/nullptr,
                                  /*error_reporter=*/&reporter);
  ASSERT_NE(interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(reporter.message(),
            "Node number 0 (fake-op-double) failed to prepare.\n");
}

TEST(TfLiteUtilsCpuTest, InvokeWithMemBuffer) {
  EdgeTpuErrorReporter reporter;
  auto model =
      LoadModelOrDie(TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  auto interpreter =
      MakeEdgeTpuInterpreterOrDie(*model, /*tpu_context=*/nullptr,
                                  /*resolver=*/nullptr,
                                  /*error_reporter=*/&reporter);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  std::vector<uint8_t> buffer;

  buffer.resize(TensorSize(*interpreter->input_tensor(0)) - 1);
  EXPECT_EQ(
      InvokeWithMemBuffer(interpreter.get(), buffer.data(), buffer.size()),
      absl::InternalError("Input buffer (150527) has fewer entries than model "
                          "input tensor (150528)."));

  buffer.resize(TensorSize(*interpreter->input_tensor(0)) + 1);
  EXPECT_EQ(
      InvokeWithMemBuffer(interpreter.get(), buffer.data(), buffer.size()),
      absl::OkStatus());
}

class TfLiteUtilsEdgeTpuTest : public EdgeTpuCacheTestBase {};

TEST_F(TfLiteUtilsEdgeTpuTest, MakeEdgeTpuInterpreter) {
  auto model =
      LoadModelOrDie(TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"));
  std::unique_ptr<tflite::Interpreter> interpreter;
  EXPECT_EQ(MakeEdgeTpuInterpreter(*model, GetTpuContextCache(),
                                   /*resolver=*/nullptr,
                                   /*error_reporter=*/nullptr, &interpreter),
            absl::OkStatus());
}

TEST_F(TfLiteUtilsEdgeTpuTest, ContainsEdgeTpuCustomOp) {
  EXPECT_TRUE(ContainsEdgeTpuCustomOp(*LoadModelOrDie(
      TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"))));
  EXPECT_FALSE(ContainsEdgeTpuCustomOp(
      *LoadModelOrDie(TestDataPath("mobilenet_v1_1.0_224_quant.tflite"))));
}

TEST_F(TfLiteUtilsEdgeTpuTest, MobilenetV1FloatInputs) {
  auto model = LoadModelOrDie(
      TestDataPath("mobilenet_v1_1.0_224_ptq_float_io_legacy_edgetpu.tflite"));
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextCache());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  CopyResizedImage(TestDataPath("cat.bmp"), *interpreter->input_tensor(0));
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  auto top = GetTopClassificationResult(*interpreter);
  EXPECT_EQ(top.id, kMobileNet_EgyptianCat);
  EXPECT_GT(top.score, 0.7);

  CopyResizedImage(TestDataPath("bird.bmp"), *interpreter->input_tensor(0));
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  top = GetTopClassificationResult(*interpreter);
  EXPECT_EQ(top.id, kMobileNet_Chickadee);
  EXPECT_GT(top.score, 0.9);
}

TEST_F(TfLiteUtilsEdgeTpuTest, MobilenetV1WithL2Norm) {
  auto model = LoadModelOrDie(
      TestDataPath("mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite"));
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextCache());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  CopyResizedImage(TestDataPath("cat.bmp"), *interpreter->input_tensor(0));
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  auto top = GetTopClassificationResult(*interpreter);
  EXPECT_EQ(top.id, kMobileNet_EgyptianCat);
  EXPECT_GT(top.score, 0.66);

  CopyResizedImage(TestDataPath("bird.bmp"), *interpreter->input_tensor(0));
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  top = GetTopClassificationResult(*interpreter);
  EXPECT_EQ(top.id, kMobileNet_Chickadee);
  EXPECT_GT(top.score, 0.9);
}

TEST_F(TfLiteUtilsEdgeTpuTest,
       TwoInterpretersSharedEdgeTpuSingleThreadInference) {
  auto model =
      LoadModelOrDie(TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"));

  // When there are multiple interpreters, they will share the Edge TPU context.
  // Ensure they can co-exist.
  auto interpreter1 = MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextCache());
  ASSERT_EQ(interpreter1->AllocateTensors(), kTfLiteOk);
  auto interpreter2 = MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextCache());
  ASSERT_EQ(interpreter2->AllocateTensors(), kTfLiteOk);

  for (int i = 0; i < 10; ++i) {
    for (auto* interpreter : {interpreter1.get(), interpreter2.get()}) {
      CopyResizedImage(TestDataPath("cat.bmp"), *interpreter->input_tensor(0));
      ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

      auto top = GetTopClassificationResult(*interpreter);
      EXPECT_EQ(top.id, kMobileNet_EgyptianCat);
      EXPECT_GT(top.score, 0.7);
    }

    for (auto* interpreter : {interpreter1.get(), interpreter2.get()}) {
      CopyResizedImage(TestDataPath("bird.bmp"), *interpreter->input_tensor(0));
      ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

      auto top = GetTopClassificationResult(*interpreter);
      EXPECT_EQ(top.id, kMobileNet_Chickadee);
      EXPECT_GT(top.score, 0.8);
    }
  }
}

// This test checks that when multiple interpreters in a multi-threaded
// environment, share the same Edge TPU. Each thread can receive correct result
// concurrently.
TEST_F(TfLiteUtilsEdgeTpuTest,
       TwoInterpretersSharedEdgeTpuMultiThreadInference) {
  static constexpr int kNumInferences = 1;

  auto tpu_context = GetTpuContextCache();
  auto model1 = LoadModelOrDie(
      TestDataPath("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"));
  auto model2 = LoadModelOrDie(
      TestDataPath("mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite"));
  const auto bird_image_path = TestDataPath("bird.bmp");
  const auto dragonfly_image_path = TestDataPath("dragonfly.bmp");

  // `job_a` runs iNat_bird model on a bird image. Sleep randomly between 2~20
  // ms after each inference.
  auto job_a = [&model1, &tpu_context, bird_image_path]() {
    const auto tid = std::this_thread::get_id();
    LOG(INFO) << "Thread: " << tid << " created.";

    auto interpreter = MakeEdgeTpuInterpreterOrDie(*model1, tpu_context);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    CopyResizedImage(bird_image_path, *interpreter->input_tensor(0));

    std::mt19937 generator(123456);
    std::uniform_int_distribution<> sleep_time_dist(2, 20);
    for (int i = 0; i < kNumInferences; ++i) {
      ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
      auto top = GetTopClassificationResult(*interpreter);
      EXPECT_EQ(top.id, kInatBird_BlackCappedChickadee);
      EXPECT_GT(top.score, 0.53);

      const auto sleep_time = sleep_time_dist(generator);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
      LOG(INFO) << "Thread: " << tid << " sleep for " << sleep_time << "ms.";
    }
    LOG(INFO) << "Thread: " << tid << " job done.";
  };

  // `job_b` runs iNat_insect model on a dragonfly image. Sleep randomly between
  // 1~10 ms. after each inference.
  auto job_b = [&model2, &tpu_context, dragonfly_image_path]() {
    const auto tid = std::this_thread::get_id();
    LOG(INFO) << "Thread: " << tid << " created.";

    auto interpreter = MakeEdgeTpuInterpreterOrDie(*model2, tpu_context);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    CopyResizedImage(dragonfly_image_path, *interpreter->input_tensor(0));

    std::mt19937 generator(654321);
    std::uniform_int_distribution<> sleep_time_dist(1, 10);
    for (int i = 0; i < kNumInferences; ++i) {
      ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
      auto top = GetTopClassificationResult(*interpreter);
      EXPECT_EQ(top.id, kInatInsect_ThornbushDasher);
      EXPECT_GT(top.score, 0.25);
      const auto sleep_time = sleep_time_dist(generator);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
      LOG(INFO) << "Thread: " << tid << " sleep for " << sleep_time << "ms.";
    }
    LOG(INFO) << "Thread: " << tid << " job done.";
  };

  std::vector<std::thread> threads;
  threads.emplace_back(job_a);
  threads.emplace_back(job_b);
  threads.emplace_back(job_a);
  threads.emplace_back(job_b);

  for (auto& thread : threads) thread.join();
}

TEST_F(TfLiteUtilsEdgeTpuTest, GetEdgetpuContext) {
  ASSERT_TRUE(GetEdgeTpuContext());
  ASSERT_TRUE(GetEdgeTpuContext(/*device=*/""));
  ASSERT_TRUE(GetEdgeTpuContext(/*device=*/":0"));
  std::set<edgetpu::DeviceType> device_types;
  for (const auto& tpu :
       edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu())
    device_types.insert(tpu.type);
  for (const auto device_type : device_types) {
    if (device_type == edgetpu::DeviceType::kApexPci) {
      ASSERT_TRUE(GetEdgeTpuContext(/*device=*/"pci"));
      ASSERT_TRUE(GetEdgeTpuContext(/*device=*/"pci:0"));
      ASSERT_TRUE(GetEdgeTpuContext(
          /*device=*/"pci:0", /*options=*/{{"Performance", "Max"}}));
      ASSERT_TRUE(GetEdgeTpuContext(
          /*device_type=*/edgetpu::DeviceType::kApexPci, /*device_index=*/0,
          /*options=*/{{"Performance", "Max"}}));
    } else if (device_type == edgetpu::DeviceType::kApexUsb) {
      ASSERT_TRUE(GetEdgeTpuContext(/*device=*/"usb"));
      ASSERT_TRUE(GetEdgeTpuContext(/*device=*/"usb:0"));
      ASSERT_TRUE(GetEdgeTpuContext(
          /*device=*/"usb:0", /*options=*/{{"Usb.MaxBulkInQueueLength", "8"}}));
      ASSERT_TRUE(GetEdgeTpuContext(
          /*device_type=*/edgetpu::DeviceType::kApexUsb, /*device_index=*/0,
          /*options=*/{{"Usb.MaxBulkInQueueLength", "8"}}));
    }
  }
}

TEST_F(TfLiteUtilsEdgeTpuTest, InvokeWithMemBufferSuccess) {
  auto model =
      LoadModelOrDie(TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"));
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, GetTpuContextCache());
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  std::vector<uint8_t> input(TensorSize(*interpreter->input_tensor(0)));
  FillRandomInt(input.begin(), input.end());

  EXPECT_EQ(InvokeWithMemBuffer(interpreter.get(), input.data(), input.size()),
            absl::OkStatus());

  std::vector<uint8_t> padded_input(TensorSize(*interpreter->input_tensor(0)) +
                                    1);
  FillRandomInt(padded_input.begin(), padded_input.end());
  auto output = TensorData<uint8_t>(*interpreter->output_tensor(0));
  EXPECT_EQ(InvokeWithMemBuffer(interpreter.get(), padded_input.data(),
                                padded_input.size()),
            absl::OkStatus());
  auto invoke_result = std::vector<uint8_t>(output.begin(), output.end());

  std::copy(padded_input.begin(), padded_input.end(),
            interpreter->typed_input_tensor<uint8_t>(0));
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(invoke_result, std::vector<uint8_t>(output.begin(), output.end()));
}

}  // namespace
}  // namespace coral
