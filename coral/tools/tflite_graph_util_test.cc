#include "coral/tools/tflite_graph_util.h"

#include "coral/test_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

// This test compares the results of model concatenation with recorded golden
// results. The test models are real production models.
void TestConcatModels(
    const std::string& graph0_path, const std::string& graph1_path,
    const std::string& expected_graph,
    const std::vector<std::string>& bypass_output_tensors = {}) {
  auto model0 =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(graph0_path).c_str());
  ASSERT_TRUE(model0);
  auto model1 =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(graph1_path).c_str());
  ASSERT_TRUE(model1);

  flatbuffers::FlatBufferBuilder fbb;
  ConcatModels(*model0->GetModel(), *model1->GetModel(), &fbb,
               bypass_output_tensors);
  auto output_bytes = std::string(
      reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize());

  auto expected = tflite::FlatBufferModel::BuildFromFile(
      TestDataPath(expected_graph).c_str());
  auto expected_bytes =
      std::string(reinterpret_cast<const char*>(expected->allocation()->base()),
                  expected->allocation()->bytes());
  EXPECT_EQ(output_bytes, expected_bytes);
}

TEST(TfliteGraphUtilTest, ConcatMobilenetClassification) {
  TestConcatModels(
      "tools/mobilenet_quant_v1_224_feature_layers-custom_op.tflite",
      "tools/mobilenet_quant_v1_224_head_layers.tflite",
      "tools/mobilenet_quant_v1_1.0_224_partial_delegation.tflite");
}

TEST(TfliteGraphUtilTest, ConcatMobilenetClassificationWithBypass) {
  // Like ConcatMobilenetClassification but also bypass the AvgPool tensor
  // out to the output.
  TestConcatModels(
      "tools/mobilenet_quant_v1_224_feature_layers-custom_op.tflite",
      "tools/mobilenet_quant_v1_224_head_layers.tflite",
      "tools/mobilenet_quant_v1_1.0_224_partial_delegation_with_bypass.tflite",
      {"MobilenetV1/Logits/AvgPool_1a/AvgPool"});
}

TEST(TfliteGraphUtilTest, ConcatMobilenetSSD) {
  TestConcatModels(
      "tools/ssd_mobilenet_v1_coco_quant_postprocess_base_custom_op.tflite",
      "tools/ssd_mobilenet_v1_coco_quant_postprocess_head_layers.tflite",
      "tools/"
      "ssd_mobilenet_v1_coco_quant_postprocess_partial_delegation.tflite");
}

}  // namespace
}  // namespace coral
