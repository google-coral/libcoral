// Tests correctness of image segmentation models.

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/flags/parse.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {
// Computes the iou of two masks.
float IntersectionOverUnion(const std::vector<uint8_t>& mask1,
                            const std::vector<uint8_t>& mask2) {
  size_t common = 0;
  for (size_t i = 0; i < mask1.size(); ++i) common += (mask1[i] == mask2[i]);
  return static_cast<float>(common) / (mask1.size() + mask2.size() - common);
}

// Computes argmax for input array.
std::vector<float> Argmax(const std::vector<float>& input, size_t size) {
  CHECK_EQ(input.size() % size, 0);
  const auto num_classes = input.size() / size;
  CHECK_GT(num_classes, 1);

  std::vector<float> argmax(size);
  for (size_t i = 0; i < argmax.size(); ++i) {
    auto from = &input[i * num_classes];
    argmax[i] = std::max_element(from, from + num_classes) - from;
  }
  return argmax;
}

// Runs given segmentation model on the given input image and returns the
// segmentation mask. kTfLiteInt64 tensor is directly converted to the mask.
// For kTfLiteUInt8 tensor Argmax is applied before conversion.
std::vector<uint8_t> RunSegmentation(const std::string& model_name,
                                     const std::string& image_name,
                                     size_t mask_size) {
  auto model = LoadModelOrDie(TestDataPath(model_name));
  auto tpu_context =
      ContainsEdgeTpuCustomOp(*model) ? GetEdgeTpuContextOrDie() : nullptr;
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  CopyResizedImage(TestDataPath(image_name), *interpreter->input_tensor(0));
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

  CHECK_EQ(interpreter->outputs().size(), 1);
  const TfLiteTensor& tensor = *interpreter->output_tensor(0);

  std::vector<uint8_t> mask;
  if (tensor.type == kTfLiteUInt8) {
    auto data = Argmax(DequantizeTensor<float>(tensor), mask_size);
    mask.resize(data.size());
    std::copy(data.begin(), data.end(), mask.begin());
  } else if (tensor.type == kTfLiteInt64) {
    // kTfLiteInt64 output tensor type for the models used in this test indicate
    // the direct output from Argmax operator.
    auto data = TensorData<int64_t>(tensor);
    mask.resize(data.size());
    std::copy(data.begin(), data.end(), mask.begin());
  } else {
    LOG(FATAL) << "Unsupported tensor type: " << tensor.type;
  }
  return mask;
}

std::vector<uint8_t> ReadSegmentation(const std::string& image_path) {
  ImageDims dims;
  auto mask = ReadBmp(TestDataPath(image_path), &dims);
  CHECK_EQ(dims.depth, 1);
  // Change 255 to 0 to be consistent with VOC2012 eval protocol.
  std::replace(mask.begin(), mask.end(), 255, 0);
  return mask;
}

TEST(ModelCorrectnessUtilsTest, IntersectionOverUnion) {
  EXPECT_FLOAT_EQ(IntersectionOverUnion({1, 1, 100, 100}, {2, 2, 220, 220}),
                  0.0);
  EXPECT_FLOAT_EQ(IntersectionOverUnion({2, 2, 100, 220}, {2, 2, 220, 220}),
                  0.6);
  EXPECT_FLOAT_EQ(IntersectionOverUnion({1, 1, 100, 100}, {1, 1, 100, 220}),
                  0.6);
  EXPECT_FLOAT_EQ(IntersectionOverUnion({1, 1, 100, 100}, {1, 1, 100, 100}),
                  1.0);
}

TEST(ModelCorrectnessUtilsTest, Argmax) {
  const std::vector<float> input0 = {
      0.4, 0.5, 0.3, 0.1, 0.2,  //  argmax=1
      0.0, 0.0, 1.0, 0.8, 0.4,  //  argmax=2
      0.9, 0.0, 0.1, 0.7, 0.6,  //  argmax=3
      0.3, 0.4, 2.0, 1.5, 1.0   //  argmax=4
  };
  EXPECT_THAT(Argmax(input0, 4),
              ::testing::ContainerEq(std::vector<float>{1.0, 2.0, 0.0, 2.0}));

  // for cases where there are multiple elements with largest value,
  // Argmax return the first (index) argmax
  const std::vector<float> input1 = {
      0.4, 0.5, 0.1, 0.5, 0.5,  //  argmax=1
      0.0, 0.0, 1.0, 1.0, 0.4,  //  argmax=2
      0.9, 0.0, 0.9, 0.7, 0.6,  //  argmax=3
      0.3, 0.4, 2.0, 1.5, 1.0   //  argmax=4
  };
  EXPECT_THAT(Argmax(input1, 4),
              ::testing::ContainerEq(std::vector<float>{1.0, 2.0, 0.0, 2.0}));
}

TEST(ModelCorrectnessTest, Deeplab513Mv2Dm1_WithArgMax) {
  // See label map: test_data/pascal_voc_segmentation_labels.txt
  const auto mask = ReadSegmentation("bird_segmentation_mask.bmp");
  const auto cpu_mask = RunSegmentation("deeplabv3_mnv2_pascal_quant.tflite",
                                        "bird_segmentation.bmp", mask.size());
  const auto edgetpu_mask =
      RunSegmentation("deeplabv3_mnv2_pascal_quant_edgetpu.tflite",
                      "bird_segmentation.bmp", mask.size());

  EXPECT_GT(IntersectionOverUnion(mask, cpu_mask), 0.9);
  EXPECT_GT(IntersectionOverUnion(mask, edgetpu_mask), 0.9);
  EXPECT_GT(IntersectionOverUnion(cpu_mask, edgetpu_mask), 0.99);
}

TEST(ModelCorrectnessTest, Deeplab513Mv2Dm05_WithArgMax) {
  // See label map: test_data/pascal_voc_segmentation_labels.txt
  const auto mask = ReadSegmentation("bird_segmentation_mask.bmp");
  const auto cpu_mask =
      RunSegmentation("deeplabv3_mnv2_dm05_pascal_quant.tflite",
                      "bird_segmentation.bmp", mask.size());
  const auto edgetpu_mask =
      RunSegmentation("deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite",
                      "bird_segmentation.bmp", mask.size());

  EXPECT_GT(IntersectionOverUnion(mask, cpu_mask), 0.9);
  EXPECT_GT(IntersectionOverUnion(mask, edgetpu_mask), 0.9);
  EXPECT_GT(IntersectionOverUnion(cpu_mask, edgetpu_mask), 0.98);
}

// Tests the corretness of an example U-Net model trained following the
// tutorial on https://www.tensorflow.org/tutorials/images/segmentation.
TEST(ModelCorrectnessTest, Keras_PostTrainingQuantization_UNet128MobilenetV2) {
  // The masks are basically labels for each pixel. Each pixel is given one
  // of three categories:
  // Class 1 : Pixel belonging to the pet.
  // Class 2 : Pixel bordering the pet.
  // Class 3 : None of the above/ Surrounding pixel.
  const auto mask = ReadSegmentation("dog_segmentation_mask.bmp");
  const auto cpu_mask =
      RunSegmentation("keras_post_training_unet_mv2_128_quant.tflite",
                      "dog_segmentation.bmp", mask.size());
  const auto edgetpu_mask =
      RunSegmentation("keras_post_training_unet_mv2_128_quant_edgetpu.tflite",
                      "dog_segmentation.bmp", mask.size());

  EXPECT_GT(IntersectionOverUnion(mask, cpu_mask), 0.86);
  EXPECT_GT(IntersectionOverUnion(mask, edgetpu_mask), 0.86);
  EXPECT_GT(IntersectionOverUnion(cpu_mask, edgetpu_mask), 0.97);
}

TEST(ModelCorrectnessTest, Keras_PostTrainingQuantization_UNet256MobilenetV2) {
  const auto mask = ReadSegmentation("dog_segmentation_mask_256.bmp");
  const auto cpu_mask =
      RunSegmentation("keras_post_training_unet_mv2_256_quant.tflite",
                      "dog_segmentation_256.bmp", mask.size());
  const auto edgetpu_mask =
      RunSegmentation("keras_post_training_unet_mv2_256_quant_edgetpu.tflite",
                      "dog_segmentation_256.bmp", mask.size());

  EXPECT_GT(IntersectionOverUnion(mask, cpu_mask), 0.83);
  EXPECT_GT(IntersectionOverUnion(mask, edgetpu_mask), 0.81);
  EXPECT_GT(IntersectionOverUnion(cpu_mask, edgetpu_mask), 0.93);
}

}  // namespace
}  // namespace coral
