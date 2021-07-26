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

#include "coral/test_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "coral/classification/adapter.h"
#include "coral/detection/adapter.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tflite/public/edgetpu.h"

ABSL_FLAG(std::string, test_data_dir, "test_data", "Test data directory");

ABSL_FLAG(std::string, tpu_device, "",
          "Edge TPU device specification string, e.g. usb:0, pci:1."
          "For multiple TPU tests, this should only specify device type, which "
          "can be empty, 'pci', or 'usb'.");

ABSL_FLAG(bool, disable_tpu_context_cache, false,
          "Whether to disable TPU context cache for EdgeTpuCacheTestBase");

namespace coral {
namespace {

constexpr size_t kBmpFileHeaderSize = 14;
constexpr size_t kBmpInfoHeaderSize = 40;
constexpr size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;

int32_t ToInt32(const uint8_t p[4]) {
  return (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
}

template <typename SrcType, typename DstType>
DstType saturate_cast(SrcType val) {
  if (val > static_cast<SrcType>(std::numeric_limits<DstType>::max())) {
    return std::numeric_limits<DstType>::max();
  }
  if (val < static_cast<SrcType>(std::numeric_limits<DstType>::lowest())) {
    return std::numeric_limits<DstType>::lowest();
  }
  return static_cast<DstType>(val);
}

inline edgetpu::DeviceType StringToDeviceType(
    const std::string& device_type_name) {
  if (device_type_name == "pci") {
    return edgetpu::DeviceType::kApexPci;
  } else {
    CHECK_EQ(device_type_name, "usb");
    return edgetpu::DeviceType::kApexUsb;
  }
}

// Converts RGB image to grayscale. Take the average.
std::vector<uint8_t> RgbToGrayscale(const std::vector<uint8_t>& in,
                                    const ImageDims& dims) {
  CHECK(dims.depth == 3 || dims.depth == 4);
  std::vector<uint8_t> out(dims.height * dims.width);
  for (int i = 0, j = 0; i < in.size(); i += dims.depth, j += 1)
    out[j] = static_cast<uint8_t>((in[i] + in[i + 1] + in[i + 2]) / 3);
  return out;
}

inline std::string Basename(const std::string& fullpath) {
  const std::size_t found = fullpath.find_last_of("/\\");
  return found == std::string::npos ? fullpath : fullpath.substr(found + 1);
}

}  // namespace

std::string TestDataPath(const std::string& name) {
  return absl::StrCat(absl::GetFlag(FLAGS_test_data_dir), "/", name);
}

bool TopKContains(const std::vector<Class>& topk, int label) {
  for (const auto& entry : topk) {
    if (entry.id == label) return true;
  }
  LOG(ERROR) << "Top K results do not contain " << label;
  for (const auto& p : topk) {
    LOG(ERROR) << p.id << ", " << p.score;
  }
  return false;
}

void TestDetection(const std::string& model_path, const std::string& image_path,
                   const BBox<float>& expected_box, int expected_label,
                   float score_threshold, float iou_threshold,
                   edgetpu::EdgeTpuContext* tpu_context) {
  LOG(INFO) << "Testing model: " << model_path;

  auto model = LoadModelOrDie(TestDataPath(model_path));
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context);

  // Additional sanity check to make sure SSD models have only one
  // postprocessing op.
  int num_pp_count = 0;
  for (int i = 0; i < interpreter->nodes_size(); ++i) {
    const char* custom_name =
        interpreter->node_and_registration(i)->second.custom_name;
    if (custom_name &&
        std::strcmp(custom_name, "TFLite_Detection_PostProcess") == 0) {
      ++num_pp_count;
    }
  }
  ASSERT_EQ(num_pp_count, 1)
      << "SSD model must have only one postprocessing operator: " << model_path;

  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  CopyResizedImage(TestDataPath(image_path), *interpreter->input_tensor(0));
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
  auto results =
      GetDetectionResults(*interpreter, score_threshold, /*top_k=*/1);
  ASSERT_EQ(results.size(), 1);
  const auto& result = results[0];
  EXPECT_EQ(result.id, expected_label);
  EXPECT_GT(result.score, score_threshold);
  EXPECT_GT(IntersectionOverUnion(result.bbox, expected_box), iou_threshold)
      << "Actual " << ToString(result.bbox) << ", expected "
      << ToString(expected_box);
}

void TestCatMsCocoDetection(const std::string& model_path,
                            float score_threshold, float iou_threshold,
                            edgetpu::EdgeTpuContext* tpu_context) {
  TestDetection(model_path, "cat.bmp",
                /*expected_box=*/{0.1, 0.1, 1.0, 0.7},
                /*expected_label=*/16, score_threshold, iou_threshold,
                tpu_context);
}

void BenchmarkModelsOnEdgeTpu(const std::vector<std::string>& model_paths,
                              benchmark::State& state) {
  auto tpu_context = GetEdgeTpuContextOrDie();
  std::vector<std::unique_ptr<tflite::Interpreter>> interpreters;
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models;
  for (int model_index = 0; model_index < model_paths.size(); ++model_index) {
    models.emplace_back(LoadModelOrDie(model_paths[model_index]));
    interpreters.emplace_back(
        MakeEdgeTpuInterpreterOrDie(*models[model_index], tpu_context.get()));
    CHECK_EQ(interpreters[model_index]->AllocateTensors(), kTfLiteOk);
    FillRandomInt(MutableTensorData<uint8_t>(
        *interpreters[model_index]->input_tensor(0)));
  }

  // Use label field to pass model paths to benchmark result reporter.
  std::vector<std::string> model_basenames(model_paths.size());
  for (int i = 0; i < model_paths.size(); ++i) {
    model_basenames[i] = Basename(model_paths[i]);
  }
  state.SetLabel(absl::StrJoin(model_basenames, ","));

  while (state.KeepRunning()) {
    for (int i = 0; i < interpreters.size(); ++i) {
      CHECK_EQ(interpreters[i]->Invoke(), kTfLiteOk);
    }
  }
}

std::vector<uint8_t> ReadBmp(const std::string& filename, ImageDims* out_dims) {
  std::ifstream file(filename, std::ios::binary);
  CHECK(file) << "Cannot open file " << filename;

  uint8_t header[kBmpHeaderSize];
  CHECK(file.read(reinterpret_cast<char*>(header), sizeof(header)));

  const uint8_t* file_header = header;
  const uint8_t* info_header = header + kBmpFileHeaderSize;

  CHECK(file_header[0] == 'B' && file_header[1] == 'M') << "Not a BMP image";

  const int channels = info_header[14] / 8;
  CHECK(channels == 1 || channels == 3 || channels == 4);
  CHECK_EQ(ToInt32(&info_header[16]), 0) << "Unsupported compression";

  const uint32_t offset = ToInt32(&file_header[10]);
  if (offset > kBmpHeaderSize)
    CHECK(file.seekg(offset - kBmpHeaderSize, std::ios::cur));

  int width = ToInt32(&info_header[4]);
  CHECK_GE(width, 1);

  int height = ToInt32(&info_header[8]);
  const bool top_down = height < 0;
  if (top_down) height = -height;
  CHECK_GE(height, 1);

  const int line_bytes = width * channels;
  const int line_padding_bytes =
      4 * ((8 * channels * width + 31) / 32) - line_bytes;

  std::vector<uint8_t> image(line_bytes * height);
  for (int i = 0; i < height; ++i) {
    uint8_t* line = &image[(top_down ? i : (height - 1 - i)) * line_bytes];
    CHECK(file.read(reinterpret_cast<char*>(line), line_bytes));
    CHECK(file.seekg(line_padding_bytes, std::ios::cur));

    // BGR => RGB or BGRA => RGBA
    if (channels == 3 || channels == 4) {
      for (int j = 0; j < width; ++j)
        std::swap(line[channels * j], line[channels * j + 2]);
    }
  }

  if (out_dims) *out_dims = {height, width, channels};
  return image;
}

void ResizeImage(const ImageDims& in_dims, const uint8_t* in,
                 const ImageDims& out_dims, uint8_t* out) {
  if (in_dims == out_dims) {
    VLOG(1) << "No resizing needed for input image.";
    std::memcpy(out, in, ImageSize(in_dims));
    return;
  }
  tflite::Interpreter interpreter;
  int base_index = 0;
  // two inputs: input and new_sizes
  interpreter.AddTensors(2, &base_index);
  // one output
  interpreter.AddTensors(1, &base_index);
  // set input and output tensors
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, in_dims.height, in_dims.width, in_dims.depth}, quant);
  interpreter.SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                           quant);
  interpreter.SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, out_dims.height, out_dims.width, out_dims.depth}, quant);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  // Setting half_pixel_centers to true is only for internal testing purpose,
  // assuming the model has the ability to tolerate such variation.
  // Any evaluation accuracy drop regarding this parameter should be considered
  // with the proper value.
  params->half_pixel_centers = true;
  interpreter.AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                    nullptr);
  CHECK_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  std::copy(in, in + ImageSize(in_dims), interpreter.typed_tensor<float>(0));
  interpreter.typed_tensor<int>(1)[0] = out_dims.height;
  interpreter.typed_tensor<int>(1)[1] = out_dims.width;
  CHECK_EQ(interpreter.Invoke(), kTfLiteOk);

  auto* output = interpreter.typed_tensor<float>(2);
  std::copy(output, output + ImageSize(out_dims), out);
}

ImageDims BrcdShapeToImageDims(absl::Span<const int> shape) {
  CHECK_EQ(shape[0], 1);
  if (shape.size() == 4) {
    return ImageDims{shape[1], shape[2], shape[3]};
  } else {
    CHECK_EQ(shape.size(), 3);
    return ImageDims{shape[1], shape[2], 1};
  }
}

std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims) {
  ImageDims image_dims;
  std::vector<uint8_t> in = ReadBmp(image_path, &image_dims);

  if (target_dims.depth == 1 &&
      (image_dims.depth == 3 || image_dims.depth == 4)) {
    in = RgbToGrayscale(in, image_dims);
    image_dims.depth = 1;
  }

  std::vector<uint8_t> result(target_dims.height * target_dims.width *
                              target_dims.depth);
  ResizeImage(image_dims, in.data(), target_dims, result.data());
  return result;
}

void CopyResizedImage(const std::string& image_path,
                      const TfLiteTensor& tensor) {
  ImageDims image_dims;
  std::vector<uint8_t> in = ReadBmp(image_path, &image_dims);

  auto tensor_dims = BrcdShapeToImageDims(TensorShape(tensor));
  if (tensor_dims.depth == 1 &&
      (image_dims.depth == 3 || image_dims.depth == 4)) {
    in = RgbToGrayscale(in, image_dims);
    image_dims.depth = 1;
  }

  if (tensor.type == kTfLiteUInt8) {
    ResizeImage(image_dims, in.data(), tensor_dims,
                MutableTensorData<uint8_t>(tensor).data());
  } else if (tensor.type == kTfLiteFloat32) {
    std::vector<uint8_t> resized(TensorSize(tensor));
    ResizeImage(image_dims, in.data(), tensor_dims, resized.data());
    auto data = MutableTensorData<float>(tensor);
    for (int i = 0; i < data.size(); ++i)
      data[i] = (resized[i] - 127.5) / 127.5;
  } else {
    LOG(FATAL) << "Unsupported tesnor type " << tensor.type;
  }
}

std::shared_ptr<edgetpu::EdgeTpuContext> GetTestEdgeTpuContextOrDie() {
  LOG(INFO) << "Initializing Edge TPU device: \""
            << absl::GetFlag(FLAGS_tpu_device) << "\"...";
  return GetEdgeTpuContextOrDie(absl::GetFlag(FLAGS_tpu_device));
}

std::shared_ptr<edgetpu::EdgeTpuContext> EdgeTpuCacheTestBase::tpu_context_;

edgetpu::EdgeTpuContext* EdgeTpuCacheTestBase::GetTpuContextCache() {
  if (tpu_context_ == nullptr ||
      absl::GetFlag(FLAGS_disable_tpu_context_cache)) {
    tpu_context_.reset();  // close the underlying device
    tpu_context_ = GetTestEdgeTpuContextOrDie();
  }
  return tpu_context_.get();
}

edgetpu::EdgeTpuContext* ModelTestBase::GetTpuContextIfNecessary() {
  edgetpu::EdgeTpuContext* tpu_context = nullptr;
  if (GetParam() == "_edgetpu.tflite") {
    tpu_context = GetTpuContextCache();
  } else if (GetParam() == ".tflite") {
    LOG(INFO) << "Testing CPU models. No need to initialize Edge TPU device.";
  } else {
    LOG(FATAL) << "Unrecognized test parameter: " << GetParam();
  }
  return tpu_context;
}

std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>>
    MultipleEdgeTpuCacheTestBase::tpu_contexts_;

std::vector<edgetpu::EdgeTpuContext*>
MultipleEdgeTpuCacheTestBase::GetTpuContextCache(int num_tpus) {
  CHECK_GT(num_tpus, 0);
  if (tpu_contexts_.size() < num_tpus ||
      absl::GetFlag(FLAGS_disable_tpu_context_cache))
    tpu_contexts_.clear();

  if (tpu_contexts_.empty()) {
    tpu_contexts_.reserve(num_tpus);
    const auto& available_tpus =
        edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    const auto& tpu_device_type = absl::GetFlag(FLAGS_tpu_device);
    for (int i = 0; i < available_tpus.size(); ++i) {
      if (!tpu_device_type.empty() &&
          StringToDeviceType(tpu_device_type) != available_tpus[i].type)
        continue;
      const auto context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
          available_tpus[i].type, available_tpus[i].path);
      if (context) {
        tpu_contexts_.push_back(context);
        LOG(INFO) << "Device " << available_tpus[i].path << " is selected.";
        if (tpu_contexts_.size() == num_tpus) break;
      }
    }
  }

  std::vector<edgetpu::EdgeTpuContext*> results;
  results.reserve(num_tpus);
  for (int i = 0;
       i < std::min(num_tpus, static_cast<int>(tpu_contexts_.size())); ++i) {
    results.push_back(tpu_contexts_[i].get());
  }
  return results;
}

void ClassificationModelTestBase::TestClassification(
    const std::string& model_path, const std::string& image_path,
    float effective_scale, const std::vector<float>& effective_means,
    bool rgb2bgr, float score_threshold, int k, int expected_topk_label) {
  LOG(INFO) << "Testing parameters:";
  LOG(INFO) << "model_path: " << model_path;
  LOG(INFO) << "image_path: " << image_path;
  LOG(INFO) << "effective_scale: " << effective_scale;
  for (int i = 0; i < effective_means.size(); ++i)
    LOG(INFO) << "effective_means: " << effective_means[i];
  LOG(INFO) << "score_threshold: " << score_threshold;
  LOG(INFO) << "k: " << k;
  LOG(INFO) << "expected_topk_label: " << expected_topk_label;
  LOG(INFO) << "rgb2bgr: " << rgb2bgr;

  auto model = LoadModelOrDie(TestDataPath(model_path));
  auto tpu_context =
      ContainsEdgeTpuCustomOp(*model) ? GetTpuContextCache() : nullptr;
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  CopyResizedImage(TestDataPath(image_path), *interpreter->input_tensor(0));

  auto input_tensor = MutableTensorData<uint8_t>(*interpreter->input_tensor(0));

  const int num_channels = effective_means.size();
  VLOG(1) << "num_channels:" << num_channels;
  if (rgb2bgr) {
    CHECK_EQ(num_channels, 3);
    for (int i = 0; i < input_tensor.size(); i += num_channels) {
      input_tensor[i] = saturate_cast<float, uint8_t>(
          (input_tensor[i + 2] - effective_means[0]) / effective_scale);
      input_tensor[i + 1] = saturate_cast<float, uint8_t>(
          (input_tensor[i + 1] - effective_means[1]) / effective_scale);
      input_tensor[i + 2] = saturate_cast<float, uint8_t>(
          (input_tensor[i] - effective_means[2]) / effective_scale);
    }
  } else {
    for (int i = 0; i < input_tensor.size(); i += num_channels) {
      for (int j = 0; j < num_channels; ++j) {
        input_tensor[i + j] = saturate_cast<float, uint8_t>(
            (input_tensor[i + j] - effective_means[j]) / effective_scale);
      }
    }
  }
  CHECK(!input_tensor.empty()) << "Input image path: " << image_path;

  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

  const auto results =
      GetClassificationResults(*interpreter, score_threshold, k);
  const bool top_k_contains_expected =
      TopKContains(results, expected_topk_label);
  if (!top_k_contains_expected) {
    LOG(ERROR) << "Top " << k << " results do not contain expected label "
               << expected_topk_label << " with threshold=" << score_threshold;
    const auto no_threshold_results = GetClassificationResults(
        *interpreter, -std::numeric_limits<float>::infinity(), k);
    LOG(ERROR) << "Without score threshold, top " << k << " results are:";
    for (const auto& p : no_threshold_results) {
      LOG(ERROR) << p.id << ", " << p.score;
    }
  }
  EXPECT_TRUE(top_k_contains_expected);
}

void ModelEquivalenceTestBase::TestModelEquivalence(
    const std::string& input_image_path, const std::string& model0_path,
    const std::string& model1_path, uint8_t tolerance) {
  LOG(INFO) << "Testing pair:\n-" << model0_path << "\n-" << model1_path;

  auto model0 = LoadModelOrDie(model0_path);
  auto interpreter0 = MakeEdgeTpuInterpreterOrDie(
      *model0,
      ContainsEdgeTpuCustomOp(*model0) ? GetTpuContextCache() : nullptr);
  ASSERT_EQ(interpreter0->inputs().size(), 1);
  ASSERT_EQ(interpreter0->AllocateTensors(), kTfLiteOk);

  auto model1 = LoadModelOrDie(model1_path);
  auto interpreter1 = MakeEdgeTpuInterpreterOrDie(
      *model1,
      ContainsEdgeTpuCustomOp(*model1) ? GetTpuContextCache() : nullptr);
  ASSERT_EQ(interpreter1->inputs().size(), 1);
  ASSERT_EQ(interpreter1->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(TensorShape(*interpreter0->input_tensor(0)),
            TensorShape(*interpreter1->input_tensor(0)));

  ASSERT_EQ(interpreter0->outputs().size(), interpreter1->outputs().size());
  for (int k = 0; k < interpreter0->outputs().size(); ++k)
    ASSERT_EQ(TensorShape(*interpreter0->output_tensor(k)),
              TensorShape(*interpreter1->output_tensor(k)))
        << " for tensor #" << k;

  if (!input_image_path.empty()) {
    CopyResizedImage(input_image_path, *interpreter0->input_tensor(0));
    CopyResizedImage(input_image_path, *interpreter1->input_tensor(0));
  } else {
    auto input_type = interpreter0->input_tensor(0)->type;
    constexpr int kRandomSeed = 12345;
    if (input_type == kTfLiteUInt8) {
      FillRandomInt(MutableTensorData<uint8_t>(*interpreter0->input_tensor(0)),
                    kRandomSeed);
      FillRandomInt(MutableTensorData<uint8_t>(*interpreter1->input_tensor(0)),
                    kRandomSeed);
    } else if (input_type == kTfLiteFloat32) {
      FillRandomReal(MutableTensorData<float>(*interpreter0->input_tensor(0)),
                     -1.0f, 1.0f, kRandomSeed);
      FillRandomReal(MutableTensorData<float>(*interpreter1->input_tensor(0)),
                     -1.0f, 1.0f, kRandomSeed);
    } else {
      LOG(FATAL) << "Unsupported input tensor type: " << input_type;
    }
  }

  ASSERT_EQ(interpreter0->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter1->Invoke(), kTfLiteOk);

  for (int k = 0; k < interpreter0->outputs().size(); ++k) {
    auto tensor0 = TensorData<uint8_t>(*interpreter0->output_tensor(k));
    auto tensor1 = TensorData<uint8_t>(*interpreter1->output_tensor(k));
    ASSERT_EQ(tensor0.size(), tensor1.size());
    for (size_t i = 0; i < tensor0.size(); ++i)
      EXPECT_NEAR(static_cast<int>(tensor0[i]), static_cast<int>(tensor1[i]),
                  static_cast<int>(tolerance))
          << "tensor byte mismatch at index " << i << " for tensor #" << k;
  }
}

}  // namespace coral
