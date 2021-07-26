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

#ifndef LIBCORAL_CORAL_TEST_UTILS_H_
#define LIBCORAL_CORAL_TEST_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "coral/bbox.h"
#include "coral/classification/adapter.h"
#include "coral/tflite_utils.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/c/common.h"
#include "tflite/public/edgetpu.h"

#define CORAL_STRINGIFY(x) #x
#define CORAL_TOSTRING(x) CORAL_STRINGIFY(x)

namespace coral {

enum CnnProcessorType { kEdgeTpu, kCpu };

// Retrieves test file path with file name.
std::string TestDataPath(const std::string& name);

// Fills given range with random int values from 0 up to
// std::numeric_limits<>::max().
template <typename ForwardIt>
void FillRandomInt(ForwardIt begin, ForwardIt end, int seed = 1) {
  using ValueType = typename std::iterator_traits<ForwardIt>::value_type;
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<> dist(0,
                                       std::numeric_limits<ValueType>::max());
  std::generate(begin, end, [&] { return dist(generator); });
}

template <typename T>
void FillRandomInt(absl::Span<T> span, int seed = 1) {
  FillRandomInt(span.begin(), span.end());
}

// Fills given range with random real values from `min` to `max`.
template <typename ForwardIt>
void FillRandomReal(
    ForwardIt begin, ForwardIt end,
    typename std::iterator_traits<ForwardIt>::value_type min = -1,
    typename std::iterator_traits<ForwardIt>::value_type max = 1,
    int seed = 1) {
  using ValueType = typename std::iterator_traits<ForwardIt>::value_type;
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<ValueType> dist(min, max);
  std::generate(begin, end, [&] { return dist(generator); });
}

template <typename T>
void FillRandomReal(absl::Span<T> span, T min = -1, T max = 1, int seed = 1) {
  FillRandomReal(span.begin(), span.end(), min, max, seed);
}

// Returns whether top k results contain a given label.
bool TopKContains(const std::vector<Class>& topk, int label);

// Tests a SSD detection model. Only checks the first detection result.
void TestDetection(const std::string& model_path, const std::string& image_path,
                   const BBox<float>& expected_box, int expected_label,
                   float score_threshold, float iou_threshold,
                   edgetpu::EdgeTpuContext* tpu_context);

// Tests a MSCOCO detection model with cat.bmp.
void TestCatMsCocoDetection(const std::string& model_path,
                            float score_threshold, float iou_threshold,
                            edgetpu::EdgeTpuContext* tpu_context);

// Benchmarks models on a sinlge EdgeTpu device. Model paths must be full paths.
void BenchmarkModelsOnEdgeTpu(const std::vector<std::string>& model_paths,
                              benchmark::State& state);

// Defines dimension of an image, in height, width, depth order.
struct ImageDims {
  int height;
  int width;
  int depth;
};

inline int ImageSize(const ImageDims& dims) {
  return dims.height * dims.width * dims.depth;
}

inline bool operator==(const ImageDims& a, const ImageDims& b) {
  return a.height == b.height && a.width == b.width && a.depth == b.depth;
}

inline bool operator!=(const ImageDims& a, const ImageDims& b) {
  return !(a == b);
}

// Converts shape (1, height, width, channels) to ImageDims
ImageDims BrcdShapeToImageDims(absl::Span<const int> shape);

// The following image-related functions take full file paths.

// Reads BMP image. It will crash upon failure.
std::vector<uint8_t> ReadBmp(const std::string& filename, ImageDims* out_dims);

// Gets input from images and resizes to `target_dims`. It will crash upon
// failure.
std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims);

// Loads image from `image_path` and copies resized pixels to the `tensor`.
// For float input tensors image data is normalized to [-1.0, 1.0) interval.
void CopyResizedImage(const std::string& image_path,
                      const TfLiteTensor& tensor);

// Returns EdgeTpu context, which can be specified with flag --tpu_device.
std::shared_ptr<edgetpu::EdgeTpuContext> GetTestEdgeTpuContextOrDie();

// Test base class that caches single Edge TPU context.
class EdgeTpuCacheTestBase : public ::testing::Test {
 public:
  static edgetpu::EdgeTpuContext* GetTpuContextCache();

 protected:
  static std::shared_ptr<edgetpu::EdgeTpuContext> tpu_context_;
};

// Edge TPU model test base.
// Test parameter is model suffix, '.tflite' or '_edgetpu.tflite'.
class ModelTestBase : public EdgeTpuCacheTestBase,
                      public ::testing::WithParamInterface<std::string> {
 public:
  // Returns pointer to EdgeTpuContext object if model suffix is
  // '_edgetpu.tflite'. It will cache the context once it's created.
  static edgetpu::EdgeTpuContext* GetTpuContextIfNecessary();
};

// Test base class that caches multiple Edge TPU contexts.
class MultipleEdgeTpuCacheTestBase : public ::testing::Test {
 public:
  // Returns vector of pointers to EdgeTpuContext. The returned vector size
  // can be smaller than 'num_tpus' if there are not enough TPUs.
  static std::vector<edgetpu::EdgeTpuContext*> GetTpuContextCache(int num_tpus);

 protected:
  static std::vector<std::shared_ptr<edgetpu::EdgeTpuContext> > tpu_contexts_;
};

class ClassificationModelTestBase : public EdgeTpuCacheTestBase {
 protected:
  // Tests a classification model with customized preprocessing.
  // Custom preprocessing is done by:
  // (v - (mean - zero_point * scale * stddev)) / (stddev * scale)
  // where zero_point and scale are the quantization parameters of the input
  // tensor, and mean and stddev are the normalization parameters of the input
  // tensor. Effective mean and scale should be
  // (mean - zero_point * scale * stddev) and (stddev * scale) respectively.
  // If rgb2bgr is true, the channels of input image will be shuffled from
  // RGB to BGR.
  void TestClassification(const std::string& model_path,
                          const std::string& image_path, float effective_scale,
                          const std::vector<float>& effective_means,
                          bool rgb2bgr, float score_threshold, int k,
                          int expected_topk_label);
};

class ModelEquivalenceTestBase : public EdgeTpuCacheTestBase {
 protected:
  // If `input_image_path` is empty, it will test with random data.
  void TestModelEquivalence(const std::string& input_image_path,
                            const std::string& model0_path,
                            const std::string& model1_path,
                            uint8_t tolerance = 0);
};

}  // namespace coral

#endif  // LIBCORAL_CORAL_TEST_UTILS_H_
