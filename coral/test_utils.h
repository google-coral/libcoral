#ifndef EDGETPU_CPP_TEST_UTILS_H_
#define EDGETPU_CPP_TEST_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "coral/bbox.h"
#include "coral/classification/adapter.h"
#include "tensorflow/lite/c/common.h"

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

// Returns whether top k results contains a given label.
bool TopKContains(const std::vector<Class>& topk, int label);

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
                        const std::vector<float>& effective_means, bool rgb2bgr,
                        float score_threshold, int k, int expected_topk_label);

// Tests a SSD detection model. Only checks the first detection result.
void TestDetection(const std::string& model_path, const std::string& image_path,
                   const BBox<float>& expected_box, int expected_label,
                   float score_threshold, float iou_threshold);

// Tests a MSCOCO detection model with cat.bmp.
void TestCatMsCocoDetection(const std::string& model_path,
                            float score_threshold, float iou_threshold);

// Benchmarks models on a sinlge EdgeTpu device.
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
}  // namespace coral

#endif  // EDGETPU_CPP_TEST_UTILS_H_
