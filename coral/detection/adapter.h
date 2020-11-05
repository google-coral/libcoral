#ifndef EDGETPU_CPP_DETECTION_ADAPTER_H_
#define EDGETPU_CPP_DETECTION_ADAPTER_H_

#include <cstddef>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "coral/bbox.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

// Detection result.
struct Object {
  int id;
  float score;
  BBox<float> bbox;
};

inline bool operator==(const Object& x, const Object& y) {
  return x.id == y.id && x.score == y.score && x.bbox == y.bbox;
}

inline bool operator!=(const Object& x, const Object& y) { return !(x == y); }

std::string ToString(const Object& obj);

inline std::ostream& operator<<(std::ostream& os, const Object& obj) {
  return os << ToString(obj);
}

// Converts inference output tensors to SSD detection results. Returns
// top_k DetectionCandidate elements ordered by score, first element has
// the highest score.
//  - 'bboxes' : bounding boxes of detected objects. Four floats per object
//  (box-corner encoding).
//  - 'ids': class identifiers of detected objects. One float per object.
//  - 'scores': confidence scores of detected objects. One float per object.
//  - 'count': number of detected objects, all tensors defined above have valid
// data only for these objects.
//  - 'threshold' : float, minimum confidence threshold for returned
//       predictions. For example, use 0.5 to receive only predictions
//       with a confidence equal-to or higher-than 0.5.
//  - 'top_k': size_t, the maximum number of predictions to return.
//
// The function will return a vector of predictions which is sorted by
// <score, label_id> in descending order.
std::vector<Object> GetDetectionResults(
    absl::Span<const float> bboxes, absl::Span<const float> ids,
    absl::Span<const float> scores, size_t count,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());

std::vector<Object> GetDetectionResults(
    const tflite::Interpreter& interpreter,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());
}  // namespace coral

#endif  // EDGETPU_CPP_DETECTION_ADAPTER_H_
