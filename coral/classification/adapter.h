#ifndef EDGETPU_CPP_CLASSIFICATION_ADAPTER_H_
#define EDGETPU_CPP_CLASSIFICATION_ADAPTER_H_

#include <cstddef>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

struct Class {
  int id;
  float score;
};

inline bool operator==(const Class& x, const Class& y) {
  return x.score == y.score && x.id == y.id;
}

inline bool operator!=(const Class& x, const Class& y) { return !(x == y); }

std::string ToString(const Class& c);

inline std::ostream& operator<<(std::ostream& os, const Class& c) {
  return os << ToString(c);
}

// Converts inference output tensors to classification results. Returns
// top_k ClassificationCandidate elements ordered by score, first element has
// the highest score.
//  - 'tensors' : vector<vector<float>>, result of RunInference() call
//  - 'threshold' : float, minimum confidence threshold for returned
//       classifications. For example, use 0.5 to receive only classifications
//       with a confidence equal-to or higher-than 0.5.
//  - 'top_k': size_t, the maximum number of classifications to return.
//
// The function will return a vector of predictions which is sorted by
// <score, label_id> in descending order.
std::vector<Class> GetClassificationResults(
    absl::Span<const float> scores,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());

std::vector<Class> GetClassificationResults(
    const tflite::Interpreter& interpreter,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());

inline Class GetTopClassificationResult(
    const tflite::Interpreter& interpreter) {
  return GetClassificationResults(
      interpreter, -std::numeric_limits<float>::infinity(), 1)[0];
}

}  // namespace coral

#endif  // EDGETPU_CPP_CLASSIFICATION_ADAPTER_H_
