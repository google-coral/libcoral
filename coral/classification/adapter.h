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

#ifndef LIBCORAL_CORAL_CLASSIFICATION_ADAPTER_H_
#define LIBCORAL_CORAL_CLASSIFICATION_ADAPTER_H_

#include <cstddef>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

// Represents a single classification result.
struct Class {
  // The class label id.
  int id;
  // The prediction score.
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

// Converts classification output tensors into a list of ordered classes.
//
// @param scores The classification output tensor (dequantized).
// @param threshold The score threshold for results. All returned results have
//   a score greater-than-or-equal-to this value.
// @param top_k The maximum number of predictions to return.
// @returns The top_k `Class` predictions, <score, label_id>, ordered by score
// (first element has the highest score).
std::vector<Class> GetClassificationResults(
    absl::Span<const float> scores,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());

// Gets results from a classification model as a list of ordered classes.
//
// @param interpreter The already-invoked interpreter for your classification
//   model.
// @param threshold The score threshold for results. All returned results have
//   a score greater-than-or-equal-to this value.
// @param top_k The maximum number of predictions to return.
// @returns The top_k `Class` predictions, <score, label_id>, ordered by score
// (first element has the highest score).
std::vector<Class> GetClassificationResults(
    const tflite::Interpreter& interpreter,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());

// Gets only the top result from a classification model.
//
// @param interpreter The already-invoked interpreter for your classification
//   model.
// @returns The top `Class` prediction, <score, label_id>.
inline Class GetTopClassificationResult(
    const tflite::Interpreter& interpreter) {
  return GetClassificationResults(
      interpreter, -std::numeric_limits<float>::infinity(), 1)[0];
}

}  // namespace coral

#endif  // LIBCORAL_CORAL_CLASSIFICATION_ADAPTER_H_
