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

#ifndef LIBCORAL_CORAL_DETECTION_ADAPTER_H_
#define LIBCORAL_CORAL_DETECTION_ADAPTER_H_

#include <cstddef>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "coral/bbox.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

// Represents a detected object.
struct Object {
  // The class label id.
  int id;
  // The prediction score.
  float score;
  // A `BBox` defining the bounding-box (ymin,xmin,ymax,xmax).
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

// Converts detection output tensors into a list of SSD results.
//
// @param bboxes Bounding boxes of detected objects. Four floats per object
//  (box-corner encoding [ymin1,xmin1,ymax1,xmax1,ymin2,xmin2,...]).
// @param ids Label identifiers of detected objects. One float per object.
// @param scores Confidence scores of detected objects. One float per object.
// @param count The number of detected objects (all tensors defined above
//   have valid data for only this number of objects).
// @param threshold The score threshold for results. All returned results have
//   a score greater-than-or-equal-to this value.
// @param top_k The maximum number of predictions to return.
// @returns The top_k `Object` predictions, <id, score, bbox>, ordered by score
// (first element has the highest score).
std::vector<Object> GetDetectionResults(
    absl::Span<const float> bboxes, absl::Span<const float> ids,
    absl::Span<const float> scores, size_t count,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());

// Gets results from a detection model as a list of ordered objects.
//
// @param interpreter The already-invoked interpreter for your detection model.
// @param threshold The score threshold for results. All returned results have
//   a score greater-than-or-equal-to this value.
// @param top_k The maximum number of predictions to return.
// @returns The top_k `Object` predictions, <id, score, bbox>, ordered by score
// (first element has the highest score).
std::vector<Object> GetDetectionResults(
    const tflite::Interpreter& interpreter,
    float threshold = -std::numeric_limits<float>::infinity(),
    size_t top_k = std::numeric_limits<size_t>::max());
}  // namespace coral

#endif  // LIBCORAL_CORAL_DETECTION_ADAPTER_H_
