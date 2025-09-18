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

#include "coral/detection/adapter.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <tuple>

#include "absl/strings/substitute.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"

namespace coral {
namespace {
struct ObjectComparator {
  bool operator()(const Object& lhs, const Object& rhs) const {
    return std::tie(lhs.score, lhs.id) > std::tie(rhs.score, rhs.id);
  }
};
}  // namespace

std::string ToString(const Object& obj) {
  return absl::Substitute("Object(id=$0,score=$1,bbox=$2)", obj.id, obj.score,
                          ToString(obj.bbox));
}

std::vector<Object> GetDetectionResults(absl::Span<const float> bboxes,
                                        absl::Span<const float> ids,
                                        absl::Span<const float> scores,
                                        size_t count, float threshold,
                                        size_t top_k) {
  CHECK_EQ(bboxes.size() % 4, 0);
  CHECK_LE(4 * count, bboxes.size());
  CHECK_LE(count, ids.size());
  CHECK_LE(count, scores.size());

  std::priority_queue<Object, std::vector<Object>, ObjectComparator> q;
  for (int i = 0; i < count; ++i) {
    const int id = std::round(ids[i]);
    const float score = scores[i];
    if (score < threshold) continue;
    const float ymin = std::max(0.0f, bboxes[4 * i]);
    const float xmin = std::max(0.0f, bboxes[4 * i + 1]);
    const float ymax = std::min(1.0f, bboxes[4 * i + 2]);
    const float xmax = std::min(1.0f, bboxes[4 * i + 3]);
    q.push(Object{id, score, BBox<float>{ymin, xmin, ymax, xmax}});
    if (q.size() > top_k) q.pop();
  }

  std::vector<Object> ret;
  ret.reserve(q.size());
  while (!q.empty()) {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

std::vector<Object> GetDetectionResults(const tflite::Interpreter& interpreter,
                                        float threshold, size_t top_k) {
  CHECK_EQ(interpreter.outputs().size(), 4);
  absl::Span<const float> bboxes, ids, scores, count;
  // If a model has signature, we use the signature output tensor names to parse
  // the results. Otherwise, we parse the results based on some assumption of
  // the output tensor order and size.
  if (!interpreter.signature_keys().empty()) {
    CHECK_EQ(interpreter.signature_keys().size(), 1);
    VLOG(1) << "Signature name: " << *interpreter.signature_keys()[0];
    const auto& signature_output_map = interpreter.signature_outputs(
        interpreter.signature_keys()[0]->c_str());
    CHECK_EQ(signature_output_map.size(), 4);
    count = TensorData<float>(
        *interpreter.tensor(signature_output_map.at("output_0")));
    scores = TensorData<float>(
        *interpreter.tensor(signature_output_map.at("output_1")));
    ids = TensorData<float>(
        *interpreter.tensor(signature_output_map.at("output_2")));
    bboxes = TensorData<float>(
        *interpreter.tensor(signature_output_map.at("output_3")));
  } else if (interpreter.output_tensor(3)->bytes / sizeof(float) == 1) {
    bboxes = TensorData<float>(*interpreter.output_tensor(0));
    ids = TensorData<float>(*interpreter.output_tensor(1));
    scores = TensorData<float>(*interpreter.output_tensor(2));
    count = TensorData<float>(*interpreter.output_tensor(3));
  } else {
    scores = TensorData<float>(*interpreter.output_tensor(0));
    bboxes = TensorData<float>(*interpreter.output_tensor(1));
    count = TensorData<float>(*interpreter.output_tensor(2));
    ids = TensorData<float>(*interpreter.output_tensor(3));
  }
  CHECK_EQ(bboxes.size(), 4 * ids.size());
  CHECK_EQ(bboxes.size(), 4 * scores.size());
  CHECK_EQ(count.size(), 1);
  return GetDetectionResults(bboxes, ids, scores, static_cast<size_t>(count[0]),
                             threshold, top_k);
}

}  // namespace coral
