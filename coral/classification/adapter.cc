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

#include "coral/classification/adapter.h"

#include <algorithm>
#include <queue>
#include <tuple>

#include "absl/strings/substitute.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"

namespace coral {
namespace {
// Defines a comparator which allows us to rank Class based on their score and
// id.
struct ClassComparator {
  bool operator()(const Class& lhs, const Class& rhs) const {
    return std::tie(lhs.score, lhs.id) > std::tie(rhs.score, rhs.id);
  }
};
}  // namespace

std::string ToString(const Class& c) {
  return absl::Substitute("Class(id=$0,score=$1)", c.id, c.score);
}

std::vector<Class> GetClassificationResults(absl::Span<const float> scores,
                                            float threshold, size_t top_k) {
  std::priority_queue<Class, std::vector<Class>, ClassComparator> q;
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] < threshold) continue;
    q.push(Class{i, scores[i]});
    if (q.size() > top_k) q.pop();
  }

  std::vector<Class> ret;
  while (!q.empty()) {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

std::vector<Class> GetClassificationResults(
    const tflite::Interpreter& interpreter, float threshold, size_t top_k) {
  const auto& tensor = *interpreter.output_tensor(0);
  if (tensor.type == kTfLiteUInt8 || tensor.type == kTfLiteInt8) {
    return GetClassificationResults(DequantizeTensor<float>(tensor), threshold,
                                    top_k);
  } else if (tensor.type == kTfLiteFloat32) {
    return GetClassificationResults(TensorData<float>(tensor), threshold,
                                    top_k);
  } else {
    LOG(FATAL) << "Unsupported tensor type: " << tensor.type;
  }
}

}  // namespace coral
