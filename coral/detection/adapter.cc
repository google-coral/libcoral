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
  auto bboxes = TensorData<float>(*interpreter.output_tensor(0));
  auto ids = TensorData<float>(*interpreter.output_tensor(1));
  auto scores = TensorData<float>(*interpreter.output_tensor(2));
  auto count = TensorData<float>(*interpreter.output_tensor(3));
  CHECK_EQ(bboxes.size(), 4 * ids.size());
  CHECK_EQ(bboxes.size(), 4 * scores.size());
  CHECK_EQ(count.size(), 1);
  return GetDetectionResults(bboxes, ids, scores, static_cast<size_t>(count[0]),
                             threshold, top_k);
}

}  // namespace coral
