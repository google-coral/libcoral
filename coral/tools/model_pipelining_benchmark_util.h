#ifndef EDGETPU_CPP_TOOLS_MODEL_PIPELINING_BENCHMARK_UTIL_H_
#define EDGETPU_CPP_TOOLS_MODEL_PIPELINING_BENCHMARK_UTIL_H_

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tflite/public/edgetpu.h"
struct IntList {
  std::vector<int> elements;
};

// Returns a textual flag value corresponding to the IntList `list`.
inline std::string AbslUnparseFlag(const IntList& list) {
  // Let flag module unparse the element type for us.
  return absl::StrJoin(list.elements, ",", [](std::string* out, int element) {
    out->append(absl::UnparseFlag(element));
  });
}

// Parses an IntList from the command line flag value `text`.
// Returns true and sets `*list` on success; returns false and sets `*error` on
// failure.
inline bool AbslParseFlag(absl::string_view text, IntList* list,
                          std::string* error) {
  // We have to clear the list to overwrite any existing value.
  list->elements.clear();
  // absl::StrSplit("") produces {""}, but we need {} on empty input.
  if (text.empty()) {
    return true;
  }
  for (const auto& part : absl::StrSplit(text, ',')) {
    // Let the flag module parse each element value for us.
    int element;
    if (!absl::ParseFlag(part, &element, error)) {
      return false;
    }
    list->elements.push_back(element);
  }
  return true;
}

enum class EdgeTpuType {
  kAny,
  kPciOnly,
  kUsbOnly,
};

// Parses an EdgeTpuType from the command line flag value. Returns `true` and
// sets `*mode` on success; returns `false` and sets `*error` on failure.
inline bool AbslParseFlag(absl::string_view text, EdgeTpuType* type,
                          std::string* error) {
  if (text == "any") {
    *type = EdgeTpuType::kAny;
    return true;
  }
  if (text == "pcionly") {
    *type = EdgeTpuType::kPciOnly;
    return true;
  }
  if (text == "usbonly") {
    *type = EdgeTpuType::kUsbOnly;
    return true;
  }
  *error = "unknown value for device_type";
  return false;
}

// Returns a textual flag value corresponding to the EdgeTpuType.
inline std::string AbslUnparseFlag(EdgeTpuType type) {
  switch (type) {
    case EdgeTpuType::kAny:
      return "any";
    case EdgeTpuType::kPciOnly:
      return "pcionly";
    case EdgeTpuType::kUsbOnly:
      return "usbonly";
    default:
      return absl::StrCat(type);
  }
}

namespace coral {

using edgetpu::EdgeTpuContext;

// num_segments, latency (in ns), latencies for all segments(in ns) tuple.
using PerfStats = std::tuple<int, int64_t, std::vector<int64_t>>;

std::vector<std::shared_ptr<EdgeTpuContext>> PrepareEdgeTpuContexts(
    int num_tpus, EdgeTpuType device_type);

PerfStats BenchmarkPartitionedModel(
    const std::vector<std::string>& model_segments_paths,
    const std::vector<std::shared_ptr<EdgeTpuContext>>* edgetpu_contexts,
    int num_inferences);

}  // namespace coral

#endif  // EDGETPU_CPP_TOOLS_MODEL_PIPELINING_BENCHMARK_UTIL_H_
