#ifndef EDGETPU_CPP_TOOLS_PARTITIONER_PARTITIONER_H_
#define EDGETPU_CPP_TOOLS_PARTITIONER_PARTITIONER_H_

#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "coral/tools/partitioner/strategy.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {

class Partitioner {
 public:
  virtual ~Partitioner() = default;
  virtual PartitionStrategy GetStrategy(int num_segments) const = 0;
};

// Base class for any partitioner that is count-based.
//
// Derived class just need to define `GetNumOpsPerSegment` function which
// returns the numbers of segment ops for each segment. For example, there's a
// total of 10 ops, `GetNumOpsPerSegment` can return [1, 1, 3, 5] as a valid
// result.
//
// Note that `input_model_content` is a serialized tflite flatbuffer and must
// outlives `CountBasedPartitioner` object.
class CountBasedPartitioner : public Partitioner {
 public:
  explicit CountBasedPartitioner(const std::vector<char>& input_model_content);

  ~CountBasedPartitioner() override = default;

  PartitionStrategy GetStrategy(int num_segments) const override;

 protected:
  virtual std::vector<int> GetNumOpsPerSegment(int num_segments) const = 0;

  const tflite::Model* model_;
};

// Parameter (size) count based partitioner
class ParameterCountBasedPartitioner : public CountBasedPartitioner {
 public:
  explicit ParameterCountBasedPartitioner(
      const std::vector<char>& input_model_content);

 protected:
  std::vector<int> GetNumOpsPerSegment(int num_segments) const final;
};
}  // namespace coral

#endif  // EDGETPU_CPP_TOOLS_PARTITIONER_PARTITIONER_H_
