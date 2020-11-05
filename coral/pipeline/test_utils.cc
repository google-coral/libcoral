#include "coral/pipeline/test_utils.h"

#include <random>

#include "absl/strings/str_cat.h"
#include "coral/test_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"

namespace coral {

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* context) {
  CHECK(context);
  auto interpreter = MakeEdgeTpuInterpreterOrDie(model, context);
  interpreter->SetNumThreads(1);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  return interpreter;
}

std::vector<std::string> SegmentsNames(const std::string& model_base_name,
                                       int num_segments) {
  std::vector<std::string> result(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    result[i] = absl::StrCat(model_base_name, "_segment_", i, "_of_",
                             num_segments, "_edgetpu.tflite");
  }
  return result;
}

std::vector<PipelineTensor> CreateRandomInputTensors(
    const tflite::Interpreter* interpreter, Allocator* allocator) {
  CHECK(interpreter);
  CHECK(allocator);
  std::vector<PipelineTensor> input_tensors;
  for (int input_index : interpreter->inputs()) {
    const auto* input_tensor = interpreter->tensor(input_index);
    PipelineTensor input_buffer;
    input_buffer.buffer = allocator->Alloc(input_tensor->bytes);
    input_buffer.bytes = input_tensor->bytes;
    input_buffer.type = input_tensor->type;
    auto* ptr = reinterpret_cast<uint8_t*>(input_buffer.buffer->ptr());
    FillRandomInt(ptr, ptr + input_buffer.bytes);
    input_tensors.push_back(input_buffer);
  }
  return input_tensors;
}

}  // namespace coral
