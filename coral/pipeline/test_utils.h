#ifndef EDGETPU_CPP_PIPELINE_TEST_UTILS_H_
#define EDGETPU_CPP_PIPELINE_TEST_UTILS_H_

#include <string>
#include <vector>

#include "coral/pipeline/allocator.h"
#include "coral/pipeline/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* context);

// Constructs model segments' names based on base name and number of segments.
std::vector<std::string> SegmentsNames(const std::string& model_base_name,
                                       int num_segments);

// If `allocator` is not provided, it uses std::malloc to allocate tensors.
std::vector<PipelineTensor> CreateRandomInputTensors(
    const tflite::Interpreter* interpreter, Allocator* allocator = nullptr);
}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_TEST_UTILS_H_
