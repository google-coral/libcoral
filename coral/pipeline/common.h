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

#ifndef LIBCORAL_CORAL_PIPELINE_COMMON_H_
#define LIBCORAL_CORAL_PIPELINE_COMMON_H_

#include <string>
#include <vector>

#include "coral/pipeline/allocator.h"
#include "glog/logging.h"
#include "tensorflow/lite/c/common.h"

namespace coral {

// A tensor in the pipeline system.
// This is a simplified version of `TfLiteTensor`.
struct PipelineTensor {
  // Unique tensor name.
  std::string name;
  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;
  // Underlying memory buffer for tensor. Allocated by `Allocator`.
  Buffer* buffer;
  // The number of bytes required to store the data of this tensor. That is:
  // `(bytes of each element) * dims[0] * ... * dims[n-1]`. For example, if
  // type is kTfLiteFloat32 and `dims = {3, 2}` then
  // `bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24`.
  size_t bytes;
};

// Performance statistics for one segment of model pipeline.
struct SegmentStats {
  // Total time spent traversing this segment so far (in nanoseconds).
  int64_t total_time_ns = 0;
  // Number of inferences processed so far.
  uint64_t num_inferences = 0;
};

// Deallocates the memory for the given tensors.
// Use this to free output tensors each time you process the results.
//
// @param tensors A vector of PipelineTensor objects to release.
// @param allocator The Allocator originally used to allocate the tensors.
inline void FreePipelineTensors(const std::vector<PipelineTensor>& tensors,
                                Allocator* allocator) {
  for (const auto& tensor : tensors) {
    VLOG(1) << "Releasing tensor "
            << " at addr: " << static_cast<void*>(tensor.buffer->ptr());
    allocator->Free(tensor.buffer);
  }
}

}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_COMMON_H_
