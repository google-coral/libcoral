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

#ifndef LIBCORAL_CORAL_TOOLS_PARTITIONER_PROFILING_BASED_PARTITIONER_ONDEVICE_LIB_H_
#define LIBCORAL_CORAL_TOOLS_PARTITIONER_PROFILING_BASED_PARTITIONER_ONDEVICE_LIB_H_

#include <string>
#include <utility>

namespace coral {

// Helper function to run profiling based partitioning, benchmark the model and
// its pipelined version, and check the speedup match expectation.
void TestProfilingBasedPartitioner(
    const std::string& model_name, int num_segments, float expected_speedup,
    int partition_search_step, int delegate_search_step = 4,
    int64_t diff_threshold_ns = 1000000,
    const std::pair<int64_t, int64_t>& initial_bounds = {-1, -1});

}  // namespace coral

#endif  // LIBCORAL_CORAL_TOOLS_PARTITIONER_PROFILING_BASED_PARTITIONER_ONDEVICE_LIB_H_
