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

#include "benchmark/benchmark.h"
#include "coral/test_utils.h"

#define TO_BENCHMARK_TEMPLATE(name, type) BENCHMARK_TEMPLATE(name, type)

namespace coral {

// All ARG_* values come from compiler defines, e.g. -DARG_TEST_NAME=my_test.
template <coral::CnnProcessorType CnnProcessor>
static void ARG_BENCHMARK_NAME(benchmark::State& state) {
  const std::string model_path =
      CnnProcessor == coral::kEdgeTpu
          ? CORAL_TOSTRING(ARG_TFLITE_EDGETPU_FILEPATH)
          : CORAL_TOSTRING(ARG_TFLITE_CPU_FILEPATH);
  coral::BenchmarkModelsOnEdgeTpu({TestDataPath(model_path)}, state);
}

#if ARG_RUN_EDGETPU_MODEL
TO_BENCHMARK_TEMPLATE(ARG_BENCHMARK_NAME, coral::kEdgeTpu);
#endif

#if ARG_RUN_CPU_MODEL
TO_BENCHMARK_TEMPLATE(ARG_BENCHMARK_NAME, coral::kCpu);
#endif

}  // namespace coral
