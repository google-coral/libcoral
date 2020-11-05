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
  coral::BenchmarkModelsOnEdgeTpu({model_path}, state);
}

#if ARG_RUN_EDGETPU_MODEL
TO_BENCHMARK_TEMPLATE(ARG_BENCHMARK_NAME, coral::kEdgeTpu);
#endif

#if ARG_RUN_CPU_MODEL
TO_BENCHMARK_TEMPLATE(ARG_BENCHMARK_NAME, coral::kCpu);
#endif

}  // namespace coral
