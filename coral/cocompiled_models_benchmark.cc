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

#include "coral/test_utils.h"

namespace coral {

enum CompilationType { kCoCompilation, kSingleCompilation };

template <CompilationType Compilation>
static void BM_Compilation_TwoSmall(benchmark::State& state) {
  const std::string model_path0 =
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
            "mobilenet_v1_0.5_160_quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite";
  const std::string model_path1 =
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.5_160_quant_cocompiled_with_"
            "mobilenet_v1_0.25_128_quant_edgetpu.tflite"
          : "mobilenet_v1_0.5_160_quant_edgetpu.tflite";
  coral::BenchmarkModelsOnEdgeTpu(
      {TestDataPath(model_path0), TestDataPath(model_path1)}, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_TwoSmall, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_TwoSmall, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_TwoLarge(benchmark::State& state) {
  const std::string model_path0 =
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v4_299_quant_cocompiled_with_inception_v3_"
            "299_quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite";
  const std::string model_path1 =
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v3_299_quant_cocompiled_with_inception_v4_"
            "299_quant_edgetpu.tflite"
          : "inception_v3_299_quant_edgetpu.tflite";
  coral::BenchmarkModelsOnEdgeTpu(
      {TestDataPath(model_path0), TestDataPath(model_path1)}, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_TwoLarge, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_TwoLarge, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Small_Large(benchmark::State& state) {
  const std::string model_path0 =
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
            "inception_v4_299_quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite";
  const std::string model_path1 =
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v4_299_quant_cocompiled_with_mobilenet_v1_"
            "0.25_128_quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite";
  coral::BenchmarkModelsOnEdgeTpu(
      {TestDataPath(model_path0), TestDataPath(model_path1)}, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Small_Large, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Small_Large, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Large_Small(benchmark::State& state) {
  const std::string model_path0 =
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v4_299_quant_cocompiled_with_mobilenet_v1_"
            "0.25_128_quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite";
  const std::string model_path1 =
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
            "inception_v4_299_quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite";
  coral::BenchmarkModelsOnEdgeTpu(
      {TestDataPath(model_path0), TestDataPath(model_path1)}, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Large_Small, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Large_Small, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Four_Small(benchmark::State& state) {
  const std::string model_path0 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_1.0_224_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_1.0_224_quant_edgetpu.tflite";
  const std::string model_path1 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_0.25_128_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite";
  const std::string model_path2 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_0.5_160_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_0.5_160_quant_edgetpu.tflite";
  const std::string model_path3 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_0.75_192_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_0.75_192_quant_edgetpu.tflite";
  coral::BenchmarkModelsOnEdgeTpu(
      {TestDataPath(model_path0), TestDataPath(model_path1),
       TestDataPath(model_path2), TestDataPath(model_path3)},
      state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Four_Small, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Four_Small, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Four_Large(benchmark::State& state) {
  const std::string model_path0 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v1_224_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v1_224_quant_edgetpu.tflite";
  const std::string model_path1 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v2_224_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v2_224_quant_edgetpu.tflite";
  const std::string model_path2 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v3_299_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v3_299_quant_edgetpu.tflite";
  const std::string model_path3 =
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v4_299_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite";
  coral::BenchmarkModelsOnEdgeTpu(
      {TestDataPath(model_path0), TestDataPath(model_path1),
       TestDataPath(model_path2), TestDataPath(model_path3)},
      state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Four_Large, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Four_Large, kSingleCompilation);

}  // namespace coral
