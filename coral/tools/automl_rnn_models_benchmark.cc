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
#include "coral/tflite_utils.h"
#include "coral/tools/automl_video_object_tracking_utils.h"
#include "coral/tools/tflite_graph_util.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {

template <CnnProcessorType CnnProcessor>
static void BM_AutoMlTrafficModelWithoutRnnLink(benchmark::State& state) {
  const std::string basename = (CnnProcessor == kEdgeTpu)
                                   ? "traffic_model_edgetpu.tflite"
                                   : "traffic_model.tflite";
  auto model =
      LoadModelOrDie(TestDataPath("automl_video_ondevice/" + basename));
  auto tpu_context = GetEdgeTpuContext();
  auto interpreter = BuildLstmEdgeTpuInterpreter(*model, tpu_context.get());
  FillRandomInt(MutableTensorData<uint8_t>(*interpreter->input_tensor(0)));
  state.SetLabel(basename);
  while (state.KeepRunning()) RunLstmInference(interpreter.get());
}
BENCHMARK_TEMPLATE(BM_AutoMlTrafficModelWithoutRnnLink, kEdgeTpu);
BENCHMARK_TEMPLATE(BM_AutoMlTrafficModelWithoutRnnLink, kCpu);

static void BM_AutoMlTrafficModelWithRnnLink(benchmark::State& state) {
  std::string output_path = "/tmp/traffic_model_recurrent_links_edgetpu.tflite";
  ASSERT_EQ(
      AppendRecurrentLinks(
          TestDataPath("automl_video_ondevice/traffic_model_edgetpu.tflite"),
          /*input_tensor_names=*/
          {"raw_inputs/init_lstm_c", "raw_inputs/init_lstm_h"},
          /*output_tensor_names=*/{"raw_outputs/lstm_c", "raw_outputs/lstm_h"},
          output_path),
      absl::OkStatus());
  coral::BenchmarkModelsOnEdgeTpu({output_path}, state);
}
BENCHMARK(BM_AutoMlTrafficModelWithRnnLink);

}  // namespace coral
