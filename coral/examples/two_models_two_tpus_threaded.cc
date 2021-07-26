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

// Example to run two models with two Edge TPUs using two threads.
// It depends only on tflite and edgetpu.h
//
// Example usage:
// 1. Create directory edgetpu_cpp_example
// 2. wget -O edgetpu_cpp_example/inat_bird_edgetpu.tflite \
//      http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite
// 3. wget -O edgetpu_cpp_example/inat_plant_edgetpu.tflite \
//      http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite
// 4. wget -O edgetpu_cpp_example/bird.jpg \
//      https://farm3.staticflickr.com/8008/7523974676_40bbeef7e3_o.jpg
// 5. wget -O edgetpu_cpp_example/plant.jpg \
//      https://c2.staticflickr.com/1/62/184682050_db90d84573_o.jpg
// 6. cd edgetpu_cpp_example && \
//    convert bird.jpg -resize 224x224! bird.rgb && \
//    convert plant.jpg -resize 224x224! plant.rgb
// 7. Build and run `two_models_two_tpus_threaded`
//
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tflite/public/edgetpu.h"

ABSL_FLAG(std::string, bird_model_path,
          "edgetpu_cpp_example/inat_bird_edgetpu.tflite",
          "Path to bird model.");
ABSL_FLAG(std::string, plant_model_path,
          "edgetpu_cpp_example/inat_plant_edgetpu.tflite",
          "Path to plant model.");
ABSL_FLAG(std::string, bird_image_path, "edgetpu_cpp_example/bird.rgb",
          "Path to bird image. The input image size must match the input size "
          "of the model and the image must be stored as RGB pixel array.");
ABSL_FLAG(std::string, plant_image_path, "edgetpu_cpp_example/plant.rgb",
          "Path to plant image. The input image size must match the input size "
          "of the model and the image must be stored as RGB pixel array.");
ABSL_FLAG(int, num_inferences, 2000, "Number of inferences to run.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto bird_model_path = absl::GetFlag(FLAGS_bird_model_path);
  const auto plant_model_path = absl::GetFlag(FLAGS_plant_model_path);
  const auto bird_image_path = absl::GetFlag(FLAGS_bird_image_path);
  const auto plant_image_path = absl::GetFlag(FLAGS_plant_image_path);
  const int num_inferences = absl::GetFlag(FLAGS_num_inferences);

  const auto& available_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  CHECK_GE(available_tpus.size(), 2)
      << "This example requires two Edge TPUs to run.";

  auto run = [num_inferences](
                 const edgetpu::EdgeTpuManager::DeviceEnumerationRecord& tpu,
                 const std::string& model_path, const std::string& image_path) {
    const auto& tid = std::this_thread::get_id();
    std::cout << "Thread: " << tid << " Using model: " << model_path
              << " Running " << num_inferences << " inferences." << std::endl;
    auto tpu_context =
        edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(tpu.type, tpu.path);
    auto model = coral::LoadModelOrDie(model_path.c_str());
    auto interpreter =
        coral::MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
    std::cout << "Thread: " << tid << " Interpreter was built." << std::endl;
    CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    auto input = coral::MutableTensorData<char>(*interpreter->input_tensor(0));
    coral::ReadFileToOrDie(image_path, input.data(), input.size());

    for (int i = 0; i < num_inferences; ++i)
      CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

    auto top = coral::GetTopClassificationResult(*interpreter);
    std::cout << "Thread: " << tid
              << " printing analysis result. Max value index: " << top.id
              << " value: " << top.score << std::endl;
  };

  const auto& start_time = std::chrono::steady_clock::now();
  std::thread bird_thread(run, available_tpus[0], bird_model_path,
                          bird_image_path);
  std::thread plant_thread(run, available_tpus[1], plant_model_path,
                           plant_image_path);
  bird_thread.join();
  plant_thread.join();
  std::chrono::duration<double> seconds =
      std::chrono::steady_clock::now() - start_time;
  std::cout << "Using two Edge TPUs, # inferences: " << num_inferences
            << " costs: " << seconds.count() << " seconds." << std::endl;
  return 0;
}
