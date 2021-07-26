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

// Example to run two models alternatively using one Edge TPU.
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
// 7. Build and run `two_models_one_tpu`
//
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"

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
ABSL_FLAG(int, batch_size, 10, "Size of the batnch.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto bird_model_path = absl::GetFlag(FLAGS_bird_model_path);
  const auto plant_model_path = absl::GetFlag(FLAGS_plant_model_path);
  const auto bird_image_path = absl::GetFlag(FLAGS_bird_image_path);
  const auto plant_image_path = absl::GetFlag(FLAGS_plant_image_path);
  const int num_inferences = absl::GetFlag(FLAGS_num_inferences);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);

  std::cout << "Running model: " << bird_model_path
            << " and model: " << plant_model_path << " for " << num_inferences
            << " inferences" << std::endl;

  const auto& start_time = std::chrono::steady_clock::now();
  // Read inputs.
  auto bird_model = coral::LoadModelOrDie(bird_model_path.c_str());
  auto plant_model = coral::LoadModelOrDie(plant_model_path.c_str());

  // This context is shared among multiple models.
  auto tpu_context = coral::GetEdgeTpuContextOrDie();

  auto bird_interpreter =
      coral::MakeEdgeTpuInterpreterOrDie(*bird_model, tpu_context.get());
  CHECK_EQ(bird_interpreter->AllocateTensors(), kTfLiteOk);
  auto bird_input =
      coral::MutableTensorData<char>(*bird_interpreter->input_tensor(0));
  coral::ReadFileToOrDie(bird_image_path, bird_input.data(), bird_input.size());

  auto plant_interpreter =
      coral::MakeEdgeTpuInterpreterOrDie(*plant_model, tpu_context.get());
  CHECK_EQ(plant_interpreter->AllocateTensors(), kTfLiteOk);
  auto plant_input =
      coral::MutableTensorData<char>(*plant_interpreter->input_tensor(0));
  coral::ReadFileToOrDie(plant_image_path, plant_input.data(),
                         plant_input.size());

  // Run inference alternately and report timing.
  int num_iterations = (num_inferences + batch_size - 1) / batch_size;
  for (int i = 0; i < num_iterations; ++i) {
    for (int j = 0; j < batch_size; ++j)
      CHECK_EQ(bird_interpreter->Invoke(), kTfLiteOk);

    for (int j = 0; j < batch_size; ++j)
      CHECK_EQ(plant_interpreter->Invoke(), kTfLiteOk);
  }

  std::chrono::duration<double> seconds =
      std::chrono::steady_clock::now() - start_time;

  // Print inference result.
  auto top_bird = coral::GetTopClassificationResult(*bird_interpreter);
  std::cout << "[Bird image analysis] max value index: " << top_bird.id
            << " value: " << top_bird.score << std::endl;

  auto top_plant = coral::GetTopClassificationResult(*plant_interpreter);
  std::cout << "[Plant image analysis] max value index: " << top_plant.id
            << " value: " << top_plant.score << std::endl;

  std::cout << "Using one Edge TPU, # inferences: " << num_inferences
            << " costs: " << seconds.count() << " seconds." << std::endl;
  return 0;
}
