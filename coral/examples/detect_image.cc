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

// An example to detect objects in an image.
// The input image size must match the input size of the model and be stored as
// RGB pixel array.
// In linux, with the imagemagick package installed, you may resize and convert an existing image to pixel array like:
//   convert kite_and_cold.jpg -resize 300x300! kite_and_cold-300x300.rgb
#include <cmath>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/detection/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "tensorflow/lite/interpreter.h"

ABSL_FLAG(std::string, model_path, "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
          "Path to the tflite model.");
ABSL_FLAG(std::string, image_path, "cat.rgb",
          "Path to the image to objects detected. The input image size must match "
          "the input size of the model and the image must be stored as RGB "
          "pixel array.");
ABSL_FLAG(std::string, labels_path, "coco_labels.txt",
          "Path to the coco labels.");
ABSL_FLAG(float, input_mean, 128, "Mean value for input normalization.");
ABSL_FLAG(float, input_std, 128, "STD value for input normalization.");
ABSL_FLAG(float, threshold, 0.2f, "Score threshold for detected objects.");
ABSL_FLAG(int, top_k, 10, "The best number of matches to return.");

int main(int argc, char* argv[]) {
	absl::ParseCommandLine(argc, argv);

	// Load the model.
	const auto model = coral::LoadModelOrDie(absl::GetFlag(FLAGS_model_path));
	auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
                             ? coral::GetEdgeTpuContextOrDie()
                             : nullptr;
	auto interpreter = coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
	CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

	// Check whether input data need to be preprocessed.
	// Image data must go through two transforms before running inference:
	// 1. normalization, f = (v - mean) / std
	// 2. quantization, q = f / scale + zero_point
	// Preprocessing combines the two steps:
	// q = (f - mean) / (std * scale) + zero_point
	// When std * scale equals 1, and mean - zero_point equals 0, the image data
	// do not need any preprocessing. In practice, it is probably okay to skip
	// preprocessing for better efficiency when the normalization and quantization
	// parameters approximate, but do not exactly meet the above conditions.
	CHECK_EQ(interpreter->inputs().size(), 1UL);
	const auto* input_tensor = interpreter->input_tensor(0);
	CHECK_EQ(input_tensor->type, kTfLiteUInt8)
		<< "Only support uint8 input type.";
	const float scale = input_tensor->params.scale;
	const float zero_point = input_tensor->params.zero_point;
	const float mean = absl::GetFlag(FLAGS_input_mean);
	const float std = absl::GetFlag(FLAGS_input_std);
	auto input = coral::MutableTensorData<uint8_t>(*input_tensor);
	const int input_size = 300;
	std::cout << "Expecting " << input_size << "x" << input_size << " input." << std::endl;
	if (std::abs(scale * std - 1) < 1e-5 && std::abs(mean - zero_point) < 1e-5) {
		// Read the image directly into input tensor as there is no preprocessing
		// needed.
		std::cout << "Input data does not require preprocessing." << std::endl;
		coral::ReadFileToOrDie(absl::GetFlag(FLAGS_image_path),
			reinterpret_cast<char*>(input.data()), input.size());
	} else {
		std::cout << "Input data requires preprocessing." << std::endl;
		std::vector<uint8_t> image_data(input.size());
		coral::ReadFileToOrDie(absl::GetFlag(FLAGS_image_path),
			reinterpret_cast<char*>(image_data.data()),
			input.size());
		for (uint8_t i = 0; i < input.size(); ++i) {
			const float tmp = (image_data[i] - mean) / (std * scale) + zero_point;
			if (tmp > 255) {
				input[i] = 255;
			} else if (tmp < 0) {
				input[i] = 0;
			} else {
				input[i] = static_cast<uint8_t>(tmp);
			}
		}
	}

	CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

	// Read the label file.
	auto labels = coral::ReadLabelFile(absl::GetFlag(FLAGS_labels_path));
	
	float threshold = absl::GetFlag(FLAGS_threshold);
	int top_k = absl::GetFlag(FLAGS_top_k);
	for (coral::Object result : coral::GetDetectionResults(*interpreter, threshold, top_k)) {
		std::cout << "---------------------------" << std::endl;
		std::cout << labels[result.id] << std::endl;
		std::cout << "Position: " << 
			"x=" << result.bbox.xmin * input_size <<
			",y=" << result.bbox.ymin * input_size <<
			",width=" << (result.bbox.xmax - result.bbox.xmin) * input_size <<
			",height=" << (result.bbox.ymax - result.bbox.ymin) * input_size << std::endl;
		std::cout << "Score: " << result.score << std::endl;
	}
	
	return 0;
}

