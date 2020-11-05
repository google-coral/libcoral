// An example to classify image.
// The input image size must match the input size of the model and be stored as
// RGB pixel array.
// In linux, you may resize and convert an existing image to pixel array like:
//   convert cat.bmp -resize 224x224! cat.rgb
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"

ABSL_FLAG(std::string, model_path, "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
          "Path to the tflite model.");
ABSL_FLAG(std::string, image_path, "cat.rgb",
          "Path to the image to be classified. The input image size must match "
          "the input size of the model and the image must be stored as RGB "
          "pixel array.");
ABSL_FLAG(std::string, labels_path, "imagenet_labels.txt",
          "Path to the imagenet labels.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Load the model.
  const auto model = coral::LoadModelOrDie(absl::GetFlag(FLAGS_model_path));
  auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
                             ? coral::GetEdgeTpuContextOrDie()
                             : nullptr;
  auto interpreter =
      coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // Read the image to input tensor.
  auto input = coral::MutableTensorData<char>(*interpreter->input_tensor(0));
  coral::ReadFileToOrDie(absl::GetFlag(FLAGS_image_path), input.data(),
                         input.size());
  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

  // Read the label file.
  auto labels = coral::ReadLabelFile(absl::GetFlag(FLAGS_labels_path));

  for (auto result :
       coral::GetClassificationResults(*interpreter, 0.0f, /*top_k=*/3)) {
    std::cout << "---------------------------" << std::endl;
    std::cout << labels[result.id] << std::endl;
    std::cout << "Score: " << result.score << std::endl;
  }
  return 0;
}
