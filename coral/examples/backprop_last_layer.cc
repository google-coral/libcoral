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

/*
A demo for on-device backprop (transfer learning) of a classification model.

This demo runs a similar task as described in TF Poets tutorial, except that
learning happens on-device.
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

Here are the steps to prepare the experiment data set:
  1) mkdir retrain
  2) curl http://download.tensorflow.org/example_images/flower_photos.tgz \
       | tar xz -C retrain
  3) find retrain -name *.jpg -exec convert {} -resize 224x224! {}.rgb \;

For more information, see
https://coral.ai/docs/edgetpu/retrain-classification-ondevice-backprop/
*/

#include <dirent.h>

#include <chrono>  // NOLINT
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/learn/backprop/softmax_regression_model.h"
#include "coral/tflite_utils.h"
#include "flatbuffers/flatbuffers.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

ABSL_FLAG(std::string, embedding_extractor_path,
          "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite",
          "Path to the embedding extractor tflite model.");
ABSL_FLAG(std::string, data_dir, "retrain/flower_photos",
          "Path to the data set.  The input image size must match the input "
          "size of the embedding model and the images must be stored as RGB "
          "pixel array.");
ABSL_FLAG(std::string, output_model_path, "trained_model_edgetpu.tflite",
          "Path to the output tflite model.");

namespace coral {
namespace {

constexpr double kValidationDataRatio = 0.1;
constexpr double kTestDataRatio = 0.1;
constexpr int kNumTrainingIterations = 500;
constexpr int kBatchSize = 100;
constexpr int kPrintEvery = 100;

struct LabelAndPath {
  int label;
  std::string image_path;
};

// Returns list of .rgb file paths under the given folder.
std::vector<std::string> ListRgbFilesUnderDir(const std::string& parent_dir) {
  std::vector<std::string> file_list;
  file_list.reserve(4096);  // reserve enough elements for the flower dataset.
  DIR* dir = opendir(parent_dir.c_str());
  if (dir) {
    struct dirent* file = nullptr;
    while ((file = readdir(dir)) != nullptr) {
      if (strstr(file->d_name, ".rgb"))
        file_list.push_back(parent_dir + "/" + file->d_name);
    }
    closedir(dir);
  }
  LOG(INFO) << file_list.size() << " rgb files found in folder " << parent_dir;
  return file_list;
}

// Returns number of classes.
int ListFilesInSubdirs(const std::string& grandparent_dir,
                       std::vector<LabelAndPath>* label_and_paths) {
  CHECK(label_and_paths);
  DIR* dir = opendir(grandparent_dir.c_str());
  int label = 0;
  if (dir) {
    struct dirent* subdir = nullptr;
    while ((subdir = readdir(dir)) != nullptr) {
      if (subdir->d_type != DT_DIR) continue;
      if (!std::strcmp(subdir->d_name, ".")) continue;
      if (!std::strcmp(subdir->d_name, "..")) continue;
      // Read samples of a new class.
      LOG(INFO) << "Read samples from subfolder " << subdir->d_name
                << " for class label " << label;
      for (const auto& file :
           ListRgbFilesUnderDir(grandparent_dir + "/" + subdir->d_name)) {
        label_and_paths->push_back({label, file});
      }

      ++label;
    }
    closedir(dir);
  }
  return label;
}

// Returns number of classes.
int SplitDataset(const std::string& data_dir,
                 std::vector<LabelAndPath>* training_label_and_paths,
                 std::vector<LabelAndPath>* validation_label_and_paths,
                 std::vector<LabelAndPath>* test_label_and_paths) {
  CHECK(training_label_and_paths);
  CHECK(validation_label_and_paths);
  CHECK(test_label_and_paths);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  std::vector<LabelAndPath> all_label_and_paths;
  const int num_classes = ListFilesInSubdirs(data_dir, &all_label_and_paths);
  for (const auto& p : all_label_and_paths) {
    const double r = dis(gen);
    if (r < kValidationDataRatio) {
      validation_label_and_paths->push_back(p);
    } else if (r < kValidationDataRatio + kTestDataRatio) {
      test_label_and_paths->push_back(p);
    } else {
      training_label_and_paths->push_back(p);
    }
  }
  LOG(INFO) << "Number of training samples: "
            << training_label_and_paths->size();
  LOG(INFO) << "Number of validation samples: "
            << validation_label_and_paths->size();
  LOG(INFO) << "Number of test samples: " << test_label_and_paths->size();
  return num_classes;
}

void ExtractEmbedding(tflite::Interpreter* extractor,
                      const std::vector<LabelAndPath>& label_and_paths,
                      int num_classes, int feature_dim,
                      Eigen::MatrixXf* embeddings, std::vector<int>* labels) {
  CHECK(extractor);
  CHECK(embeddings);
  CHECK(labels);

  embeddings->resize(label_and_paths.size(), feature_dim);
  labels->resize(label_and_paths.size(), -1);

  auto input = MutableTensorData<char>(*extractor->input_tensor(0));
  for (int i = 0; i < label_and_paths.size(); ++i) {
    const auto& entry = label_and_paths[i];

    CHECK_LT(entry.label, num_classes);

    ReadFileToOrDie(entry.image_path, input.data(), input.size());
    CHECK_EQ(extractor->Invoke(), kTfLiteOk);
    auto embedding = DequantizeTensor<float>(*extractor->output_tensor(0));
    CHECK_EQ(embedding.size(), feature_dim);
    embeddings->row(i) =
        Eigen::Map<const Eigen::VectorXf>(embedding.data(), embedding.size());

    // Create one-hot label vector.
    (*labels)[i] = entry.label;
  }
}

// Returns classification accuracy.
float EvaluateTrainedModel(const flatbuffers::FlatBufferBuilder& fbb,
                           const std::vector<LabelAndPath>& label_and_paths) {
  auto model = LoadModelOrDie(fbb);
  auto tpu_context =
      ContainsEdgeTpuCustomOp(*model) ? GetEdgeTpuContextOrDie() : nullptr;
  auto interpreter = MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  auto input = MutableTensorData<char>(*interpreter->input_tensor(0));
  int num_hits = 0;
  for (auto& entry : label_and_paths) {
    ReadFileToOrDie(entry.image_path, input.data(), input.size());
    CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
    auto top = GetTopClassificationResult(*interpreter);
    if (top.id == entry.label) ++num_hits;
  }
  return static_cast<float>(num_hits) / label_and_paths.size();
}

void TrainAndEvaluate(const std::string& embedding_extractor_path,
                      const std::string& data_dir,
                      const std::string& output_model_path) {
  auto model = LoadModelOrDie(embedding_extractor_path);
  auto tpu_context =
      ContainsEdgeTpuCustomOp(*model) ? GetEdgeTpuContextOrDie() : nullptr;
  auto embedding_extractor =
      MakeEdgeTpuInterpreterOrDie(*model, tpu_context.get());
  CHECK_EQ(embedding_extractor->AllocateTensors(), kTfLiteOk);

  CHECK_EQ(embedding_extractor->outputs().size(), 1);
  const int feature_dim = TensorSize(*embedding_extractor->output_tensor(0));

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<LabelAndPath> training_label_and_paths,
      validation_label_and_paths, test_label_and_paths;
  const int num_classes =
      SplitDataset(data_dir, &training_label_and_paths,
                   &validation_label_and_paths, &test_label_and_paths);

  TrainingData training_data;
  ExtractEmbedding(embedding_extractor.get(), training_label_and_paths,
                   num_classes, feature_dim, &training_data.training_data,
                   &training_data.training_labels);
  ExtractEmbedding(embedding_extractor.get(), validation_label_and_paths,
                   num_classes, feature_dim, &training_data.validation_data,
                   &training_data.validation_labels);

  // Run training
  const auto t1 = std::chrono::steady_clock::now();
  SoftmaxRegressionModel regression_model(feature_dim, num_classes);
  regression_model.Train(
      training_data,
      /*train_config=*/{kNumTrainingIterations, kBatchSize, kPrintEvery},
      /*learning_rate=*/.01);

  const auto t2 = std::chrono::steady_clock::now();
  LOG(INFO)
      << "Time to get embedding vectors (ms): "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  LOG(INFO)
      << "Time to train last layer (ms): "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  // Append learned weights to input model and save as tflite format.
  flatbuffers::FlatBufferBuilder fbb;
  regression_model.AppendLayersToEmbeddingExtractor(*model->GetModel(), &fbb);

  // Evaluate the trained model.
  LOG(INFO) << "Accuracy on training data: "
            << EvaluateTrainedModel(fbb, training_label_and_paths);
  LOG(INFO) << "Accuracy on validation data: "
            << EvaluateTrainedModel(fbb, validation_label_and_paths);
  LOG(INFO) << "Accuracy on test data: "
            << EvaluateTrainedModel(fbb, test_label_and_paths);
}

}  // namespace
}  // namespace coral

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  coral::TrainAndEvaluate(absl::GetFlag(FLAGS_embedding_extractor_path),
                          absl::GetFlag(FLAGS_data_dir),
                          absl::GetFlag(FLAGS_output_model_path));
}
