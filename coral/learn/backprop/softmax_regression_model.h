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

#ifndef LIBCORAL_CORAL_LEARN_BACKPROP_SOFTMAX_REGRESSION_MODEL_H_
#define LIBCORAL_CORAL_LEARN_BACKPROP_SOFTMAX_REGRESSION_MODEL_H_

#include <functional>
#include <memory>
#include <vector>

#include "coral/learn/backprop/layers.h"
#include "tensorflow/lite/model.h"

namespace coral {

// A struct to define the training configuration to be used for last-layer
// training.
struct TrainConfig {
  TrainConfig(int num_iter, int batch_size, int print_every)
      : num_iter(num_iter), batch_size(batch_size), print_every(print_every) {}
  int num_iter;
  int batch_size;
  int print_every;
};

// A struct to define the training and validation input to a model.
struct TrainingData {
  Eigen::MatrixXf training_data;
  Eigen::MatrixXf validation_data;
  std::vector<int> training_labels;
  std::vector<int> validation_labels;
};

// An implementation of the softmax regression function (multinominal logistic
// regression) that operates as the last layer of your classification model, and
// allows for on-device training with backpropagation (for this layer only).
// Input for this layer must be an image embedding, which should be the output
// of your embedding extractor (the backbone of your model). Once given here,
// the input is fed to a fully-connected layer where weights and bias are
// applied, and then passed to the softmax function to receive the final
// probability for each class.
class SoftmaxRegressionModel {
 public:
  // Inputs: feature_dim (int): The dimension of the input feature (length of
  // the feature vector).
  // num_classes (int): The number of output classes.
  // weight_scale (float): A weight factor for computing new weights. The
  // backpropagated weights are drawn from standard normal distribution, then
  // multiplied by this number to keep the scale small.
  // reg (float): The regularization strength.
  explicit SoftmaxRegressionModel(int feature_dim = 0, int num_classes = 0,
                                  float weight_scale = 0.01, float reg = 0.0);

  // Computes how many of the labels are classified correctly for a given input.
  // Inputs: tensor mat_x of size NXD of input data, Vector labels of size N.
  // Output: a decimal float between 0 and 1 of the accuracy.
  // Accuracy = number classified correctly / total inputs
  float GetAccuracy(const Eigen::MatrixXf& mat_x,
                    const std::vector<int>& correct_labels);

  // Runs an inference using the current weights.
  // Inputs: tensor mat_x of size NxD of input data (image embeddings)
  // Outputs: Vector of inferred label index for each input embedding
  std::vector<int> RunInference(const Eigen::MatrixXf& mat_x);

  // Finds the optimal weights and biases for the last layer according to the
  // specifications of weight_updater (sgd_updater for this model) and
  // train_config
  void Train(const TrainingData& data, const TrainConfig& train_config,
             float learning_rate);

  // Appends the learned weight and bias values to embedding extractor tflite
  // model and saves to file.
  void AppendLayersToEmbeddingExtractor(
      const tflite::Model& embedding_extractor_model,
      flatbuffers::FlatBufferBuilder* fbb) const;

 private:
  // Calculates the loss of the current model for the given data, using a
  // cross-entropy loss function.
  // Inputs: tensor mat_x of size NxD of input data (image embeddings) to test
  // wehere N is number of inputs and D is the dimension of each input. Tensor
  // labels of size NxC of labels in one-hot vector format where C is the number
  // of classes possible in the model
  // Output: value of the averaged cross entropy loss for each input embedding.
  float GetLoss(const Eigen::MatrixXf& mat_x, const Eigen::MatrixXf& labels);

  // Calculates the gradients of the current weights using cross entropy loss as
  // the loss function and backpropogation using CrossEntropyGradient,
  // SoftmaxGradient, and FullyConnectedGradient.
  // Uses same inputs as get_loss(). Outputs: vector of weights of the model:
  // mat_w_ and vec_b_
  std::vector<Eigen::MatrixXf> GetGrads(const Eigen::MatrixXf& mat_x,
                                        const Eigen::MatrixXf& labels);

  // Weights and biases of the model that we want to learn.
  Eigen::MatrixXf mat_w_;
  Eigen::MatrixXf vec_b_;

  // Minimum / maximum values of the last fully connected layer's output. These
  // values will be learned from datga and used to quantize the output of the
  // learned fully-connected layer.
  float logit_min_;
  float logit_max_;

  // Parameters set when initializing model.
  int feature_dim_ = 0;
  int num_classes_ = 0;
  float weight_scale_ = 0.01;
  float reg_ = 0.0;

  // Caches to improve backpropogation performance by storing intermediate
  // results of forward operators, in order of logits, probs, loss.
  std::vector<Eigen::MatrixXf> caches_;
};  // namespace

}  // namespace coral
#endif  // LIBCORAL_CORAL_LEARN_BACKPROP_SOFTMAX_REGRESSION_MODEL_H_
