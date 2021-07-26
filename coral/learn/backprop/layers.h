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

#ifndef LIBCORAL_CORAL_LEARN_BACKPROP_LAYERS_H_
#define LIBCORAL_CORAL_LEARN_BACKPROP_LAYERS_H_

#include <vector>

#include "Eigen/Core"

namespace coral {

// Computes the cross entropy between two probability distributions
// using CE(c,p) = - sum(c*log(p)) and returns the average loss of the batch.
// Inputs: vector of size 2 of [c, p]
// Both c and p tensors are size NxC where N is the number of distributions
// and C is the length of each distribution.
// When used with softmax, p is the probability output from softmax and
// c is a batch of one-hot vectors for class labels.
// Output: vector of size 1; tensor is 1x1 containing the average loss value.
Eigen::MatrixXf CrossEntropyLoss(const Eigen::MatrixXf& c,
                                 const Eigen::MatrixXf& p);

// This class computes the gradient of the Cross Entropy Loss with respect to
// each of the elements in probability distribution p
// A good reference for this is: https://deepnotes.io/softmax-crossentropy
// Inputs: vector of size 2 of [c, p]
// c and p described in CrossEntropyLoss class; Loss is output of the Compute
// method in CrossEntropyLoss class.
// Output: vector of size 1; tensor is NxC gradient with respect to p
Eigen::MatrixXf CrossEntropyGradient(const Eigen::MatrixXf& c,
                                     const Eigen::MatrixXf& p);

// Forward pass operator for the fully connected layer that computes Y = X*W + b
// A good reference for this is: http://cs231n.github.io/linear-classify/#score
// Input: vector of tensors in order of data mat_x, weights mat_w, and bias
// vec_b. mat_x is size NxD where N is number of inputs and D is number of
// dimensions. mat_w is size DxC. vec_b is size 1xC.
// Output: vector of size 1 that is layer output Y
Eigen::MatrixXf FullyConnected(const Eigen::MatrixXf& mat_x,
                               const Eigen::MatrixXf& mat_w,
                               const Eigen::MatrixXf& mat_b);

// Backward pass operator that computes gradients of the inputs to the fully
// connected layer
// A good reference for this is:
// http://cs231n.stanford.edu/2017/handouts/linear-backprop.pdf
// Input: vector of tensors in order of data mat_x, weights mat_w, bias b,
// grad dmat_y. The tensors mat_x, mat_w, vec_b are as described in
// FullyConnected class, dmat_y is size NxC.
// Output: vector of tensors of gradients in order of dmat_x, dmat_w, dvec_b
// and correspond in size to mat_x, mat_w, vec_b respectively
std::vector<Eigen::MatrixXf> FullyConnectedGradient(
    const Eigen::MatrixXf& mat_x, const Eigen::MatrixXf& mat_w,
    const Eigen::MatrixXf& b, const Eigen::MatrixXf& dmat_y);

// Forward pass operator for the softmax classifier layer that
// computes the probibilities of each sample in the tensor being in each class.
// A good reference for this is:
// http://cs231n.github.io/linear-classify/#softmax
// Input: vector of size 1 of unnormalized logits; tensor is NxC array
// where N is number of inputs and C is number of classes.
// Output: vector of size 1 of normalized probabilities; tensor is NxC array.
Eigen::MatrixXf Softmax(const Eigen::MatrixXf& logits);

// This class computes the gradient of the Softmax operator with respect to
// each of the elements in the vector of unnormalized logits.
// A good reference for this is: https://deepnotes.io/softmax-crossentropy

// Input: vector of size 2 of tensors [logits, dprobs].
// logits is NxC array where N is number of inputs and C is number of classes.
// dprobs is NXC array of gradients of Loss with respect to softmax output.
// Output: vector of size 1; tensor is NxC gradient of Loss with respect to
// logits.
Eigen::MatrixXf SoftmaxGradient(const Eigen::MatrixXf& logits,
                                const Eigen::MatrixXf& dprobs);

// Helper function to compute local gradient of softmax.
Eigen::MatrixXf SoftmaxLocalGradient(Eigen::MatrixXf::RowXpr prob);

// Updates the value of weights based on grads.
// Inputs: grads is a vector of tensors of gradients to be used to update the
// weights in a particular layer of a neural net, and weights is a vector of
// tensors of the weights that we want to update. Each element grads[i] is the
// same size as its corresponding element weights[i].
// When used to update a fully connected layer, the grads are dW and db from
// the output of FullyConnectedGradient.Compute and the weights are W and b.
// The learning rate is how fast the model learns; this value determines how
// much the weights are changed based on their gradient.
void SgdUpdate(const std::vector<Eigen::MatrixXf>& grads, float learning_rate,
               std::vector<Eigen::MatrixXf*> weights);
}  // namespace coral
#endif  // LIBCORAL_CORAL_LEARN_BACKPROP_LAYERS_H_
