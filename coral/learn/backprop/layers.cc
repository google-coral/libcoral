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

#include "coral/learn/backprop/layers.h"

namespace coral {
namespace {
using ::Eigen::MatrixXf;

// Computes cross entropy loss between two probability distributions c and p.
float GetLoss(const MatrixXf& c, const MatrixXf& p) {
  MatrixXf logp = p.array().log();
  MatrixXf loss = -c.cwiseProduct(logp).array().rowwise().sum();
  return loss.mean();
}
}  // namespace

MatrixXf CrossEntropyLoss(const MatrixXf& c, const MatrixXf& p) {
  return MatrixXf::Constant(1, 1, GetLoss(c, p));
}

// Gradient of loss with respect to each element ij in input p is:
// dloss/d(pij) = 1/n * -cij/pij where n is the number of rows in p.
MatrixXf CrossEntropyGradient(const MatrixXf& c, const MatrixXf& p) {
  return 1.0 / p.rows() * -c.array() / p.array();
}

MatrixXf FullyConnected(const MatrixXf& mat_x, const MatrixXf& mat_w,
                        const MatrixXf& mat_b) {
  MatrixXf mat_y = mat_x * mat_w;
  mat_y.array().rowwise() += mat_b.array()(0, Eigen::indexing::all);
  return mat_y;
}

std::vector<MatrixXf> FullyConnectedGradient(const MatrixXf& mat_x,
                                             const MatrixXf& mat_w,
                                             const MatrixXf& b,
                                             const MatrixXf& dmat_y) {
  // Outputs: tensors of [dmat_x, dmat_w, dvec_b]
  // dmat_x = dmat_y * mat_w^T
  // dmat_w = mat_x^T * dmat_y
  // dvec_b = dmat_y^T * [1]
  MatrixXf dmat_x = dmat_y * mat_w.transpose();
  MatrixXf dmat_w = mat_x.transpose() * dmat_y;
  MatrixXf dmat_b =
      (dmat_y.transpose() * MatrixXf::Ones(mat_x.rows(), 1)).transpose();
  return {dmat_x, dmat_w, dmat_b};
}

MatrixXf Softmax(const MatrixXf& logits) {
  MatrixXf exps =
      (logits.array().colwise() - logits.array().rowwise().maxCoeff()).exp();
  return exps.array().colwise() / exps.array().rowwise().sum();
}

// Helper function to compute the local gradient dprobs/dlogits.
// Given a single logit input prob of dimension 1XC, the local gradient is size
// CxC where C is the number of classes.
// dprobi/dlogitj = probi*(kij - probj) where probi is output of softmax at
// index i, logitj is input logit to softmax at index j, kij is kronecker_delta
// function at position ij
MatrixXf SoftmaxLocalGradient(MatrixXf::RowXpr prob) {
  MatrixXf kronecker_delta = MatrixXf::Identity(prob.size(), prob.size());
  MatrixXf local = kronecker_delta.array().rowwise() - prob.array();
  return prob.asDiagonal() * local;
}

// Multiplies dloss/dprobs by dprobs/dlogits to output dloss/dlogits = grad.
MatrixXf SoftmaxGradient(const MatrixXf& logits, const MatrixXf& dprobs) {
  MatrixXf probs = Softmax(logits);
  MatrixXf grad = MatrixXf::Zero(probs.rows(), probs.cols());
  for (int i = 0; i < probs.rows(); i++) {
    MatrixXf local_grad = SoftmaxLocalGradient(probs.row(i));
    grad.row(i) = (dprobs.row(i) * local_grad);
  }
  return grad;
}

void SgdUpdate(const std::vector<MatrixXf>& grads, float learning_rate,
               std::vector<MatrixXf*> weights) {
  for (int i = 0; i < weights.size(); i++)
    *weights[i] -= learning_rate * grads[i];
}

}  // namespace coral
