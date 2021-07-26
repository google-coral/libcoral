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

#include "coral/learn/backprop/test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using ::Eigen::MatrixXf;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(CrossEntropyLossTest, OneInput) {
  MatrixXf probs = MatrixXf::Constant(1, 10, 0.1);
  MatrixXf labels = MatrixXf::Zero(1, 10);
  labels(0, 4) = 1;
  MatrixXf loss_expected(1, 1);
  loss_expected << -std::log(.1);

  EXPECT_THAT(CrossEntropyLoss(labels, probs).reshaped(),
              Pointwise(FloatEq(), loss_expected.reshaped()));
}

TEST(CrossEntropyLossTest, TwoInputs) {
  MatrixXf probs(2, 10);
  probs << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.02, 0.01,
      0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
  MatrixXf labels = MatrixXf::Zero(2, 10);
  labels(0, 4) = 1;
  labels(1, 0) = 1;
  MatrixXf loss_expected(1, 1);
  loss_expected << -1 * (std::log(.1) + std::log(.9)) / probs.rows();

  EXPECT_THAT(CrossEntropyLoss(labels, probs).reshaped(),
              Pointwise(FloatEq(), loss_expected.reshaped()));
}

TEST(CrossEntropyLossGradientTest, GradientofLoss) {
  MatrixXf probs(2, 10);
  probs << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.02, 0.01,
      0.01, 0.01, 0.01, 0.01, 0.9, 0.01, 0.01;
  MatrixXf labels = MatrixXf::Zero(2, 10);
  labels(0, 4) = 1;
  labels(1, 7) = 1;
  auto calculate_loss = [&labels](const MatrixXf& x) {
    return CrossEntropyLoss(labels, x)(0, 0);
  };

  EXPECT_THAT(CrossEntropyGradient(labels, probs).reshaped(),
              Pointwise(FloatNear(1e-3),
                        NumericalGradient(calculate_loss, &probs).reshaped()));
}

TEST(FullyConnectedTest, SimpleInputs) {
  MatrixXf mat_x(2, 5);
  mat_x << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXf mat_w = mat_x.transpose();
  MatrixXf vec_b(1, 2);
  vec_b << 1, 1;
  MatrixXf mat_y_expected(2, 2);
  mat_y_expected << 31, 81, 81, 256;

  EXPECT_THAT(FullyConnected(mat_x, mat_w, vec_b).reshaped(),
              Pointwise(FloatEq(), mat_y_expected.reshaped()));
}

TEST(FullyConnectedGradientTest, GradientOfAveragingElementsOfY) {
  MatrixXf mat_x(2, 3);
  mat_x << 0, 1, 2, 3, 4, 5;
  MatrixXf mat_w(3, 1);
  mat_w << 0, 1, 2;
  MatrixXf vec_b(1, 1);
  vec_b << 7;

  // dmat_y is a vector of constants equal to 1/numElements, because the
  // derivative of the Average value with respect to each element of mat_y is
  // 1/numElements
  MatrixXf dmat_y = MatrixXf::Ones(mat_x.rows(), mat_w.cols());
  dmat_y *= 1.0 / dmat_y.size();

  auto x_fc_avg = [&mat_w, &vec_b](const MatrixXf& x) {
    return FullyConnected(x, mat_w, vec_b).mean();
  };
  auto w_fc_avg = [&mat_x, &vec_b](const MatrixXf& w) {
    return FullyConnected(mat_x, w, vec_b).mean();
  };
  auto b_fc_avg = [&mat_x, &mat_w](const MatrixXf& b) {
    return FullyConnected(mat_x, mat_w, b).mean();
  };
  MatrixXf dx_numerical = NumericalGradient(x_fc_avg, &mat_x);
  MatrixXf dw_numerical = NumericalGradient(w_fc_avg, &mat_w);
  MatrixXf db_numerical = NumericalGradient(b_fc_avg, &vec_b);

  std::vector<MatrixXf> outputs =
      FullyConnectedGradient(mat_x, mat_w, vec_b, dmat_y);
  const auto& dmat_x = outputs[0];
  const auto& dmat_w = outputs[1];
  const auto& dvec_b = outputs[2];

  const float inf_norm_expected_dx = dmat_x.lpNorm<Eigen::Infinity>();
  const float inf_norm_diff_dx =
      (dmat_x - dx_numerical).lpNorm<Eigen::Infinity>();
  const float kEpsilon = 0.001f;
  EXPECT_LT(inf_norm_diff_dx / inf_norm_expected_dx, kEpsilon);

  const float inf_norm_expected_dw = dmat_w.lpNorm<Eigen::Infinity>();
  const float inf_norm_diff_dw =
      (dmat_w - dw_numerical).lpNorm<Eigen::Infinity>();
  EXPECT_LT(inf_norm_diff_dw / inf_norm_expected_dw, kEpsilon);

  const float inf_norm_expected_db = dvec_b.lpNorm<Eigen::Infinity>();
  const float inf_norm_diff_db =
      (dvec_b - db_numerical).lpNorm<Eigen::Infinity>();
  EXPECT_LT(inf_norm_diff_db / inf_norm_expected_db, kEpsilon);
}

TEST(FullyConnectedGradientTest, GradientOfSummingElementsOfY) {
  MatrixXf mat_x(3, 3);
  mat_x << 0, 1, 2, 3, 4, 5, 6, 7, 8;
  MatrixXf mat_w(3, 5);
  mat_w << 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2;
  MatrixXf vec_b(1, 5);
  vec_b << 4, 3, 2, 1, 0;

  // dmat_y is a vector of ones, because the derivative of the Summed value with
  // respect to each element of mat_y is 1.
  MatrixXf dmat_y = MatrixXf::Ones(mat_x.rows(), mat_w.cols());

  auto x_fc_sum = [&mat_w, &vec_b](const MatrixXf& x) {
    return FullyConnected(x, mat_w, vec_b).sum();
  };
  auto w_fc_sum = [&mat_x, &vec_b](const MatrixXf& w) {
    return FullyConnected(mat_x, w, vec_b).sum();
  };
  auto b_fc_sum = [&mat_x, &mat_w](const MatrixXf& b) {
    return FullyConnected(mat_x, mat_w, b).sum();
  };
  MatrixXf dx_numerical = NumericalGradient(x_fc_sum, &mat_x);
  MatrixXf dw_numerical = NumericalGradient(w_fc_sum, &mat_w);
  MatrixXf db_numerical = NumericalGradient(b_fc_sum, &vec_b);

  std::vector<MatrixXf> outputs =
      FullyConnectedGradient(mat_x, mat_w, vec_b, dmat_y);
  const auto& dmat_x = outputs[0];
  const auto& dmat_w = outputs[1];
  const auto& dvec_b = outputs[2];
}

TEST(SoftmaxTest, OneInput) {
  EXPECT_THAT(Softmax(MatrixXf::Ones(1, 10)).reshaped(),
              Pointwise(FloatEq(), MatrixXf::Constant(1, 10, 0.1).reshaped()));
}

TEST(SoftmaxTest, TwoInputs) {
  EXPECT_THAT(Softmax(MatrixXf::Ones(2, 5)).reshaped(),
              Pointwise(FloatEq(), MatrixXf::Constant(2, 5, 0.2).reshaped()));
}

TEST(SoftmaxTest, LargeInputs) {
  EXPECT_THAT(Softmax(MatrixXf::Constant(2, 5, 1000000)).reshaped(),
              Pointwise(FloatEq(), MatrixXf::Constant(2, 5, 0.2).reshaped()));
}

// Helper function to check that the local gradient of softmax used in
// SoftmaxGradient.Compute is calculated correctly
MatrixXf SoftmaxLocalGradientNaive(const MatrixXf& prob_row) {
  MatrixXf local(prob_row.size(), prob_row.size());
  MatrixXf kronecker = MatrixXf::Identity(prob_row.size(), prob_row.size());
  for (int i = 0; i < prob_row.size(); i++) {
    for (int j = 0; j < prob_row.size(); j++) {
      local(i, j) = prob_row(0, i) * (kronecker(i, j) - prob_row(0, j));
    }
  }
  return local;
}

TEST(SoftmaxGradientTest, LocalGradient) {
  MatrixXf test(1, 4);
  test << 7, 3, 8, 9;
  MatrixXf grad_naive = SoftmaxLocalGradientNaive(test);
  MatrixXf grad = SoftmaxLocalGradient(test.row(0));
  EXPECT_THAT(grad_naive.reshaped(),
              Pointwise(FloatNear(1e-3), grad.reshaped()));
}

TEST(SoftmaxGradientTest, GradientOfSummingElements) {
  MatrixXf logits(1, 10);
  logits << 5, 2, 8, 9, 1, -15, 0, 0, -6, 5.4;
  // dprobs is a vector of ones, because the derivative of the summed value with
  // respect to each element of probs is 1.
  MatrixXf dprobs = MatrixXf::Ones(logits.rows(), logits.cols());

  auto x_softmax_sum = [](const MatrixXf& x) { return Softmax(x).sum(); };

  EXPECT_THAT(SoftmaxGradient(logits, dprobs).reshaped(),
              Pointwise(FloatNear(1e-3),
                        NumericalGradient(x_softmax_sum, &logits).reshaped()));
}

TEST(SoftmaxGradientTest, GradientOfSummingElementsTwoInputs) {
  MatrixXf logits(2, 5);
  logits << 5, 2, 8, 9, 1, -15, 0, 0, -6, 5.4;
  // dprobs is a vector of ones, because the derivative of the summed value with
  // respect to each element of probs is 1.
  MatrixXf dprobs = MatrixXf::Ones(logits.rows(), logits.cols());

  auto x_softmax_sum = [](const MatrixXf& x) { return Softmax(x).sum(); };

  EXPECT_THAT(SoftmaxGradient(logits, dprobs).reshaped(),
              Pointwise(FloatNear(1e-3),
                        NumericalGradient(x_softmax_sum, &logits).reshaped()));
}

TEST(SoftmaxGradientTest, GradientOfCrossEntropyLoss) {
  MatrixXf logits(1, 10);
  logits << 5, 2, 8, 9, 1, -15, 0, 0, -6, 5.4;
  MatrixXf labels = MatrixXf::Zero(1, 10);
  labels(0, 4) = 1;

  auto x_softmax_cel = [&labels](const MatrixXf& x) {
    return CrossEntropyLoss(labels, Softmax(x))(0, 0);
  };

  MatrixXf dprobs = CrossEntropyGradient(labels, Softmax(logits));
  EXPECT_THAT(SoftmaxGradient(logits, dprobs).reshaped(),
              Pointwise(FloatNear(1e-3),
                        NumericalGradient(x_softmax_cel, &logits).reshaped()));
}

TEST(SgdUpdaterTest, UpdateWeights) {
  MatrixXf mat_w(2, 5);
  mat_w << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXf vec_b(1, 2);
  vec_b << 1, 1;

  MatrixXf mat_w_expected = mat_w / 2;
  MatrixXf vec_b_expected = vec_b / 2;

  SgdUpdate({mat_w / 2, vec_b / 2}, /*learning_rate=*/1.0, {&mat_w, &vec_b});

  EXPECT_THAT(mat_w.reshaped(),
              Pointwise(FloatNear(1e-3), mat_w_expected.reshaped()));
  EXPECT_THAT(vec_b.reshaped(),
              Pointwise(FloatNear(1e-3), vec_b_expected.reshaped()));
}

}  // namespace
}  // namespace coral
