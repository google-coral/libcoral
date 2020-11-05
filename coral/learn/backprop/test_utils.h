#ifndef EDGETPU_CPP_LEARN_BACKPROP_TEST_UTILS_H_
#define EDGETPU_CPP_LEARN_BACKPROP_TEST_UTILS_H_

#include <functional>

#include "Eigen/Core"
#include "coral/learn/backprop/softmax_regression_model.h"

namespace coral {

// Func takes in tensor and outputs a scalar value
using Func = std::function<float(const Eigen::MatrixXf&)>;

// Gets numerical gradient of f at point x to use in backprop gradient checking
// uses dx = f(x+h)-f(x-h)/(2*h) as numerical approximation where h is epsilon
Eigen::MatrixXf NumericalGradient(Func f, Eigen::MatrixXf* x,
                                  float epsilon = 1e-3);

// Helper function to generate data in which examples from the same class are
// drawn from the same MultiVariate Normal (MVN) Distribution.
// Note that this function generates real random values. If this leads to
// flakiness consider change it to create same pseudo random value sequence.
TrainingData GenerateMvnRandomData(const std::vector<int>& class_sizes,
                                   const std::vector<Eigen::VectorXf>& means,
                                   const std::vector<Eigen::MatrixXf>& cov_mats,
                                   int num_train);

// Helper function to generate data in which examples are drawn from Uniform
// Distribution. Note that this function generates real random values. If this
// leads to flakiness consider change it to create same pseudo random value
// sequence.
TrainingData GenerateUniformRandomData(const std::vector<int>& class_sizes,
                                       int feature_dim, int num_train);

}  // namespace coral
#endif  // EDGETPU_CPP_LEARN_BACKPROP_TEST_UTILS_H_
