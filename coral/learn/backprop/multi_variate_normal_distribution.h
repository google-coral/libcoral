#ifndef EDGETPU_CPP_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
#define EDGETPU_CPP_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
#include <random>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"

namespace coral {

// Multi variate normal distribution implemented with Eigen library.
class MultiVariateNormalDistribution {
 public:
  MultiVariateNormalDistribution(const Eigen::VectorXf& mean,
                                 const Eigen::MatrixXf& cov);

  // Samples 'num' samples from distribution.
  // Returns a [dim, num] shape matrix.
  Eigen::MatrixXf Sample(int num);

 private:
  // Mean of the distribution.
  Eigen::VectorXf mean_;

  // Covariance matrix of the distribution.
  Eigen::MatrixXf cov_;

  // Multiplies this matrix with a random variable X which is drawn from
  // N(0, I) will produce a sample drawn from N(0, cov_).
  Eigen::MatrixXf p_;

  // The dimension of the covariance matrix.
  int dim_;

  // Eigen solver which is used to compute eigen value and eigen vectors.
  Eigen::EigenSolver<Eigen::MatrixXf> solver_;

  // Gaussian random number generator.
  std::normal_distribution<float> rand_gaussian_;
};

}  // namespace coral

#endif  // EDGETPU_CPP_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
