#include "LA.h"
#include <Eigen/QR>
Eigen::VectorXd spin_orbital::linsolve(const Eigen::MatrixXd& kernel, const Eigen::VectorXd& rhs) {
  Eigen::VectorXd result(kernel.rows());
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(kernel);
  result = cod.solve(rhs);
  return result;
}
