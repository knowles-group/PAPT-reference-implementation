#include "utility.h"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <iomanip>
Eigen::VectorXd spin_orbital::linsolve(const Eigen::MatrixXd& kernel, const Eigen::VectorXd& rhs) {
  Eigen::VectorXd result(kernel.rows());
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(kernel);
  if (cod.rank() != kernel.rows() - 2) {
    std::cout << "problem size " << kernel.rows() << std::endl;
    std::cout << "rank " << cod.rank() << std::endl;
    Eigen::EigenSolver<Eigen::MatrixXd> solver(kernel);
    for (int i = 0; i < kernel.rows(); ++i) {
      if (std::abs(solver.eigenvalues()(i)) < 1e-8) {
        std::cout << "Small eigenvalue " << solver.eigenvalues()(i) << std::endl;
        std::cout << "Eigenvector " << solver.eigenvectors().col(i) << std::endl;
      }
    }
  }
  result = cod.solve(rhs);
  return result;
}

void spin_orbital::result(const molpro::PluginGuest& molproPlugin, const std::string& name, double value) {
  if (molproPlugin.active()) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(14);
    molproPlugin.send(std::string{"TAKE PROPERTY "}+ name);
    ss << value;
    molproPlugin.send(ss.str());
  }
}
