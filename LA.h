#ifndef SPIN_ORBITAL__LA_H_
#define SPIN_ORBITAL__LA_H_
#include <Eigen/Core>
namespace spin_orbital {
Eigen::VectorXd linsolve(const Eigen::MatrixXd& kernel, const Eigen::VectorXd& rhs);
}
#endif // SPIN_ORBITAL__LA_H_
