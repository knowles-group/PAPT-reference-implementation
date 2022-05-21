#ifndef SPIN_ORBITAL__AMPLITUDES_H_
#define SPIN_ORBITAL__AMPLITUDES_H_

#include <unsupported/Eigen/CXX11/Tensor>
namespace spin_orbital {

class Hamiltonian;
class Amplitudes {
public:
  Eigen::Tensor<double, 2> t1;
  Eigen::Tensor<double, 4> t2;
  Amplitudes(const Hamiltonian& hamiltonian);
  Amplitudes MP1(const Hamiltonian& hamiltonian);
  double energy(const Hamiltonian& hamiltonian) const;
  double operator*(const Amplitudes& other) const;
  std::string str(bool oneElectron = true, bool twoElectron = true) const;
  Amplitudes transform(const Eigen::MatrixXd rot) const;
  const Hamiltonian& reference_hamiltonian;
};
} // namespace spin_orbital

#endif // SPIN_ORBITAL__AMPLITUDES_H_
