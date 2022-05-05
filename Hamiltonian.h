#ifndef SPIN_ORBITAL_MAIN_CPP_HAMILTONIAN_H_
#define SPIN_ORBITAL_MAIN_CPP_HAMILTONIAN_H_
#include <molpro/FCIdump.h>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
namespace spin_orbital {
class Hamiltonian {
public:
  int norb;
  int nelec;
  Hamiltonian(const molpro::FCIdump&);
  Hamiltonian(size_t basis_size, bool oneElectron = true, bool twoElectron = true);
  void dump(const std::string& filename) const;
  double ecore;
  double e0;
  double e1;
  bool uhf;
  int spin_multiplicity;
  std::vector<std::vector<int>> spin_orbital;
  Eigen::Tensor<double, 2> h;
  Eigen::Tensor<double, 2> f;
  Eigen::Tensor<double, 4> dirac;
  Eigen::Tensor<double, 4> mulliken;
};
} // namespace spin_orbital
#endif // SPIN_ORBITAL_MAIN_CPP_HAMILTONIAN_H_
