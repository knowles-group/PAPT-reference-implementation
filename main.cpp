#include "Amplitudes.h"
#include "Hamiltonian.h"
#include "LA.h"
#include "ManyBody.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <molpro/FCIdump.h>

using molpro::FCIdump;
using namespace spin_orbital;

int main(int argc, char* argv[]) {
  std::cout << std::fixed << std::setprecision(8);
  if (argc < 2)
    throw std::out_of_range("must give FCIdump filename as first command-line argument");
  std::cout << "fcidump " << argv[1] << std::endl;
  auto fcidump = molpro::FCIdump(argv[1]);

  Hamiltonian hamiltonian(fcidump);

  std::cout << "Reference energy: " << hamiltonian.e0 + hamiltonian.e1 << std::endl;
  std::cout << "MP0 energy: " << hamiltonian.e0 << std::endl;

  Amplitudes Kijab(hamiltonian);
  auto amplitudes = Kijab.MP1(hamiltonian);
  const auto MP1action1 = SDaction(hamiltonian, amplitudes, true, false);
  const auto MP1action2 = SDaction(hamiltonian, amplitudes, false, true);
  const auto MP1action12 = SDaction(hamiltonian, amplitudes, true, true);

  std::cout << "MP2 energy contribution: " << amplitudes.energy(hamiltonian) << " or " << -(MP1action1 * amplitudes)
            << std::endl;
  std::cout << "MP2 energy: " << hamiltonian.e0 + hamiltonian.e1 + amplitudes.energy(hamiltonian) << std::endl;

  std::cout << "MP3 energy contribution: " << amplitudes * MP1action2 << std::endl;
  std::cout << "MP3 energy: "
            << hamiltonian.e0 + hamiltonian.e1 + amplitudes.energy(hamiltonian) + amplitudes * MP1action2 << std::endl;

  const auto papt_rhs = PAPT_action(amplitudes, MP1action12);

  auto papt_dimension = papt_rhs.rows();
  Eigen::MatrixXd papt_kernel(papt_dimension, papt_dimension);
  for (size_t i = 0; i < papt_dimension; ++i) {
    Eigen::VectorXd value(papt_dimension);
    value.setZero();
    value[i] = 1;
    auto line = PAPT_kernel_action(value, amplitudes);
    for (size_t j = 0; j < papt_dimension; ++j)
      papt_kernel(j, i) = line(j);
  }

  //  std::cout << "papt_rhs\n" << papt_rhs << std::endl;
  //  std::cout << "papt_kernel\n" << papt_kernel << std::endl;
  auto papt_solution = linsolve(papt_kernel, papt_rhs);
  //  std::cout << "papt_solution\n" << papt_rhs << std::endl;

  std::cout << "check: " << (papt_kernel * papt_solution - papt_rhs).norm() << std::endl;
  auto papt_operator = PAPT_unpack(papt_solution, hamiltonian);
  //  auto solver_raw = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(
  //      Eigen::Map<Eigen::MatrixXd>(papt_operator.f.data(), papt_operator.norb, papt_operator.norb));
  //  std::cout << "PAPT operator eigenvalues before shift: " << solver_raw.eigenvalues().transpose() << std::endl;
  //  std::cout << "hamiltonian.e0 " << hamiltonian.e0 << std::endl;
  //  std::cout << "papt_operator.e0 " << papt_operator.e0 << std::endl;
  for (int i = 0; i < hamiltonian.norb; ++i)
    papt_operator.f(i, i) += (hamiltonian.e0 - hamiltonian.ecore - papt_operator.e0) / hamiltonian.nelec;
  auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(
      Eigen::Map<Eigen::MatrixXd>(papt_operator.f.data(), papt_operator.norb, papt_operator.norb));
  std::cout << "PAPT operator eigenvalues: " << solver.eigenvalues().transpose() << std::endl;
  auto papt_operator_rotated = papt_operator;
  const Eigen::VectorXd eigval = solver.eigenvalues().eval();
  const Eigen::MatrixXd eigvalmat = eigval.asDiagonal();
  papt_operator_rotated.f =
      Eigen::TensorMap<const Eigen::Tensor<double, 2>>(eigvalmat.data(), eigval.rows(), eigval.rows());
  auto rotated_Kijab = Kijab.transform(solver.eigenvectors());
  auto amplitudes_PAPT1 = rotated_Kijab.MP1(papt_operator_rotated);
  double epapt2 = rotated_Kijab * amplitudes_PAPT1;
  std::cout << "PAPT2 energy contribution and total: " << epapt2 << " " << hamiltonian.e0 + hamiltonian.e1 + epapt2
            << std::endl;
  auto amplitudes_PAPT1_backrotated = amplitudes_PAPT1.transform(solver.eigenvectors().transpose());
  std::cout << "PAPT2 energy contribution: "
            << -(SDaction(papt_operator, amplitudes_PAPT1_backrotated, true, false) * amplitudes_PAPT1_backrotated)
            << std::endl;
  double epapt3 =
      SDaction(hamiltonian, amplitudes_PAPT1_backrotated, true, true) * amplitudes_PAPT1_backrotated + epapt2;
  std::cout << "PAPT3 energy contribution and total: " << epapt3 << " "
            << hamiltonian.e0 + hamiltonian.e1 + epapt2 + epapt3 << std::endl;
  if (argc > 3 && std::abs(epapt2 - std::stod(argv[3])) > 1e-6 || std::isnan(epapt2))
    throw std::runtime_error(std::string{"Second order energy does not match reference value "} + argv[3]);
  if (argc > 4 && std::abs(epapt3 - std::stod(argv[4])) > 1e-6 || std::isnan(epapt3))
    throw std::runtime_error(std::string{"Third order energy does not match reference value "} + argv[4]);

  if (argc > 2)
    papt_operator.dump(argv[2]);
  return 0;
}
