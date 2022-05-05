#include "Hamiltonian.h"
#include <iostream>
#include <molpro/FCIdump.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

using namespace spin_orbital;

Hamiltonian::Hamiltonian(const molpro::FCIdump& fcidump) {
  auto orbsym = fcidump.parameter("ORBSYM", std::vector<int>(1, 0));
  for (const auto& s : orbsym)
    if (s != 1)
      throw std::invalid_argument("Orbital symmetries must be 1");
  auto norb_space = fcidump.parameter("NORB", std::vector<int>(1, 0))[0];
  norb = norb_space * 2;
  nelec = fcidump.parameter("NELEC", std::vector<int>(1, 0))[0];
  auto iuhf = fcidump.parameter("IUHF", std::vector<int>(1, 0))[0];
  uhf = iuhf != 0;
  std::cout << (uhf ? "UHF" : "RHF") << " orbitals" << std::endl;
  auto nalpha = fcidump.parameter("OCC", std::vector<int>(1, 0))[0];
  auto nbeta = fcidump.parameter("CLOSED", std::vector<int>(1, 0))[0];
  spin_multiplicity = nalpha - nbeta + 1;
  if (nalpha + nbeta != nelec)
    throw std::invalid_argument("wrong numbers of electrons");
  std::cout << "spatial orbitals: " << norb_space << ", alpha electrons: " << nalpha << ", beta electrons: " << nbeta
            << std::endl;
  h = Eigen::Tensor<double, 2>(norb, norb);
  h.setZero();
  f = Eigen::Tensor<double, 2>(norb, norb);
  dirac = Eigen::Tensor<double, 4>(norb, norb, norb, norb);
  mulliken = Eigen::Tensor<double, 4>(norb, norb, norb, norb);
  mulliken.setZero();
  for (int spin = 0; spin < 2; ++spin)
    spin_orbital.emplace_back(1);
  int orb = 0;
  for (int i = 0; i < nbeta; ++i) {
    spin_orbital[0].push_back(orb++);
    spin_orbital[1].push_back(orb++);
  }
  for (int i = nbeta; i < nalpha; ++i)
    spin_orbital[0].push_back(orb++);
  for (int i = nbeta; i < nalpha; ++i)
    spin_orbital[1].push_back(orb++);
  for (int i = nalpha; i < norb_space; ++i) {
    spin_orbital[0].push_back(orb++);
    spin_orbital[1].push_back(orb++);
  }
  if (false) {
    for (int spin = 0; spin < 2; ++spin) {
      std::cout << "orbital map for spin=" << spin << std::endl;
      for (int i = 1; i < norb_space + 1; ++i)
        std::cout << " " << spin_orbital[spin][i];
      std::cout << std::endl;
    }
  }

  int i, j, k, l;
  double value;
  molpro::FCIdump::integralType type;
  fcidump.rewind();
  bool print = false;
  while ((type = fcidump.nextIntegral(i, j, k, l, value)) != molpro::FCIdump::endOfFile) {
    if (print) {
      if (type == molpro::FCIdump::I2aa or type == molpro::FCIdump::I2ab or type == molpro::FCIdump::I2bb)
        std::cout << "found " << ((type != molpro::FCIdump::I2bb) ? "alpha" : "beta") << "-"
                  << ((type == molpro::FCIdump::I2aa) ? "alpha" : "beta") << " integral (" << i << j << "|" << k << l
                  << ")=" << value << std::endl;
      else if (k == 0 && l == 0)
        std::cout << "found " << ((type != molpro::FCIdump::I1b) ? "alpha" : "beta") << " integral <" << i << "|h|" << j
                  << ">=" << value << std::endl;
      else if (type == molpro::FCIdump::I0)
        std::cout << "found "
                  << "scalar integral " << value << std::endl;
    }
    if (type == molpro::FCIdump::I2aa) {
      mulliken(spin_orbital[0][i], spin_orbital[0][j], spin_orbital[0][k], spin_orbital[0][l]) = value;
      mulliken(spin_orbital[0][j], spin_orbital[0][i], spin_orbital[0][k], spin_orbital[0][l]) = value;
      mulliken(spin_orbital[0][i], spin_orbital[0][j], spin_orbital[0][l], spin_orbital[0][k]) = value;
      mulliken(spin_orbital[0][j], spin_orbital[0][i], spin_orbital[0][l], spin_orbital[0][k]) = value;
      mulliken(spin_orbital[0][k], spin_orbital[0][l], spin_orbital[0][i], spin_orbital[0][j]) = value;
      mulliken(spin_orbital[0][l], spin_orbital[0][k], spin_orbital[0][i], spin_orbital[0][j]) = value;
      mulliken(spin_orbital[0][k], spin_orbital[0][l], spin_orbital[0][j], spin_orbital[0][i]) = value;
      mulliken(spin_orbital[0][l], spin_orbital[0][k], spin_orbital[0][j], spin_orbital[0][i]) = value;
      if (iuhf == 0)
        type = molpro::FCIdump::I2ab;
    }
    if (type == molpro::FCIdump::I2ab) {
      mulliken(spin_orbital[0][i], spin_orbital[0][j], spin_orbital[1][k], spin_orbital[1][l]) = value;
      mulliken(spin_orbital[0][j], spin_orbital[0][i], spin_orbital[1][k], spin_orbital[1][l]) = value;
      mulliken(spin_orbital[0][i], spin_orbital[0][j], spin_orbital[1][l], spin_orbital[1][k]) = value;
      mulliken(spin_orbital[0][j], spin_orbital[0][i], spin_orbital[1][l], spin_orbital[1][k]) = value;
      mulliken(spin_orbital[1][k], spin_orbital[1][l], spin_orbital[0][i], spin_orbital[0][j]) = value;
      mulliken(spin_orbital[1][l], spin_orbital[1][k], spin_orbital[0][i], spin_orbital[0][j]) = value;
      mulliken(spin_orbital[1][k], spin_orbital[1][l], spin_orbital[0][j], spin_orbital[0][i]) = value;
      mulliken(spin_orbital[1][l], spin_orbital[1][k], spin_orbital[0][j], spin_orbital[0][i]) = value;
      if (iuhf == 0) {
        mulliken(spin_orbital[1][i], spin_orbital[1][j], spin_orbital[0][k], spin_orbital[0][l]) = value;
        mulliken(spin_orbital[1][j], spin_orbital[1][i], spin_orbital[0][k], spin_orbital[0][l]) = value;
        mulliken(spin_orbital[1][i], spin_orbital[1][j], spin_orbital[0][l], spin_orbital[0][k]) = value;
        mulliken(spin_orbital[1][j], spin_orbital[1][i], spin_orbital[0][l], spin_orbital[0][k]) = value;
        mulliken(spin_orbital[0][k], spin_orbital[0][l], spin_orbital[1][i], spin_orbital[1][j]) = value;
        mulliken(spin_orbital[0][l], spin_orbital[0][k], spin_orbital[1][i], spin_orbital[1][j]) = value;
        mulliken(spin_orbital[0][k], spin_orbital[0][l], spin_orbital[1][j], spin_orbital[1][i]) = value;
        mulliken(spin_orbital[0][l], spin_orbital[0][k], spin_orbital[1][j], spin_orbital[1][i]) = value;
        type = molpro::FCIdump::I2bb;
      }
    }
    if (type == molpro::FCIdump::I2bb) {
      mulliken(spin_orbital[1][i], spin_orbital[1][j], spin_orbital[1][k], spin_orbital[1][l]) = value;
      mulliken(spin_orbital[1][j], spin_orbital[1][i], spin_orbital[1][k], spin_orbital[1][l]) = value;
      mulliken(spin_orbital[1][i], spin_orbital[1][j], spin_orbital[1][l], spin_orbital[1][k]) = value;
      mulliken(spin_orbital[1][j], spin_orbital[1][i], spin_orbital[1][l], spin_orbital[1][k]) = value;
      mulliken(spin_orbital[1][k], spin_orbital[1][l], spin_orbital[1][i], spin_orbital[1][j]) = value;
      mulliken(spin_orbital[1][l], spin_orbital[1][k], spin_orbital[1][i], spin_orbital[1][j]) = value;
      mulliken(spin_orbital[1][k], spin_orbital[1][l], spin_orbital[1][j], spin_orbital[1][i]) = value;
      mulliken(spin_orbital[1][l], spin_orbital[1][k], spin_orbital[1][j], spin_orbital[1][i]) = value;
    }
    if (type == molpro::FCIdump::I1a) {
      h(spin_orbital[0][i], spin_orbital[0][j]) = value;
      h(spin_orbital[0][j], spin_orbital[0][i]) = value;
      if (iuhf == 0)
        type = molpro::FCIdump::I1b;
    }
    if (type == molpro::FCIdump::I1b) {
      h(spin_orbital[1][i], spin_orbital[1][j]) = value;
      h(spin_orbital[1][j], spin_orbital[1][i]) = value;
    }
    if (type == molpro::FCIdump::I0) {
      ecore = value;
    }
  }

  for (int i = 0; i < norb; ++i)
    for (int j = 0; j < norb; ++j)
      for (int k = 0; k < norb; ++k)
        for (int l = 0; l < norb; ++l)
          dirac(i, j, k, l) = mulliken(i, k, j, l) - mulliken(i, l, j, k);

  f.setZero();
  for (int i = 0; i < norb; ++i)
    for (int j = 0; j < norb; ++j) {
      f(i, j) = h(i, j);
      for (int k = 0; k < nelec; ++k)
        f(i, j) += dirac(i, k, j, k);
    }
  e0 = ecore;
  e1 = 0;
  for (int i = 0; i < nelec; ++i) {
    e0 += f(i, i);
    e1 += double(0.5) * (h(i, i) - f(i, i));
  }

  bool fockdiag = true;
  for (int i = 0; i < norb; ++i)
    for (int j = 0; j < i; ++j)
      fockdiag = fockdiag and (std::abs(f(i, j)) < 1e-7);
  if (not fockdiag)
    std::cout << "Warning: Fock matrix is not diagonal" << std::endl;

  //  std::cout << "1-electron hamiltonian\n"<<h<<std::endl;
  //  std::cout << "Fock matrix\n";//<<f<<std::endl;
  //  for (int i = 0; i < norb; ++i)
  //    std::cout
  //        << f.slice(Eigen::array<Eigen::Index, 2>{i, 0}, Eigen::array<Eigen::Index, 2>{1, norb})
  //        << std::endl;
}

Hamiltonian::Hamiltonian(size_t basis_size, bool oneElectron, bool twoElectron) : norb(basis_size) {
  if (oneElectron) {
    h = Eigen::Tensor<double, 2>(norb, norb);
  }
  if (twoElectron) {
    dirac = Eigen::Tensor<double, 4>(norb, norb, norb, norb);
    mulliken = Eigen::Tensor<double, 4>(norb, norb, norb, norb);
  }
}
void Hamiltonian::dump(const std::string& filename) const {
  {
    auto dump = molpro::FCIdump();
    auto norb_space = norb / 2;
    dump.addParameter("NORB", norb_space);
    dump.addParameter("NELEC", nelec);
    dump.addParameter("ORBSYM", std::vector<int>(norb_space, 1));
    dump.addParameter("IUHF", uhf ? 1 : 0);
    dump.addParameter("CLOSED", (nelec - spin_multiplicity + 1) / 2);
    dump.addParameter("OCC", (nelec + spin_multiplicity - 1) / 2);
    dump.write(filename, molpro::FCIdump::FileFormatted, false);
    if (uhf)
      for (int i = 0; i < 3; ++i)
        dump.writeIntegral(0, 0, 0, 0, 0);
    for (int spin = 0; spin < (uhf ? 2 : 1); ++spin) {
      for (int i = 1; i <= norb_space; ++i) {
        auto ii = spin_orbital[spin][i];
        //        std::cout << "spin=" << spin << " space=" << i << " spin-orbital=" << ii << std::endl;
        for (int j = 1; j <= i; ++j) {
          auto jj = spin_orbital[spin][j];
          if (std::abs(f(ii, jj)) > 1e-13)
            dump.writeIntegral(i, j, 0, 0, f(ii, jj));
        }
      }
      if (uhf)
        dump.writeIntegral(0, 0, 0, 0, 0);
    }
    dump.writeIntegral(0, 0, 0, 0, ecore);
  }
  bool validate = true;
  if (validate) {
    auto dump = molpro::FCIdump(filename);
    auto hamiltonian = Hamiltonian(dump);
    for (int i = 0; i < norb; ++i)
      for (int j = 0; j < norb; ++j)
        if (std::abs(hamiltonian.f(i, j) - f(i, j)) > 1e-10) {
          std::cout << "original f\n" << f << std::endl;
          std::cout << "read in f\n" << hamiltonian.f << std::endl;
          throw std::runtime_error("faulty fcidump output");
        }
  }
}
