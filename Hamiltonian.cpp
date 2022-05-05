#include "Hamiltonian.h"
#include <iostream>
#include <molpro/FCIdump.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

using namespace spin_orbital;

Hamiltonian::Hamiltonian(const molpro::FCIdump &fcidump)
    : norb(fcidump.parameter("NORB", std::vector<int>(1, 0))[0] * 2),
      nelec(fcidump.parameter("NELEC", std::vector<int>(1, 0))[0]),
      occ(fcidump.parameter("OCC", std::vector<int>(1, 0))),
      closed(fcidump.parameter("CLOSED", std::vector<int>(1, 0))),
      orbsym(fcidump.parameter("ORBSYM", std::vector<int>(1, 0))),
      uhf(fcidump.parameter("IUHF", std::vector<int>(1, 0))[0] != 0),
      spin_multiplicity(std::accumulate(occ.begin(), occ.end(), 0) - std::accumulate(closed.begin(), closed.end(), 0) +
                        1),
      h(Eigen::Tensor<double, 2>(norb, norb)), f(Eigen::Tensor<double, 2>(norb, norb)),
      dirac(Eigen::Tensor<double, 4>(norb, norb, norb, norb)),
      mulliken(Eigen::Tensor<double, 4>(norb, norb, norb, norb)) {
  auto norb_space = norb / 2;
  std::cout << (uhf ? "UHF" : "RHF") << " orbitals" << std::endl;
  std::cout << "spatial orbitals: " << norb_space << ", alpha electrons: " << (nelec + spin_multiplicity - 1) / 2
            << ", beta electrons: " << (nelec - spin_multiplicity + 1) / 2 << std::endl;
  h.setZero();
  mulliken.setZero();
  spin_orbital.clear();
  for (int spin = 0; spin < 2; ++spin)
    spin_orbital.emplace_back(norb_space + 1, -1);
  int orb = 0;
  // select occupied orbitals
  for (int sym = 0; sym < occ.size(); ++sym) {
    int nsym = 0;
    for (int i = 0; i < norb_space; ++i) {
      if (orbsym[i] == sym + 1) {
        if (nsym < occ[sym]) {
          spin_orbital[0][i + 1] = orb++;
        }
        if (nsym < closed[sym]) {
          spin_orbital[1][i + 1] = orb++;
        }
        ++nsym;
      }
    }
  }
  for (int sym = 0; sym < occ.size(); ++sym) {
    int nsym = 0;
    for (int i = 0; i < norb_space; ++i) {
      if (orbsym[i] == sym + 1) {
        ++nsym;
        if (nsym > occ[sym])
          spin_orbital[0][i + 1] = orb++;
        if (nsym > closed[sym])
          spin_orbital[1][i + 1] = orb++;
      }
    }
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
      if (!uhf)
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
      if (!uhf) {
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
      if (!uhf)
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
void Hamiltonian::dump(const std::string &filename) const {
  {
    auto dump = molpro::FCIdump();
    auto norb_space = norb / 2;
    dump.addParameter("NORB", norb_space);
    dump.addParameter("NELEC", nelec);
    dump.addParameter("ORBSYM", orbsym);
    dump.addParameter("IUHF", uhf ? 1 : 0);
    dump.addParameter("CLOSED", closed);
    dump.addParameter("OCC", occ);
    dump.write(filename, molpro::FCIdump::FileFormatted, false);
    if (uhf)
      for (int i = 0; i < 3; ++i)
        dump.writeIntegral(0, 0, 0, 0, 0);
    for (int spin = 0; spin < (uhf ? 2 : 1); ++spin) {
      for (int i = 1; i <= norb_space; ++i) {
        auto ii = spin_orbital[spin][i];
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
