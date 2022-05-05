#include "Amplitudes.h"
#include "Hamiltonian.h"
#include <iostream>
using namespace spin_orbital;
Amplitudes::Amplitudes(const Hamiltonian& hamiltonian) {
  t1 = decltype(t1)(hamiltonian.norb - hamiltonian.nelec, hamiltonian.nelec);
  for (int a = hamiltonian.nelec; a < hamiltonian.norb; ++a)
    for (int i = 0; i < hamiltonian.nelec; ++i)
      t1(a - hamiltonian.nelec, i) = hamiltonian.f(a, i);
  t2 = decltype(t2)(hamiltonian.norb - hamiltonian.nelec, hamiltonian.norb - hamiltonian.nelec, hamiltonian.nelec,
                    hamiltonian.nelec);
  for (int a = hamiltonian.nelec; a < hamiltonian.norb; ++a)
    for (int b = hamiltonian.nelec; b < hamiltonian.norb; ++b)
      for (int i = 0; i < hamiltonian.nelec; ++i)
        for (int j = 0; j < hamiltonian.nelec; ++j) {
          t2(a - hamiltonian.nelec, b - hamiltonian.nelec, i, j) = hamiltonian.dirac(a, b, i, j);
        }
}

double Amplitudes::energy(const Hamiltonian& hamiltonian) const {
  double result = 0;
  const auto& no = hamiltonian.nelec;
  const auto& m = hamiltonian.norb;
  for (int a = no; a < m; ++a)
    for (int i = 0; i < no; ++i)
      result += t1(a - no, i) * hamiltonian.f(a, i);
  for (int j = 0; j < no; ++j)
    for (int i = 0; i < no; ++i)
      for (int b = no; b < m; ++b)
        for (int a = no; a < m; ++a)
          result += double(0.25) * t2(a - no, b - no, i, j) * hamiltonian.dirac(a, b, i, j);
  return result;
}
double Amplitudes::operator*(const Amplitudes& other) const {
  double result = 0;
  const auto& no = t1.dimension(1);
  const auto& nv = t1.dimension(0);
  for (int i = 0; i < no; ++i)
    for (int a = 0; a < nv; ++a)
      result += t1(a, i) * other.t1(a, i);
  for (int j = 0; j < no; ++j)
    for (int i = 0; i < j; ++i)
      for (int b = 0; b < nv; ++b)
        for (int a = 0; a < b; ++a) {
          result += t2(a, b, i, j) * other.t2(a, b, i, j);
          //          if (
          //              t2(a, b, i, j) != -t2(b, a, i, j)
          //                  or t2(a, b, i, j) != -t2(a, b, j, i)
          //                  or t2(a, b, i, j) != t2(b, a, j, i))
          //            std::cout <<
          //                      "Missing asymmetry this " << a << b << i << j << " " << t2(a, b, i, j) << " " << t2(b,
          //                      a, i, j)
          //                      << " "
          //                      << t2(a, b, j, i) << " " << t2(b, a, j, i) << std::endl;
          //          if (
          //              other.t2(a, b, i, j) != -other.t2(b, a, i, j)
          //                  or other.t2(a, b, i, j) != -other.t2(a, b, j, i)
          //                  or other.t2(a, b, i, j) != other.t2(b, a, j, i))
          //            std::cout <<
          //                      "Missing asymmetry other " << a << b << i << j << " " << other.t2(a, b, i, j) << " "
          //                      << other.t2(b, a, i, j)
          //                      << " "
          //                      << other.t2(a, b, j, i) << " " << other.t2(b, a, j, i) << std::endl;
        }
  return result;
}
std::string Amplitudes::str(bool oneElectron, bool twoElectron) const {
  std::stringstream ss;
  const auto& no = t1.dimension(1);
  const auto& nv = t1.dimension(0);
  ss << "no=" << no << " nv=" << nv << "\n";
  if (oneElectron) {
    ss << "Amplitudes Tai:\n";
    for (int i = 0; i < no; ++i)
      for (int a = 0; a < nv; ++a)
        if (t1(a, i) != 0)
          ss << "a=" << a << " i=" << i << " " << t1(a, i) << "\n";
  }
  if (twoElectron) {
    ss << "Amplitudes Tabij:\n";
    for (int j = 0; j < no; ++j)
      for (int i = 0; i < j; ++i)
        for (int b = 0; b < nv; ++b)
          for (int a = 0; a < b; ++a) {
            if (t2(a, b, i, j) != 0)
              ss << "a=" << a << " b=" << b << " i=" << i << " j=" << j << " " << t2(a, b, i, j) << "\n";
            if (std::abs(t2(a, b, i, j) + t2(b, a, i, j)) > 1e-12 or
                std::abs(t2(a, b, i, j) + t2(a, b, j, i)) > 1e-12 or std::abs(t2(a, b, i, j) - t2(b, a, j, i)) > 1e-12)
              ss << "Amplitudes.str(): Missing asymmetry this " << a << b << i << j << " " << t2(a, b, i, j) << " "
                 << t2(b, a, i, j) + t2(a, b, i, j) << " " << t2(a, b, j, i) + t2(a, b, i, j) << " "
                 << t2(b, a, j, i) - t2(a, b, i, j) << std::endl;
          }
  }
  return ss.str();
}
Amplitudes Amplitudes::transform(const Eigen::MatrixXd rot) const {
  spin_orbital::Amplitudes result(*this);
  const auto no = t1.dimension(1);
  const auto nv = t1.dimension(0);
  result.t1.setZero();
  auto temp1 = result.t1;
  temp1.setZero();
  for (int i = 0; i < no; ++i)
    for (int a = 0; a < nv; ++a)
      for (int b = 0; b < nv; ++b)
        temp1(a, i) += t1(b, i) * rot(b + no, a + no);
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int a = 0; a < nv; ++a)
        result.t1(a, i) += temp1(a, j) * rot(j, i);
  auto temp = result.t2;
  temp.setZero();
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
          for (int k = 0; k < no; ++k)
            temp(a, b, i, j) += t2(a, b, i, k) * rot(k, j);
  result.t2.setZero();
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
          for (int k = 0; k < no; ++k)
            result.t2(a, b, i, j) += temp(a, b, k, j) * rot(k, i);
  temp.setZero();
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
          for (int c = 0; c < nv; ++c)
            temp(a, b, i, j) += result.t2(a, c, i, j) * rot(c + no, b + no);
  result.t2.setZero();
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
          for (int c = 0; c < nv; ++c)
            result.t2(a, b, i, j) += temp(c, b, i, j) * rot(c + no, a + no);
  return result;
}

Amplitudes Amplitudes::MP1(const Hamiltonian& hamiltonian) {
  auto result = *this;
  const auto no = t1.dimension(1);
  const auto nv = t1.dimension(0);
  for (int a = no; a < no + nv; ++a)
    for (int i = 0; i < no; ++i)
      result.t1(a - no, i) = t1(a - no, i) / (hamiltonian.f(i, i) - hamiltonian.f(a, a));
  for (int a = no; a < no + nv; ++a)
    for (int b = no; b < no + nv; ++b)
      for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
          result.t2(a - no, b - no, i, j) = t2(a - no, b - no, i, j) / (hamiltonian.f(i, i) + hamiltonian.f(j, j) -
                                                                        hamiltonian.f(a, a) - hamiltonian.f(b, b));
  return result;
}
