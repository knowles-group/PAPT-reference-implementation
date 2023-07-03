#include <iostream>
#include "ADC.h"
#include <Eigen/Dense>

double IP_ADC(const spin_orbital::Hamiltonian &hamiltonian, const spin_orbital::Amplitudes &t1, int order) {
    std::cout << "IP-ADC(" << order << ")" << std::endl;
//    std::cout << "hamiltonian.h\n" << hamiltonian.h << std::endl;
//    std::cout << "hamiltonian.f\n" << hamiltonian.f << std::endl;
//    std::cout << "t1\n" << t1.t2 <<std::endl;
    spin_orbital::Amplitudes Kijab(hamiltonian);
    if (order > 0) std::cout << "ground state second order energy " << Kijab * t1 << std::endl;
    const auto &no = hamiltonian.nelec;
    const auto nv = hamiltonian.norb - hamiltonian.nelec;
    const auto no2 = (no * (no - 1)) / 2;
    Eigen::MatrixXd m(no + no2 * nv, no + no2 * nv);
    m.setZero();

    for (int i = 0; i < no; ++i) {
        m(i, i) = hamiltonian.f(i, i);
        if (order > 1) {
            for (int j = 0; j < no; ++j)
                for (int k = 0; k < no; ++k)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            m(i, j) +=
                                    0.25 * t1.t2(a, b, i, k) * Kijab.t2(a, b, j, k)
                                    + 0.25 * t1.t2(a, b, j, k) * Kijab.t2(a, b, i, k);
        }
    }

    int jka = no;
    for (int j = 0; j < no; ++j)
        for (int k = 0; k < j; ++k)
            for (int a = 0; a < nv; ++a) {
                if (order > 1)
                    for (int i = 0; i < no; ++i)
                        m(jka, i) = m(i, jka) = hamiltonian.dirac(j, k, a + no, i);
                m(jka, jka) = hamiltonian.f(j, j) + hamiltonian.f(k, k) - hamiltonian.f(no + a, no + a);
                ++jka;
            }

//    std::cout << "m\n" << m << std::endl;
    auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(m);
//    std::cout << "Eigenvalues: " << solver.eigenvalues().transpose() << std::endl;
    std::cout << "IP: " << -solver.eigenvalues()[m.cols() - 1] << std::endl;
    return -solver.eigenvalues()[m.cols() - 1];
}

double EA_ADC(const spin_orbital::Hamiltonian &hamiltonian, const spin_orbital::Amplitudes &t1, int order) {
    std::cout << "EA-ADC(" << order << ")" << std::endl;
//    std::cout << "hamiltonian.h\n" << hamiltonian.h << std::endl;
//    std::cout << "hamiltonian.f\n" << hamiltonian.f << std::endl;
//    std::cout << "t1\n" << t1.t2 <<std::endl;
    spin_orbital::Amplitudes Kijab(hamiltonian);
    if (order > 0) std::cout << "ground state second order energy " << Kijab * t1 << std::endl;
    const auto &no = hamiltonian.nelec;
    const auto nv = hamiltonian.norb - hamiltonian.nelec;
    const auto nv2 = (nv * (nv - 1)) / 2;
    Eigen::MatrixXd m(nv + nv2 * no, nv + nv2 * no);
    m.setZero();

    for (int a = 0; a < nv; ++a) {
        m(a, a) = hamiltonian.f(no + a, no + a);
        if (order > 1) {
            for (int b = 0; b < nv; ++b)
                for (int c = 0; c < nv; ++c)
                    for (int i = 0; i < no; ++i)
                        for (int j = 0; j < no; ++j)
                            m(a, b) -=
                                    0.25 * t1.t2(a, c, i, j) * Kijab.t2(b, c, i, j)
                                    + 0.25 * t1.t2(b, c, i, j) * Kijab.t2(a, c, i, j);
        }
    }

    int bci = nv;
    for (int b = 0; b < nv; ++b)
        for (int c = 0; c < b; ++c)
            for (int i = 0; i < no; ++i) {
                if (order > 1)
                    for (int a = 0; a < nv; ++a)
                        m(bci, a) = m(a, bci) = hamiltonian.dirac(b + no, c + no, a + no, i);
                m(bci, bci) = hamiltonian.f(b + no, b + no) + hamiltonian.f(c + no, c + no) - hamiltonian.f(i, i);
                ++bci;
            }

//    std::cout << "m\n" << m << std::endl;
    auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(m);
//    std::cout << "Eigenvalues: " << solver.eigenvalues().transpose() << std::endl;
    std::cout << "EA: " << solver.eigenvalues()[0] << std::endl;
    return solver.eigenvalues()[0];
}
