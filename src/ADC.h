#ifndef PAPT_ADC_H
#define PAPT_ADC_H

#include "Hamiltonian.h"
#include "Amplitudes.h"

double IP_ADC(const spin_orbital::Hamiltonian &hamiltonian, const spin_orbital::Amplitudes &t1, int order=2);
double EA_ADC(const spin_orbital::Hamiltonian &hamiltonian, const spin_orbital::Amplitudes &t1, int order=2);

#endif //PAPT_ADC_H
