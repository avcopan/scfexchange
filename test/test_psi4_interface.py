from scfexchange.psi4_interface import Integrals
from scfexchange.test import integral_tests


def test__integral_interface():
    integral_tests.run_test__interface(Integrals)


def test__integral_get_ao_1e_overlap():
    integral_tests.run_test__get_ao_1e_overlap(Integrals)


def test__integral_get_ao_1e_kinetic():
    integral_tests.run_test__get_ao_1e_kinetic(Integrals)


def test__integral_get_ao_1e_potential():
    integral_tests.run_test__get_ao_1e_potential(Integrals)


def test__integral_get_ao_1e_dipole():
    integral_tests.run_test__get_ao_1e_dipole(Integrals)


def test__integral_get_ao_1e_core_hamiltonian():
    integral_tests.run_test__get_ao_1e_core_hamiltonian(Integrals)


def test__integral_get_ao_2e_repulsion():
    integral_tests.run_test__get_ao_2e_repulsion(Integrals)
