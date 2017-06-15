from scfexchange.psi4_interface import AOIntegrals
from scfexchange.test import integral_tests


def test__integral_interface():
    integral_tests.run_test__interface(AOIntegrals)


def test__integral_get_ao_1e_overlap():
    integral_tests.run_test__overlap(AOIntegrals)


def test__integral_get_ao_1e_kinetic():
    integral_tests.run_test__kinetic(AOIntegrals)


def test__integral_get_ao_1e_potential():
    integral_tests.run_test__potential(AOIntegrals)


def test__integral_get_ao_1e_dipole():
    integral_tests.run_test__dipole(AOIntegrals)


def test__integral_get_ao_1e_core_hamiltonian():
    integral_tests.run_test__core_hamiltonian(AOIntegrals)


def test__integral_get_ao_2e_repulsion():
    integral_tests.run_test__electron_repulsion(AOIntegrals)
