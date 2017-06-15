import scfexchange.pyscf_interface as interface
from scfexchange.test import ao_tests


def test__ao_interface():
    ao_tests.run_test__interface(interface)


def test__ao_overlap():
    ao_tests.run_test__overlap(interface)


def test__ao_kinetic():
    ao_tests.run_test__kinetic(interface)


def test__ao_potential():
    ao_tests.run_test__potential(interface)


def test__ao_dipole():
    ao_tests.run_test__dipole(interface)


def test__ao_electron_repulsion():
    ao_tests.run_test__electron_repulsion(interface)


def test__ao_core_hamiltonian():
    ao_tests.run_test__core_hamiltonian(interface)


def test__ao_mean_field():
    ao_tests.run_test__mean_field(interface)


def test__ao_fock():
    ao_tests.run_test__fock(interface)

