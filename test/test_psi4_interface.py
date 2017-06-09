from scfexchange.psi4_interface import Integrals, Orbitals
from scfexchange.test import integral_tests
from scfexchange.test import orbital_tests


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


def test__orbital_interface():
    orbital_tests.run_test__interface(Integrals, Orbitals)


def test__orbital_rotate():
    orbital_tests.run_test__rotate(Integrals, Orbitals)


def test__orbital_get_mo_count():
    orbital_tests.run_test__get_mo_count(Integrals, Orbitals)


def test__orbital_get_mo_slice():
    orbital_tests.run_test__get_mo_slice(Integrals, Orbitals)


def test__orbital_get_mo_coefficients():
    orbital_tests.run_test__get_mo_coefficients(Integrals, Orbitals)


def test__orbital_get_mo_fock_diagonal():
    orbital_tests.run_test__get_mo_fock_diagonal(Integrals, Orbitals)


def test__orbital_get_mo_1e_kinetic():
    orbital_tests.run_test__get_mo_1e_kinetic(Integrals, Orbitals)


def test__orbital_get_mo_1e_potential():
    orbital_tests.run_test__get_mo_1e_potential(Integrals, Orbitals)


def test__orbital_get_mo_1e_dipole():
    orbital_tests.run_test__get_mo_1e_dipole(Integrals, Orbitals)


def test__orbital_get_mo_1e_core_hamiltonian():
    orbital_tests.run_test__get_mo_1e_core_hamiltonian(Integrals, Orbitals)


def test__orbital_get_mo_1e_mean_field():
    orbital_tests.run_test__get_mo_1e_mean_field(Integrals, Orbitals)


def test__orbital_get_mo_1e_fock():
    orbital_tests.run_test__get_mo_1e_fock(Integrals, Orbitals)


def test__orbital_get_mo_2e_repulsion():
    orbital_tests.run_test__get_mo_2e_repulsion(Integrals, Orbitals)


def test__orbital_get_ao_1e_hf_density():
    orbital_tests.run_test__get_ao_1e_hf_density(Integrals, Orbitals)


def test__orbital_get_ao_1e_mean_field():
    orbital_tests.run_test__get_ao_1e_mean_field(Integrals, Orbitals)


def test__orbital_get_ao_1e_fock():
    orbital_tests.run_test__get_ao_1e_fock(Integrals, Orbitals)


def test__orbital_get_energy():
    orbital_tests.run_test__get_energy(Integrals, Orbitals)

