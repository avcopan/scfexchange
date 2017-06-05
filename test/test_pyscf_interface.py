from scfexchange.pyscf_interface import Integrals, Orbitals
from scfexchange.test import integral_tests
from scfexchange.test import orbital_tests


def test__integral_interface():
    integral_tests.run_interface_check(Integrals)


def test__integral_get_ao_1e_overlap():
    integral_tests.run_ao_1e_overlap_check(Integrals)


def test__integral_get_ao_1e_kinetic():
    integral_tests.run_ao_1e_kinetic_check(Integrals)


def test__integral_get_ao_1e_potential():
    integral_tests.run_ao_1e_potential_check(Integrals)


def test__integral_get_ao_1e_dipole():
    integral_tests.run_ao_1e_dipole_check(Integrals)


def test__integral_get_ao_1e_core_hamiltonian():
    integral_tests.run_ao_1e_core_hamiltonian_check(Integrals)


def test__integral_get_ao_2e_repulsion():
    integral_tests.run_ao_2e_repulsion_check(Integrals)


def test__orbital_interface():
    orbital_tests.run_interface_check(Integrals, Orbitals)


def test__orbital_mo_counting():
    orbital_tests.run_mo_counting_check(Integrals, Orbitals)


def test__orbital_mo_slicing():
    orbital_tests.run_mo_slicing_check(Integrals, Orbitals)


def test__orbital_mo_fock_diagonal():
    orbital_tests.run_mo_fock_diagonal_check(Integrals, Orbitals)


def test__orbital_mo_coefficients():
    orbital_tests.run_mo_coefficients_check(Integrals, Orbitals)


def test__orbital_mo_rotation():
    orbital_tests.run_mo_rotation_check(Integrals, Orbitals)


def test__orbital_mo_1e_kinetic():
    orbital_tests.run_mo_1e_kinetic_check(Integrals, Orbitals)


def test__orbital_mo_1e_potential():
    orbital_tests.run_mo_1e_potential_check(Integrals, Orbitals)


def test__orbital_mo_1e_dipole():
    orbital_tests.run_mo_1e_dipole_check(Integrals, Orbitals)


def test__orbital_mo_1e_fock():
    orbital_tests.run_mo_1e_fock_check(Integrals, Orbitals)


def test__orbital_mo_1e_core_hamiltonian():
    orbital_tests.run_mo_1e_core_hamiltonian_check(Integrals, Orbitals)


def test__orbital_mo_1e_core_field():
    orbital_tests.run_mo_1e_core_field_check(Integrals, Orbitals)


def test__orbital_mo_2e_repulsion():
    orbital_tests.run_mo_2e_repulsion_check(Integrals, Orbitals)


def test__orbital_ao_1e_density():
    orbital_tests.run_ao_1e_density_check(Integrals, Orbitals)


def test__orbital_ao_1e_mean_field():
    orbital_tests.run_ao_1e_mean_field_check(Integrals, Orbitals)


def test__orbital_ao_1e_fock():
    orbital_tests.run_ao_1e_fock_check(Integrals, Orbitals)


def test__orbital_hf_energy():
    orbital_tests.run_hf_energy_check(Integrals, Orbitals)


def test__orbital_core_energy():
    orbital_tests.run_core_energy_check(Integrals, Orbitals)

