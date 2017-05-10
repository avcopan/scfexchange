import scfexchange
from scfexchange.test import integral_tests
from scfexchange.test import orbital_tests
from scfexchange.pyscf_interface import Integrals, Orbitals


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


def test__integral_get_ao_2e_repulsion():
    integral_tests.run_ao_2e_repulsion_check(Integrals)


def test__orbital_interface():
    orbital_tests.run_interface_check(Integrals, Orbitals)


def test__orbital_mo_slicing():
    orbital_tests.run_mo_slicing_check(Integrals, Orbitals)


def test__orbital_core_energy():
    orbital_tests.run_core_energy_check(Integrals, Orbitals)


def test__orbital_mp2_energy():
    orbital_tests.run_mp2_energy_check(Integrals, Orbitals)


def test__orbital_dipole_moment():
    orbital_tests.run_dipole_moment_check(Integrals, Orbitals)