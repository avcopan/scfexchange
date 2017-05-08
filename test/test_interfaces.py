import scfexchange
from scfexchange.test import integral_tests
from scfexchange.test import orbital_tests


def test__pyscf_integral_interface():
    from scfexchange.pyscf_interface import Integrals
    integral_tests.run_interface_check(Integrals)


def test__pyscf_integral_save_option():
    from scfexchange.pyscf_interface import Integrals
    integral_tests.run_save_option_check(Integrals)


def test__pyscf_orbital_interface():
    from scfexchange.pyscf_interface import Integrals, Orbitals
    orbital_tests.run_interface_check(Integrals, Orbitals)


def test__pyscf_orbital_core_energy():
    from scfexchange.pyscf_interface import Integrals, Orbitals
    orbital_tests.run_core_energy_check(Integrals, Orbitals)


def test__pyscf_orbital_mp2_energy():
    from scfexchange.pyscf_interface import Integrals, Orbitals
    orbital_tests.run_mp2_energy_check(Integrals, Orbitals)


def test__psi4_integral_interface():
    from scfexchange.psi4_interface import Integrals
    integral_tests.run_interface_check(Integrals)


def test__psi4_integral_save_option():
    from scfexchange.psi4_interface import Integrals
    integral_tests.run_save_option_check(Integrals)


def test__psi4_orbital_interface():
    from scfexchange.psi4_interface import Integrals, Orbitals
    orbital_tests.run_interface_check(Integrals, Orbitals)


def test__psi4_orbital_core_energy():
    from scfexchange.psi4_interface import Integrals, Orbitals
    orbital_tests.run_core_energy_check(Integrals, Orbitals)


def test__psi4_orbital_mp2_energy():
    from scfexchange.psi4_interface import Integrals, Orbitals
    orbital_tests.run_mp2_energy_check(Integrals, Orbitals)
