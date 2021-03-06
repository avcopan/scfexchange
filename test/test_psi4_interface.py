import chem.elec
import scfexchange.psi4_interface as interface
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


def test__ao_electronic_energy():
    ao_tests.run_test__electronic_energy(interface)


def test__ao_electronic_dipole_moment():
    ao_tests.run_test__electronic_dipole_moment(interface)


def test__hf_mo_coefficients():
    import numpy as np
    import scfexchange as scfx

    nuc_labels = ("O", "H", "H")
    nuc_coords = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                           [0.0000000000, -1.4343021349, 0.9864370414],
                           [0.0000000000, 1.4343021349, 0.9864370414]])

    nuc_energy = scfx.chem.nuc.repulsion_energy(nuc_labels, nuc_coords)
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    energies = iter([
        -74.963343795087553, -74.963343795087553, -74.654712456959132,
        -74.656730208992315])

    for n in (0, 1):
        for restr in [True, False]:
            mo_coeffs = interface.hf_mo_coefficients(aoints, mol_charge=n,
                                                     nunp=n,  restricted=restr)
            nao, nbo = scfx.chem.elec.count_spins(nuc_labels, mol_charge=n,
                                                  nunp=n)
            alpha_coeffs = mo_coeffs[0, :, :nao]
            beta_coeffs = mo_coeffs[1, :, :nbo]
            elec_energy = aoints.electronic_energy(alpha_coeffs,
                                                   bc=beta_coeffs)
            energy = elec_energy + nuc_energy
            assert(np.isclose(energy, next(energies)))
