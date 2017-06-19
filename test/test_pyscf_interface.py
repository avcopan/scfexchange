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


def test__ao_electronic_energy():
    ao_tests.run_test__electronic_energy(interface)


def test__ao_electronic_dipole_moment():
    ao_tests.run_test__electronic_dipole_moment(interface)


def test__hf_mo_coefficients():
    import numpy
    import scfexchange.molecule as mol

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])

    nuc_energy = mol.nuclear_repulsion_energy(nuc_labels, nuc_coords)
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    energies = iter([
        -74.963343795087553, -74.963343795087553, -74.654712456959132,
        -74.656730208992315])

    for charge, multp in [(0, 1), (1, 2)]:
        for restr in [True, False]:
            mo_coeffs = interface.hf_mo_coefficients(aoints, charge=charge,
                                                     multp=multp,
                                                     restricted=restr)
            naocc, nbocc = mol.electron_spin_count(nuc_labels,
                                                   mol_charge=charge,
                                                   multp=multp)
            alpha_coeffs = mo_coeffs[0, :, :naocc]
            beta_coeffs = mo_coeffs[1, :, :nbocc]
            elec_energy = aoints.electronic_energy(alpha_coeffs,
                                                   beta_coeffs=beta_coeffs)
            energy = elec_energy + nuc_energy
            assert(numpy.isclose(energy, next(energies)))
