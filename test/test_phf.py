from scfexchange.examples.phf import perturbed_uhf_mo_coefficients


def test__hf_mo_coefficients():
    import numpy
    import scfexchange.molecule as scfxmol
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])

    nuc_energy = scfxmol.nuclear_repulsion_energy(nuc_labels, nuc_coords)
    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    energies = iter([-74.963343795087553, -74.656730208992315])

    for charge, multp in [(0, 1), (1, 2)]:
        mo_coeffs = perturbed_uhf_mo_coefficients(aoints, charge=charge,
                                                  multp=multp)
        naocc, nbocc = scfxmol.electron_spin_count(nuc_labels,
                                                   mol_charge=charge,
                                                   multp=multp)
        alpha_coeffs = mo_coeffs[0, :, :naocc]
        beta_coeffs = mo_coeffs[1, :, :nbocc]
        elec_energy = aoints.electronic_energy(alpha_coeffs,
                                               beta_coeffs=beta_coeffs)
        energy = elec_energy + nuc_energy
        assert(numpy.isclose(energy, next(energies)))


def test__hellmann_feynman_theorem():
    import numpy
    import numdifftools
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])

    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
