import scfexchange.examples.puhf as puhf


def test__hf_mo_coefficients():
    import numpy
    import scfexchange.chem as scfxmol
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])

    nuc_energy = scfxmol.nuclear_repulsion_energy(nuc_labels, nuc_coords)
    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    energies = iter([-74.963343795087553, -74.656730208992315])

    for charge, multp in [(0, 1), (1, 2)]:
        mo_coeffs = puhf.mo_coefficients(aoints, charge=charge, multp=multp)
        naocc, nbocc = scfxmol.electron_spin_count(nuc_labels,
                                                   mol_charge=charge,
                                                   multp=multp)
        ac_o = mo_coeffs[0, :, :naocc]
        bc_o = mo_coeffs[1, :, :nbocc]
        elec_energy = aoints.electronic_energy(ac_o, beta_coeffs=bc_o)
        energy = elec_energy + nuc_energy
        assert(numpy.isclose(energy, next(energies)))


def test__hellmann_feynman():
    import numpy
    import numdifftools
    import scfexchange.chem as scfxmol
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    mol_charge = 1
    multp = 2

    # Compute the electronic dipole moment analytically
    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = scfxif.hf_mo_coefficients(aoints, charge=mol_charge,
                                          multp=multp, restricted=False,
                                          d_threshold=1e-9)
    naocc, nbocc = scfxmol.electron_spin_count(nuc_labels,
                                               mol_charge=mol_charge,
                                               multp=multp)
    ac_o = mo_coeffs[0, :, :naocc]
    bc_o = mo_coeffs[1, :, :nbocc]

    m = aoints.electronic_dipole_moment(ac_o, beta_coeffs=bc_o)

    # Compute the electric field gradient numerically
    energy_fn = puhf.electronic_energy_function(aoints, charge=mol_charge,
                                                multp=multp, niter=150,
                                                e_threshold=1e-14,
                                                d_threshold=1e-12)
    grad_fn = numdifftools.Gradient(energy_fn, step=0.005, order=4)
    grad = grad_fn(numpy.r_[0., 0., 0.])

    # By the Hellmann-Feynman theorem, `grad == -m`
    assert(numpy.allclose(grad, -m))
