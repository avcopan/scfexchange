import scfexchange.examples.rpa as rpa


def test__hellmann_feynman():
    import numpy
    import numdifftools
    import scfexchange as scfx
    import scfexchange.pyscf_interface as scfxif
    import scfexchange.examples.puhf as puhf

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    n = 1

    # Compute the dipole polarizability analytically
    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = scfxif.hf_mo_coefficients(aoints, mol_charge=n, nunp=n,
                                          restricted=False, d_threshold=1e-9)
    nao, nbo = scfx.chem.elec.count_spins(nuc_labels, mol_charge=n, nunp=n)
    moints = scfx.mo.MOIntegrals(aoints, mo_coeffs, nao=nao, nbo=nbo)
    t = rpa.dipole_gradient(moints)
    r = rpa.property_response(moints, prop_grad=t)
    alpha = t.T.dot(r)
    alphadiag = alpha.diagonal()

    # Compute the electric field Hessian numerically
    energy_fn = puhf.energy_function(aoints, mol_charge=n, nunp=n, niter=150,
                                     e_threshold=1e-14, d_threshold=1e-12)
    hessdiag_fn = numdifftools.Hessdiag(energy_fn, step=0.005, order=6)
    hessdiag = hessdiag_fn(numpy.r_[0., 0., 0.])

    # By the Hellmann-Feynman theorem, `hessdiag == -alphadiag`
    assert(numpy.allclose(hessdiag, -alphadiag))
