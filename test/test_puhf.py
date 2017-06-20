import scfexchange.examples.puhf as puhf


def test__hf_mo_coefficients():
    import numpy as np
    import scfexchange as scfx
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                           [0.0000000000, -1.4343021349, 0.9864370414],
                           [0.0000000000, 1.4343021349, 0.9864370414]])

    nuc_energy = scfx.chem.nuc.repulsion_energy(nuc_labels, nuc_coords)
    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    energies = iter([-74.963343795087553, -74.656730208992315])

    for n in (0, 1):
        mo_coeffs = puhf.mo_coefficients(aoints, mol_charge=n, nunp=n)
        nao, nbo = scfx.chem.elec.count_spins(nuc_labels, mol_charge=n, nunp=n)
        ac_o = mo_coeffs[0, :, :nao]
        bc_o = mo_coeffs[1, :, :nbo]
        elec_energy = aoints.electronic_energy(ac_o, bc=bc_o)
        energy = elec_energy + nuc_energy
        assert(np.isclose(energy, next(energies)))


def test__hellmann_feynman():
    import numpy as np
    import numdifftools as ndt
    import scfexchange as scfx
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                           [0.0000000000, -1.4343021349, 0.9864370414],
                           [0.0000000000, 1.4343021349, 0.9864370414]])
    mol_charge = 1
    nunp = 1

    # Compute the electronic dipole moment analytically
    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = scfxif.hf_mo_coefficients(aoints, mol_charge=mol_charge,
                                          nunp=nunp, restricted=False,
                                          d_threshold=1e-9)
    nao, nbo = scfx.chem.elec.count_spins(nuc_labels, mol_charge=mol_charge,
                                          nunp=nunp)
    ac_o = mo_coeffs[0, :, :nao]
    bc_o = mo_coeffs[1, :, :nbo]

    m = aoints.electronic_dipole_moment(ac_o, bc=bc_o)

    # Compute the electric field gradient numerically
    energy_fn = puhf.energy_function(aoints, mol_charge=mol_charge,
                                     nunp=nunp, niter=150,
                                     e_threshold=1e-14,
                                     d_threshold=1e-12)
    grad_fn = ndt.Gradient(energy_fn, step=0.005, order=4)
    grad = grad_fn(np.r_[0., 0., 0.])

    # By the Hellmann-Feynman theorem, `grad == -m`
    assert(np.allclose(grad, -m))
