import numpy as np
import scipy.linalg as spla
import tensorutils as tu


def a_matrix(moints):
    g_vovo = moints.electron_repulsion(mo_block='v,o,v,o', spin_sector='s,s',
                                       antisymmetrize=True)
    f_oo = moints.fock(mo_block='o,o')
    f_vv = moints.fock(mo_block='v,v')
    nocc = moints.mo_count(mo_space='o')
    nvir = moints.mo_count(mo_space='v')
    i_o = np.identity(nocc)
    i_v = np.identity(nvir)
    a = (
        +
        tu.einsum('ab,ij->iajb', f_vv, i_o)
        -
        tu.einsum('ji,ab->iajb', f_oo, i_v)
        -
        tu.einsum('ajbi->iajb', g_vovo)
    )
    return a.reshape((nocc * nvir, nocc * nvir))


def b_matrix(moints):
    g_oovv = moints.electron_repulsion(mo_block='o,o,v,v', spin_sector='s,s',
                                       antisymmetrize=True)
    nocc = moints.mo_count(mo_space='o')
    nvir = moints.mo_count(mo_space='v')
    b = tu.einsum('ijab->iajb', g_oovv)
    return b.reshape((nocc * nvir, nocc * nvir))


def tda_spectrum(moints):
    a = a_matrix(moints)
    spectrum = spla.eigvalsh(a)
    return spectrum


def rpa_spectrum(moints):
    a = a_matrix(moints)
    b = b_matrix(moints)
    h = (a + b).dot(a - b)
    spectrum = np.sqrt(spla.eigvals(h).real)
    spectrum.sort()
    return spectrum


def dipole_gradient(moints):
    nocc = moints.mo_count(mo_space='o')
    nvir = moints.mo_count(mo_space='v')
    t = moints.dipole(mo_block='o,v', spin_sector='s')
    return t.transpose((1, 2, 0)).reshape((nocc * nvir, 3))


def property_response(moints, prop_grad=None):
    if prop_grad is None:
        t = dipole_gradient(moints)
    else:
        t = prop_grad
    a = a_matrix(moints)
    b = b_matrix(moints)
    return spla.solve(a + b, 2 * t, sym_pos=True)


def _main():
    import numpy
    import numdifftools
    import scfexchange as scfx
    import scfexchange.pyscf_interface as scfxif
    import scfexchange.examples.puhf as puhf

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    mol_charge = 1
    nunp = 1

    # Compute the dipole polarizability analytically
    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = scfxif.hf_mo_coefficients(aoints, mol_charge=mol_charge,
                                          nunp=nunp, restricted=False,
                                          d_threshold=1e-9)
    nao, nbo = scfx.chem.elec.count_spins(nuc_labels, mol_charge=mol_charge,
                                          nunp=nunp)
    moints = scfx.mo.MOIntegrals(aoints, mo_coeffs, nao=nao, nbo=nbo)
    t = dipole_gradient(moints)
    r = property_response(moints, prop_grad=t)
    alpha = t.T.dot(r)
    alphadiag = alpha.diagonal()

    # Compute the electric field Hessian numerically
    energy_fn = puhf.energy_function(aoints, mol_charge=mol_charge,
                                     nunp=nunp, niter=150,
                                     e_threshold=1e-14,
                                     d_threshold=1e-12)
    hessdiag_fn = numdifftools.Hessdiag(energy_fn, step=0.005, order=6)
    hessdiag = hessdiag_fn(numpy.r_[0., 0., 0.])

    print(hessdiag.round(10))
    print(alphadiag.round(10))
    # By the Hellmann-Feynman theorem, `hessdiag == -alphadiag`
    assert(numpy.allclose(hessdiag, -alphadiag))

if __name__ == "__main__":
    _main()
