import warnings

import numpy as np
import scipy.linalg as spla


def density(c, slc=slice(None)):
    return np.dot(c[:, slc], c[:, slc].T)


def mean_field(g, ad, bd):
    j = np.tensordot(g, ad + bd, axes=[(1, 3), (0, 1)])
    ak = np.tensordot(g, ad, axes=[(1, 2), (0, 1)])
    bk = np.tensordot(g, bd, axes=[(1, 2), (0, 1)])
    return np.array([j - ak, j - bk])


def fock(h, g, ad, bd):
    aw, bw = mean_field(g, ad, bd)
    return np.array([h + aw, h + bw])


def rohf_fock(s, h, g, ad, bd):
    n = s.shape[0]
    assert s.shape == h.shape == ad.shape == bd.shape == (n, n)
    assert g.shape == (n, n, n, n)
    af, bf = fock(h, g, ad, bd)
    abf = (af + bf) / 2.
    p_c = np.dot(bd, s)
    p_o = np.dot(ad - bd, s)
    p_v = np.eye(n) - np.dot(ad, s)
    f = (+ np.linalg.multi_dot([p_c.T, abf, p_c]) / 2.
         + np.linalg.multi_dot([p_o.T, abf, p_o]) / 2.
         + np.linalg.multi_dot([p_v.T, abf, p_v]) / 2.
         + np.linalg.multi_dot([p_o.T, bf, p_c])
         + np.linalg.multi_dot([p_o.T, af, p_v])
         + np.linalg.multi_dot([p_v.T, abf, p_c]))
    return np.array([f, f])


def energy(h, g, ad, bd):
    aw, bw = mean_field(g, ad, bd)
    return np.sum((h + aw / 2) * ad + (h + bw / 2) * bd)


def dipole_moment(p, ad, bd):
    return np.tensordot(p, ad + bd, axes=[(1, 2), (0, 1)])


def uhf_orb_grad(s, af, bf, ad, bd):
    ar = af.dot(ad.dot(s)) - s.dot(ad.dot(af))
    br = bf.dot(bd.dot(s)) - s.dot(bd.dot(bf))
    return np.array([ar, br])


def uhf_mo_coefficients(s, af, bf):
    _, ac = spla.eigh(af, b=s)
    _, bc = spla.eigh(bf, b=s)
    return np.array([ac, bc])


def mo_coefficients(s, h, g, na, nb, guess_density=None, niter=100,
                    e_threshold=1e-12, d_threshold=1e-6, print_conv=False,
                    diis_start=3, ndiis_vecs=6):
    nbf = s.shape[0]
    if guess_density is None:
        guess_density = np.zeros((2, nbf, nbf))

    assert nbf >= na >= nb
    assert s.shape == h.shape == (nbf, nbf)
    assert g.shape == (nbf, nbf, nbf, nbf)
    assert guess_density.shape == (2, nbf, nbf)

    ad, bd = guess_density

    e = e0 = de = r_norm = 0.
    iteration = 0
    converged = False
    f_series = []
    r_series = []
    for iteration in range(niter):
        # Update orbitals
        af, bf = rohf_fock(s, h, g, ad, bd)
        ac, bc = uhf_mo_coefficients(s, af, bf)
        ad = density(ac, slc=slice(na))
        bd = density(bc, slc=slice(nb))

        # Get energy change
        e = energy(h, g, ad, bd)
        de = np.fabs(e - e0)
        e0 = e

        # Get orbital gradient (MO basis)
        ar, br = uhf_orb_grad(s, af, bf, ad, bd)
        r_norm = spla.norm([ar, br])

        if diis_start is not None:
            f_series.append((af, bf))
            r_series.append((ar, br))

        # Check convergence
        converged = (de < e_threshold and r_norm < d_threshold)
        if converged:
            break

    if not converged:
        warnings.warn("Did not converge! (dE: {:7.1e}, orb grad: {:7.1e})"
                      .format(de, r_norm))

    if print_conv:
        print("E={:20.15f} ({:-3d} iterations, dE: {:7.1e}, orb grad: {:7.1e})"
              .format(e, iteration, de, r_norm))

    return np.array([ac, bc])


def _main():
    import scfexchange as scfx
    import scfexchange.interface.pyscf as scfxif

    atoms = ("O", "H", "H")
    centers = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                        [0.0000000000, -1.4343021349, 0.9864370414],
                        [0.0000000000, 1.4343021349, 0.9864370414]])
    charge = spin = 1

    s = scfxif.overlap("sto-3g", atoms, centers)
    t = scfxif.kinetic("sto-3g", atoms, centers)
    v = scfxif.potential("sto-3g", atoms, centers)
    g = scfxif.electron_repulsion("sto-3g", atoms, centers)
    na, nb = scfx.chem.elec.count_spins(atoms, charge=charge, spin=spin)
    h = t + v

    ac, bc = mo_coefficients(s, h, g, na, nb, print_conv=True)
    ad = density(ac, slc=slice(na))
    bd = density(bc, slc=slice(nb))
    e_elec = energy(h, g, ad, bd)
    e_nuc = scfx.chem.nuc.repulsion_energy(atoms, centers)
    print(e_elec + e_nuc)


if __name__ == "__main__":
    _main()
