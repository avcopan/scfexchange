import warnings

import numpy as np
import scipy.linalg as spla
from scfexchange import chem as mol


def mo_coefficients(aoints, charge=0, multp=1, electric_field=None,
                    niter=100, e_threshold=1e-12, d_threshold=1e-6,
                    print_info=False):
    s = aoints.overlap()
    x = spla.inv(spla.sqrtm(s))
    naocc, nbocc = mol.electron_spin_count(aoints.nuc_labels, mol_charge=charge,
                                           multp=multp)
    ac_o = np.zeros((aoints.nbf, naocc))
    bc_o = np.zeros((aoints.nbf, nbocc))

    last_elec_energy = 0.
    converged = False
    for iteration in range(niter):
        # Update orbitals
        af, bf = aoints.fock(ac_o, beta_coeffs=bc_o,
                             electric_field=electric_field)
        taf = x.dot(af.dot(x))
        tbf = x.dot(bf.dot(x))
        ae, tac = spla.eigh(taf)
        be, tbc = spla.eigh(tbf)
        ac = x.dot(tac)
        bc = x.dot(tbc)
        ac_o = ac[:, :naocc]
        bc_o = bc[:, :nbocc]

        # Get energy change
        elec_energy = aoints.electronic_energy(ac_o, beta_coeffs=bc_o,
                                               electric_field=electric_field)
        energy_change = np.fabs(elec_energy - last_elec_energy)
        last_elec_energy = elec_energy

        # Get orbital gradient (MO basis)
        ac_v = ac[:, naocc:]
        bc_v = bc[:, nbocc:]
        amo_orb_grad = ac_o.T.dot(af.dot(ac_v))
        bmo_orb_grad = bc_o.T.dot(bf.dot(bc_v))
        orb_grad_norm = np.sqrt(np.linalg.norm(amo_orb_grad) ** 2 +
                                np.linalg.norm(bmo_orb_grad) ** 2)

        # Check convergence
        converged = (energy_change < e_threshold and
                     orb_grad_norm < d_threshold)
        if converged:
            break

    if not converged:
        warnings.warn("Orbitals did not converge!")

    if print_info:
        nuc_energy = mol.nuclear_repulsion_energy(aoints.nuc_labels,
                                                  aoints.nuc_coords)
        energy = nuc_energy + elec_energy
        print("E={:20.15f} ({:-3d} iterations, dE: {:7.1e}, orb grad: {:7.1e})"
              .format(energy, iteration, energy_change, orb_grad_norm))

    return np.array([ac, bc])


def electronic_energy_function(aoints, charge=0, multp=1, niter=100,
                               e_threshold=1e-12, d_threshold=1e-6,
                               print_info=False):

    def electronic_energy_function(electric_field=(0., 0., 0.)):
        mo_coeffs = mo_coefficients(aoints, charge=charge, multp=multp,
                                    electric_field=electric_field,
                                    niter=niter, e_threshold=e_threshold,
                                    d_threshold=d_threshold,
                                    print_info=print_info)
        naocc, nbocc = mol.electron_spin_count(aoints.nuc_labels,
                                               mol_charge=charge, multp=multp)
        ac_o = mo_coeffs[0, :, :naocc]
        bc_o = mo_coeffs[1, :, :nbocc]
        elec_energy = aoints.electronic_energy(ac_o, beta_coeffs=bc_o,
                                               electric_field=electric_field)
        return elec_energy

    return electronic_energy_function


def _main():
    import numpy
    import numdifftools
    import scfexchange as scfx
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    mol_charge = 1
    multp = 2

    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = scfxif.hf_mo_coefficients(aoints, charge=mol_charge,
                                          multp=multp, restricted=False,
                                          d_threshold=1e-9)
    naocc, nbocc = scfx.chem.electron_spin_count(nuc_labels,
                                                 mol_charge=mol_charge,
                                                 multp=multp)
    ac_o = mo_coeffs[0, :, :naocc]
    bc_o = mo_coeffs[1, :, :nbocc]

    m = aoints.electronic_dipole_moment(ac_o, beta_coeffs=bc_o)

    energy_fn = electronic_energy_function(aoints,
                                           charge=mol_charge,
                                           multp=multp,
                                           niter=150,
                                           e_threshold=1e-14,
                                           d_threshold=1e-12)
    grad_fn = numdifftools.Gradient(energy_fn, step=0.005, order=4)
    grad = grad_fn(np.r_[0., 0., 0.])
    print(m.round(12))
    print(grad.round(12))


if __name__ == "__main__":
    _main()
