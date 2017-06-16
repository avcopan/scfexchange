import warnings

import numpy as np
import scipy.linalg as spla
from scfexchange import molecule as mol


def perturbed_uhf_mo_coefficients(aoints, charge=0, multp=1,
                                  electric_field=None, niter=100,
                                  e_threshold=1e-12, d_threshold=1e-6):
    s = aoints.overlap()
    x = spla.inv(spla.sqrtm(s))
    nuc_labels = aoints.nuc_labels
    naocc, nbocc = mol.electron_spin_count(nuc_labels, mol_charge=charge,
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

    return np.array([ac, bc])


def _main():
    import numpy
    from scfexchange.pyscf_interface import AOIntegrals

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])

    aoints = AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    perturbed_uhf_mo_coefficients(aoints, charge=1, multp=2)


if __name__ == "__main__":
    _main()
