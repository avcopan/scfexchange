import warnings

import numpy as np
import scipy.linalg as spla
import scfexchange as scfx


def mo_coefficients(aoints, mol_charge=0, nunp=0, electric_field=None,
                    niter=100, e_threshold=1e-12, d_threshold=1e-6,
                    print_conv=False):
    s = aoints.overlap()
    x = spla.inv(spla.sqrtm(s))
    nao, nbo = scfx.chem.elec.count_spins(aoints.nuc_labels,
                                          mol_charge=mol_charge, nunp=nunp)
    ac = np.zeros((aoints.nbf, aoints.nbf))
    bc = np.zeros((aoints.nbf, aoints.nbf))
    ac_o = ac[:, :nao]
    bc_o = bc[:, :nbo]

    elec_energy = last_elec_energy = energy_change = orb_grad_norm = 0.
    iteration = 0
    converged = False
    for iteration in range(niter):
        # Update orbitals
        af, bf = aoints.fock(ac_o, bc=bc_o,
                             electric_field=electric_field)
        taf = x.dot(af.dot(x))
        tbf = x.dot(bf.dot(x))
        ae, tac = spla.eigh(taf)
        be, tbc = spla.eigh(tbf)
        ac = x.dot(tac)
        bc = x.dot(tbc)
        ac_o = ac[:, :nao]
        bc_o = bc[:, :nbo]

        # Get energy change
        elec_energy = aoints.electronic_energy(ac_o, bc=bc_o,
                                               electric_field=electric_field)
        energy_change = np.fabs(elec_energy - last_elec_energy)
        last_elec_energy = elec_energy

        # Get orbital gradient (MO basis)
        ac_v = ac[:, nao:]
        bc_v = bc[:, nbo:]
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
        warnings.warn("Did not converge! (dE: {:7.1e}, orb grad: {:7.1e})"
                      .format(energy_change, orb_grad_norm))

    if print_conv:
        nuc_energy = scfx.chem.nuc.repulsion_energy(aoints.nuc_labels,
                                                    aoints.nuc_coords)
        energy = nuc_energy + elec_energy
        print("E={:20.15f} ({:-3d} iterations, dE: {:7.1e}, orb grad: {:7.1e})"
              .format(energy, iteration, energy_change, orb_grad_norm))

    return np.array([ac, bc])


def energy_function(aoints, mol_charge=0, nunp=0, niter=100,
                    e_threshold=1e-12, d_threshold=1e-6,
                    print_conv=False):

    def energy_fn(electric_field=(0., 0., 0.)):
        mo_coeffs = mo_coefficients(aoints, mol_charge=mol_charge, nunp=nunp,
                                    electric_field=electric_field, niter=niter,
                                    e_threshold=e_threshold,
                                    d_threshold=d_threshold,
                                    print_conv=print_conv)
        nao, nbo = scfx.chem.elec.count_spins(aoints.nuc_labels,
                                              mol_charge=mol_charge, nunp=nunp)
        ac_o = mo_coeffs[0, :, :nao]
        bc_o = mo_coeffs[1, :, :nbo]
        nuc_energy = scfx.chem.nuc.repulsion_energy(aoints.nuc_labels,
                                                    aoints.nuc_coords)
        elec_energy = aoints.electronic_energy(ac_o, bc=bc_o,
                                               electric_field=electric_field)
        return elec_energy + nuc_energy

    return energy_fn


def _main():
    import numpy
    import numdifftools
    import scfexchange.pyscf_interface as scfxif

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    mol_charge = 1
    nunp = 1

    aoints = scfxif.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = scfxif.hf_mo_coefficients(aoints, mol_charge=mol_charge,
                                          nunp=nunp, restricted=False,
                                          d_threshold=1e-9)
    nao, nbo = scfx.chem.elec.count_spins(nuc_labels, mol_charge=mol_charge,
                                          nunp=nunp)
    ac_o = mo_coeffs[0, :, :nao]
    bc_o = mo_coeffs[1, :, :nbo]

    m = aoints.electronic_dipole_moment(ac_o, bc=bc_o)

    energy_fn = energy_function(aoints, mol_charge=mol_charge, nunp=nunp,
                                niter=150, e_threshold=1e-14, d_threshold=1e-12)
    grad_fn = numdifftools.Gradient(energy_fn, step=0.005, order=4)
    grad = grad_fn(np.r_[0., 0., 0.])
    print(m.round(12))
    print(grad.round(12))


if __name__ == "__main__":
    _main()
