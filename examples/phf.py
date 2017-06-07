import warnings

import numpy as np
import scipy.linalg as spla
from scfexchange import OrbitalsInterface


class PerturbedHartreeFock(OrbitalsInterface):

    def solve(self, niter=40, e_threshold=1e-12, d_threshold=1e-8,
              electric_field=None):
        """Solve for electrically perturbed Hartree-Fock orbitals.
        
        Args:
            niter (int): Maximum number of iterations allowed.
            e_threshold (float): Energy convergence threshold.
            d_threshold (float): Density convergence threshold.
            electric_field (numpy.ndarray): A three-component vector specifying 
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added 
                to the core Hamiltonian.
        """
        if self.molecule.multiplicity is not 0 and self.spin_is_restricted:
            raise NotImplementedError("Perturbed ROHF is not implemented.")
        s = self.integrals.get_ao_1e_overlap()
        x = spla.inv(spla.sqrtm(s))
        energy = 0.0
        converged = False
        for iteration in range(niter):
            af = self.get_ao_1e_fock(mo_space='co', spin_sector='a',
                                     electric_field=electric_field)
            bf = self.get_ao_1e_fock(mo_space='co', spin_sector='b',
                                     electric_field=electric_field)
            taf = x.dot(af.dot(x))
            tbf = x.dot(bf.dot(x))
            ae, tac = spla.eigh(taf)
            be, tbc = spla.eigh(tbf)
            ac = x.dot(tac)
            bc = x.dot(tbc)
            self.mo_coefficients = np.array([ac, bc])
            # Check convergence
            # 1. Determine the energy difference
            previous_energy = energy
            energy = self.get_energy(electric_field=electric_field)
            energy_change = np.fabs(energy - previous_energy)
            # 2. Determine the orbital gradient
            w = self.get_mo_1e_fock(mo_block='o,v', spin_sector='s',
                                    electric_field=electric_field)
            orb_grad_norm = np.linalg.norm(w)
            # 3. Quit if converged
            converged = (energy_change < e_threshold and
                         orb_grad_norm < d_threshold)
            if converged:
                break

        print(electric_field)
        print("E={:20.15f} (iter: {:3d}, dE: {:7.0e}, orb grad: {:7.1e})"
              .format(energy, iteration, energy_change, orb_grad_norm))
        if not converged:
            warnings.warn("UHF algorithm did not converge!")

    def get_energy_field_function(self, niter=40, e_threshold=1e-12,
                                  d_threshold=1e-8):

        def energy_fn(field=(0., 0., 0.)):
            self.solve(niter=niter, e_threshold=e_threshold,
                       d_threshold=d_threshold, electric_field=field)
            return self.get_energy(electric_field=field)

        return energy_fn


if __name__ == "__main__":
    from scfexchange.pyscf_interface import Integrals
    from scfexchange.molecule import Nuclei
    from scfexchange import DeterminantDensity

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = Integrals(nuclei, "sto-3g")
    phf = PerturbedHartreeFock(integrals, charge=1, multiplicity=2,
                               restrict_spin=False)
    phf.solve(niter=100, e_threshold=1e-14, d_threshold=1e-12)
    density = DeterminantDensity(phf)
    mu = density.get_dipole_moment()

    import numdifftools as ndt

    energy_fn = phf.get_energy_field_function(niter=300, e_threshold=1e-13,
                                              d_threshold=1e-12)
    grad = ndt.Gradient(energy_fn, step=0.005, order=8)
    print(grad(np.r_[0., 0., 0.]))
    print(-mu)

