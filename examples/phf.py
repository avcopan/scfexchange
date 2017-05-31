import warnings

import numpy as np
import scipy.linalg as spla
from scfexchange.orbitals import OrbitalsInterface


class PerturbedHartreeFock(OrbitalsInterface):
    def solve(self, niter=40, e_threshold=1e-12, d_threshold=1e-8,
              electric_field=None):
        """
        
        Args:
            niter (int): Maximum number of iterations allowed.
            e_threshold (float): Energy convergence threshold.
            d_threshold (float): Density convergence threshold.
            electric_field (numpy.ndarray): A three-component vector specifying 
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added 
                to the core Hamiltonian.
        """
        s = self.integrals.get_ao_1e_overlap()
        x = spla.inv(spla.sqrtm(s))
        energy = 0.0
        converged = False
        for iteration in range(niter):
            af = self.get_ao_1e_fock('alpha', electric_field=electric_field)
            bf = self.get_ao_1e_fock('beta', electric_field=electric_field)
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
            energy = self.get_hf_energy(electric_field=electric_field)
            energy_change = np.fabs(energy - previous_energy)
            # 2. Determine the orbital gradient
            w = self.get_mo_1e_fock(mo_type='spinorb', mo_block='o,v',
                                    electric_field=electric_field)
            orb_grad_norm = np.linalg.norm(w)
            # 3. Quit if converged
            converged = (energy_change < e_threshold and
                         orb_grad_norm < d_threshold)
            if converged:
                print("@UHF converged energy = {:20.15f}".format(energy))
                break
            # Print step info
            print('@UHF {:-3d} {:20.15f} {:20.15f} {:20.15f}'
                  .format(iteration, energy, energy_change, orb_grad_norm))
        if not converged:
            warnings.warn("UHF algorithm did not converge!")


if __name__ == "__main__":
    from scfexchange.pyscf_interface import Integrals
    from scfexchange.molecule import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = Integrals(nuclei, "sto-3g")
    orbitals = PerturbedHartreeFock(integrals, charge=1, multiplicity=2,
                                    restrict_spin=False)
    orbitals.solve(niter=100, e_threshold=1e-14, d_threshold=1e-12)
    d_occ = orbitals.get_mo_1e_dipole(mo_type='spinorb', mo_block='o,o')
    mu = [np.trace(d_occ_x) for d_occ_x in d_occ]

    import scipy.misc


    def energy(field_value=0., field_component=0):
        e_field = np.zeros((3,))
        e_field[field_component] = field_value
        orbitals.solve(niter=100, e_threshold=1e-15,
                       d_threshold=1e-13, electric_field=e_field)
        return -orbitals.get_hf_energy(electric_field=e_field)


    comp = 2
    dedx = scipy.misc.derivative(energy, 0., dx=0.005, n=1, args=(comp,),
                                 order=7)
    print(dedx)
    print(mu[comp])
