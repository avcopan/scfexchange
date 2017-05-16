import scipy.linalg as spla
import numpy as np

from scfexchange.integrals import IntegralsInterface
from scfexchange.orbitals import OrbitalsInterface
from scfexchange.molecule import Molecule


class HartreeFock(OrbitalsInterface):

    def __init__(self, integrals, charge=0, multiplicity=1, restrict_spin=True,
                 n_iterations=40, e_threshold=1e-12, d_threshold=1e-6,
                 n_frozen_orbitals=0, electric_field=None):
        """Initialize HartreeFock object.
        
        Args:
            integrals (:obj:`scfexchange.integrals.IntegralsInterface`): The
                atomic-orbital integrals object.
            charge (int): Total molecular charge.
            multiplicity (int): Spin multiplicity.
            restrict_spin (bool): Spin-restrict the orbitals?
            n_iterations (int): Maximum number of Hartree-Fock iterations 
                allowed before the orbitals are considered unconverged.
            e_threshold (float): Energy convergence threshold.
            d_threshold (float): Density convergence threshold, based on the 
                norm of the orbital gradient
            n_frozen_orbitals (int): How many core orbitals should be set to 
                `frozen`.
            electric_field (np.ndarray): A three-component vector specifying 
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added 
                to the core Hamiltonian.
        """
        if not isinstance(integrals, IntegralsInterface):
            raise ValueError("'integrals' must be an instance of the "
                             "IntegralsInterface base class.")
        self.integrals = integrals
        self.options = {
            'restrict_spin': restrict_spin,
            'n_iterations': n_iterations,
            'e_threshold': e_threshold,
            'd_threshold': d_threshold
        }
        self.molecule = Molecule(self.integrals.nuclei, charge, multiplicity)
        # Determine the orbital counts (total, frozen, and occupied)
        self.nfrz = n_frozen_orbitals
        self.naocc = self.molecule.nalpha - self.nfrz
        self.nbocc = self.molecule.nbeta - self.nfrz
        self.norb = self.integrals.nbf - self.nfrz

