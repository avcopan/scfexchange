import abc
import numpy as np
import scipy.linalg as spla

from six import with_metaclass
from .molecule import Molecule
from .util import compute_if_unknown


class IntegralsInterface(with_metaclass(abc.ABCMeta)):
    """Molecular integrals.
    
    Attributes:
        nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): Specifies the
            positions of the atomic centers.
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nbf (int): The number of basis functions.
    """

    def _compute_ao_1e(self, name, compute_ints, integrate_spin=True,
                       save=False, ncomp=3):
        ao_name = "_ao_1e_{:s}".format(name)
        aso_name = "_aso_1e_{:s}".format(name)
        ints = compute_if_unknown(self, ao_name, compute_ints, save)

        def convert_to_aso():
            return IntegralsInterface.convert_1e_ao_to_aso(ints, ncomp=3)

        if not integrate_spin:
            ints = compute_if_unknown(self, aso_name, convert_to_aso, save)
        return ints

    def _compute_ao_2e(self, name, compute_ints, integrate_spin=True,
                       save=False, antisymmetrize=False):
        ao_name = "_ao_2e_chem_{:s}".format(name)
        aso_name = "_aso_2e_chem_{:s}".format(name)
        chem_ints = compute_if_unknown(self, ao_name, compute_ints, save)

        def convert_to_aso():
            return IntegralsInterface.convert_2e_ao_to_aso(chem_ints)

        if not integrate_spin:
            chem_ints = compute_if_unknown(self, aso_name, convert_to_aso, save)
        ints = chem_ints.transpose((0, 2, 1, 3))
        if antisymmetrize:
            ints = ints - ints.transpose((0, 1, 3, 2))
        return ints

    @abc.abstractmethod
    def get_ao_1e_overlap(self, integrate_spin=True, save=False):
        """Compute overlap integrals for the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A nbf x nbf array of overlap integrals,
            < mu(1) | nu(1) >.
        """
        return

    @abc.abstractmethod
    def get_ao_1e_kinetic(self, integrate_spin=True, save=False):
        """Compute kinetic energy operator in the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A nbf x nbf array of kinetic energy operator integrals,
            < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
        """
        return

    @abc.abstractmethod
    def get_ao_1e_potential(self, integrate_spin=True, save=False):
        """Compute nuclear potential operator in the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A nbf x nbf array of nuclear potential operator integrals,
            < mu(1) | sum_A Z_A / r_1A | nu(1) >.
        """
        return

    '''
    @abc.abstractmethod
    def get_ao_1e_dipole(self, integrate_spin=True, save=False):
        """Compute the dipole operator in the atomic orbital basis.
        
        Args:
            integrate_spin (bool): Use spatial orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A 3 x nbf x nbf array of dipole operator integrals,
            < mu(1) | [x, y, z] | nu(1) >
        """
        pass
    '''

    @abc.abstractmethod
    def get_ao_2e_repulsion(self, integrate_spin=True, save=False,
                            antisymmetrize=False):
        """Compute electron-repulsion operator in the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
            save (bool): Save the computed array for later use?
            antisymmetrize (bool): Antisymmetrize the repulsion integrals?
    
        Returns:
            A nbf x nbf x nbf x nbf array of electron
            repulsion operator integrals,
            < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
        """
        return

    @staticmethod
    def convert_1e_ao_to_aso(ao_1e_operator, ncomp=1):
        """Convert one-electron operator to the atomic spin-orbital basis.
        
        Args:
            ao_1e_operator (np.ndarray): An AO basis one-electron operator.
    
        Returns:
            An np.ndarray with shape (2*nbf, 2*nbf) where nbf is `self.nbf`.
        """
        comps = ao_1e_operator
        if ncomp is 1 and ao_1e_operator.ndim is 2:
            comps = [ao_1e_operator]
        return np.array([spla.block_diag(comp, comp) for comp in comps])

    @staticmethod
    def convert_2e_ao_to_aso(ao_2e_chem_operator):
        """Convert two-electron operator to the atomic spin-orbital basis.

        Args:
            ao_2e_chem_operator (np.ndarray): An AO basis two-electron operator
                with axes ordered in chemist's notation.
    
        Returns:
            An np.ndarray with shape (2*nbf, 2*nbf, 2*nbf, 2*nbf) where nbf is
            `self.nbf`.
        """
        aso_2e_chem_operator = np.kron(np.identity(2),
                                       np.kron(np.identity(2),
                                               ao_2e_chem_operator).T)
        return aso_2e_chem_operator

