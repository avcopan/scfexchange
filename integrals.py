import abc

import numpy as np
import scipy.linalg as spla
from six import with_metaclass

import tensorutils as tu


class IntegralsInterface(with_metaclass(abc.ABCMeta)):
    """Molecular integrals.
    
    Attributes:
        nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): Specifies the
            positions of the atomic centers.
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nbf (int): The number of basis functions.
    """

    def _compute_ao_1e(self, name, integrate, use_spinorbs=False,
                       recompute=False, ncomp=None):
        """Compute one-electron integrals in the atomic orbital basis.
        
        These are matrices of the form <mu(1)|O_1|nu(1)> where O_1 is a one-
        electron operator.
        
        Args:
            name (str): Unique name for the type of integral, i.e. 'kinetic'.
            integrate: A callable object that computes integrals in the spatial
                AO basis when called.
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            ncomp (int): For multi-component integrals, this specifies the 
                number of components.  For example, dipole integrals have three
                components -- x, y, and z.

        Returns:
            np.ndarray: The integrals.
        """
        if use_spinorbs and hasattr(self, "_aso_1e_" + name) and not recompute:
            return getattr(self, "_aso_1e_" + name)
        elif hasattr(self, "_ao_1e_" + name) and not recompute:
            integrals = getattr(self, "_ao_1e_" + name)
        else:
            integrals = integrate()
            setattr(self, "_ao_1e_" + name, integrals)
        # If requested, transform the integrals to the spin-orbital basis and
        # store them as an attribute.
        if use_spinorbs:
            if ncomp is None:
                integrals = tu.construct_spinorb_integrals(integrals)
            else:
                integrals = np.array([tu.construct_spinorb_integrals(comp)
                                      for comp in integrals])
            setattr(self, "_aso_1e_" + name, integrals)
        return integrals

    def _compute_ao_2e(self, name, integrate, use_spinorbs=False,
                       recompute=False):
        """Compute two-electron integrals in the atomic orbital basis.
        
        Assumes chemist's notation, (mu(1) rh(1)| O_12 |nu(2) si(2)).
        
        Args:
            name (str): Unique name for the type of integral, i.e. 'kinetic'.
            integrate: A callable object that computes integrals in the spatial
                AO basis when called.
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?

        Returns:
            np.ndarray: The integrals.
        """
        if use_spinorbs and hasattr(self, "_aso_2e_" + name) and not recompute:
            return getattr(self, "_aso_2e_" + name)
        elif hasattr(self, "_ao_2e_" + name) and not recompute:
            integrals = getattr(self, "_ao_2e_" + name)
        else:
            integrals = integrate()
            setattr(self, "_ao_2e_" + name, integrals)
        # If requested, transform the integrals to the spin-orbital basis and
        # store them as an attribute.
        if use_spinorbs:
            integrals = tu.construct_spinorb_integrals(integrals)
            setattr(self, "_aso_2e_" + name, integrals)
        return integrals

    @abc.abstractmethod
    def get_ao_1e_overlap(self, use_spinorbs=False, recompute=False):
        """Get the overlap integrals.
       
        Returns the overlap matrix of the atomic-orbital basis, <mu(1)|nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        return

    @abc.abstractmethod
    def get_ao_1e_kinetic(self, use_spinorbs=False, recompute=False):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the atomic-orbital basis, <mu(1)| - 1 / 2 * nabla_1^2 |nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        return

    @abc.abstractmethod
    def get_ao_1e_potential(self, use_spinorbs=False, recompute=False):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        atomic-orbital basis, <mu(1)| sum_A Z_A / ||r_1 - r_A|| |nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        return

    @abc.abstractmethod
    def get_ao_1e_dipole(self, use_spinorbs=False, recompute=False):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        atomic-orbital basis, <mu(1)| [-x, -y, -z] |nu(1)>.
        
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        pass

    @abc.abstractmethod
    def get_ao_2e_repulsion(self, use_spinorbs=False, recompute=False,
                            antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        atomic-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            antisymmetrize (bool): Antisymmetrize the integral tensor?
    
        Returns:
            np.ndarray: The integrals.
        """
        return
