import abc

import numpy as np
import tensorutils as tu
from six import with_metaclass


class IntegralsInterface(with_metaclass(abc.ABCMeta)):
    """Molecular integrals base class.
    
    Subclasses should override the methods for returning integrals as numpy 
    arrays (such as `get_ao_1e_kinetic`).
    
    Attributes:
        nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): Specifies the
            positions of the atomic centers.
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nbf (int): The number of basis functions.
    """

    def _get_ints(self, name, integrate, use_spinorbs=False, recompute=False,
                  ncomp=None):
        """Retrieve a set of integrals, computing them only when necessary.
        
        The first time this function is called for a given set of integrals,
        they are computed and stored as an attribute.  If the function gets
        called at any later time, the attribute is returned without recomputing,
        unless the user specifically requests otherwise.
        
        Args:
            name (str): A unique name for the type of integral, such as
                '1e_kinetic' or '2e_repulsion'.
            integrate: A callable object that computes spatial electronic 
                integrals.  Only gets called if `recompute` is True or if we
                haven't previously computed integrals under this value of
                `name`.
            use_spinorbs (bool): Expand integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            ncomp (int): For multi-component integrals, this specifies the 
                number of components.  For example, dipole integrals have three 
                components, x, y, and z.

        Returns:
            numpy.ndarray: The integrals.
        """
        # If spinorb integrals are requested and we have them, return them.
        if use_spinorbs and hasattr(self, "_aso_" + name) and not recompute:
            return getattr(self, "_aso_" + name)
        # Otherwise, compute or retrieve the spatial integrals.
        if hasattr(self, "_ao_" + name) and not recompute:
            integrals = getattr(self, "_ao_" + name)
        else:
            integrals = integrate()
        setattr(self, "_ao_" + name, integrals)
        # If requested, construct spin-orbital integrals from the spatial ones.
        if use_spinorbs:
            if ncomp is None:
                integrals = tu.construct_spinorb_integrals(integrals)
            else:
                integrals = np.array([tu.construct_spinorb_integrals(comp)
                                      for comp in integrals])
            setattr(self, "_aso_" + name, integrals)
        return integrals

    def get_ao_1e_core_hamiltonian(self, use_spinorbs=False, recompute=False,
                                   electric_field=None):
        """Get the core Hamiltonian integrals.

        Returns the one-particle contribution to the Hamiltonian, i.e.
        everything except for two-electron repulsion.  May include an external
        static electric field in the dipole approximation.
        
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            electric_field (numpy.ndarray): A three-component vector specifying 
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added 
                to the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        t = self.get_ao_1e_kinetic(use_spinorbs=use_spinorbs,
                                   recompute=recompute)
        v = self.get_ao_1e_potential(use_spinorbs=use_spinorbs,
                                     recompute=recompute)
        h = t + v
        if electric_field is not None:
            d = self.get_ao_1e_dipole(use_spinorbs=use_spinorbs,
                                      recompute=recompute)
            h += -np.tensordot(d, electric_field, axes=(0, 0))
        return h

    @abc.abstractmethod
    def get_ao_1e_overlap(self, use_spinorbs=False, recompute=False):
        """Get the overlap integrals.
       
        Returns the overlap matrix of the atomic-orbital basis, <mu(1)|nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
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
            numpy.ndarray: The integrals.
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
            numpy.ndarray: The integrals.
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
            numpy.ndarray: The integrals.
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
            numpy.ndarray: The integrals.
        """
        return
