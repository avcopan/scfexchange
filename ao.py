import abc

import numpy as np
import scipy.linalg as spla
import tensorutils as tu
from six import with_metaclass


def hf_density(alpha_coeffs, beta_coeffs=None, spinorb=False):
    """Get the Hartree-Fock density of a set of moints.

    The AO-basis Hartree-Fock density has the form `d = c.dot(c.T)` where the
    `c` is the orbital coefficient matrix.  This is not the same as the one-
    particle reduced density matrix, which is `s.dot(d.dot(s))`, where `s` is
    the atomic-orbital overlap matrix.

    Args:
        alpha_coeffs (numpy.ndarray): Alpha orbital coefficients.
        beta_coeffs (numpy.ndarray): Beta orbital coefficients.  If `None`,
            these are assumed to be the same as the alpha coefficients.
        spinorb (bool): Return the density in the spin-orbital basis?

    Returns:
        numpy.ndarray: The alpha and beta densities.
    """
    if beta_coeffs is None:
        ac = bc = alpha_coeffs
    else:
        ac = alpha_coeffs
        bc = beta_coeffs
    ad = ac.dot(ac.T)
    bd = bc.dot(bc.T)
    if spinorb:
        return spla.block_diag(ad, bd)
    else:
        return np.array([ad, bd])


class AOIntegralsInterface(with_metaclass(abc.ABCMeta)):
    """Molecular integrals base class.
    
    Subclasses should override the methods for returning integrals as numpy 
    arrays (such as `kinetic`).
    
    Attributes:
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates.
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nbf (int): The number of basis functions.
    """

    def _compute(self, name, integrate, spinorb=False, recompute=False,
                 multicomp=False):
        """Retrieve a set of integrals, computing them only when necessary.
        
        The first time this function is called for a given set of integrals,
        they are computed and stored as an attribute.  If the function gets
        called at any later time, the attribute is returned without recomputing,
        unless the user specifically requests otherwise.
        
        Args:
            name (str): A unique name for the type of integral, such as
                '_kinetic' or '_electron_repulsion'.
            integrate: A callable object that computes spatial electronic 
                integrals.  Only gets called if `recompute` is True or if we
                haven't previously computed integrals under this value of
                `name`.
            spinorb (bool): Expand integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            multicomp (bool): Is this a multi-component operator?

        Returns:
            numpy.ndarray: The integrals.
        """
        if hasattr(self, name) and not recompute:
            ints = getattr(self, name)
        else:
            ints = integrate()
        setattr(self, name, ints)
        # If requested, construct spin-orbital integrals from the spatial ones.
        if spinorb:
            if multicomp:
                ints = np.array([tu.construct_spinorb_integrals(ints_x)
                                 for ints_x in ints])
            else:
                ints = tu.construct_spinorb_integrals(ints)
        return ints

    @abc.abstractmethod
    def overlap(self, spinorb=False, recompute=False):
        """Get the overlap integrals.
       
        Returns the overlap matrix of the atomic-orbital basis, <mu(1)|nu(1)>.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """
        return

    @abc.abstractmethod
    def kinetic(self, spinorb=False, recompute=False):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the atomic-orbital basis, <mu(1)| - 1 / 2 * nabla_1^2 |nu(1)>.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """
        return

    @abc.abstractmethod
    def potential(self, spinorb=False, recompute=False):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        atomic-orbital basis, <mu(1)| sum_A Z_A / ||r_1 - r_A|| |nu(1)>.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """
        return

    @abc.abstractmethod
    def dipole(self, spinorb=False, recompute=False):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        atomic-orbital basis, <mu(1)| [-x, -y, -z] |nu(1)>.
        
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """
        pass

    @abc.abstractmethod
    def electron_repulsion(self, spinorb=False, recompute=False,
                           antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        atomic-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            antisymmetrize (bool): Antisymmetrize the integral tensor?
    
        Returns:
            numpy.ndarray: The integrals.
        """
        return

    def core_hamiltonian(self, spinorb=False, recompute=False,
                         electric_field=None):
        """Get the core Hamiltonian integrals.

        Returns the one-particle contribution to the Hamiltonian, i.e.
        everything except for two-electron repulsion.  May include an external
        static electric field in the dipole approximation.

        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        t = self.kinetic(spinorb=spinorb, recompute=recompute)
        v = self.potential(spinorb=spinorb, recompute=recompute)
        h = t + v
        if electric_field is not None:
            d = self.dipole(spinorb=spinorb, recompute=recompute)
            h += -tu.contract(d, electric_field)
        return h

    def mean_field(self, alpha_coeffs, beta_coeffs=None, spinorb=False,
                   recompute=False):
        """Get the mean field integrals for a set of moints.

        Returns the electronic mean field of a set of moints, which are
        <mu(1)|J(1) - K(1)|nu(1)> where J and K are the corresponding
        Coulomb and exchange operators.

        Args:
            alpha_coeffs (numpy.ndarray): Alpha orbital coefficients.
            beta_coeffs (numpy.ndarray): Beta orbital coefficients.  If `None`,
                these are assumed to be the same as the alpha coefficients.
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?

        Returns:
            numpy.ndarray: The alpha and beta mean-field integrals.
        """
        ad, bd = hf_density(alpha_coeffs, beta_coeffs=beta_coeffs,
                            spinorb=False)
        g = self.electron_repulsion(spinorb=False, recompute=recompute)
        # Compute the Coulomb and exchange matrices.
        j = np.tensordot(g, ad + bd, axes=[(1, 3), (1, 0)])
        ak = np.tensordot(g, ad, axes=[(1, 2), (1, 0)])
        bk = np.tensordot(g, bd, axes=[(1, 2), (1, 0)])
        if spinorb:
            return spla.block_diag(j - ak, j - bk)
        else:
            return np.array([j - ak, j - bk])

    def fock(self, alpha_coeffs, beta_coeffs=None, spinorb=False,
             recompute=False, electric_field=None):
        """Get the Fock operator integrals for a set of moints.

        Returns the core Hamiltonian plus the mean field of the moints.  The
        core Hamiltonian may include an external static electric field in the
        dipole approximation.

        Args:
            alpha_coeffs (numpy.ndarray): Alpha orbital coefficients.
            beta_coeffs (numpy.ndarray): Beta orbital coefficients.  If `None`,
                these are assumed to be the same as the alpha coefficients.
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.

        Returns:
            numpy.ndarray: The alpha and beta Fock integrals.
        """
        h = self.core_hamiltonian(spinorb=False, recompute=recompute,
                                  electric_field=electric_field)
        aw, bw = self.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs,
                                 spinorb=False, recompute=recompute)
        if spinorb:
            return spla.block_diag(h + aw, h + bw)
        else:
            return np.array([h + aw, h + bw])

    def electronic_energy(self, alpha_coeffs, beta_coeffs=None,
                          electric_field=None, recompute=False):
        """Get the mean field energy of a set of moints.

        Args:
            alpha_coeffs (numpy.ndarray): Alpha orbital coefficients.
            beta_coeffs (numpy.ndarray): Beta orbital coefficients.  If `None`,
                these are assumed to be the same as the alpha coefficients.
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.
            recompute (bool): Recompute the integrals, if we already have them?

        Returns:
            float: The energy.
        """
        h = self.core_hamiltonian(spinorb=False, recompute=recompute,
                                  electric_field=electric_field)
        aw, bw = self.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs,
                                 spinorb=False, recompute=recompute)
        ad, bd = hf_density(alpha_coeffs, beta_coeffs=beta_coeffs,
                            spinorb=False)
        return np.sum((h + aw / 2) * ad + (h + bw / 2) * bd)

    def electronic_dipole_moment(self, alpha_coeffs, beta_coeffs=None,
                                 recompute=False):
        """Get the electric dipole moment of a set of moints.

        Args:
            alpha_coeffs (numpy.ndarray): Alpha orbital coefficients.
            beta_coeffs (numpy.ndarray): Beta orbital coefficients.  If `None`,
                these are assumed to be the same as the alpha coefficients.
            recompute (bool): Recompute the integrals, if we already have them?

        Returns:
            numpy.ndarray: The dipole moment.
        """
        p = self.dipole(spinorb=False, recompute=recompute)
        ad, bd = hf_density(alpha_coeffs, beta_coeffs=beta_coeffs,
                            spinorb=False)
        return np.array([np.sum(p_x * ad + p_x * bd) for p_x in p])
