import abc
import numpy as np
import tensorutils as tu
import scipy.linalg as spla

from six import with_metaclass
from .integrals import IntegralsInterface
from .util import compute_if_unknown


class OrbitalsInterface(with_metaclass(abc.ABCMeta)):
    """Molecular orbitals.
    
    Attributes:
        integrals (:obj:`scfexchange.integrals.Integrals`): Contributions to the
            Hamiltonian operator, in the molecular orbital basis.
        molecule (:obj:`scfexchange.nuclei.Molecule`): A Molecule object
            specifying the molecular charge and multiplicity
        options (dict): A dictionary of options, by keyword argument.
        nfrz (int): The number of frozen (spatial) orbitals.  This can be set
            with the option 'n_frozen_orbitals'.  Alternatively, if
            'freeze_core' is True and the number of frozen orbitals is not set,
            this defaults to the number of core orbitals, as determined by the
            nuclei object.
        norb (int): The total number of non-frozen (spatial) orbitals.  That is,
            the number of basis functions minus the number of frozen orbitals.
        naocc (int): The number of occupied non-frozen alpha orbitals.
        nbocc (int): The number of occupied non-frozen beta orbitals.
        core_energy (float): Hartree-Fock energy of the frozen core, including
            nuclear repulsion energy.
        hf_energy (float): The total Hartree-Fock energy.
    """

    def get_mo_slice(self, mo_type='alpha', mo_space='ov'):
        """Return the slice for a specific subset of molecular orbitals.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
    
        Returns:
            slice: The slice for the requested MO space.
        """
        # Check the arguments to make sure all is kosher
        if not mo_type in ('spinorb', 'alpha', 'beta'):
            raise ValueError(
                "Invalid mo_type argument '{:s}'.  Please use 'alpha', "
                "'beta', or 'spinorb'.".format(mo_type))
        if not mo_space in ('c', 'o', 'v', 'co', 'ov', 'cov'):
            raise ValueError(
                "Invalid mo_space argument '{:s}'.  Please use 'c', "
                "'o', 'v', 'co', 'ov', or 'cov'.".format(mo_space))
        # Assign slice start point
        if mo_space.startswith('c'):
            start = None
        elif mo_type is 'spinorb':
            start = 2 * self.nfrz if mo_space.startswith(
                'o') else 2 * self.nfrz + self.naocc + self.nbocc
        elif mo_type is 'alpha':
            start = self.nfrz if mo_space.startswith(
                'o') else self.nfrz + self.naocc
        elif mo_type is 'beta':
            start = self.nfrz if mo_space.startswith(
                'o') else self.nfrz + self.nbocc
        # Assign slice end point
        if mo_space.endswith('v'):
            end = None
        elif mo_type is 'spinorb':
            end = 2 * self.nfrz if mo_space.endswith(
                'c') else 2 * self.nfrz + self.naocc + self.nbocc
        elif mo_type is 'alpha':
            end = self.nfrz if mo_space.endswith(
                'c') else self.nfrz + self.naocc
        elif mo_type is 'beta':
            end = self.nfrz if mo_space.endswith(
                'c') else self.nfrz + self.nbocc
        return slice(start, end)

    def get_mo_energies(self, mo_type='alpha', mo_space='ov'):
        """Return the molecular orbital energies.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
    
        Returns:
            np.ndarray: The orbital energies.
        """
        slc = self.get_mo_slice(mo_type=mo_type, mo_space=mo_space)
        if mo_type is 'alpha':
            return self._mo_energies[0][slc]
        elif mo_type is 'beta':
            return self._mo_energies[1][slc]
        elif mo_type is 'spinorb':
            return self._mso_energies[slc]

    def get_mo_coefficients(self, mo_type='alpha', mo_space='ov',
                            transformation=None):
        """Return the molecular orbital coefficients.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients prior to transformation.  Either
                a single matrix or a pair of matrices for alpha and beta spins.
    
        Returns:
            np.ndarray: The orbital coefficients.
        """
        slc = self.get_mo_slice(mo_type=mo_type, mo_space=mo_space)
        # Grab the appropriate set of coefficients
        if mo_type is 'alpha':
            c = self._mo_coefficients[0]
        elif mo_type is 'beta':
            c = self._mo_coefficients[1]
        elif mo_type is 'spinorb':
            c = self._mso_coefficients
        # Check the transformation matrix.
        t_array = np.array(transformation, dtype=np.float64)
        if t_array.ndim is 0:
            t = 1.
        elif t_array.ndim is 2:
            t = t_array
        elif t_array.ndim is 3 and t_array.shape[0] is 2 and mo_type is 'alpha':
            t = t_array[0]
        elif t_array.ndim is 3 and t_array.shape[0] is 2 and mo_type is 'beta':
            t = t_array[1]
        else:
            raise ValueError("If 'transformation' is set, it must be a matrix "
                             "or matrix pair.")
        # Apply the transformation and return the appropriate slice
        c = c.dot(t)
        return c[:, slc]

    def get_mo_1e_core_field(self, mo_type='alpha', mo_block='ov,ov',
                             transformation=None):
        """Get the core field integrals.
        
        Returns the representation of the mean field of the core electrons in
        the molecular-orbital basis, <p(1)|J_c(1) + K_c(1)|q(1)>.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  The MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients prior to transformation.  Either
                a single matrix or a pair of matrices for alpha and beta spins.

        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        va, vb = self._compute_ao_1e_core_field()
        if mo_type is 'alpha':
            v = va
        elif mo_type is 'beta':
            v = vb
        elif mo_type is 'spinorb':
            v = spla.block_diag(va, vb)
        return c1.T.dot(v.dot(c2))

    def get_mo_1e_kinetic(self, mo_type='alpha', mo_block='ov,ov',
                          transformation=None):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the molecular-orbital basis, <p(1)| - 1 / 2 * nabla_1^2 |q(1)>.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  The MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients prior to transformation.  Either
                a single matrix or a pair of matrices for alpha and beta spins.
    
        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        use_spinorbs = (mo_type is 'spinorb')
        t_ao = self.integrals.get_ao_1e_kinetic(use_spinorbs)
        t_mo = c1.T.dot(t_ao.dot(c2))
        return t_mo

    def get_mo_1e_potential(self, mo_type='alpha', mo_block='ov,ov',
                            transformation=None):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        molecular-orbital basis, <p(1)| sum_A Z_A / ||r_1 - r_A|| |q(1)>.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients prior to transformation.  Either
                a single matrix or a pair of matrices for alpha and beta spins.
    
        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        use_spinorbs = (mo_type is 'spinorb')
        v_ao = self.integrals.get_ao_1e_potential(use_spinorbs)
        v_mo = c1.T.dot(v_ao.dot(c2))
        return v_mo

    def get_mo_1e_dipole(self, mo_type='alpha', mo_block='ov,ov',
                         transformation=None):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        molecular-orbital basis, <mu(1)| [-x, -y, -z] |nu(1)>.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients prior to transformation.  Either
                a single matrix or a pair of matrices for alpha and beta spins.
    
        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        use_spinorbs = (mo_type is 'spinorb')
        mu_ao = self.integrals.get_ao_1e_dipole(use_spinorbs)
        mu_mo = np.array([c1.T.dot(mu_ao_x.dot(c2)) for mu_ao_x in mu_ao])
        return mu_mo

    def get_mo_2e_repulsion(self, mo_type='alpha', mo_block='ov,ov,ov,ov',
                            transformation=None, antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        molecular-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients prior to transformation.  Either
                a single matrix or a pair of matrices for alpha and beta spins.
            antisymmetrize (bool): Antisymmetrize the integral tensor?
    
        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        use_spinorbs = (mo_type is 'spinorb')
        g = self.integrals.get_ao_2e_repulsion(use_spinorbs, recompute=False,
                                               antisymmetrize=antisymmetrize)
        if mo_type is 'mixed':
            c1 = self.get_mo_coefficients(mo_type='alpha',
                                          mo_space=mo_spaces[0],
                                          transformation=transformation)
            c2 = self.get_mo_coefficients(mo_type='beta',
                                          mo_space=mo_spaces[1],
                                          transformation=transformation)
            c3 = self.get_mo_coefficients(mo_type='alpha',
                                          mo_space=mo_spaces[2],
                                          transformation=transformation)
            c4 = self.get_mo_coefficients(mo_type='beta',
                                          mo_space=mo_spaces[3],
                                          transformation=transformation)
        else:
            c1 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[0],
                                          transformation=transformation)
            c2 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[1],
                                          transformation=transformation)
            c3 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[2],
                                          transformation=transformation)
            c4 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[3],
                                          transformation=transformation)
        return tu.einsum("mntu,mp,nq,tr,us->pqrs", g, c1, c2, c3, c4)

    def _get_ao_1e_hf_density(self, mo_type='alpha', mo_space='ov',
                              transformation=None):
        """Get the Hartree-Fock density.
        
        Returns the Hartree-Fock density, D_mu,nu = sum_i C_mu,i C_nu,i^*, 
        in the atomic-orbital basis.  This is not quite equal to the 
        one-particle density matrix, which is S * D * S with S denoting the
        atomic-orbital overlap matrix.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients prior to transformation.  Either
                a single matrix or a pair of matrices for alpha and beta spins.
    
        Returns:
            np.ndarray: The matrix.
        """
        c = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_space,
                                     transformation=transformation)
        d = c.dot(c.T)
        return d

    def _compute_ao_1e_core_field(self):
        """Get the core field integrals.
        
        Returns the representation of the mean field of the core electrons in
        the atomic-orbital basis, <p(1)|J_c(1) + K_c(1)|q(1)>.
        
        Returns:
            np.ndarray: The integrals.
        """
        da = self._get_ao_1e_hf_density('alpha', mo_space='c')
        db = self._get_ao_1e_hf_density('beta', mo_space='c')
        g = self.integrals.get_ao_2e_repulsion(use_spinorbs=False)
        j = np.tensordot(g, da + db, axes=[(1, 3), (1, 0)])
        va = j - np.tensordot(g, da, axes=[(1, 2), (1, 0)])
        vb = j - np.tensordot(g, db, axes=[(1, 2), (1, 0)])
        return np.array([va, vb])

    def _compute_core_energy(self):
        da = self._get_ao_1e_hf_density('alpha', mo_space='c')
        db = self._get_ao_1e_hf_density('beta', mo_space='c')
        va, vb = self._compute_ao_1e_core_field()
        h = (self.integrals.get_ao_1e_kinetic(use_spinorbs=False)
             + self.integrals.get_ao_1e_potential(use_spinorbs=False))
        core_energy = np.sum((h + va / 2) * da + (h + vb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return core_energy + e_nuc
