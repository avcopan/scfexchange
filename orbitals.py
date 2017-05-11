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

    def get_mo_slice(self, mo_type='alpha', mo_block='ov'):
        """Return the slice for a specific block of molecular orbitals.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
    
        Returns:
            A `slice` object with appropriate start and end points for the
            specified MO type and block.
        """
        # Check the arguments to make sure all is kosher
        if not mo_type in ('spinorb', 'alpha', 'beta'):
            raise ValueError(
                "Invalid mo_type argument '{:s}'.  Please use 'alpha', "
                "'beta', or 'spinorb'.".format(mo_type))
        if not mo_block in ('c', 'o', 'v', 'co', 'ov', 'cov'):
            raise ValueError(
                "Invalid mo_block argument '{:s}'.  Please use 'c', "
                "'o', 'v', 'co', 'ov', or 'cov'.".format(mo_block))
        # Assign slice start point
        if mo_block.startswith('c'):
            start = None
        elif mo_type is 'spinorb':
            start = 2 * self.nfrz if mo_block.startswith(
                'o') else 2 * self.nfrz + self.naocc + self.nbocc
        elif mo_type is 'alpha':
            start = self.nfrz if mo_block.startswith(
                'o') else self.nfrz + self.naocc
        elif mo_type is 'beta':
            start = self.nfrz if mo_block.startswith(
                'o') else self.nfrz + self.nbocc
        # Assign slice end point
        if mo_block.endswith('v'):
            end = None
        elif mo_type is 'spinorb':
            end = 2 * self.nfrz if mo_block.endswith(
                'c') else 2 * self.nfrz + self.naocc + self.nbocc
        elif mo_type is 'alpha':
            end = self.nfrz if mo_block.endswith(
                'c') else self.nfrz + self.naocc
        elif mo_type is 'beta':
            end = self.nfrz if mo_block.endswith(
                'c') else self.nfrz + self.nbocc
        return slice(start, end)

    def get_mo_energies(self, mo_type='alpha', mo_block='ov'):
        """Return the molecular orbital energies.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
    
        Returns:
            An array of orbital energies for the given MO type and block.
        """
        slc = self.get_mo_slice(mo_type=mo_type, mo_block=mo_block)
        if mo_type is 'alpha':
            return self._mo_energies[0][slc]
        elif mo_type is 'beta':
            return self._mo_energies[1][slc]
        elif mo_type is 'spinorb':
            return self._mso_energies[slc]

    def get_mo_coefficients(self, mo_type='alpha', mo_block='ov',
                            r_matrix=None):
        """Return the molecular orbital coefficients.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Orbital rotation matrix to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            An array of orbital coefficients for the given MO type and block.
        """
        slc = self.get_mo_slice(mo_type=mo_type, mo_block=mo_block)
        try:
            if mo_type is 'alpha':
                c = self._mo_coefficients[0]
                if not r_matrix is None:
                    r = r_matrix if isinstance(r_matrix, np.ndarray) else \
                    r_matrix[0]
                    c = c.dot(r)
            elif mo_type is 'beta':
                c = self._mo_coefficients[1]
                if not r_matrix is None:
                    r = r_matrix if isinstance(r_matrix, np.ndarray) else \
                    r_matrix[1]
                    c = c.dot(r)
            elif mo_type is 'spinorb':
                c = self._mso_coefficients
                if not r_matrix is None:
                    r = r_matrix
                    c = c.dot(r)
            return c[:, slc]
        except:
            raise ValueError(
                "'r_matrix' must either be numpy array or a pair of "
                "numpy arrays for each spin.")

    def _get_ao_1e_density_matrix(self, mo_type='alpha', mo_block='ov',
                                  r_matrix=None):
        """Return the one-particle density matrix *in the AO basis*.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Orbital rotation matrix to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            An array containing the one-particle density matrix.
        """
        c = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_block,
                                     r_matrix=r_matrix)
        d = c.dot(c.T)
        return d

    def _compute_ao_1e_core_field(self):
        """Get the core field integrals.
        
        Returns the representation of the mean field of the core electrons in
        the molecular-orbital basis, <p(1)|J_c(1) + K_c(1)|q(1)>.
        
        Returns:
            np.ndarray: The integrals.
        """
        da = self._get_ao_1e_density_matrix('alpha', mo_block='c')
        db = self._get_ao_1e_density_matrix('beta', mo_block='c')
        g = self.integrals.get_ao_2e_repulsion(use_spinorbs=False)
        j = np.tensordot(g, da + db, axes=[(1, 3), (1, 0)])
        va = j - np.tensordot(g, da, axes=[(1, 2), (1, 0)])
        vb = j - np.tensordot(g, db, axes=[(1, 2), (1, 0)])
        return np.array([va, vb])

    def get_mo_1e_core_field(self, mo_type='alpha', mo_block='ov,ov',
                             r_matrix=None):
        """Get the core field integrals.
        
        Returns the representation of the mean field of the core electrons in
        the molecular-orbital basis, <p(1)|J_c(1) + K_c(1)|q(1)>.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Orbital rotation matrix to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.

        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        va, vb = self._compute_ao_1e_core_field()
        if mo_type is 'alpha':
            v = va
        elif mo_type is 'beta':
            v = vb
        elif mo_type is 'spinorb':
            v = spla.block_diag(va, vb)
        return c1.T.dot(v.dot(c2))

    def get_mo_1e_kinetic(self, mo_type='alpha', mo_block='ov,ov',
                          r_matrix=None):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the molecular-orbital basis, <p(1)| - 1 / 2 * nabla_1^2 |q(1)>.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Orbital rotation matrix to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        use_spinorbs = (mo_type is 'spinorb')
        t_ao = self.integrals.get_ao_1e_kinetic(use_spinorbs)
        t_mo = c1.T.dot(t_ao.dot(c2))
        return t_mo

    def get_mo_1e_potential(self, mo_type='alpha', mo_block='ov,ov',
                            r_matrix=None):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        molecular-orbital basis, <p(1)| sum_A Z_A / ||r_1 - r_A|| |q(1)>.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Orbital rotation matrix to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        use_spinorbs = (mo_type is 'spinorb')
        v_ao = self.integrals.get_ao_1e_potential(use_spinorbs)
        v_mo = c1.T.dot(v_ao.dot(c2))
        return v_mo

    def get_mo_1e_dipole(self, mo_type='alpha', mo_block='ov,ov',
                         r_matrix=None):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        molecular-orbital basis, <mu(1)| [-x, -y, -z] |nu(1)>.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Orbital rotation matrix to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        use_spinorbs = (mo_type is 'spinorb')
        mu_ao = self.integrals.get_ao_1e_dipole(use_spinorbs)
        mu_mo = np.array([c1.T.dot(mu_ao_x.dot(c2)) for mu_ao_x in mu_ao])
        return mu_mo

    def get_mo_2e_repulsion(self, mo_type='alpha', mo_block='ov,ov,ov,ov',
                            r_matrix=None, antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        molecular-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Orbital rotation matrix to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
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
                                          mo_block=mo_spaces[0],
                                          r_matrix=r_matrix)
            c2 = self.get_mo_coefficients(mo_type='beta',
                                          mo_block=mo_spaces[1],
                                          r_matrix=r_matrix)
            c3 = self.get_mo_coefficients(mo_type='alpha',
                                          mo_block=mo_spaces[2],
                                          r_matrix=r_matrix)
            c4 = self.get_mo_coefficients(mo_type='beta',
                                          mo_block=mo_spaces[3],
                                          r_matrix=r_matrix)
        else:
            c1 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_block=mo_spaces[0],
                                          r_matrix=r_matrix)
            c2 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_block=mo_spaces[1],
                                          r_matrix=r_matrix)
            c3 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_block=mo_spaces[2],
                                          r_matrix=r_matrix)
            c4 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_block=mo_spaces[3],
                                          r_matrix=r_matrix)
        return tu.einsum("mntu,mp,nq,tr,us->pqrs", g, c1, c2, c3, c4)

    def _compute_core_energy(self):
        da = self._get_ao_1e_density_matrix('alpha', mo_block='c')
        db = self._get_ao_1e_density_matrix('beta', mo_block='c')
        va, vb = self._compute_ao_1e_core_field()
        h = (self.integrals.get_ao_1e_kinetic(use_spinorbs=False)
             + self.integrals.get_ao_1e_potential(use_spinorbs=False))
        core_energy = np.sum((h + va / 2) * da + (h + vb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return core_energy + e_nuc
