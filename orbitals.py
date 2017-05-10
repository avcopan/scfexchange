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
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
    
        Returns:
            A `slice` object with appropriate start and end points for the
            specified MO type and block.
        """
        # Check the arguments to make sure all is kosher
        if not mo_type in ('spinor', 'alpha', 'beta'):
            raise ValueError(
                "Invalid mo_type argument '{:s}'.  Please use 'alpha', "
                "'beta', or 'spinor'.".format(mo_type))
        if not mo_block in ('c', 'o', 'v', 'co', 'ov', 'cov'):
            raise ValueError(
                "Invalid mo_block argument '{:s}'.  Please use 'c', "
                "'o', 'v', 'co', 'ov', or 'cov'.".format(mo_block))
        # Assign slice start point
        if mo_block.startswith('c'):
            start = None
        elif mo_type is 'spinor':
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
        elif mo_type is 'spinor':
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
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
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
        elif mo_type is 'spinor':
            return self._mso_energies[slc]

    def get_mo_coefficients(self, mo_type='alpha', mo_block='ov',
                            r_matrix=None):
        """Return the molecular orbital coefficients.
    
        Args:
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Molecular orbital rotation to be
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
            elif mo_type is 'spinor':
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
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Molecular orbital rotation to be
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
        da = self._get_ao_1e_density_matrix('alpha', mo_block='c')
        db = self._get_ao_1e_density_matrix('beta', mo_block='c')
        g = self.integrals.get_ao_2e_repulsion(integrate_spin=True)
        j = np.tensordot(g, da + db, axes=[(1, 3), (1, 0)])
        va = j - np.tensordot(g, da, axes=[(1, 2), (1, 0)])
        vb = j - np.tensordot(g, db, axes=[(1, 2), (1, 0)])
        return np.array([va, vb])

    def _compute_core_energy(self):
        da = self._get_ao_1e_density_matrix('alpha', mo_block='c')
        db = self._get_ao_1e_density_matrix('beta', mo_block='c')
        va, vb = compute_if_unknown(self, '_ao_core_field',
                                    self._compute_ao_1e_core_field)
        h = (self.integrals.get_ao_1e_kinetic(integrate_spin=True)
             + self.integrals.get_ao_1e_potential(integrate_spin=True))
        core_energy = np.sum((h + va / 2) * da + (h + vb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return core_energy + e_nuc

    def get_mo_1e_core_field(self, mo_type='alpha', mo_block='ov,ov',
                             r_matrix=None):
        """Compute mean-field of the core electrons in the molecular orbital basis.
    
        Args:
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Molecular orbital rotation to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            A nbf x nbf array of core field integrals,
            < p(1) | j_a + j_b + k_a | q(1) >.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        va, vb = compute_if_unknown(self, '_ao_1e_core_field',
                                    self._compute_ao_1e_core_field)
        if mo_type is 'alpha':
            v = va
        elif mo_type is 'beta':
            v = vb
        elif mo_type is 'spinor':
            v = spla.block_diag(va, vb)
        return c1.T.dot(v.dot(c2))

    def get_mo_1e_kinetic(self, mo_type='alpha', mo_block='ov,ov',
                          r_matrix=None):
        """Compute kinetic energy operator in the molecular orbital basis.
    
        Args:
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Molecular orbital rotation to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            A nbf x nbf array of kinetic energy operator integrals,
            < p(1) | - 1 / 2 * nabla_1^2 | q(1) >.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        t = self.integrals.get_ao_1e_kinetic(
            integrate_spin=(mo_type is not 'spinor'),
        )
        return c1.T.dot(t.dot(c2))

    def get_mo_1e_potential(self, mo_type='alpha', mo_block='ov,ov',
                            r_matrix=None):
        """Compute nuclear potential operator in the molecular orbital basis.
    
        Args:
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Molecular orbital rotation to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            A nbf x nbf array of nuclear potential operator integrals,
            < p(1) | sum_A Z_A / r_1A | q(1) >.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        v = self.integrals.get_ao_1e_potential(
            integrate_spin=(mo_type is not 'spinor'),
        )
        return c1.T.dot(v.dot(c2))

    def get_mo_1e_dipole(self, mo_type='alpha', mo_block='ov,ov',
                         r_matrix=None):
        """Compute nuclear potential operator in the molecular orbital basis.
    
        Args:
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Molecular orbital rotation to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
    
        Returns:
            A nbf x nbf array of nuclear potential operator integrals,
            < p(1) | sum_A Z_A / r_1A | q(1) >.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[0],
                                      r_matrix=r_matrix)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_block=mo_spaces[1],
                                      r_matrix=r_matrix)
        mu_comps = self.integrals.get_ao_1e_dipole(
            integrate_spin=(mo_type is not 'spinor'),
        )
        return np.array([c1.T.dot(mu_comp.dot(c2)) for mu_comp in mu_comps])

    def get_mo_2e_repulsion(self, mo_type='alpha', mo_block='ov,ov,ov,ov',
                            r_matrix=None, antisymmetrize=False):
        """Compute electron-repulsion operator in the molecular orbital basis.
    
        Args:
            mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            r_matrix (np.ndarray or tuple): Molecular orbital rotation to be
                applied to the MO coefficients prior to transformation.  Must
                have the full dimension of the spatial or spin-orbital basis,
                including frozen orbitals.  For spatial orbitals, this can be a
                pair of arrays, one for each spin.
            antisymmetrize (bool): Whether or not to symmetrize the repulsion
                integrals as < p q || r s > = < p q | r s > - < p q | s r >.
    
        Returns:
            A nbf x nbf x nbf x nbf array of electron repulsion integrals,
            < p(1) q(2) | 1 / r_12 | r(1) s(2) >.
        """
        mo_spaces = mo_block.split(',')
        kwargs = {
        }
        g = self.integrals.get_ao_2e_repulsion(
            integrate_spin=(mo_type is not 'spinor'),
            antisymmetrize=antisymmetrize
        )
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

