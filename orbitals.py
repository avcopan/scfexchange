import abc
import numpy as np
import tensorutils as tu
import scipy.linalg as spla

from six import with_metaclass
from .integrals import IntegralsInterface


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

    def get_mo_count(self, mo_type='alpha', mo_space='ov'):
        """Return the number of orbitals in a given space.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.

        Returns:
            np.ndarray: The orbital energies.
        """
        count = 0
        if mo_type in ('alpha', 'spinorb'):
            if 'c' in mo_space:
                count += self.n_frozen_orbitals
            if 'o' in mo_space:
                count += self.molecule.nalpha - self.n_frozen_orbitals
            if 'v' in mo_space:
                count += self.integrals.nbf - self.molecule.nalpha
        if mo_type in ('beta', 'spinorb'):
            if 'c' in mo_space:
                count += self.n_frozen_orbitals
            if 'o' in mo_space:
                count += self.molecule.nbeta - self.n_frozen_orbitals
            if 'v' in mo_space:
                count += self.integrals.nbf - self.molecule.nbeta
        return count

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
        if mo_type not in ('spinorb', 'alpha', 'beta'):
            raise ValueError("Invalid 'mo_type' argument.")
        if mo_space not in ('c', 'o', 'v', 'co', 'ov', 'cov'):
            raise ValueError("Invalid 'mo_space' argument.")
        i_start = 'cov'.index(mo_space[0])
        i_end = 'cov'.index(mo_space[-1])
        start = self.get_mo_count(mo_type, 'cov'[:i_start])
        end = self.get_mo_count(mo_type, 'cov'[:i_end + 1])
        return slice(start, end)

    def get_spinorb_order(self):
        """Determine indices that would sort the concatenated orbital energies.
        
        Returns:
            np.ndarray: The sorting indices.
        """
        spinorb_energies = np.concatenate(self.mo_energies)
        spinorb_order = np.argsort(spinorb_energies)
        return spinorb_order

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
            e = self.mo_energies[0]
        elif mo_type is 'beta':
            e = self.mo_energies[1]
        elif mo_type is 'spinorb':
            spinorb_order = self.get_spinorb_order()
            e = np.concatenate(self.mo_energies)[spinorb_order]
        return e[slc]

    def get_mo_coefficients(self, mo_type='alpha', mo_space='ov',
                            transformation=None):
        """Return the molecular orbital coefficients.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
        Returns:
            np.ndarray: The orbital coefficients.
        """
        slc = self.get_mo_slice(mo_type=mo_type, mo_space=mo_space)
        # Grab the appropriate set of coefficients
        if mo_type is 'alpha':
            c = self.mo_coefficients[0]
        elif mo_type is 'beta':
            c = self.mo_coefficients[1]
        elif mo_type is 'spinorb':
            spinorb_order = self.get_spinorb_order()
            c = spla.block_diag(*self.mo_coefficients)[:, spinorb_order]
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
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.

        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        w_ao = self.get_ao_1e_core_field(mo_type=mo_type,
                                         transformation=transformation)
        w_mo = c1.T.dot(w_ao.dot(c2))
        return w_mo

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
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
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
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
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
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
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

    def get_mo_1e_core_hamiltonian(self, mo_type='alpha', mo_block='ov,ov',
                                   transformation=None, electric_field=None,
                                   add_core_repulsion=True):
        """Get the core Hamiltonian integrals.

        Returns the one-particle contribution to the Hamiltonian, i.e.
        everything except for two-electron repulsion.  May include an external
        static electric field in the dipole approximation and/or the mean 
        field of the frozen core electrons.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            electric_field (np.ndarray): A three-component vector specifying the
                magnitude of an external static electric field.  Its dot product
                with the dipole integrals will be added to the core Hamiltonian.
            add_core_repulsion (bool): Add in the core electron mean field?

        Returns:
            np.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        use_spinorbs = (mo_type is 'spinorb')
        h_ao = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=use_spinorbs,
            recompute=False,
            electric_field=electric_field
        )
        if add_core_repulsion:
            w_ao = self.get_ao_1e_core_field(mo_type=mo_type,
                                             transformation=transformation)
            h_ao += w_ao
        h_mo = c1.T.dot(h_ao.dot(c2))
        return h_mo

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
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
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

    def get_ao_1e_hf_density(self, mo_type='alpha', mo_space='ov',
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
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
        Returns:
            np.ndarray: The matrix.
        """
        c = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_space,
                                     transformation=transformation)
        d = c.dot(c.T)
        return d

    def get_ao_1e_core_field(self, mo_type='alpha', transformation=None):
        """Get the core field integrals.
        
        Returns the representation of the mean field of the core electrons in
        the atomic-orbital basis, <mu(1)*spin|J_c(1) - K_c(1)|nu(1)*spin>.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.

        Returns:
            np.ndarray: The integrals
        """
        da = self.get_ao_1e_hf_density('alpha', mo_space='c',
                                       transformation=transformation)
        db = self.get_ao_1e_hf_density('beta', mo_space='c',
                                       transformation=transformation)
        g = self.integrals.get_ao_2e_repulsion(use_spinorbs=False)
        # Compute the Coulomb and exchange matrices.
        j = np.tensordot(g, da + db, axes=[(1, 3), (1, 0)])
        ka = np.tensordot(g, da, axes=[(1, 2), (1, 0)])
        kb = np.tensordot(g, db, axes=[(1, 2), (1, 0)])
        # Return the result.
        if mo_type is 'alpha':
            return j - ka
        elif mo_type is 'beta':
            return j - kb
        elif mo_type is 'spinorb':
            return spla.block_diag(j - ka, j - kb)

    def get_core_energy(self, transformation=None):
        """Get the total mean-field energy of the core electrons.
        
        Includes the nuclear repulsion energy.
        
        Args:
            transformation (np.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.

        Returns:
            float: The core energy.
        """
        da = self.get_ao_1e_hf_density('alpha', mo_space='c',
                                       transformation=transformation)
        db = self.get_ao_1e_hf_density('beta', mo_space='c',
                                       transformation=transformation)
        va = self.get_ao_1e_core_field('alpha', transformation=transformation)
        vb = self.get_ao_1e_core_field('beta', transformation=transformation)
        t = self.integrals.get_ao_1e_kinetic(use_spinorbs=False)
        v = self.integrals.get_ao_1e_potential(use_spinorbs=False)
        h = t + v
        e_elec = np.sum((h + va / 2) * da + (h + vb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return e_elec + e_nuc

