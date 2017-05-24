import abc
import numpy as np
import tensorutils as tu
import scipy.linalg as spla

from six import with_metaclass
from .integrals import IntegralsInterface
from .molecule import Molecule


class OrbitalsInterface(with_metaclass(abc.ABCMeta)):
    """Molecular orbitals base class.
    
    Subclasses should override the `solve` method, which sets the value 
    of `mo_coefficients`.
    
    Attributes:
        integrals (:obj:`scfexchange.Integrals`): The integrals object.
        molecule (:obj:`scfexchange.Molecule`): A Molecule object specifying
            the total molecular charge and spin multiplicity of the system.
        mo_coefficients (numpy.ndarray): The orbital expansion coefficients.
        mo_energies (numpy.ndarray): The orbital energies.
        spin_is_restricted (bool): Are the orbital spin-restricted?
        ncore (int): The number of low-energy orbitals assigned to the core
            orbital space.
    """

    def __init__(self, integrals, charge=0, multiplicity=1, restrict_spin=False,
                 mo_coefficients=None, mo_energies=None, ncore=0):
        """Initialize an instance of the OrbitalsInterface.
        
        Args:
            integrals (:obj:`scfexchange.Integrals`): The integrals object.
            charge (int): Total molecular charge.
            multiplicity (int): Spin multiplicity.
            restrict_spin (bool): Spin-restrict the orbitals?
            mo_coefficients (numpy.ndarray): The orbital expansion coefficients.
            mo_energies (numpy.ndarray): The orbital energies.
            ncore (int): The number of low-energy orbitals to be assigned to the 
                core orbital space.
        """
        self.integrals = integrals
        self.molecule = Molecule(integrals.nuclei, charge, multiplicity)
        self.mo_coefficients = mo_coefficients
        self.mo_energies = mo_energies
        self.spin_is_restricted = restrict_spin
        self.ncore = ncore
        if not isinstance(integrals, IntegralsInterface):
            raise ValueError("Invalid 'integrals' argument.")
        nbf = integrals.nbf
        if self.mo_energies is None:
            self.mo_energies = np.zeros((2, nbf))
        elif not (isinstance(self.mo_energies, np.ndarray)
                  and self.mo_energies.shape == (2, nbf)):
            raise ValueError("Invalid 'mo_energies' argument.")
        if self.mo_coefficients is None:
            self.mo_coefficients = np.zeros((2, nbf, nbf))
        elif not (isinstance(self.mo_coefficients, np.ndarray)
                  and self.mo_coefficients.shape == (2, nbf, nbf)):
            raise ValueError("Invalid 'mo_coefficients' argument.")

    @abc.abstractmethod
    def solve(self, **options):
        """Solve for the orbitals.

        Sets `self.mo_coefficients` to a new value.
        
        Args:
            **options: Parameters defining the behavior of the solution 
                algorithm, such as convergence criteria.
        """
        return

    def get_mo_count(self, mo_type='alpha', mo_space='ov'):
        """Return the number of orbitals in a given space.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.

        Returns:
            numpy.ndarray: The orbital energies.
        """
        count = 0
        if mo_type in ('alpha', 'spinorb'):
            if 'c' in mo_space:
                count += self.ncore
            if 'o' in mo_space:
                count += self.molecule.nalpha - self.ncore
            if 'v' in mo_space:
                count += self.integrals.nbf - self.molecule.nalpha
        if mo_type in ('beta', 'spinorb'):
            if 'c' in mo_space:
                count += self.ncore
            if 'o' in mo_space:
                count += self.molecule.nbeta - self.ncore
            if 'v' in mo_space:
                count += self.integrals.nbf - self.molecule.nbeta
        return count

    def get_mo_slice(self, mo_type='alpha', mo_space='ov'):
        """Return the slice for a given orbital space.
    
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
            numpy.ndarray: The sorting indices.
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
            numpy.ndarray: The orbital energies.
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

    def get_mo_fock_diagonal(self, mo_type='alpha', mo_space='ov',
                             transformation=None, electric_field=None):
        """Get the Fock operator integrals.

        Returns the diagonal elements of the Fock matrix.  For canonical 
        Hartree-Fock orbitals, these will be the orbital energies.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        c = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_space,
                                     transformation=transformation)
        f_ao = self.get_ao_1e_fock(mo_type=mo_type,
                                   transformation=transformation,
                                   electric_field=electric_field)
        e = tu.einsum('mn,mi,ni->i', f_ao, c, c)
        return e

    def get_mo_coefficients(self, mo_type='alpha', mo_space='ov',
                            transformation=None):
        """Return the molecular orbital coefficients.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
        Returns:
            numpy.ndarray: The orbital coefficients.
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
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
        Returns:
            numpy.ndarray: The integrals.
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
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
        Returns:
            numpy.ndarray: The integrals.
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
        molecular-orbital basis, <p(1)| [-x, -y, -z] |q(1)>.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        use_spinorbs = (mo_type is 'spinorb')
        d_ao = self.integrals.get_ao_1e_dipole(use_spinorbs)
        d_mo = np.array([c1.T.dot(d_ao_x.dot(c2)) for d_ao_x in d_ao])
        return d_mo

    def get_mo_1e_fock(self, mo_type='alpha', mo_block='ov,ov',
                       transformation=None, electric_field=None):
        """Get the Fock operator integrals.
        
        Returns the core Hamiltonian plus the mean field of the electrons in 
        the molecular-orbital basis, <p(1)*spin|h(1) + J(1) - K(1)|q(1)*spin>.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        f_ao = self.get_ao_1e_fock(mo_type=mo_type,
                                   transformation=transformation,
                                   electric_field=electric_field)
        f_mo = c1.T.dot(f_ao.dot(c2))
        return f_mo

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
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.
            add_core_repulsion (bool): Add in the core electron mean field?

        Returns:
            numpy.ndarray: The integrals.
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
            w_ao = self.get_ao_1e_mean_field(mo_type=mo_type, mo_space='c',
                                             transformation=transformation)
            h_ao += w_ao
        h_mo = c1.T.dot(h_ao.dot(c2))
        return h_mo

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
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0],
                                      transformation=transformation)
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1],
                                      transformation=transformation)
        w_ao = self.get_ao_1e_mean_field(mo_type=mo_type, mo_space='c',
                                         transformation=transformation)
        w_mo = c1.T.dot(w_ao.dot(c2))
        return w_mo

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
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            antisymmetrize (bool): Antisymmetrize the integral tensor?
    
        Returns:
            numpy.ndarray: The integrals.
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

    def get_ao_1e_density(self, mo_type='alpha', mo_space='co',
                          transformation=None):
        """Get the electronic density matrix.
        
        Returns the SCF density matrix, D_mu,nu = sum_i C_mu,i C_nu,i^*, in the
        atomic-orbital basis.  This is not quite equal to the one-particle
        reduced density matrix, which is S * D * S with S denoting the atomic
        orbital overlap matrix.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual), defining the space of 
                electrons generating the field.  Defaults to all 'co', 
                which includes all frozen and unfrozen occupied electrons.
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
    
        Returns:
            numpy.ndarray: The matrix.
        """
        c = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_space,
                                     transformation=transformation)
        d = c.dot(c.T)
        return d

    def get_ao_1e_mean_field(self, mo_type='alpha', mo_space='co',
                             transformation=None):
        """Get the electron mean-field integrals.
        
        Returns the representation of electronic mean field of a given orbital 
        space in the atomic-orbital basis, <mu(1)*spin|J(1) - K(1)|nu(1)*spin>.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual), defining the space of 
                electrons generating the field.  Defaults to all 'co', 
                which includes all frozen and unfrozen occupied electrons.
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.

        Returns:
            numpy.ndarray: The integrals
        """
        da = self.get_ao_1e_density('alpha', mo_space=mo_space,
                                    transformation=transformation)
        db = self.get_ao_1e_density('beta', mo_space=mo_space,
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

    def get_ao_1e_fock(self, mo_type='alpha', transformation=None,
                       electric_field=None):
        """Get the Fock operator integrals.
        
        Returns the core Hamiltonian plus the mean field of the electrons in 
        the atomic-orbital basis, <mu(1)*spin|h(1) + J(1) - K(1)|nu(1)*spin>.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        use_spinorbs = (mo_type == 'spinorb')
        h = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=use_spinorbs, electric_field=electric_field)
        w = self.get_ao_1e_mean_field(mo_type=mo_type, mo_space='co',
                                      transformation=transformation)
        return h + w

    def get_hf_energy(self, transformation=None, electric_field=None):
        """Get the total mean-field energy of the occupied electrons.
        
        Includes the nuclear repulsion energy.
        
        Args:
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.
        Returns:
            float: The energy.
        """
        h = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=False, electric_field=electric_field)
        da = self.get_ao_1e_density('alpha', mo_space='co',
                                    transformation=transformation)
        db = self.get_ao_1e_density('beta', mo_space='co',
                                    transformation=transformation)
        wa = self.get_ao_1e_mean_field(mo_type='alpha', mo_space='co',
                                       transformation=transformation)
        wb = self.get_ao_1e_mean_field(mo_type='beta', mo_space='co',
                                       transformation=transformation)
        e_elec = np.sum((h + wa / 2) * da + (h + wb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return e_elec + e_nuc

    def get_core_energy(self, transformation=None, electric_field=None):
        """Get the total mean-field energy of the core electrons.
        
        Includes the nuclear repulsion energy.
        
        Args:
            transformation (numpy.ndarray): Orbital transformation matrix to be
                applied to the MO coefficients.  Either a single matrix or a 
                pair of matrices for alpha and beta spins.
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.
        Returns:
            float: The energy.
        """
        h = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=False, electric_field=electric_field)
        da = self.get_ao_1e_density('alpha', mo_space='c',
                                    transformation=transformation)
        db = self.get_ao_1e_density('beta', mo_space='c',
                                    transformation=transformation)
        wa = self.get_ao_1e_mean_field(mo_type='alpha', mo_space='c',
                                       transformation=transformation)
        wb = self.get_ao_1e_mean_field(mo_type='beta', mo_space='c',
                                       transformation=transformation)
        e_elec = np.sum((h + wa / 2) * da + (h + wb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return e_elec + e_nuc
