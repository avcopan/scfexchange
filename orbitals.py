import abc

import more_itertools as mit
import numpy as np
import scipy.linalg as spla
import tensorutils as tu
import permutils as pu
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
        spin_is_restricted (bool): Are the orbital spin-restricted?
        ncore (int): The number of low-energy orbitals assigned to the core
            orbital space.
    """

    def __init__(self, integrals, charge=0, multiplicity=1, restrict_spin=False,
                 mo_coefficients=None, ncore=0):
        """Initialize an instance of the OrbitalsInterface.
        
        Args:
            integrals (:obj:`scfexchange.Integrals`): The integrals object.
            charge (int): Total molecular charge.
            multiplicity (int): Spin multiplicity.
            restrict_spin (bool): Spin-restrict the orbitals?
            mo_coefficients (numpy.ndarray): The orbital expansion coefficients.
            ncore (int): The number of low-energy orbitals to be assigned to the
                core orbital space.
        """
        self.integrals = integrals
        self.molecule = Molecule(integrals.nuclei, charge, multiplicity)
        self.mo_coefficients = mo_coefficients
        self.spin_is_restricted = bool(restrict_spin)
        self.ncore = int(ncore)
        if not isinstance(integrals, IntegralsInterface):
            raise ValueError("Invalid 'integrals' argument.")
        nbf = integrals.nbf
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

    def get_spinorb_order(self, invert=False):
        """Get the spin-orbital sort order.

        Args:
            invert (bool): Return the inverse of the sorting permutation?

        Returns:
            tuple: The sorting indices.
        """
        nbf = self.integrals.nbf
        alumo_idx = self.molecule.nalpha
        blumo_idx = self.molecule.nbeta + nbf
        occ_indices = tuple(mit.interleave_longest(range(0, alumo_idx),
                                                   range(nbf, blumo_idx)))
        vir_indices = tuple(mit.interleave_longest(range(alumo_idx, nbf),
                                                   range(blumo_idx, 2 * nbf)))
        spinorb_order = occ_indices + vir_indices

        # If requested, return the inverse of the sorting permutation.
        perm_helper = pu.PermutationHelper(range(2 * nbf))
        spinorb_inv_order = perm_helper.get_inverse(spinorb_order)
        return spinorb_order if not invert else spinorb_inv_order

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

    def get_mo_coefficients(self, mo_type='alpha', mo_space='ov'):
        """Return the molecular orbital coefficients.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.

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
        return c[:, slc]

    def rotate(self, rotation_matrix):
        """Rotate the orbitals with a unitary transformation.

        Args:
            rotation_matrix (numpy.ndarray): Orbital rotation matrix. This can
                be a single square matrix of dimension `norb`, a pair of such
                matrices for alpha and beta spins, or a square matrix of
                dimension `2 * norb`, where `norb` is the total number of
                spatial orbitals.  The final case corresponds to a rotation
                in the spin-orbital basis, but note that it must not combine
                alpha and beta spins.
        """
        norb = self.get_mo_count(mo_space='cov')
        rot_mat = np.array(rotation_matrix)
        if rot_mat.shape == (norb, norb):
            arot_mat = brot_mat = rot_mat
        elif rot_mat.shape == (2, norb, norb):
            arot_mat, brot_mat = rot_mat
        elif rot_mat.shape == (2 * norb, 2 * norb):
            spinorb_inv_order = self.get_spinorb_order(invert=True)
            blocked_rot_mat = rot_mat[spinorb_inv_order, :]
            arot_mat = blocked_rot_mat[:norb, :norb]
            brot_mat = blocked_rot_mat[norb:, norb:]
            if not (np.allclose(blocked_rot_mat[:norb, norb:], 0.) and
                    np.allclose(blocked_rot_mat[norb:, :norb], 0.)):
                raise ValueError("Spin-orbital rotation matrix mixes spins.")
        else:
            raise ValueError("Invalid 'rotation_matrix' argument.")
        ac, bc = self.mo_coefficients
        self.mo_coefficients = np.array([ac.dot(arot_mat), bc.dot(brot_mat)])

    def get_mo_fock_diagonal(self, mo_type='alpha', mo_space='ov',
                             electric_field=None):
        """Get the Fock operator integrals.

        Returns the diagonal elements of the Fock matrix.  For canonical
        Hartree-Fock orbitals, these will be the orbital energies.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its
                negative dot product with the dipole integrals will be added to
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        c = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_space)
        f_ao = self.get_ao_1e_fock(mo_type=mo_type,
                                   electric_field=electric_field)
        e = tu.einsum('mn,mi,ni->i', f_ao, c, c)
        return e

    def get_mo_1e_kinetic(self, mo_type='alpha', mo_block='ov,ov'):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the molecular-orbital basis, <p(1)| - 1 / 2 * nabla_1^2 |q(1)>.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0])
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1])
        use_spinorbs = (mo_type is 'spinorb')
        t_ao = self.integrals.get_ao_1e_kinetic(use_spinorbs)
        t_mo = c1.T.dot(t_ao.dot(c2))
        return t_mo

    def get_mo_1e_potential(self, mo_type='alpha', mo_block='ov,ov'):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        molecular-orbital basis, <p(1)| sum_A Z_A / ||r_1 - r_A|| |q(1)>.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0])
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1])
        use_spinorbs = (mo_type is 'spinorb')
        v_ao = self.integrals.get_ao_1e_potential(use_spinorbs)
        v_mo = c1.T.dot(v_ao.dot(c2))
        return v_mo

    def get_mo_1e_dipole(self, mo_type='alpha', mo_block='ov,ov'):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        molecular-orbital basis, <p(1)| [-x, -y, -z] |q(1)>.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0])
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1])
        use_spinorbs = (mo_type is 'spinorb')
        d_ao = self.integrals.get_ao_1e_dipole(use_spinorbs)
        d_mo = np.array([c1.T.dot(d_ao_x.dot(c2)) for d_ao_x in d_ao])
        return d_mo

    def get_mo_1e_fock(self, mo_type='alpha', mo_block='ov,ov',
                       electric_field=None):
        """Get the Fock operator integrals.
        
        Returns the core Hamiltonian plus the mean field of the electrons in 
        the molecular-orbital basis, <p(1)*spin|h(1) + J(1) - K(1)|q(1)*spin>.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0])
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1])
        f_ao = self.get_ao_1e_fock(mo_type=mo_type,
                                   electric_field=electric_field)
        f_mo = c1.T.dot(f_ao.dot(c2))
        return f_mo

    def get_mo_1e_core_hamiltonian(self, mo_type='alpha', mo_block='ov,ov',
                                   electric_field=None,
                                   add_core_repulsion=True):
        """Get the core Hamiltonian integrals.

        Returns the one-particle contribution to the Hamiltonian, i.e.
        everything except for two-electron repulsion.  May include an external
        static electric field in the dipole approximation and/or the mean 
        field of the frozen core electrons.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.
            add_core_repulsion (bool): Add in the core electron mean field?

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0])
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1])
        use_spinorbs = (mo_type is 'spinorb')
        h_ao = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=use_spinorbs,
            recompute=False,
            electric_field=electric_field
        )
        if add_core_repulsion:
            w_ao = self.get_ao_1e_mean_field(mo_type=mo_type, mo_space='c')
            h_ao += w_ao
        h_mo = c1.T.dot(h_ao.dot(c2))
        return h_mo

    def get_mo_1e_core_field(self, mo_type='alpha', mo_block='ov,ov'):
        """Get the core field integrals.
        
        Returns the representation of the mean field of the core electrons in
        the molecular-orbital basis, <p(1)|J_c(1) + K_c(1)|q(1)>.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).

        Returns:
            numpy.ndarray: The integrals.
        """
        mo_spaces = mo_block.split(',')
        c1 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[0])
        c2 = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_spaces[1])
        w_ao = self.get_ao_1e_mean_field(mo_type=mo_type, mo_space='c')
        w_mo = c1.T.dot(w_ao.dot(c2))
        return w_mo

    def get_mo_2e_repulsion(self, mo_type='alpha', mo_block='ov,ov,ov,ov',
                            antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        molecular-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', 'mixed', or 'spinorb'.
            mo_block (str): A comma-separated list of four MO spaces.  Each MO
                space is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
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
                                          mo_space=mo_spaces[0])
            c2 = self.get_mo_coefficients(mo_type='beta',
                                          mo_space=mo_spaces[1])
            c3 = self.get_mo_coefficients(mo_type='alpha',
                                          mo_space=mo_spaces[2])
            c4 = self.get_mo_coefficients(mo_type='beta',
                                          mo_space=mo_spaces[3])
        else:
            c1 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[0])
            c2 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[1])
            c3 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[2])
            c4 = self.get_mo_coefficients(mo_type=mo_type,
                                          mo_space=mo_spaces[3])
        return tu.einsum("mntu,mp,nq,tr,us->pqrs", g, c1, c2, c3, c4)

    def get_ao_1e_density(self, mo_type='alpha', mo_space='co'):
        """Get the electronic density matrix.
        
        Returns the SCF density matrix, D_mu,nu = sum_i C_mu,i C_nu,i^*, in the
        atomic-orbital basis.  This is not the same as the one-particle reduced
        density matrix, which is S * D * S where S is the atomic orbital overlap
        matrix.
    
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual), defining the space of 
                electrons generating the field.  Defaults to all 'co', 
                which includes all frozen and unfrozen occupied electrons.

        Returns:
            numpy.ndarray: The matrix.
        """
        c = self.get_mo_coefficients(mo_type=mo_type, mo_space=mo_space)
        d = c.dot(c.T)
        return d

    def get_ao_1e_mean_field(self, mo_type='alpha', mo_space='co'):
        """Get the electron mean-field integrals.
        
        Returns the representation of electronic mean field of a given orbital 
        space in the atomic-orbital basis, <mu(1)*spin|J(1) - K(1)|nu(1)*spin>.

        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual), defining the space of 
                electrons generating the field.  Defaults to all 'co', 
                which includes all frozen and unfrozen occupied electrons.

        Returns:
            numpy.ndarray: The integrals
        """
        da = self.get_ao_1e_density('alpha', mo_space=mo_space)
        db = self.get_ao_1e_density('beta', mo_space=mo_space)
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

    def get_ao_1e_fock(self, mo_type='alpha', electric_field=None):
        """Get the Fock operator integrals.
        
        Returns the core Hamiltonian plus the mean field of the electrons in 
        the atomic-orbital basis, <mu(1)*spin|h(1) + J(1) - K(1)|nu(1)*spin>.
        
        Args:
            mo_type (str): Orbital type, 'alpha', 'beta', or 'spinorb'.
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
        w = self.get_ao_1e_mean_field(mo_type=mo_type, mo_space='co')
        return h + w

    def get_hf_energy(self, electric_field=None):
        """Get the total mean-field energy of the occupied electrons.
        
        Includes the nuclear repulsion energy.
        
        Args:
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.
        Returns:
            float: The energy.
        """
        h = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=False, electric_field=electric_field)
        da = self.get_ao_1e_density('alpha', mo_space='co')
        db = self.get_ao_1e_density('beta', mo_space='co')
        wa = self.get_ao_1e_mean_field(mo_type='alpha', mo_space='co')
        wb = self.get_ao_1e_mean_field(mo_type='beta', mo_space='co')
        e_elec = np.sum((h + wa / 2) * da + (h + wb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return e_elec + e_nuc

    def get_core_energy(self, electric_field=None):
        """Get the total mean-field energy of the core electrons.
        
        Includes the nuclear repulsion energy.
        
        Args:
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            float: The energy.
        """
        h = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=False, electric_field=electric_field)
        da = self.get_ao_1e_density('alpha', mo_space='c')
        db = self.get_ao_1e_density('beta', mo_space='c')
        wa = self.get_ao_1e_mean_field(mo_type='alpha', mo_space='c')
        wb = self.get_ao_1e_mean_field(mo_type='beta', mo_space='c')
        e_elec = np.sum((h + wa / 2) * da + (h + wb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return e_elec + e_nuc
