import abc

import more_itertools as mit
import numpy as np
import scipy.linalg as spla
from six import with_metaclass
from functools import reduce

import tensorutils as tu
import permutils as pu

from .integrals import IntegralsInterface
from .molecule import Molecule


class OrbitalsInterface(with_metaclass(abc.ABCMeta)):
    """Molecular orbitals base class.
    
    Subclasses should override the `solve` method, which sets the value 
    of `mo_coefficients`.
    
    Attributes:
        integrals (:obj:`scfexchange.IntegralsInterface`): The integrals.
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
            integrals (:obj:`scfexchange.IntegralsInterface`): The integrals.
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
            **options: Convergence thresholds, etc.
        """
        return

    def get_block_keys(self, mo_block, spin_sector):
        spin_keys = spin_sector.split(',') * 2
        space_keys = mo_block.split(',')
        if len(space_keys) is not len(spin_keys):
            raise ValueError("Invalid 'mo_block'/'spin_sector' combination.")
        if 's' in spin_keys and not all(spin == 's' for spin in spin_keys):
            raise NotImplementedError("Mixed spatial/spin-orbital block.")
        return space_keys, spin_keys

    def _transform_ints(self, ao_ints, mo_block, spin_sector, ndim=0):
        """Transform atomic-orbital integrals to the molecular-orbital basis.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            ndim (int): The number of axes that refer to operator components
                rather than basis functions.  For example, diple integrals will
                have ndim=1 since the first axis refers to the Cartesian
                component of the dipole operator.  Secon-derivative integrals
                will have ndim=1.

        Returns:
            numpy.array: The integrals.
        """
        space_keys, spin_keys = self.get_block_keys(mo_block, spin_sector)
        cs = tuple(self.get_mo_coefficients(mo_space, spin)
                   for mo_space, spin in zip(space_keys, spin_keys))

        # Before transforming, move the component axes to the end.
        ao_ints = np.moveaxis(ao_ints, range(0, ndim), range(-ndim, 0))
        assert(ao_ints.ndim is len(cs) + ndim)
        mo_ints = reduce(tu.contract, cs, ao_ints)
        return mo_ints

    def rotate(self, rotation_matrix):
        """Rotate the orbitals with a unitary transformation.

        Args:
            rotation_matrix (numpy.ndarray): Orbital rotation matrix. This can
                be a single square matrix of dimension `nbf`, a pair of such
                matrices for alpha and beta spins, or a square matrix of
                dimension `2 * nbf`, where `nbf` is the total number of
                spatial orbitals.  The final case corresponds to a rotation
                in the spin-orbital basis, but note that it must not combine
                alpha and beta spins.
        """
        nbf = self.integrals.nbf
        rot_mat = np.array(rotation_matrix)
        if rot_mat.shape == (nbf, nbf):
            arot_mat = brot_mat = rot_mat
        elif rot_mat.shape == (2, nbf, nbf):
            arot_mat, brot_mat = rot_mat
        elif rot_mat.shape == (2 * nbf, 2 * nbf):
            spinorb_inv_order = self.get_spinorb_order(invert=True)
            blocked_rot_mat = rot_mat[spinorb_inv_order, :]
            arot_mat = blocked_rot_mat[:nbf, :nbf]
            brot_mat = blocked_rot_mat[nbf:, nbf:]
            if not (np.allclose(blocked_rot_mat[:nbf, nbf:], 0.) and
                        np.allclose(blocked_rot_mat[nbf:, :nbf], 0.)):
                raise ValueError("Spin-orbital rotation matrix mixes spins.")
        else:
            raise ValueError("Invalid 'rotation_matrix' argument.")
        ac, bc = self.mo_coefficients
        self.mo_coefficients = np.array([ac.dot(arot_mat), bc.dot(brot_mat)])

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

    def get_mo_count(self, mo_space='ov', spin='s'):
        """Return the number of orbitals in a given space.

        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            int: The number of orbitals.
        """
        count = 0
        if spin in ('a', 's'):
            if 'c' in mo_space:
                count += self.ncore
            if 'o' in mo_space:
                count += self.molecule.nalpha - self.ncore
            if 'v' in mo_space:
                count += self.integrals.nbf - self.molecule.nalpha
        if spin in ('b', 's'):
            if 'c' in mo_space:
                count += self.ncore
            if 'o' in mo_space:
                count += self.molecule.nbeta - self.ncore
            if 'v' in mo_space:
                count += self.integrals.nbf - self.molecule.nbeta
        return count

    def get_mo_slice(self, mo_space='ov', spin='s'):
        """Return the slice for a given orbital space.
    
        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov', 
                which denotes all unfrozen orbitals.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            slice: The slice.
        """
        # Check the arguments to make sure all is kosher
        if spin not in ('a', 'b', 's'):
            raise ValueError("Invalid 'mo_type' argument.")
        if mo_space not in ('', 'c', 'o', 'v', 'co', 'ov', 'cov'):
            raise ValueError("Invalid 'mo_space' argument.")
        if mo_space == '':
            return slice(0)
        start_str = 'cov'[:'cov'.index(mo_space[0])]
        end_str = 'cov'[:'cov'.index(mo_space[-1]) + 1]
        start = self.get_mo_count(mo_space=start_str, spin=spin)
        end = self.get_mo_count(mo_space=end_str, spin=spin)
        return slice(start, end)

    def get_mo_coefficients(self, mo_space='ov', spin='s'):
        """Return the molecular orbital coefficients.
    
        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov',
                which denotes all unfrozen orbitals.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The orbital coefficients.
        """
        slc = self.get_mo_slice(mo_space=mo_space, spin=spin)
        # Grab the appropriate set of coefficients
        if spin == 'a':
            c = self.mo_coefficients[0]
        elif spin == 'b':
            c = self.mo_coefficients[1]
        elif spin == 's':
            spinorb_order = self.get_spinorb_order()
            c = spla.block_diag(*self.mo_coefficients)[:, spinorb_order]
        else:
            raise ValueError("Invalid 'spin' argument.")
        return c[:, slc]

    def get_mo_fock_diagonal(self, mo_space='ov', spin='s',
                             electric_field=None):
        """Get the Fock operator integrals.

        Returns the diagonal elements of the Fock matrix.  For canonical
        Hartree-Fock orbitals, these will be the orbital energies.

        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov',
                which denotes all unfrozen orbitals.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its
                negative dot product with the dipole integrals will be added to
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        c = self.get_mo_coefficients(mo_space=mo_space, spin=spin)
        f_ao = self.get_ao_1e_fock(mo_space='co', spin_sector=spin,
                                   electric_field=electric_field)
        e = tu.einsum('mn,mi,ni->i', f_ao, c, c)
        return e

    def get_mo_1e_kinetic(self, mo_block='ov,ov', spin_sector='s'):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the molecular-orbital basis, <p(1)| - 1 / 2 * nabla_1^2 |q(1)>.
    
        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The integrals.
        """
        t_ao = self.integrals.get_ao_1e_kinetic(spin_sector == 's')
        return self._transform_ints(t_ao, mo_block, spin_sector)

    def get_mo_1e_potential(self, mo_block='ov,ov', spin_sector='s'):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        molecular-orbital basis, <p(1)| sum_A Z_A / ||r_1 - r_A|| |q(1)>.
    
        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The integrals.
        """
        v_ao = self.integrals.get_ao_1e_potential(spin_sector == 's')
        return self._transform_ints(v_ao, mo_block, spin_sector)

    def get_mo_1e_dipole(self, mo_block='ov,ov', spin_sector='s'):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        molecular-orbital basis, <p(1)| [-x, -y, -z] |q(1)>.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The integrals.
        """
        d_ao = self.integrals.get_ao_1e_dipole(spin_sector == 's')
        return self._transform_ints(d_ao, mo_block, spin_sector, ndim=1)

    def get_mo_1e_core_hamiltonian(self, mo_block='ov,ov', spin_sector='s',
                                   electric_field=None):
        """Get the core Hamiltonian integrals.

        Returns the one-particle contribution to the Hamiltonian, i.e.
        everything except for two-electron repulsion.  May include an external
        static electric field in the dipole approximation and/or the mean
        field of the frozen core electrons.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its
                negative dot product with the dipole integrals will be added to
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        h_ao = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=(spin_sector == 's'), electric_field=electric_field)
        return self._transform_ints(h_ao, mo_block, spin_sector)

    def get_mo_1e_mean_field(self, mo_block='ov,ov', spin_sector='s',
                             mo_space='co'):
        """Get the core field integrals.

        Returns the representation of electronic mean field of a given orbital
        space in the molecular-orbital basis, <p(1)*spin|J(1) - K(1)|q(1)*spin>.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            mo_space (str): The MO space applying the field, 'c' (core),
                'o' (occupied), and 'v' (virtual).

        Returns:
            numpy.ndarray: The integrals.
        """
        w_ao = self.get_ao_1e_mean_field(mo_space=mo_space,
                                         spin_sector=spin_sector)
        return self._transform_ints(w_ao, mo_block, spin_sector)

    def get_mo_1e_fock(self, mo_block='ov,ov', spin_sector='s', mo_space='co',
                       electric_field=None):
        """Get the Fock operator integrals.
        
        Returns the core Hamiltonian plus the mean field of an orbital space in
        the molecular-orbital basis, <p(1)*spin|h(1) + J(1) - K(1)|q(1)*spin>.
        
        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            mo_space (str): The MO space applying the field, 'c' (core),
                'o' (occupied), and 'v' (virtual).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        f_ao = self.get_ao_1e_fock(mo_space=mo_space, spin_sector=spin_sector,
                                   electric_field=electric_field)
        return self._transform_ints(f_ao, mo_block, spin_sector)

    def get_mo_2e_repulsion(self, mo_block='ov,ov,ov,ov', spin_sector='s,s',
                            antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        molecular-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            antisymmetrize (bool): Antisymmetrize the integral tensor?
    
        Returns:
            numpy.ndarray: The integrals.
        """
        s0, s1 = spin_sector.split(',')
        g_ao = self.integrals.get_ao_2e_repulsion(
            use_spinorbs=(spin_sector == 's,s'),
            antisymmetrize=(s0 == s1))
        return self._transform_ints(g_ao, mo_block, spin_sector)

    def get_ao_1e_hf_density(self, mo_space='co', spin_sector='s'):
        """Get the electronic density matrix.
        
        Returns the SCF density matrix, D_mu,nu = sum_i C_mu,i C_nu,i^*, in the
        atomic-orbital basis.  This is not the same as the one-particle reduced
        density matrix, which is S * D * S where S is the atomic orbital overlap
        matrix.
    
        Args:
            mo_space (str): The MO space of the electrons, 'c' (core),
                'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The matrix.
        """
        c = self.get_mo_coefficients(mo_space=mo_space, spin=spin_sector)
        d = c.dot(c.T)
        return d

    def get_ao_1e_mean_field(self, mo_space='co', spin_sector='s'):
        """Get the electron mean-field integrals.
        
        Returns the representation of electronic mean field of a given orbital 
        space in the atomic-orbital basis, <mu(1)*spin|J(1) - K(1)|nu(1)*spin>.

        Args:
            mo_space (str): The MO space applying the field, 'c' (core),
                'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The integrals
        """
        da = self.get_ao_1e_hf_density(mo_space=mo_space, spin_sector='a')
        db = self.get_ao_1e_hf_density(mo_space=mo_space, spin_sector='b')
        g = self.integrals.get_ao_2e_repulsion(use_spinorbs=False)
        # Compute the Coulomb and exchange matrices.
        j = np.tensordot(g, da + db, axes=[(1, 3), (1, 0)])
        ka = np.tensordot(g, da, axes=[(1, 2), (1, 0)])
        kb = np.tensordot(g, db, axes=[(1, 2), (1, 0)])
        # Return the result.
        if spin_sector == 'a':
            return j - ka
        elif spin_sector == 'b':
            return j - kb
        elif spin_sector == 's':
            return spla.block_diag(j - ka, j - kb)

    def get_ao_1e_fock(self, mo_space='co', spin_sector='s',
                       electric_field=None):
        """Get the Fock operator integrals.
        
        Returns the core Hamiltonian plus the mean field of an orbital space in
        the atomic-orbital basis, <mu(1)*spin|h(1) + J(1) - K(1)|nu(1)*spin>.
        
        Args:
            mo_space (str): The MO space applying the field, 'c' (core),
                'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        h = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=(spin_sector == 's'), electric_field=electric_field)
        w = self.get_ao_1e_mean_field(mo_space=mo_space,
                                      spin_sector=spin_sector)
        return h + w

    def get_energy(self, mo_space='co', electric_field=None):
        """Get the total mean-field energy of a given orbital space.
        
        Includes the nuclear repulsion energy.
        
        Args:
            mo_space (str): The MO space of the electrons, 'c' (core),
                'o' (occupied), and 'v' (virtual).
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its 
                negative dot product with the dipole integrals will be added to 
                the core Hamiltonian.

        Returns:
            float: The energy.
        """
        h = self.integrals.get_ao_1e_core_hamiltonian(
            use_spinorbs=False, electric_field=electric_field)
        da = self.get_ao_1e_hf_density(mo_space=mo_space, spin_sector='a')
        db = self.get_ao_1e_hf_density(mo_space=mo_space, spin_sector='b')
        wa = self.get_ao_1e_mean_field(mo_space=mo_space, spin_sector='a')
        wb = self.get_ao_1e_mean_field(mo_space=mo_space, spin_sector='b')
        e_elec = np.sum((h + wa / 2) * da + (h + wb / 2) * db)
        e_nuc = self.molecule.nuclei.get_nuclear_repulsion_energy()
        return e_elec + e_nuc

