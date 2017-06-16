import more_itertools as mit
import numpy as np
import scipy.linalg as spla
from functools import reduce

import tensorutils as tu
import permutils as pu

from .ao import AOIntegralsInterface


class MOIntegrals(object):
    """Molecular orbitals class.

    Subclasses should override the `solve` method, which sets the value
    of `mo_coeffs`.

    Attributes:
        aoints (:obj:`scfexchange.AOIntegralsInterface`): The integrals.
        molecule (:obj:`scfexchange.Molecule`): A Molecule object specifying
            the total molecular charge and spin multiplicity of the system.
        mo_coeffs (numpy.ndarray): The orbital expansion coefficients.
        spin_is_restricted (bool): Are the orbital spin-restricted?
        ncore (int): The number of low-energy orbitals assigned to the core
            orbital space.
    """

    def __init__(self, aoints, mo_coeffs, naocc, nbocc, ncore=0):
        self.aoints = aoints
        self.mo_coeffs = mo_coeffs
        self.naocc = naocc
        self.nbocc = nbocc
        self.ncore = ncore
        self.norb = self.aoints.nbf
        if not isinstance(aoints, AOIntegralsInterface):
            raise ValueError("Invalid 'integrals' argument.")
        if not (isinstance(mo_coeffs, np.ndarray) and
                mo_coeffs.shape == (2, self.norb, self.norb)):
            raise ValueError("Invalid 'mo_coeffs' argument.")

    def block_keys(self, mo_block, spin_sector):
        spin_keys = spin_sector.split(',') * 2
        space_keys = mo_block.split(',')
        if len(space_keys) is not len(spin_keys):
            raise ValueError("Invalid 'mo_block'/'spin_sector' combination.")
        if 's' in spin_keys and not all(spin == 's' for spin in spin_keys):
            raise NotImplementedError("Mixed spatial/spin-orbital block.")
        return space_keys, spin_keys

    def _transform(self, ao_ints, mo_block, spin_sector, ndim=0):
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
        space_keys, spin_keys = self.block_keys(mo_block, spin_sector)
        cs = tuple(self.mo_coefficients(mo_space, spin)
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
        nbf = self.aoints.nbf
        rot_mat = np.array(rotation_matrix)
        if rot_mat.shape == (nbf, nbf):
            arot_mat = brot_mat = rot_mat
        elif rot_mat.shape == (2, nbf, nbf):
            arot_mat, brot_mat = rot_mat
        elif rot_mat.shape == (2 * nbf, 2 * nbf):
            spinorb_inv_order = self.spinorb_order(invert=True)
            blocked_rot_mat = rot_mat[spinorb_inv_order, :]
            arot_mat = blocked_rot_mat[:nbf, :nbf]
            brot_mat = blocked_rot_mat[nbf:, nbf:]
            if not (np.allclose(blocked_rot_mat[:nbf, nbf:], 0.) and
                        np.allclose(blocked_rot_mat[nbf:, :nbf], 0.)):
                raise ValueError("Spin-orbital rotation matrix mixes spins.")
        else:
            raise ValueError("Invalid 'rotation_matrix' argument.")
        ac, bc = self.mo_coeffs
        self.mo_coeffs = np.array([ac.dot(arot_mat), bc.dot(brot_mat)])

    def spinorb_order(self, invert=False):
        """Get the spin-orbital sort order.

        Args:
            invert (bool): Return the inverse of the sorting permutation?

        Returns:
            tuple: The sorting indices.
        """
        alumo_idx = self.naocc
        blumo_idx = self.nbocc + self.norb
        occ_indices = mit.interleave_longest(range(0, alumo_idx),
                                             range(self.norb, blumo_idx))
        vir_indices = mit.interleave_longest(range(alumo_idx, self.norb),
                                             range(blumo_idx, 2 * self.norb))
        spinorb_order = tuple(occ_indices) + tuple(vir_indices)

        # If requested, return the inverse of the sorting permutation.
        perm_helper = pu.PermutationHelper(range(2 * self.norb))
        spinorb_inv_order = perm_helper.get_inverse(spinorb_order)
        return spinorb_order if not invert else spinorb_inv_order

    def mo_count(self, mo_space='ov', spin='s'):
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
                count += self.naocc - self.ncore
            if 'v' in mo_space:
                count += self.norb - self.naocc
        if spin in ('b', 's'):
            if 'c' in mo_space:
                count += self.ncore
            if 'o' in mo_space:
                count += self.nbocc - self.ncore
            if 'v' in mo_space:
                count += self.norb - self.nbocc
        return count

    def mo_slice(self, mo_space='ov', spin='s'):
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
        start = self.mo_count(mo_space=start_str, spin=spin)
        end = self.mo_count(mo_space=end_str, spin=spin)
        return slice(start, end)

    def mo_coefficients(self, mo_space='ov', spin='s'):
        """Return the molecular orbital coefficients.

        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov',
                which denotes all unfrozen orbitals.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The orbital coefficients.
        """
        slc = self.mo_slice(mo_space=mo_space, spin=spin)
        # Grab the appropriate set of coefficients
        if spin == 'a':
            c = self.mo_coeffs[0]
        elif spin == 'b':
            c = self.mo_coeffs[1]
        elif spin == 's':
            spinorb_order = self.spinorb_order()
            c = spla.block_diag(*self.mo_coeffs)[:, spinorb_order]
        else:
            raise ValueError("Invalid 'spin' argument.")
        return c[:, slc]

    def kinetic(self, mo_block='ov,ov', spin_sector='s'):
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
        t_ao = self.aoints.kinetic(spin_sector == 's')
        return self._transform(t_ao, mo_block, spin_sector)

    def potential(self, mo_block='ov,ov', spin_sector='s'):
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
        v_ao = self.aoints.potential(spin_sector == 's')
        return self._transform(v_ao, mo_block, spin_sector)

    def dipole(self, mo_block='ov,ov', spin_sector='s'):
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
        d_ao = self.aoints.dipole(use_spinorbs=(spin_sector == 's'))
        return self._transform(d_ao, mo_block, spin_sector, ndim=1)

    def core_hamiltonian(self, mo_block='ov,ov', spin_sector='s',
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
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        h_ao = self.aoints.core_hamiltonian(
            use_spinorbs=(spin_sector == 's'), electric_field=electric_field)
        return self._transform(h_ao, mo_block, spin_sector)

    def mean_field(self, mo_block='ov,ov', spin_sector='s',
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
        ac = self.mo_coefficients(mo_space=mo_space, spin='a')
        bc = self.mo_coefficients(mo_space=mo_space, spin='b')
        if spin_sector == 'a':
            w_ao, _ = self.aoints.mean_field(
                alpha_coeffs=ac, beta_coeffs=bc, use_spinorbs=False)
        elif spin_sector == 'b':
            _, w_ao = self.aoints.mean_field(
                alpha_coeffs=ac, beta_coeffs=bc, use_spinorbs=False)
        elif spin_sector == 's':
            w_ao = self.aoints.mean_field(
                alpha_coeffs=ac, beta_coeffs=bc, use_spinorbs=True)
        else:
            raise ValueError("Invalid 'spin_sector' argument.")

        return self._transform(w_ao, mo_block, spin_sector)

    def fock(self, mo_block='ov,ov', spin_sector='s', mo_space='co',
             electric_field=None, split_diagonal=False):
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
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.
            split_digaonal (bool): Split the matrix into a diagonal vector and
                an off-diagonal matrix?

        Returns:
            numpy.ndarray: The integrals.
        """
        ac = self.mo_coefficients(mo_space=mo_space, spin='a')
        bc = self.mo_coefficients(mo_space=mo_space, spin='b')
        if spin_sector == 'a':
            f_ao, _ = self.aoints.fock(
                alpha_coeffs=ac, beta_coeffs=bc, use_spinorbs=False,
                electric_field=electric_field)
        elif spin_sector == 'b':
            _, f_ao = self.aoints.fock(
                alpha_coeffs=ac, beta_coeffs=bc, use_spinorbs=False,
                electric_field=electric_field)
        elif spin_sector == 's':
            f_ao = self.aoints.fock(
                alpha_coeffs=ac, beta_coeffs=bc, use_spinorbs=True,
                electric_field=electric_field)
        else:
            raise ValueError("Invalid 'spin_sector' argument.")

        f_mo = self._transform(f_ao, mo_block, spin_sector)

        if split_diagonal:
            f_mo_diag = f_mo.diagonal().copy()
            np.fill_diagonal(f_mo, 0.)
            return f_mo_diag, f_mo
        else:
            return f_mo

    def electron_repulsion(self, mo_block='ov,ov,ov,ov', spin_sector='s,s',
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
        g_ao = self.aoints.electron_repulsion(
            use_spinorbs=(spin_sector == 's,s'),
            antisymmetrize=(antisymmetrize and s0 == s1))
        return self._transform(g_ao, mo_block, spin_sector)

    def mean_field_energy(self, mo_space='co', electric_field=None):
        """Get the total mean-field energy of a given orbital space.

        Note that this does *not* include the nuclear repulsion energy.

        Args:
            mo_space (str): The MO space of the electrons, 'c' (core),
                'o' (occupied), and 'v' (virtual).
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.

        Returns:
            float: The energy.
        """
        blk = ','.join([mo_space, mo_space])
        ah = self.core_hamiltonian(mo_block=blk, spin_sector='a',
                                   electric_field=electric_field)
        bh = self.core_hamiltonian(mo_block=blk, spin_sector='b',
                                   electric_field=electric_field)
        aw = self.mean_field(mo_block=blk, spin_sector='a',
                             mo_space=mo_space)
        bw = self.mean_field(mo_block=blk, spin_sector='b',
                             mo_space=mo_space)
        return np.trace(ah + aw / 2.) + np.trace(bh + bw / 2.)
