from functools import reduce

import more_itertools as mit
import numpy as np
import permutils as pu
import scipy.linalg as spla
import tensorutils as tu

from . import ao


class MOIntegrals(object):
    """Molecular moints class.

    Subclasses should override the `solve` method, which sets the value
    of `mo_coeffs`.

    Attributes:
        aoints (:obj:`scfexchange.ao.AOIntegralsInterface`): The integrals.
        mo_coeffs (numpy.ndarray): The orbital expansion coefficients, as a pair
            of alpha and beta arrays.
        nao: The number of alpha electrons.
        nbo: The number of beta electrons.
        nc: The number of electron pairs to be designated 'c' (core).
    """

    def __init__(self, aoints, mo_coeffs, nao, nbo, nc=0):
        """Initalize an MOIntegrals object.

        Args:
            aoints (:obj:`scfexchange.ao.AOIntegralsInterface`): The integrals.
            mo_coeffs (numpy.ndarray): The orbital expansion coefficients,
                as a pair of alpha and beta arrays.
            nao: The number of alpha electrons.
            nbo: The number of beta electrons.
            nc: The number of electron pairs to be designated 'c' (core).
        """
        self.aoints = aoints
        self.mo_coeffs = np.array(mo_coeffs)
        self.nao = int(nao)
        self.nbo = int(nbo)
        self.nc = int(nc)
        self.nbf = int(self.aoints.nbf)
        if not isinstance(aoints, ao.AOIntegralsInterface):
            raise ValueError("Invalid 'integrals' argument.")
        if not (isinstance(self.mo_coeffs, np.ndarray) and
                        self.mo_coeffs.shape == (2, self.nbf, self.nbf)):
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
        assert (ao_ints.ndim is len(cs) + ndim)
        mo_ints = reduce(tu.contract, cs, ao_ints)
        return mo_ints

    def rotate(self, rotation_matrix):
        """Rotate the moints with a unitary transformation.

        Args:
            rotation_matrix (numpy.ndarray): Orbital rotation matrix. This can
                be a single square matrix of dimension `nbf`, a pair of such
                matrices for alpha and beta spins, or a square matrix of
                dimension `2 * nbf`, where `nbf` is the total number of
                spatial moints.  The final case corresponds to a rotation
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
        alumo_idx = self.nao
        blumo_idx = self.nbo + self.nbf
        occ_indices = mit.interleave_longest(range(0, alumo_idx),
                                             range(self.nbf, blumo_idx))
        vir_indices = mit.interleave_longest(range(alumo_idx, self.nbf),
                                             range(blumo_idx, 2 * self.nbf))
        spinorb_order = tuple(occ_indices) + tuple(vir_indices)

        # If requested, return the inverse of the sorting permutation.
        perm_helper = pu.PermutationHelper(range(2 * self.nbf))
        spinorb_inv_order = perm_helper.get_inverse(spinorb_order)
        return spinorb_order if not invert else spinorb_inv_order

    def mo_count(self, mo_space='ov', spin='s'):
        """Return the number of moints in a given space.

        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov',
                which denotes all unfrozen moints.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            int: The number of moints.
        """
        # Check the arguments to make sure all is kosher
        if spin not in ('a', 'b', 's'):
            raise ValueError("Invalid 'mo_type' argument.")
        if mo_space not in ('', 'c', 'o', 'v', 'co', 'ov', 'cov'):
            raise ValueError("Invalid 'mo_space' argument.")

        count = 0
        if spin in ('a', 's'):
            if 'c' in mo_space:
                count += self.nc
            if 'o' in mo_space:
                count += self.nao - self.nc
            if 'v' in mo_space:
                count += self.nbf - self.nao
        if spin in ('b', 's'):
            if 'c' in mo_space:
                count += self.nc
            if 'o' in mo_space:
                count += self.nbo - self.nc
            if 'v' in mo_space:
                count += self.nbf - self.nbo
        return count

    def mo_slice(self, mo_space='ov', spin='s'):
        """Return the slice for a given orbital space.

        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov',
                which denotes all unfrozen moints.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            slice: The slice.
        """
        if mo_space is '':
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
                which denotes all unfrozen moints.
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
        d_ao = self.aoints.dipole(spinorb=(spin_sector == 's'))
        return self._transform(d_ao, mo_block, spin_sector, ndim=1)

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
            spinorb=(spin_sector == 's,s'),
            antisymmetrize=(antisymmetrize and s0 == s1))
        return self._transform(g_ao, mo_block, spin_sector)

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
            spinorb=(spin_sector == 's'), electric_field=electric_field)
        return self._transform(h_ao, mo_block, spin_sector)

    def mean_field(self, mo_block='ov,ov', spin_sector='s',
                   src_mo_space='co'):
        """Get the mean field integrals of an orbital space.

        Returns the representation of electronic mean field of a given orbital
        space in the molecular-orbital basis, <p(1)*spin|J(1) - K(1)|q(1)*spin>.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            src_mo_space (str): The MO space generating the field, a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).

        Returns:
            numpy.ndarray: The integrals.
        """
        ac = self.mo_coefficients(mo_space=src_mo_space, spin='a')
        bc = self.mo_coefficients(mo_space=src_mo_space, spin='b')
        if spin_sector == 'a':
            w_ao, _ = self.aoints.mean_field(
                ac=ac, bc=bc, spinorb=False)
        elif spin_sector == 'b':
            _, w_ao = self.aoints.mean_field(
                ac=ac, bc=bc, spinorb=False)
        elif spin_sector == 's':
            w_ao = self.aoints.mean_field(
                ac=ac, bc=bc, spinorb=True)
        else:
            raise ValueError("Invalid 'spin_sector' argument.")

        return self._transform(w_ao, mo_block, spin_sector)

    def fock(self, mo_block='ov,ov', spin_sector='s', src_mo_space='co',
             electric_field=None):
        """Get the Fock integrals of an orbital space.

        Returns the core Hamiltonian plus the mean field of an orbital space in
        the molecular-orbital basis, <p(1)*spin|h(1) + J(1) - K(1)|q(1)*spin>.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            src_mo_space (str): The MO space generating the field, a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.

        Returns:
            numpy.ndarray: The integrals.
        """
        ac = self.mo_coefficients(mo_space=src_mo_space, spin='a')
        bc = self.mo_coefficients(mo_space=src_mo_space, spin='b')
        if spin_sector == 'a':
            f_ao, _ = self.aoints.fock(
                ac=ac, bc=bc, spinorb=False,
                electric_field=electric_field)
        elif spin_sector == 'b':
            _, f_ao = self.aoints.fock(
                ac=ac, bc=bc, spinorb=False,
                electric_field=electric_field)
        elif spin_sector == 's':
            f_ao = self.aoints.fock(
                ac=ac, bc=bc, spinorb=True,
                electric_field=electric_field)
        else:
            raise ValueError("Invalid 'spin_sector' argument.")

        return self._transform(f_ao, mo_block, spin_sector)

    def fock_diagonal(self, mo_space='ov', spin='s', src_mo_space='co',
                      electric_field=None):
        """Get the Fock integral diagonals of an orbital space.

        Args:
            mo_space (str): Any contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).  Defaults to 'ov',
                which denotes all unfrozen moints.
            spin (str): 'a' (alpha), 'b' (beta), or 's' (spin-orbital).
            src_mo_space (str): The MO space generating the field, a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            electric_field (tuple): A three-component vector specifying the
                magnitude of an external static electric field.  Its negative
                dot product with the dipole integrals will be added to the core
                Hamiltonian.

        Returns:
            numpy.ndarray: The Fock diagonal.
        """
        ac = self.mo_coefficients(mo_space=src_mo_space, spin='a')
        bc = self.mo_coefficients(mo_space=src_mo_space, spin='b')
        if spin == 'a':
            f_ao, _ = self.aoints.fock(
                ac=ac, bc=bc, spinorb=False,
                electric_field=electric_field)
        elif spin == 'b':
            _, f_ao = self.aoints.fock(
                ac=ac, bc=bc, spinorb=False,
                electric_field=electric_field)
        elif spin == 's':
            f_ao = self.aoints.fock(
                ac=ac, bc=bc, spinorb=True,
                electric_field=electric_field)
        else:
            raise ValueError("Invalid 'spin' argument.")

        mo_block = ','.join([mo_space, mo_space])
        return self._transform(f_ao, mo_block, spin).diagonal()

    def electronic_energy(self, mo_space='co', electric_field=None):
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
                             src_mo_space=mo_space)
        bw = self.mean_field(mo_block=blk, spin_sector='b',
                             src_mo_space=mo_space)
        return np.trace(ah + aw / 2.) + np.trace(bh + bw / 2.)

    def electronic_dipole_moment(self, mo_space='co'):
        """Get the electronic dipole moment of a given orbital space.

        Args:
            mo_space (str): The MO space of the electrons, 'c' (core),
                'o' (occupied), and 'v' (virtual).

        Returns:
            numpy.ndarray: The dipole moment.
        """
        blk = ','.join([mo_space, mo_space])
        ap = self.dipole(mo_block=blk, spin_sector='a')
        bp = self.dipole(mo_block=blk, spin_sector='b')
        am = np.trace(ap, axis1=1, axis2=2)
        bm = np.trace(bp, axis1=1, axis2=2)
        return am + bm
