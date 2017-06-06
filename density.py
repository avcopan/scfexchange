import abc
import numpy as np
import tensorutils as tu
from six import with_metaclass
from .orbitals import OrbitalsInterface


class DensityInterface(with_metaclass(abc.ABCMeta)):
    """
    """

    def __init__(self, orbitals):
        """Initialize a DensityInterface instance.

        Args:
            orbitals (scfexchange.OrbitalsInterface): The orbital basis.
        """
        self.orbitals = orbitals
        if not isinstance(orbitals, OrbitalsInterface):
            raise ValueError("Invalid 'orbitals' argument.")

    @abc.abstractmethod
    def solve(self, **options):
        """Solve for parameters needed to compute the density.

        Args:
            **options: Convergence thresholds, etc.
        """
        return

    @abc.abstractmethod
    def get_mo_1e_moment(self, mo_block='ov,ov', spin_sector='s'):
        """Get the one-particle density moment array.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        return

    @abc.abstractmethod
    def get_mo_2e_moment(self, mo_block='ov,ov,ov,ov', spin_sector='s,s'):
        """Get the two-particle density moment array.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        return

    def get_energy(self, electric_field=None):
        """Get the total energy of this electron distribution.

        Args:
            electric_field (numpy.ndarray): A three-component vector specifying
                the magnitude of an external static electric field.  Its
                negative dot product with the dipole integrals will be added to
                the core Hamiltonian.

        Returns:
            float: The energy.
        """
        ah = self.orbitals.get_mo_1e_fock(spin_sector='a', mo_space='c',
                                          electric_field=electric_field)
        bh = self.orbitals.get_mo_1e_fock(spin_sector='b', mo_space='c',
                                          electric_field=electric_field)
        aag = self.orbitals.get_mo_2e_repulsion(spin_sector='a,a',
                                                antisymmetrize=True)
        abg = self.orbitals.get_mo_2e_repulsion(spin_sector='a,b',
                                                antisymmetrize=True)
        bbg = self.orbitals.get_mo_2e_repulsion(spin_sector='b,b',
                                                antisymmetrize=True)
        agamma1 = self.get_mo_1e_moment(spin_sector='a')
        bgamma1 = self.get_mo_1e_moment(spin_sector='b')
        aagamma2 = self.get_mo_2e_moment(spin_sector='a,a')
        abgamma2 = self.get_mo_2e_moment(spin_sector='a,b')
        bbgamma2 = self.get_mo_2e_moment(spin_sector='b,b')
        e_1e = np.sum(ah * agamma1 + bh * bgamma1)
        e_2e = 1. / 4 * np.sum(aag * aagamma2 + 4 * abg * abgamma2 +
                               bbg * bbgamma2)
        e_core = self.orbitals.get_energy(mo_space='c',
                                          electric_field=electric_field)
        return e_1e + e_2e + e_core

    def get_dipole(self):
        raise NotImplementedError


class DeterminantDensity(DensityInterface):

    def solve(self):
        return

    def get_mo_1e_moment(self, mo_block='ov,ov', spin_sector='s'):
        """Get the one-particle density moment array.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        norb = self.orbitals.get_mo_count(mo_space='cov', spin=spin_sector)
        gamma1 = np.zeros((norb, norb))

        # The determinant OPDM is a diagonal projection onto the occupied space.
        o = self.orbitals.get_mo_slice(mo_space='co', spin=spin_sector)
        np.fill_diagonal(gamma1[o, o], 1)

        space_keys, spin_keys = self.orbitals.get_block_keys(mo_block,
                                                             spin_sector)
        slices = tuple(self.orbitals.get_mo_slice(mo_space, spin)
                       for mo_space, spin in zip(space_keys, spin_keys))
        return gamma1[slices]

    def get_mo_2e_moment(self, mo_block='ov,ov,ov,ov', spin_sector='s,s'):
        """Get the two-particle density moment array.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        s0, s1 = spin_sector.split(',')
        gamma1s0 = self.get_mo_1e_moment(mo_block='cov,cov', spin_sector=s0)
        gamma1s1 = self.get_mo_1e_moment(mo_block='cov,cov', spin_sector=s1)
        gamma2 = tu.einsum('pr,qs->pqrs', gamma1s0, gamma1s1)
        if s0 == s1:
            gamma2 = gamma2 - gamma2.transpose((0, 1, 3, 2))

        space_keys, spin_keys = self.orbitals.get_block_keys(mo_block,
                                                             spin_sector)
        slices = tuple(self.orbitals.get_mo_slice(mo_space, spin)
                       for mo_space, spin in zip(space_keys, spin_keys))
        return gamma2[slices]


if __name__ == "__main__":
    import itertools as it
    from .pyscf_interface import Integrals, Orbitals
    from .molecule import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    energies = iter([
        -74.963343795087525, -74.963343795087511, -74.654712456959146,
        -74.656730208992286
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = Orbitals(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        density = DeterminantDensity(orbitals)
        energy = density.get_energy()
        assert (np.isclose(energy, next(energies)))
        print(energy)

