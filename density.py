import abc
import numpy as np
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
    def get_mo_1e_moment(self, mo_block='ov,ov', spin_block=None):
        """Get the one-particle density moment array.

        Args:
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            spin_block (str): A comma-separated pair of spins, 'a' (alpha) or
                'b' (beta).  Otherwise, None (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        return

    @abc.abstractmethod
    def get_mo_2e_moment(self, mo_block='ov,ov,ov,ov', spin_block=None):
        """Get the two-particle density moment array.

        Args:
            mo_block (str): A comma-separated list of four MO spaces.  Each MO
                space is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            spin_block (str): A comma-separated list of four spins, 'a' (alpha)
                or 'b' (beta).  Otherwise, None (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        return

    def get_energy(self):
        raise NotImplementedError

    def get_dipole(self):
        raise NotImplementedError


class DeterminantDensity(DensityInterface):

    def solve(self):
        return

    def get_mo_1e_moment(self, mo_block='ov,ov', spin_block=None):
        """Get the one-particle density moment array.

        Args:
            mo_block (str): A comma-separated pair of MO spaces.  Each MO space
                is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            spin_block (str): A comma-separated pair of spins, 'a' (alpha) or
                'b' (beta).  Otherwise, None (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        mo_spaces, spins = self.orbitals.get_block_args(mo_block, spin_block)
        norb = self.orbitals.get_mo_count(mo_space='cov', spin=spins[0])
        gamma1 = np.zeros((norb, norb))

        # The determinant OPDM is a diagonal projection onto the occupied space.
        o = self.orbitals.get_mo_slice(mo_space='co', spin=spins[0])
        np.fill_diagonal(gamma1[o, o], 1)

        slices = tuple(self.orbitals.get_mo_slice(mo_space, spin)
                       for mo_space, spin in zip(mo_spaces, spins))
        return gamma1[slices]

    def get_mo_2e_moment(self, mo_block='ov,ov,ov,ov', spin_block=None):
        """Get the two-particle density moment array.

        Args:
            mo_block (str): A comma-separated list of four MO spaces.  Each MO
                space is specified as a contiguous combination of 'c' (core),
                'o' (occupied), and 'v' (virtual).
            spin_block (str): A comma-separated list of four spins, 'a' (alpha)
                or 'b' (beta).  Otherwise, None (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        mo_spaces, spins = self.orbitals.get_block_args(mo_block, spin_block)
        gamma1s0 = self.get_determinant_density(mo_block='cov,cov', spin=spins[0])
        gamma1s1 = self.get_
        return


if __name__ == "__main__":
    from .pyscf_interface import Integrals, Orbitals
    from .molecule import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    orbitals = Orbitals(integrals, charge=1, multiplicity=2)
    orbitals.solve()
    density = DeterminantDensity(orbitals)
    gamma1 = density.get_mo_1e_moment()
    print(gamma1)

