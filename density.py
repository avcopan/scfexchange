import abc
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
