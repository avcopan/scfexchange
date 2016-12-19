from .molecule import Molecule
from .util import (abstractmethod, contract, with_metaclass,
                   check_common_attributes, AttributeContractMeta)


class IntegralsCommonInterface(with_metaclass(AttributeContractMeta, object)):
  """Abstract base class defining a consistent interface for integrals.

  Not sure if this is good OO design, but it made sense to me.

  Attributes:
    basis_label (str): The basis set label (e.g. 'sto-3g').
    molecule: Together with `self.basis_label`, this specifies the atomic
      orbitals entereing the integral computation.
    nbf (int): The number of basis functions.
  """
  _common_attributes = {
    'basis_label': str,
    'molecule': Molecule,
    'nbf': int
  }

  def _check_common_attributes(self):
    """Make sure the common attributes of IntegralsBase have been defined."""
    check_common_attributes(self, IntegralsCommonInterface._common_attributes)

  @abstractmethod
  @contract(returns='array[NxN](float64)')
  def get_ao_1e_overlap_integrals(self):
    """
    Compute overlap integrals for this molecule and basis set.

    < mu(1) | nu(1) >
    """
    return

  @abstractmethod
  @contract(returns='array[NxN](float64)')
  def get_ao_1e_potential_integrals(self):
    """
    Compute nuclear potential operator for this molecule and basis set.

    < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >
    """
    return

  @abstractmethod
  @contract(returns='array[NxN](float64)')
  def get_ao_1e_kinetic_integrals(self):
    """
    Compute kinetic energy operator for this molecule and basis set.

    < mu(1) | sum_A Z_A / r_1A | nu(1) >
    """
    return

  @abstractmethod
  @contract(returns='array[NxNxNxN](float64)')
  def get_ao_2e_repulsion_integrals(self):
    """
    Compute electron-repulsion operator for this molecule and basis set.

    Returns AO basis integrals as 4d array in physicist's notation, i.e.
    < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >
    """
    return


