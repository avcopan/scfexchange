from .molecule import Molecule
from .util import (abstractmethod, contract, with_metaclass, check_attributes,
                   AttributeContractMeta)


class IntegralsCommonInterface(with_metaclass(AttributeContractMeta, object)):
  """Abstract base class defining a consistent interface for integrals.

  This is perhaps not very Pythonic, but seemed like a convenient way to force
  myself to be consistent.  Due to limitations of abstract base classes,
  initialization is not subject to contract, so here's the desired signature:

  >>> class Integrals(IntegralsCommonInterface): 
  >>> 
  >>>   def __init__(self, molecule, basis_label):
  >>>     '''Initialize Integrals object.
  >>> 
  >>>     Args:
  >>>       molecule (:obj:`scfexchange.molecule.Molecule`)
  >>>       basis_label (str)
  >>>     '''
  >>>     ...
  >>> 
  >>>     self.molecule = molecule
  >>>     self.basis_label = basis_label

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

  def _check_attribute_contract(self):
    """Make sure common attributes are correctly initialized."""
    check_attributes(self, IntegralsCommonInterface._common_attributes)

  @abstractmethod
  @contract(returns = 'array[NxN](float64)')
  def get_ao_1e_overlap_integrals(self):
    """Compute overlap integrals for this molecule and basis set.

    Returns:
      A `self.nbf` x `self.nbf` array of overlap integrals,
      < mu(1) | nu(1) >.
    """
    return

  @abstractmethod
  @contract(returns = 'array[NxN](float64)')
  def get_ao_1e_kinetic_integrals(self):
    """Compute kinetic energy operator for this molecule and basis set.

    Returns:
      A `self.nbf` x `self.nbf` array of kinetic energy operator integrals,
      < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
    """
    return

  @abstractmethod
  @contract(returns = 'array[NxN](float64)')
  def get_ao_1e_potential_integrals(self):
    """Compute nuclear potential operator for this molecule and basis set.

    Returns:
      A `self.nbf` x `self.nbf` array of nuclear potential operator integrals,
      < mu(1) | sum_A Z_A / r_1A | nu(1) >.
    """
    return

  @abstractmethod
  @contract(returns = 'array[NxNxNxN](float64)')
  def get_ao_2e_repulsion_integrals(self):
    """Compute electron-repulsion operator for this molecule and basis set.

    Returns:
      A `self.nbf` x `self.nbf` x `self.nbf` x `self.nbf` array of electron
      repulsion operator integrals,
      < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
    """
    return


