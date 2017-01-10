import numpy as np
import scipy.linalg as spla
from .molecule import Molecule
from .util import (abstractmethod, contract, with_metaclass, check_attributes,
                   AttributeContractMeta)


class IntegralsInterface(with_metaclass(AttributeContractMeta, object)):
  """Abstract base class defining a consistent interface for integrals.

  Due to limitations of the ABC/PyContracts libraries, initialization is not
  subject to contract.  Here's the desired signature:

  >>> class Integrals(IntegralsInterface): 
  >>> 
  >>>   def __init__(self, molecule, basis_label):
  >>>     self.molecule = molecule
  >>>     self.basis_label = basis_label
  >>>     ...

  Attributes:
    basis_label (str): The basis set label (e.g. 'sto-3g').
    molecule: Together with `self.basis_label`, this specifies the atomic
      orbitals entereing the integral computation.
    nbf (int): The number of basis functions.
  """

  _attribute_types = {
    'basis_label': str,
    'molecule': Molecule,
    'nbf': int
  }

  def _check_attribute_contract(self):
    """Make sure common attributes are correctly initialized."""
    check_attributes(self, IntegralsInterface._attribute_types)

  @abstractmethod
  @contract(spinor = 'bool', returns = 'array[NxN](float64)')
  def get_ao_1e_overlap(self, spinor = False):
    """Compute overlap integrals for the atomic orbital basis.

    Args:
      spinor (bool): Convert to atomic spin-orbital basis?

    Returns:
      A nbf x nbf array of overlap integrals,
      < mu(1) | nu(1) >.
    """
    return

  @abstractmethod
  @contract(spinor = 'bool', returns = 'array[NxN](float64)')
  def get_ao_1e_kinetic(self, spinor = False):
    """Compute kinetic energy operator in the atomic orbital basis.

    Args:
      spinor (bool): Convert to atomic spin-orbital basis?

    Returns:
      A nbf x nbf array of kinetic energy operator integrals,
      < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
    """
    return

  @abstractmethod
  @contract(spinor = 'bool', returns = 'array[NxN](float64)')
  def get_ao_1e_potential(self, spinor = False):
    """Compute nuclear potential operator in the atomic orbital basis.

    Args:
      spinor (bool): Convert to atomic spin-orbital basis?

    Returns:
      A nbf x nbf array of nuclear potential operator integrals,
      < mu(1) | sum_A Z_A / r_1A | nu(1) >.
    """
    return

  @abstractmethod
  @contract(spinor = 'bool', returns = 'array[NxNxNxN](float64)')
  def get_ao_2e_repulsion(self, spinor = False):
    """Compute electron-repulsion operator in the atomic orbital basis.

    Args:
      spinor (bool): Convert to atomic spin-orbital basis?

    Returns:
      A nbf x nbf x nbf x nbf array of electron
      repulsion operator integrals,
      < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
    """
    return

  @staticmethod
  def convert_1e_ao_to_aso(ao_1e_operator):
    """Convert AO basis one-electron operator to the atomic spin-orbital basis.

    Returns:
      An np.ndarray with shape (2*nbf, 2*nbf) where nbf is `self.nbf`.
    """
    return spla.block_diag(ao_1e_operator, ao_1e_operator)

  @staticmethod
  def convert_2e_ao_to_aso(ao_2e_operator):
    """Convert AO basis two-electron operator to the atomic spin-orbital basis.

    Returns:
      An np.ndarray with shape (2*nbf, 2*nbf, 2*nbf, 2*nbf) where nbf is
      `self.nbf`.
    """
    ao_2e_chem_operator = ao_2e_operator.transpose((0, 2, 1, 3))
    aso_2e_chem_operator = np.kron(np.identity(2),
                                   np.kron(np.identity(2),
                                           ao_2e_chem_operator).T)
    return aso_2e_chem_operator.transpose((0, 2, 1, 3))
