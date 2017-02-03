import numpy as np
import scipy.linalg as spla
from .molecule import Molecule
from .util import (abstractmethod, contract, with_metaclass, check_attributes,
                   AttributeContractMeta, compute_if_unknown)


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

  def _compute_ao_1e(self, name, compute_ints, integrate_spin = True,
                     save = False):
    ao_name  = "ao_1e_{:s}".format(name)
    aso_name = "aso_1e_{:s}".format(name)
    ints = compute_if_unknown(self, ao_name, compute_ints, save)
    def convert_to_aso():
      return IntegralsInterface.convert_1e_ao_to_aso(ints)
    if not integrate_spin:
      ints = compute_if_unknown(self, aso_name, convert_to_aso, save)
    return ints

  def _compute_ao_2e(self, name, compute_ints, integrate_spin = True,
                     save = False, antisymmetrize = False):
    ao_name  = "ao_2e_chem_{:s}".format(name)
    aso_name = "aso_2e_chem_{:s}".format(name)
    chem_ints = compute_if_unknown(self, ao_name, compute_ints, save)
    def convert_to_aso():
      return IntegralsInterface.convert_2e_ao_to_aso(chem_ints)
    if not integrate_spin:
      chem_ints = compute_if_unknown(self, aso_name, convert_to_aso, save)
    ints = chem_ints.transpose((0, 2, 1, 3))
    if antisymmetrize:
      ints = ints - ints.transpose((0, 1, 3, 2))
    return ints

  @abstractmethod
  @contract(integrate_spin = 'bool', save = 'bool',
            returns = 'array[NxN](float64)')
  def get_ao_1e_overlap(self, integrate_spin = True, save = False):
    """Compute overlap integrals for the atomic orbital basis.

    Args:
      integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
      save (bool): Save the computed array for later use?

    Returns:
      A nbf x nbf array of overlap integrals,
      < mu(1) | nu(1) >.
    """
    return

  @abstractmethod
  @contract(integrate_spin = 'bool', save = 'bool',
            returns = 'array[NxN](float64)')
  def get_ao_1e_potential(self, integrate_spin = True, save = False):
    """Compute nuclear potential operator in the atomic orbital basis.

    Args:
      integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
      save (bool): Save the computed array for later use?

    Returns:
      A nbf x nbf array of nuclear potential operator integrals,
      < mu(1) | sum_A Z_A / r_1A | nu(1) >.
    """
    return

  @abstractmethod
  @contract(integrate_spin = 'bool', save = 'bool',
            returns = 'array[NxN](float64)')
  def get_ao_1e_kinetic(self, integrate_spin = True, save = False):
    """Compute kinetic energy operator in the atomic orbital basis.

    Args:
      integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
      save (bool): Save the computed array for later use?

    Returns:
      A nbf x nbf array of kinetic energy operator integrals,
      < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
    """
    return

  @abstractmethod
  @contract(integrate_spin = 'bool', save = 'bool', antisymmetrize = 'bool',
            returns = 'array[NxNxNxN](float64)')
  def get_ao_2e_repulsion(self, integrate_spin = False, save = False,
                          antisymmetrize = False):
    """Compute electron-repulsion operator in the atomic orbital basis.

    Args:
      integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
      save (bool): Save the computed array for later use?
      antisymmetrize (bool): Antisymmetrize the repulsion integrals?

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
  def convert_2e_ao_to_aso(ao_2e_chem_operator):
    """Convert AO basis two-electron operator to the atomic spin-orbital basis.

    Returns:
      An np.ndarray with shape (2*nbf, 2*nbf, 2*nbf, 2*nbf) where nbf is
      `self.nbf`.
    """
    aso_2e_chem_operator = np.kron(np.identity(2),
                                   np.kron(np.identity(2),
                                           ao_2e_chem_operator).T)
    return aso_2e_chem_operator
