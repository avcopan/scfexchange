import numpy as np
from .integrals import IntegralsCommonInterface
from .util import (abstractmethod, contract, with_metaclass, check_attributes,
                   AttributeContractMeta)

class HartreeFockCommonInterface(with_metaclass(AttributeContractMeta, object)):
  """Abstract base class defining a consistent interface for integrals.

  Due to limitations of the ABC/PyContracts libraries, initialization is not
  subject to contract.  Here's the desired signature:

  >>>  class HatreeFock(HartreeFockCommonInterface): 
  >>>    def __init__(self, integrals, using_spinor_basis = True):
  >>>      '''Initialize HatreeFock object.
  >>> 
  >>>      Args:
  >>>        integrals (:obj:`scfexchange.integrals.Integrals`)
  >>>        using_spinor_basis (bool)
  >>>      '''
  >>>      ...
  >>>   
  >>>      self.integrals = integrals
  >>>      self.using_spinor_basis = using_spinor_basis

  Attributes:
    integrals (:obj:`scfexchange.integrals.Integrals`): Contributions to the
      Hamiltonian operator, in the atomic orbital basis.
    using_spinor_basis (bool): Identifies whether we're using spin or spatial
      molecular orbitals.
    mo_coefficients (np.ndarray): Spinor molecular orbital coefficients.
    mo_alpha_coefficients (np.ndarray): Alpha molecular orbital coefficients.
    mo_beta_coefficients (np.ndarray): Beta molecular orbital coefficients.
  """

  _common_attributes = {
    'integrals': IntegralsCommonInterface,
    'using_spinor_basis': bool
  }

  _spinorb_attributes = {
    'mo_coefficients': np.ndarray
  }

  _spaceorb_attributes = {
    'mo_alpha_coefficients': np.ndarray,
    'mo_beta_coefficients' : np.ndarray
  }

  def _check_attribute_contract(self):
    """Make sure common attributes are correctly initialized."""
    check_attributes(self, HartreeFockCommonInterface._common_attributes)
    if self.using_spinor_basis:
      check_attributes(self, HartreeFockCommonInterface._spinorb_attributes)
    else:
      check_attributes(self, HartreeFockCommonInterface._spaceorb_attributes)

  @abstractmethod
  @contract(spin = 'str', returns = 'array[NxN](float64)')
  def get_mo_1e_core_hamiltonian(spin = ''):
    """Compute core Hamiltonian in the MO basis.

    Args:
      spin (str): The desired MO spin block, either 'a' (alpha) or 'b' (beta).
        Not used for spinor basis integrals (defaults to '').

    Returns:
      < p(1) | - 1 / 2 * nabla_1^2 + sum_A Z_A / r_1A | q(1) >, a square n x n
      array where n is `2 * self.integrals.nbf` or `self.integrals.nbf`,
      depending on whether or not we're using a spinor basis.  Defaults to alpha
      spin in the spin-integrated case.

    Raises:
      ValueError: Complains when the requested spin is not available or not
        consistent with the MO basis (spinor vs. spin-integrated).
    """
    pass

  @abstractmethod
  @contract(spin = 'str', returns = 'array[NxNxNxN](float64)')
  def get_mo_2e_repulsion_integrals(spin = ''):
    """Compute electron-repulsion operator in the MO basis.

    Args:
      spin (str): The desired MO spin block -- 'aaaa', 'abab', or 'bbbb'.  Not
        used for spinor basis integrals (defaults to '').

    Returns:
      < p(1) q(2) | 1 / r_12 | r(1) s(2) >, an n x n x n x n array where n is
      `2 * self.integrals.nbf` or `self.integrals.nbf`, depending on whether or
      not we're using a spinor basis.  Defaults to alpha spin in the spin-
      integrated case.

    Raises:
      ValueError: Complains when the requested spin is not available or not
        consistent with the MO basis (spinor vs. spin-integrated).
    """
    pass



if __name__ == "__main__":
  import numpy as np
  from .molecule import Molecule
  from .pyscf_interface import Integrals

  class HatreeFock(HartreeFockCommonInterface):

    def __init__(self, integrals, using_spinor_basis = True):
      self.integrals = integrals
      self.using_spinor_basis = using_spinor_basis
      self.mo_coefficients = np.zeros((5, 5))

    def get_mo_1e_core_hamiltonian(self, spin = ''):
      return np.zeros((5, 5))

    def get_mo_2e_repulsion_integrals(self, spin = ''):
      return np.zeros((5, 5, 5, 5))
 
  units = "angstrom"
  charge = +1
  multiplicity = 2
  labels = ("O", "H", "H")
  coordinates = np.array([[0.000,  0.000, -0.066],
                          [0.000, -0.759,  0.522],
                          [0.000,  0.759,  0.522]])

  molecule = Molecule(labels, coordinates, units = units, charge = charge,
                      multiplicity = multiplicity)

  integrals = Integrals(molecule, "sto-3g")

  hf = HatreeFock(integrals)
  h = hf.get_mo_1e_core_hamiltonian()
  g = hf.get_mo_2e_repulsion_integrals()
  print(h.shape)
  print(g.shape)
