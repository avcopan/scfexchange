import numpy as np
import scipy.linalg as spla
from .integrals import IntegralsInterface
from .util import (abstractmethod, contract, with_metaclass, check_attributes,
                   AttributeContractMeta)

class OrbitalsInterface(with_metaclass(AttributeContractMeta, object)):
  """Abstract base class defining a consistent interface for molecular orbitals.

  Due to limitations of the ABC/PyContracts libraries, initialization is not
  subject to contract.  Here's the desired signature:

  >>>  class Orbitals(OrbitalsInterface): 
  >>>    def __init__(self, integrals, using_restricted_orbitals = False)
  >>>      self.integrals = integrals
  >>>      self.using_restricted_orbitals = using_restricted_orbitals
  >>>      ...

  Attributes:
    integrals (:obj:`scfexchange.integrals.Integrals`): Contributions to the
      Hamiltonian operator, in the atomic orbital basis.
    using_restricted_orbitals (bool): Identifies whether we're using spin-
      restricted molecular orbitals.
    mo_coefficients (np.ndarray): Molecular orbital coefficients, either as
      a 2*nbf x 2*nbf array of spin-MOs or a 2 x nbf x nbf array of alpha and
      beta spatial MOs.
  """

  _attributes = {
    'integrals': IntegralsInterface,
    'using_restricted_orbitals': bool,
    'mo_energies': np.ndarray,
    'mo_coefficients': np.ndarray,
    'mso_energies': np.ndarray,
    'mso_coefficients': np.ndarray
  }

  def _check_attribute_contract(self):
    """Make sure common attributes are correctly initialized."""
    check_attributes(self, OrbitalsInterface._attributes)

  def _get_mso_energies_and_coefficients(self):
    mso_energies = np.concatenate(self.mo_energies)
    mso_coefficients = spla.block_diag(*self.mo_coefficients)
    sorting_indices = mso_energies.argsort()
    return mso_energies[sorting_indices], mso_coefficients[:, sorting_indices]

  def get_mo_energies(self, mo_type = 'alpha'):
    """Return the molecular orbital energies.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
    """
    if mo_type is 'alpha':
      return self.mo_energies[0]
    elif mo_type is 'beta':
      return self.mo_energies[1]
    elif mo_type is 'spinor':
      return self.mso_energies
    else:
      raise ValueError("Invalid mo_type argument '{:s}'.  Please use 'alpha', "
                       "'beta', or 'spinor'.".format(mo_type))

  def get_mo_coefficients(self, mo_type = 'alpha'):
    """Return the molecular orbital coefficients.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
    """
    if mo_type is 'alpha':
      return self.mo_coefficients[0]
    elif mo_type is 'beta':
      return self.mo_coefficeints[1]
    elif mo_type is 'spinor':
      return self.mso_coefficients
    else:
      raise ValueError("Invalid mo_type argument '{:s}'.  Please use 'alpha', "
                       "'beta', or 'spinor'.".format(mo_type))

  def get_mo_1e_kinetic(self, mo_type = 'alpha'):
    """Compute kinetic energy operator in the molecular orbital basis.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.

    Returns:
      A nbf x nbf array of kinetic energy operator integrals,
      < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
    """
    t = self.integrals.get_ao_1e_kinetic(spinor = (mo_type is 'spinor'))
    c = self.get_mo_coefficients(mo_type = mo_type)
    return c.T.dot(t.dot(c))

  def get_mo_1e_potential(self, mo_type = 'alpha'):
    """Compute nuclear potential operator in the atomic orbital basis.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.

    Returns:
      A nbf x nbf array of nuclear potential operator integrals,
      < mu(1) | sum_A Z_A / r_1A | nu(1) >.
    """
    v = self.integrals.get_ao_1e_potential(spinor = (mo_type is 'spinor'))
    c = self.get_mo_coefficients(mo_type = mo_type)
    return c.T.dot(v.dot(c))

  def get_mo_2e_repulsion(self, mo_type = 'alpha'):
    """Compute electron-repulsion operator in the atomic orbital basis.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.

    Returns:
      A nbf x nbf x nbf x nbf array of electron
      repulsion operator integrals,
      < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
    """
    g = self.integrals.get_ao_2e_repulsion(spinor = (mo_type is 'spinor'))
    c = self.get_mo_coefficients(mo_type = mo_type)
    ctr = lambda a, b: np.tensordot(a, b, axes = (0, 0))
    return ctr(ctr(ctr(ctr(g, c), c), c), c)


if __name__ == "__main__":
  import numpy as np
  from .molecule import Molecule
  from .pyscf_interface import Integrals

  class Orbitals(OrbitalsInterface):

    def __init__(self, integrals):
      self.integrals = integrals
      self.mo_coefficients = np.zeros((5, 5))

    def get_mo_1e_core_hamiltonian(self, spin = ''):
      return np.zeros((5, 5))

    def get_mo_2e_repulsion(self, spin = ''):
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

  mo = Orbitals(integrals)
  h = mo.get_mo_1e_core_hamiltonian()
  g = mo.get_mo_2e_repulsion()
  print(h.shape)
  print(g.shape)
