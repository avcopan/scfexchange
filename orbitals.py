import numpy as np
import scipy.linalg as spla
from .integrals import IntegralsInterface
from .util import (abstractmethod, contract, with_metaclass, check_attributes,
                   process_options, AttributeContractMeta)

class OrbitalsInterface(with_metaclass(AttributeContractMeta, object)):
  """Abstract base class defining a consistent interface for molecular orbitals.

  Due to limitations of the ABC/PyContracts libraries, initialization is not
  subject to contract.  Here's the desired signature:

  >>>  class Orbitals(OrbitalsInterface): 
  >>>    def __init__(self, integrals, **options)
  >>>      self.integrals = integrals
  >>>      self.options = options
  >>>      ...

  Attributes:
    integrals (:obj:`scfexchange.integrals.Integrals`): Contributions to the
      Hamiltonian operator, in the atomic orbital basis.
    options (dict): A dictionary of options, by keyword argument.
    mo_coefficients (np.ndarray): Molecular orbital coefficients, given as a
      2 x nbf x nbf array of alpha and beta spatial MOs.
    mso_coeffieicnts (np.ndarray): Molecular spin-orbital coefficients, given as
      a (2*nbf) x (2*nbf) array of spinor coefficients, in which the columns are
      sorted by orbital energy.
    mo_energies (np.ndarray): Molecular orbital energies, given as a 2 x nbf
      array.
    mso_energies (np.ndarray): Molecular spin-orbital energies, given as an
      array of length 2*nbf which is sorted in increasing order.
  """

  _attributes = {
    'integrals': IntegralsInterface,
    'options': dict,
    'mo_energies': np.ndarray,
    'mo_coefficients': np.ndarray,
    'mso_energies': np.ndarray,
    'mso_coefficients': np.ndarray
  }

  _options = {
    'restrict_spin': True,
    'n_iterations': 20,
    'e_threshold': 1e-7
  }

  def _check_attribute_contract(self):
    """Make sure common attributes are correctly initialized."""
    check_attributes(self, OrbitalsInterface._attributes)

  def _process_options(self, options):
    return process_options(options, OrbitalsInterface._options)

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

    def __init__(self, integrals, **options):
      self.integrals = integrals
      self.options = self._process_options(options)
      self.mo_energies = np.zeros((5,))
      self.mo_coefficients = np.zeros((5, 5))
      self.mso_energies = np.zeros((10,))
      self.mso_coefficients = np.zeros((10, 10))

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

  orbitals = Orbitals(integrals)
  h = orbitals.get_mo_1e_core_hamiltonian()
  g = orbitals.get_mo_2e_repulsion()
  print(h.shape)
  print(g.shape)
  print(orbitals.options)
