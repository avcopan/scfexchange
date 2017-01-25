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
      Hamiltonian operator, in the molecular orbital basis.
    options (dict): A dictionary of options, by keyword argument.
    nfrz (int): The number of frozen (spatial) orbitals.  This can be set with
      the option 'n_frozen_orbitals'.  Alternatively, if 'freeze_core' is True
      and the number of frozen orbitals is not set, this defaults to the number
      of core orbitals, as determined by the molecule object.
    norb (int): The total number of non-frozen (spatial) orbitals.  That is, the
      number of basis functions minus the number of frozen orbitals.
    naocc (int): The number of occupied non-frozen alpha orbitals.
    nbocc (int): The number of occupied non-frozen beta orbitals.
    mo_coefficients (np.ndarray): Molecular orbital coefficients, given as a
      2 x nbf x nbf array of alpha and beta spatial MOs.
    mso_coeffieicnts (np.ndarray): Molecular spin-orbital coefficients, given as
      a (2*nbf) x (2*nbf) array of spinor coefficients, in which the columns are
      sorted by orbital energy.
    mo_energies (np.ndarray): Molecular orbital energies, given as a 2 x nbf
      array.
    mso_energies (np.ndarray): Molecular spin-orbital energies, given as an
      array of length 2*nbf which is sorted in increasing order.
    core_mo_energies (np.ndarray): Energies of the frozen core orbitals.
    core_mo_coefficients (np.ndarray): Coefficients of the frozen core orbitals.
    core_energy (float): Hartree-Fock energy of the frozen core, including
      nuclear repulsion energy.
  """

  _attribute_types = {
    'integrals': IntegralsInterface,
    'options': dict,
    'nfrz': int,
    'norb': int,
    'naocc': int,
    'nbocc': int,
    'mo_energies': np.ndarray,
    'mo_coefficients': np.ndarray,
    'mso_energies': np.ndarray,
    'mso_coefficients': np.ndarray,
    'core_mo_energies': np.ndarray,
    'core_mo_coefficients': np.ndarray,
    'core_energy': float
  }

  _option_defaults = {
    'restrict_spin': True,
    'n_iterations': 40,
    'e_threshold': 1e-12,
    'd_threshold': 1e-6,
    'freeze_core': False,
    'n_frozen_orbitals': 0
  }

  def _check_attribute_contract(self):
    """Make sure common attributes are correctly initialized."""
    check_attributes(self, OrbitalsInterface._attribute_types)

  def _process_options(self, options):
    return process_options(options, OrbitalsInterface._option_defaults)

  def _count_orbitals(self):
    """Count the number of frozen, unfrozen, and occupied orbitals.

    Returns:
      int, int, (int, int) corresponding to the number of frozen, unfrozen, and
        occupied (alpha, beta) orbitals.
    """
    # First, determine the number of frozen core orbitals
    nfrz = (0 if not self.options['freeze_core'] else
            self.integrals.molecule.ncore)
    if self.options['n_frozen_orbitals'] != 0:
      nfrz = self.options['n_frozen_orbitals']
    if nfrz is None:
      raise Exception("Could not determine the number of frozen orbitals.  "
                      "Please set this value using 'n_frozen_orbitals'.")
    # Now, determine the rest
    norb = self.integrals.nbf - nfrz
    naocc = self.integrals.molecule.nalpha - nfrz
    nbocc = self.integrals.molecule.nbeta  - nfrz
    return nfrz, norb, (naocc, nbocc)

  def _compute_ao_1e_core_density(self):
    ca, cb = self.core_mo_coefficients
    da = ca.dot(ca.T)
    db = cb.dot(cb.T)
    return da, db

  def _compute_ao_1e_core_field(self):
    da, db = self._compute_ao_1e_core_density()
    h = (  self.integrals.get_ao_1e_kinetic(spinor = False)
         + self.integrals.get_ao_1e_potential(spinor = False))
    g = self.integrals.get_ao_2e_repulsion(spinor = False)
    j  = np.tensordot(g, da + db, axes = [(1, 3), (1, 0)])
    va = j - np.tensordot(g, da, axes = [(1, 2), (1, 0)])
    vb = j - np.tensordot(g, db, axes = [(1, 2), (1, 0)])
    return va, vb

  def _compute_core_energy(self):
    da, db = self._compute_ao_1e_core_density()
    va, vb = self.ao_core_field
    h = (  self.integrals.get_ao_1e_kinetic(spinor = False)
         + self.integrals.get_ao_1e_potential(spinor = False))
    core_energy = np.sum((h + va/2) * da + (h + vb/2) * db)
    return core_energy + self.integrals.molecule.nuclear_repulsion_energy

  def get_mo_energies(self, mo_type = 'spinor'):
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

  def get_mo_coefficients(self, mo_type = 'spinor'):
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

  def get_mo_1e_core_field(self, mo_block = 'spinor', mo_rotation = None):
    """Compute mean-field of the core electrons in the molecular orbital basis.

    Args:
      mo_block (str): Molecular orbital block, 'alpha', 'beta', or 'spinor'.
      mo_rotation (np.ndarray): Molecular orbital rotation to be applied to the
        MO coefficients prior to transformation.

    Returns:
      A nbf x nbf array of kinetic energy operator integrals,
      < mu(1) | j_a + j_b + k_a | nu(1) >.
    """
    c = self.get_mo_coefficients(mo_type = mo_block)
    if mo_block is 'alpha':
      v = self.ao_core_field[0]
    elif mo_block is 'beta':
      v = self.ao_core_field[1]
    elif mo_block is 'spinor':
      v = spla.block_diag(*self.ao_core_field)
    if not mo_rotation is None:
      c = c.dot(mo_rotation)
    return c.T.dot(v.dot(c))

  def get_mo_1e_kinetic(self, mo_block = 'spinor', mo_rotation = None):
    """Compute kinetic energy operator in the molecular orbital basis.

    Args:
      mo_block (str): Molecular orbital block, 'alpha', 'beta', or 'spinor'.
      mo_rotation (np.ndarray): Molecular orbital rotation to be applied to the
        MO coefficients prior to transformation.

    Returns:
      A nbf x nbf array of kinetic energy operator integrals,
      < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
    """
    t = self.integrals.get_ao_1e_kinetic(spinor = (mo_block is 'spinor'))
    c = self.get_mo_coefficients(mo_type = mo_block)
    if not mo_rotation is None:
      c = c.dot(mo_rotation)
    return c.T.dot(t.dot(c))

  def get_mo_1e_potential(self, mo_block = 'spinor', mo_rotation = None):
    """Compute nuclear potential operator in the molecular orbital basis.

    Args:
      mo_block (str): Molecular orbital block, 'alpha', 'beta', or 'spinor'.
      mo_rotation (np.ndarray): Molecular orbital rotation to be applied to the
        MO coefficients prior to transformation.

    Returns:
      A nbf x nbf array of nuclear potential operator integrals,
      < mu(1) | sum_A Z_A / r_1A | nu(1) >.
    """
    v = self.integrals.get_ao_1e_potential(spinor = (mo_block is 'spinor'))
    c = self.get_mo_coefficients(mo_type = mo_block)
    if not mo_rotation is None:
      c = c.dot(mo_rotation)
    return c.T.dot(v.dot(c))

  def get_mo_2e_repulsion(self, mo_block = 'spinor', mo_rotation = None):
    """Compute electron-repulsion operator in the molecular orbital basis.

    Args:
      mo_block (str): Molecular orbital block, 'alpha', 'beta', 'mixed', or
        'spinor'.  The 'mixed' block refers to the mixed alpha/beta block of the
        repulsion integrals, i.e. <a b | a b>.
      mo_rotation (tuple or np.ndarray): Molecular orbital rotation to be
        applied to the MO coefficients prior to transformation.  If mo_block is
        'mixed', this should be a pair of rotation matrices for the alpha and
        beta coefficients, respectively.

    Returns:
      A nbf x nbf x nbf x nbf array of electron
      repulsion operator integrals,
      < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
    """
    g = self.integrals.get_ao_2e_repulsion(spinor = (mo_block is 'spinor'))
    if mo_block is 'mixed':
      c1 = self.get_mo_coefficients(mo_type = 'alpha')
      c2 = self.get_mo_coefficients(mo_type = 'beta')
      if not mo_rotation is None:
        u1, u2 = mo_rotation
        c1 = c1.dot(u1)
        c2 = c2.dot(u2)
    else:
      c1 = c2 = self.get_mo_coefficients(mo_type = mo_block)
      if not mo_rotation is None:
        c1 = c2 = c1.dot(mo_rotation)
    ctr = lambda a, b: np.tensordot(a, b, axes = (0, 0))
    return ctr(ctr(ctr(ctr(g, c1), c2), c1), c2)
