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
    'mso_coefficients': np.ndarray#,
    #'core_energy': float
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
    norb = int(self.integrals.nbf - nfrz)
    naocc = int(self.integrals.molecule.nalpha - nfrz)
    nbocc = int(self.integrals.molecule.nbeta  - nfrz)
    return nfrz, norb, (naocc, nbocc)

  def _compute_ao_1e_core_density(self):
    ca = self.get_mo_coefficients(mo_type = 'alpha', mo_block = 'c')
    cb = self.get_mo_coefficients(mo_type =  'beta', mo_block = 'c')
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

  def get_mo_slice(self, mo_type = 'spinor', mo_block = 'ov'):
    """Return the slice for a specific block of molecular orbitals.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
      mo_block (str): Any contiguous combination of 'c' (core), 'o' (occupied),
        and 'v' (virtual): 'c', 'o', 'v', 'co', 'ov', or 'cov'.  Defaults to
        'ov', which denotes all unfrozen orbitals.

    Returns:
      A `slice` object with appropriate start and end points for the specified
      MO type and block.
    """
    # Check the arguments to make sure all is kosher
    if not mo_type in ('spinor', 'alpha', 'beta'):
      raise ValueError("Invalid mo_type argument '{:s}'.  Please use 'alpha', "
                       "'beta', or 'spinor'.".format(mo_type))
    if not mo_block in ('c', 'o', 'v', 'co', 'ov', 'cov'):
      raise ValueError("Invalid mo_block argument '{:s}'.  Please use 'c', "
                       "'o', 'v', 'co', 'ov', or 'cov'.".format(mo_block))
    # Assign slice start point
    if mo_block.startswith('c'):
      start = None
    elif mo_type is 'spinor':
      start = 2 * self.nfrz if mo_block.startswith('o') else self.naocc + self.nbocc
    elif mo_type is 'alpha':
      start = self.nfrz if mo_block.startswith('o') else self.naocc
    elif mo_type is 'beta':
      start = self.nfrz if mo_block.startswith('o') else self.nbocc
    # Assign slice end point
    if mo_block.endswith('v'):
      end = None
    elif mo_type is 'spinor':
      end = 2 * self.nfrz if mo_block.endswith('c') else self.naocc + self.nbocc
    elif mo_type is 'alpha':
      end = self.nfrz if mo_block.endswith('c') else self.naocc
    elif mo_type is 'beta':
      end = self.nfrz if mo_block.endswith('c') else self.nbocc

    return slice(start, end)

  def get_mo_energies(self, mo_type = 'spinor', mo_block = 'ov'):
    """Return the molecular orbital energies.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
      mo_block (str): Any contiguous combination of 'c' (core), 'o' (occupied),
        and 'v' (virtual): 'c', 'o', 'v', 'co', 'ov', or 'cov'.  Defaults to
        'ov', which denotes all unfrozen orbitals.

    Returns:
      An array of orbital energies for the given MO type and block.
    """
    slc = self.get_mo_slice(mo_type = mo_type, mo_block = mo_block)
    if mo_type is 'alpha':
      return self.mo_energies[0][slc]
    elif mo_type is 'beta':
      return self.mo_energies[1][slc]
    elif mo_type is 'spinor':
      return self.mso_energies[slc]

  def get_mo_coefficients(self, mo_type = 'spinor', mo_block = 'ov', r_matrix = None):
    """Return the molecular orbital coefficients.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
      mo_block (str): Any contiguous combination of 'c' (core), 'o' (occupied),
        and 'v' (virtual): 'c', 'o', 'v', 'co', 'ov', or 'cov'.  Defaults to
        'ov', which denotes all unfrozen orbitals.
      r_matrix (np.ndarray or tuple): Molecular orbital rotation to be applied
        to the MO coefficients prior to transformation.  Must have the full
        dimension of the spatial or spin-orbital basis, including frozen
        orbitals.  For spatial orbitals, this can be a pair of arrays, one for
        each spin.

    Returns:
      An array of orbital coefficients for the given MO type and block.
    """
    if not isinstance(r_matrix, np.ndarray) or (mo_type in ('alpha', 'beta') and
                                                hasattr(r_matrix, 'len') and
                                                getattr(r_matrix, 'len') is 2):
      raise ValueError("'r_matrix' must either be numpy array or a pair of "
                       "numpy arrays for each spin.")
    slc = self.get_mo_slice(mo_type = mo_type, mo_block = mo_block)
    if mo_type is 'alpha':
      c = self.mo_coefficients[0]
      if not r_matrix is None:
        r = r_matrix if isinstance(r_matrix, np.ndarray) else r_matrix[0]
        c = c.dot(r)
    elif mo_type is 'beta':
      c = self.mo_coefficients[1]
      if not r_matrix is None:
        r = r_matrix if isinstance(r_matrix, np.ndarray) else r_matrix[1]
        c = c.dot(r)
    elif mo_type is 'spinor':
      c = self.mso_coefficients
      if not r_matrix is None:
        r = r_matrix

  def get_mo_1e_core_field(self, mo_type = 'spinor', r_matrix = None):
    """Compute mean-field of the core electrons in the molecular orbital basis.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
      r_matrix (np.ndarray): Molecular orbital rotation to be applied to the
        MO coefficients prior to transformation.

    Returns:
      A nbf x nbf array of kinetic energy operator integrals,
      < mu(1) | j_a + j_b + k_a | nu(1) >.
    """
    c = self.get_mo_coefficients(mo_type = mo_type)
    if mo_type is 'alpha':
      v = self.ao_core_field[0]
    elif mo_type is 'beta':
      v = self.ao_core_field[1]
    elif mo_type is 'spinor':
      v = spla.block_diag(*self.ao_core_field)
    if not r_matrix is None:
      c = c.dot(r_matrix)
    return c.T.dot(v.dot(c))

  def get_mo_1e_kinetic(self, mo_type = 'spinor', r_matrix = None):
    """Compute kinetic energy operator in the molecular orbital basis.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
      r_matrix (np.ndarray): Molecular orbital rotation to be applied to the
        MO coefficients prior to transformation.

    Returns:
      A nbf x nbf array of kinetic energy operator integrals,
      < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
    """
    t = self.integrals.get_ao_1e_kinetic(spinor = (mo_type is 'spinor'))
    c = self.get_mo_coefficients(mo_type = mo_type)
    if not r_matrix is None:
      c = c.dot(r_matrix)
    return c.T.dot(t.dot(c))

  def get_mo_1e_potential(self, mo_type = 'spinor', r_matrix = None):
    """Compute nuclear potential operator in the molecular orbital basis.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', or 'spinor'.
      r_matrix (np.ndarray): Molecular orbital rotation to be applied to the
        MO coefficients prior to transformation.

    Returns:
      A nbf x nbf array of nuclear potential operator integrals,
      < mu(1) | sum_A Z_A / r_1A | nu(1) >.
    """
    v = self.integrals.get_ao_1e_potential(spinor = (mo_type is 'spinor'))
    c = self.get_mo_coefficients(mo_type = mo_type)
    if not r_matrix is None:
      c = c.dot(r_matrix)
    return c.T.dot(v.dot(c))

  def get_mo_2e_repulsion(self, mo_type = 'spinor', r_matrix = None):
    """Compute electron-repulsion operator in the molecular orbital basis.

    Args:
      mo_type (str): Molecular orbital type, 'alpha', 'beta', 'mixed', or
        'spinor'.  The 'mixed' block refers to the mixed alpha/beta block of the
        repulsion integrals, i.e. <a b | a b>.
      r_matrix (tuple or np.ndarray): Molecular orbital rotation to be
        applied to the MO coefficients prior to transformation.  If mo_type is
        'mixed', this should be a pair of rotation matrices for the alpha and
        beta coefficients, respectively.

    Returns:
      A nbf x nbf x nbf x nbf array of electron
      repulsion operator integrals,
      < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
    """
    g = self.integrals.get_ao_2e_repulsion(spinor = (mo_type is 'spinor'))
    if mo_type is 'mixed':
      c1 = self.get_mo_coefficients(mo_type = 'alpha')
      c2 = self.get_mo_coefficients(mo_type = 'beta')
      if not r_matrix is None:
        u1, u2 = r_matrix
        c1 = c1.dot(u1)
        c2 = c2.dot(u2)
    else:
      c1 = c2 = self.get_mo_coefficients(mo_type = mo_type)
      if not r_matrix is None:
        c1 = c2 = c1.dot(r_matrix)
    ctr = lambda a, b: np.tensordot(a, b, axes = (0, 0))
    return ctr(ctr(ctr(ctr(g, c1), c2), c1), c2)

