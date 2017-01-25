import pyscf
import numpy as np
import scipy.linalg as spla
from .integrals import IntegralsInterface
from .orbitals  import OrbitalsInterface
from .util import with_doc

class Integrals(IntegralsInterface):
  __doc__ = """**IntegralsInterface.__doc__**

{:s}

**Integrals.__doc__**

Interface to PySCF integrals.

  Attributes:
    _pyscf_molecule (:obj:`pyscf.gto.Mole`): Used to access PySCF integrals.

  """.format(IntegralsInterface.__doc__)

  def __init__(self, molecule, basis_label):
    """Initialize Integrals object (PySCF interface).

    Args:
      molecule (:obj:`scfexchange.molecule.Molecule`): The molecule.
      basis_label (str): What basis set to use.
    """
    self._pyscf_molecule = pyscf.gto.Mole(atom = list(iter(molecule)),
                                          unit = molecule.units,
                                          basis = basis_label,
                                          charge = molecule.charge,
                                          spin = molecule.multiplicity - 1)
    self._pyscf_molecule.build()

    self.molecule = molecule
    self.basis_label = basis_label
    self.nbf = int(self._pyscf_molecule.nao_nr())

  @with_doc(IntegralsInterface.get_ao_1e_overlap.__doc__)
  def get_ao_1e_overlap(self, spinor = False):
    ao_1e_overlap = self._pyscf_molecule.intor('cint1e_ovlp_sph')
    return (ao_1e_overlap if not spinor
            else IntegralsInterface.convert_1e_ao_to_aso(ao_1e_overlap))

  @with_doc(IntegralsInterface.get_ao_1e_potential.__doc__)
  def get_ao_1e_potential(self, spinor = False):
    ao_1e_potential = self._pyscf_molecule.intor('cint1e_nuc_sph')
    return (ao_1e_potential if not spinor
            else IntegralsInterface.convert_1e_ao_to_aso(ao_1e_potential))

  @with_doc(IntegralsInterface.get_ao_1e_kinetic.__doc__)
  def get_ao_1e_kinetic(self, spinor = False):
    ao_1e_kinetic = self._pyscf_molecule.intor('cint1e_kin_sph')
    return (ao_1e_kinetic if not spinor
            else IntegralsInterface.convert_1e_ao_to_aso(ao_1e_kinetic))

  @with_doc(IntegralsInterface.get_ao_2e_repulsion.__doc__)
  def get_ao_2e_repulsion(self, spinor = False):
    # PySCF returns these as a nbf*nbf x nbf*nbf matrix, so reshape and
    # transpose from chemist's to physicist's notation.
    ao_2e_chem_repulsion = (self._pyscf_molecule.intor('cint2e_sph')
                            .reshape((self.nbf, self.nbf, self.nbf, self.nbf)))
    ao_2e_repulsion = ao_2e_chem_repulsion.transpose((0, 2, 1, 3))
    return (ao_2e_repulsion if not spinor
            else IntegralsInterface.convert_2e_ao_to_aso(ao_2e_repulsion))



class Orbitals(OrbitalsInterface): 
  __doc__ = """**OrbitalsInterface.__doc__**

{:s}

**Orbitals.__doc__**

Interface for accessing PySCF molecular orbitals.

  Attributes:
    _pyscf_hf (:obj:`pyscf.scf.SCF`): Used to access PySCF orbitals.

  """.format(OrbitalsInterface.__doc__)

  def __init__(self, integrals, **options):
    """Initialize Orbitals object (PySCF interface).

    Args:
      integrals (:obj:`scfexchange.pyscf_interface.Integrals`): AO integrals.
      restrict_spin (:obj:`bool`, optional): Use spin-restricted orbitals?
      n_iterations (:obj:`int`, optional): Maximum number of iterations allowed
        before considering the orbitals unconverged.
      e_threshold (:obj:`float`, optional): Energy convergence threshold.
      d_threshold (:obj:`float`, optional): Density convergence threshold, based
        on the norm of the orbital gradient.
      freeze_core (:obj:`bool`, optional): Whether or not to cut core orbitals
        out of the MO coefficients.
      n_frozen_orbitals (:obj:`int`, optional): How many core orbitals to cut
        out from the MO coefficients.
    """

    if not isinstance(integrals, Integrals):
      raise ValueError("Please use an integrals object from this interface.")
    self.integrals = integrals
    self.options = self._process_options(options)
    # Determine the orbital counts (total, frozen, and occupied)
    self.nfrz, self.norb, (self.naocc, self.nbocc) = self._count_orbitals()
    # Build PySCF HF object and compute the energy.
    if self.options['restrict_spin']:
      self._pyscf_hf = pyscf.scf.RHF(integrals._pyscf_molecule)
    else:
      self._pyscf_hf = pyscf.scf.UHF(integrals._pyscf_molecule)
    self._pyscf_hf.conv_tol = self.options['e_threshold']
    self._pyscf_hf.conv_tol_grad = self.options['d_threshold']
    self._pyscf_hf.max_cycle = self.options['n_iterations']
    self._pyscf_hf.kernel()
    self.mo_energies = self._pyscf_hf.mo_energy
    self.mo_coefficients = self._pyscf_hf.mo_coeff
    if self.options['restrict_spin']:
      self.mo_energies = np.array([mo_energies] * 2)
      self.mo_coefficients = np.array([mo_coefficients] * 2)
    # Build spin-orbital energy and coefficient arrays, sorted by orbital energy
    mso_energies = np.concatenate(self.mo_energies)
    mso_coefficients = spla.block_diag(*self.mo_coefficients)
    sorting_indices = mso_energies.argsort()
    self.mso_energies = mso_energies[sorting_indices]
    self.mso_coefficients = mso_coefficients[:, sorting_indices]
    # Get the core field and energy
    #self.ao_core_field = self._compute_ao_1e_core_field()
    #self.core_energy = self._compute_core_energy()

if __name__ == "__main__":
  import numpy as np
  from . import Molecule

  units = "angstrom"
  charge = 1
  multiplicity = 2
  labels = ("O", "H", "H")
  coordinates = np.array([[0.000,  0.000, -0.066],
                          [0.000, -0.759,  0.522],
                          [0.000,  0.759,  0.522]])

  mol = Molecule(labels, coordinates, units = units, charge = charge,
                 multiplicity = multiplicity)
  integrals = Integrals(mol, "cc-pvdz")
  orbital_options = {
    'freeze_core': False,
    'n_frozen_orbitals': 1,
    'e_threshold': 1e-14,
    'n_iterations': 50,
    'restrict_spin': False
  }
  orbitals = Orbitals(integrals, **orbital_options)
  #core_energy = orbitals.core_energy
  nocc = orbitals.naocc + orbitals.nbocc
  o = slice(None, nocc)
  h = (orbitals.get_mo_1e_kinetic() + orbitals.get_mo_1e_potential())[o, o]
  g = orbitals.get_mo_2e_repulsion()[o,o,o,o]
  g = g - g.transpose((0, 2, 1, 3))
  valence_energy = np.trace(h) + 1./2 * np.einsum("ijij", g) + orbitals.integrals.molecule.nuclear_repulsion_energy
  #v = orbitals.get_mo_1e_core_field()[o, o]
  #core_valence_energy = np.trace(v)
  #total_energy = valence_energy + core_energy + core_valence_energy
  #print("Core energy:            {:20.15f}".format(core_energy))
  print("Valence energy:         {:20.15f}".format(valence_energy))
  #print("C-V interaction energy: {:20.15f}".format(core_valence_energy))
  #print("Total energy:           {:20.15f}".format(total_energy))
  print(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'c').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'o').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'v').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'co').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'ov').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'cov').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'c').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'o').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'v').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'co').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'ov').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'cov').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'c').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'o').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'v').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'co').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'ov').shape)
  print(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'cov').shape)
