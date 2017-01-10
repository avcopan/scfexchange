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
    """

    if not isinstance(integrals, Integrals):
      raise ValueError("Please use an integrals object from this interface.")
    self.integrals = integrals
    self.options = self._process_options(options)
    if self.options['restrict_spin']:
      self._pyscf_hf = pyscf.scf.RHF(integrals._pyscf_molecule)
    else:
      self._pyscf_hf = pyscf.scf.UHF(integrals._pyscf_molecule)
    self._pyscf_hf.conv_tol = self.options['e_threshold']
    self._pyscf_hf.max_cycle = self.options['n_iterations']
    self._pyscf_hf.kernel()
    self.mo_energies = self._pyscf_hf.mo_energy
    self.mo_coefficients = self._pyscf_hf.mo_coeff
    if self.options['restrict_spin']:
      self.mo_energies = np.array([self.mo_energies] * 2)
      self.mo_coefficients = np.array([self.mo_coefficients] * 2)
    self.mso_energies, self.mso_coefficients = \
      self._get_mso_energies_and_coefficients()


if __name__ == "__main__":
  import numpy as np
  from .molecule import Molecule

  units = "angstrom"
  charge = +1
  multiplicity = 2
  labels = ("O", "H", "H")
  coordinates = np.array([[0.000,  0.000, -0.066],
                          [0.000, -0.759,  0.522],
                          [0.000,  0.759,  0.522]])

  mol = Molecule(labels, coordinates, units = units, charge = charge,
                 multiplicity = multiplicity)

  integrals = Integrals(mol, "sto-3g")
  s = integrals.get_ao_1e_overlap()
  g = integrals.get_ao_2e_repulsion()

  options = {
    'restrict_spin': True,
    'n_iterations': 20,
    'e_threshold': 1e-12
  }
  orbitals = Orbitals(integrals, **options)
  print(orbitals.mso_coefficients.round(1))
  print(orbitals.mso_energies)
  print(orbitals.get_mo_2e_repulsion().shape)
  print(orbitals.get_mo_2e_repulsion('spinor').shape)
  print(Orbitals.__init__.__doc__)
