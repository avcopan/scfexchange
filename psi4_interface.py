import numpy as np
import psi4.core
from .integrals import IntegralsInterface
from .orbitals import OrbitalsInterface
from .util import with_doc

class Integrals(IntegralsInterface):
  __doc__ = """**IntegralsInterface.__doc__**

{:s}

**Integrals.__doc__**

Interface to Psi4 integrals.

  Attributes:
    _mints_helper (:obj:`psi.core.MintsHelper`): Used to call the molecular
      integrals code in Psi4.
    _psi4_molecule (:obj:`psi.core.Molecule`): Psi4's molecule object.
  """.format(IntegralsInterface.__doc__)

  def __init__(self, molecule, basis_label):
    """Initialize Integrals object (PySCF interface).

    Args:
      molecule (:obj:`scfexchange.molecule.Molecule`): The molecule.
      basis_label (str): What basis set to use.
    """
    molstr = str(molecule)
    self._psi4_molecule = psi4.core.Molecule.create_molecule_from_string(molstr)
    self._psi4_molecule.set_molecular_charge(molecule.charge)
    self._psi4_molecule.set_multiplicity(molecule.multiplicity)
    self._psi4_molecule.reset_point_group("c1")
    self._psi4_molecule.update_geometry()
    basisset = psi4.core.BasisSet.build(self._psi4_molecule, "BASIS",
                                        basis_label)
    self._mints_helper = psi4.core.MintsHelper(basisset)
    self.molecule = molecule
    self.basis_label = basis_label
    self.nbf = int(self._mints_helper.nbf())

  @with_doc(IntegralsInterface.get_ao_1e_overlap.__doc__)
  def get_ao_1e_overlap(self, spinor = False):
    ao_1e_overlap = np.array(self._mints_helper.ao_overlap())
    return (ao_1e_overlap if not spinor
            else IntegralsInterface.convert_1e_ao_to_aso(ao_1e_overlap))

  @with_doc(IntegralsInterface.get_ao_1e_potential.__doc__)
  def get_ao_1e_potential(self, spinor = False):
    ao_1e_potential = np.array(self._mints_helper.ao_potential())
    return (ao_1e_potential if not spinor
            else IntegralsInterface.convert_1e_ao_to_aso(ao_1e_potential))

  @with_doc(IntegralsInterface.get_ao_1e_kinetic.__doc__)
  def get_ao_1e_kinetic(self, spinor = False):
    ao_1e_kinetic = np.array(self._mints_helper.ao_kinetic())
    return (ao_1e_kinetic if not spinor
            else IntegralsInterface.convert_1e_ao_to_aso(ao_1e_kinetic))

  @with_doc(IntegralsInterface.get_ao_2e_repulsion.__doc__)
  def get_ao_2e_repulsion(self, spinor = False):
    # Psi4 returns these in chemist's notation, (mu rh | nu si), so transpose to
    # physicist's notation, <mu nu | rh si>.
    ao_2e_chem_repulsion = np.array(self._mints_helper.ao_eri())
    ao_2e_repulsion = ao_2e_chem_repulsion.transpose((0, 2, 1, 3))
    return (ao_2e_repulsion if not spinor
            else IntegralsInterface.convert_2e_ao_to_aso(ao_2e_repulsion))


class Orbitals(OrbitalsInterface): 
  __doc__ = """**OrbitalsInterface.__doc__**

{:s}

**Orbitals.__doc__**

Interface for accessing Psi4 molecular orbitals.

  Attributes:
    _psi4_hf (:obj:`psi4.core.HF`): Used to access Psi4 orbitals.

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
    # Determine the number of frozen, occupied, virtual orbitals.
    nfrz = self._determine_n_frozen_orbitals()
    self.nspocc = self.integrals.molecule.nelec - 2 * nfrz
    self.nsporb = 2 * (self.integrals.nbf - nfrz)
    self.nspvir = self.nsporb - self.nspocc
    # Build Psi4 HF object and compute the energy.
    wfn = psi4.core.Wavefunction.build(integrals._psi4_molecule,
                                       integrals.basis_label)
    sf, _ = psi4.driver.dft_functional.build_superfunctional("HF")
    psi4.core.set_global_option("guess", "gwh")
    psi4.core.set_global_option("e_convergence", self.options['e_threshold'])
    psi4.core.set_global_option("d_convergence", self.options['d_threshold'])
    psi4.core.set_global_option("maxiter", self.options['n_iterations'])
    if self.options['restrict_spin']:
      if integrals.molecule.multiplicity is 1:
        psi4.core.set_global_option("reference", "RHF")
        self._psi4_hf = psi4.core.RHF(wfn, sf)
      else:
        psi4.core.set_global_option("reference", "ROHF")
        self._psi4_hf = psi4.core.ROHF(wfn, sf)
    else:
      psi4.core.set_global_option("reference", "UHF")
      self._psi4_hf = psi4.core.UHF(wfn, sf)
    self._psi4_hf.compute_energy()
    # Get MO energies and coefficients and put them in the right format
    mo_alpha_energies = self._psi4_hf.epsilon_a().to_array()[nfrz:]
    mo_beta_energies = self._psi4_hf.epsilon_b().to_array()[nfrz:]
    mo_alpha_coeffs = np.array(self._psi4_hf.Ca())[:, nfrz:]
    mo_beta_coeffs = np.array(self._psi4_hf.Cb())[:, nfrz:]
    self.mo_energies = np.array([mo_alpha_energies, mo_beta_energies])
    self.mo_coefficients = np.array([mo_alpha_coeffs, mo_beta_coeffs])
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
    'restrict_spin': False,
    'n_iterations': 20,
    'e_threshold': 1e-12
  }
  orbitals = Orbitals(integrals, **options)
  print(orbitals.mso_coefficients.round(1))
  print(orbitals.mso_energies)
  print(orbitals.get_mo_2e_repulsion().shape)
  print(orbitals.get_mo_2e_repulsion('spinor').shape)
  print(Orbitals.__init__.__doc__)
