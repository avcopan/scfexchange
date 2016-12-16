import numpy as np
import psi4.core
from psi4.core import Molecule as Psi4Molecule
from .integrals import IntegralsBase

class Integrals(IntegralsBase):
  """Interface to Psi4 integrals.

  Attributes:
    _mints_helper: A `psi4.core.MintsHelper` object, used to call the molecular
      integrals code in Psi4.
  """
  def __init__(self, molecule, basis_label):
    """Initialize Integrals object for the Psi4 interface.

    Calls base class constructor and then builds Psi4's `MintsHelper` object.
    """
    IntegralsBase.__init__(self, molecule, basis_label)

    psi4_molecule = Psi4Molecule.create_molecule_from_string(str(self.molecule))
    basisset = psi4.core.BasisSet.build(psi4_molecule, "BASIS", basis_label)
    self._mints_helper = psi4.core.MintsHelper(basisset)

  def get_ao_1e_overlap_integrals(self):
    return np.array(self._mints_helper.ao_overlap())

  def get_ao_1e_potential_integrals(self):
    return np.array(self._mints_helper.ao_potential())

  def get_ao_1e_kinetic_integrals(self):
    return np.array(self._mints_helper.ao_kinetic())

  def get_ao_2e_repulsion_integrals(self):
    # transpose from chemist's to physicist's notation
    return np.array(self._mints_helper.ao_eri()).transpose((0, 2, 1, 3))

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
  s = integrals.get_ao_1e_overlap_integrals()
  g = integrals.get_ao_2e_repulsion_integrals()
  print(s.shape)
  print(g.shape)
