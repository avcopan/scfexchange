import numpy as np
from .integrals import IntegralsBase
from psi4.core import Molecule as PsiMolecule, BasisSet, MintsHelper

class Integrals(IntegralsBase):

  def __init__(self, molecule, basis_label):
    IntegralsBase.__init__(self, molecule, basis_label)

    mol_str = str(self.molecule) + "\nunits {:s}".format(self.molecule.units)
    psi_molecule = PsiMolecule.create_molecule_from_string(mol_str)
    basisset = BasisSet.build(psi_molecule, "BASIS", basis_label)
    self.mints_helper = MintsHelper(basisset)

  def get_ao_1e_overlap_integrals(self):
    return np.array(self.mints_helper.ao_overlap())

  def get_ao_1e_potential_integrals(self):
    return np.array(self.mints_helper.ao_potential())

  def get_ao_1e_kinetic_integrals(self):
    return np.array(self.mints_helper.ao_kinetic())

  def get_ao_2e_repulsion_integrals(self):
    return np.array(self.mints_helper.ao_eri()).transpose((0, 2, 1, 3))

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
