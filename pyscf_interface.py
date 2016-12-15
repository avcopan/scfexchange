from .integrals import IntegralsBase
from pyscf.gto import Mole as PySCFMolecule

class Integrals(IntegralsBase):

  def __init__(self, molecule, basis_label):
    IntegralsBase.__init__(self, molecule, basis_label)
    atoms = list(zip(molecule.labels, molecule.coordinates))
    self.pyscf_molecule = PySCFMolecule(atom = atoms,
                                        unit = self.molecule.units,
                                        charge = self.molecule.charge,
                                        spin = self.molecule.multiplicity - 1,
                                        basis = basis_label)
    self.pyscf_molecule.build()
    self.nbf = self.pyscf_molecule.nao_nr()

  def get_ao_1e_overlap_integrals(self):
    return self.pyscf_molecule.intor('cint1e_ovlp_sph')

  def get_ao_1e_potential_integrals(self):
    return self.pyscf_molecule.intor('cint1e_nuc_sphi')

  def get_ao_1e_kinetic_integrals(self):
    return self.pyscf_molecule.intor('cint1e_kin_sph')

  def get_ao_2e_repulsion_integrals(self):
    return (self.pyscf_molecule.intor('cint2e_sph')
            .reshape((self.nbf, self.nbf, self.nbf, self.nbf))
            .transpose((0, 2, 1, 3)))


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

  print(s)
