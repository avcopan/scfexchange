from .integrals import IntegralsBase
from pyscf.gto import Mole as PySCFMolecule

class Integrals(IntegralsBase):
  """Interface to PySCF integrals.

  Attributes:
    _pyscf_molecule: A `pyscf.gto.Mole` object, used to access PySCF integrals.
    _nbf: An integer specifying the number of basis functions.
  """

  def __init__(self, molecule, basis_label):
    IntegralsBase.__init__(self, molecule, basis_label)
    self._pyscf_molecule = PySCFMolecule(atom = list(iter(molecule)),
                                         unit = self.molecule.units,
                                         basis = basis_label)
    self._pyscf_molecule.build()
    self._nbf = self._pyscf_molecule.nao_nr()

  def get_ao_1e_overlap_integrals(self):
    return self._pyscf_molecule.intor('cint1e_ovlp_sph')

  def get_ao_1e_potential_integrals(self):
    return self._pyscf_molecule.intor('cint1e_nuc_sphi')

  def get_ao_1e_kinetic_integrals(self):
    return self._pyscf_molecule.intor('cint1e_kin_sph')

  def get_ao_2e_repulsion_integrals(self):
    # PySCF returns these as a nbf*nbf x nbf*nbf matrix, so reshape and
    # transpose from chemist's to physicist's notation
    return (self._pyscf_molecule.intor('cint2e_sph')
            .reshape((self._nbf, self._nbf, self._nbf, self._nbf))
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
