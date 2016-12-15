import numpy as np

bohr2angstrom = 0.52917721067

class Molecule(object):

  def __init__(self, labels, coordinates, units = "angstrom", charge = 0,
               multiplicity = 1):
    self.labels = tuple(labels)
    self.coordinates = np.array(coordinates)
    self.units = str(units.lower())
    self.charge = int(charge)
    self.multiplicity = int(multiplicity)
    if not self.units in ("angstrom", "bohr"):
      raise Exception("{:s} is not a valid entry for self.units.  "
                      "Try 'bohr' or 'angstrom'.".format(self.units))

  def set_units(self, units):
    if units == "angstrom" and self.units == "bohr":
      self.units  = "angstrom"
      self.coordinates  *= bohr2angstrom
    elif units == "bohr" and self.units == "angstrom":
      self.units  = "bohr"
      self.coordinates  /= bohr2angstrom

  def __iter__(self):
    for label, coordinate in zip(self.labels, self.coordinates):
      yield label, coordinate

  def __str__(self):
    geom_string = "units {:s}\n"
    geom_line_template = "{:2s} {: >15.10f} {: >15.10f} {: >15.10f}\n"
    for label, coordinate in self:
      geom_string += geom_line_template.format(label, *coordinate)
    return geom_string

  def __repr__(self):
    return self.__str__()

  @classmethod
  def build_molecule_from_xyz_string(cls, xyz_string, units = "angstrom"):
    pass

  @classmethod
  def build_molecule_from_zmat_string(cls, zmat_string, units = "angstrom"):
    pass

if __name__ == "__main__":
  units = "angstrom"
  charge = +1
  multiplicity = 2
  labels = ("O", "H", "H")
  coordinates = np.array([[0.000,  0.000, -0.066],
                          [0.000, -0.759,  0.522],
                          [0.000,  0.759,  0.522]])

  mol = Molecule(labels, coordinates, units = units, charge = charge,
                 multiplicity = multiplicity)
  mol.set_units("bohr")
  print(list(zip(mol.labels, mol.coordinates)))
