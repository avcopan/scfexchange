import numpy as np

bohr2angstrom = 0.52917721067

class Molecule(object):
  """A class to store information about a chemical system.

  Attributes:
    natoms: An integer indicating the number of atoms in the molecule.
    labels: A tuple of strings indicating the atomic symbol of each atom.
    coordinates: An natoms x 3 array indicating the Cartesian coordinates of
      each atom.
    units: Either 'angstrom' or 'bohr', indicating the units of
      `self.coordinates`.
    charge: An integer indicating the total charge of the molecule.
    multiplicity: An integer indicating the spin state of the molecule, 2*S + 1.
  """

  def __init__(self, labels, coordinates, units = "angstrom", charge = 0,
               multiplicity = 1):
    self.natoms = len(labels)
    self.labels = tuple(labels)
    self.coordinates = np.array(coordinates)
    self.units = str(units.lower())
    self.charge = int(charge)
    self.multiplicity = int(multiplicity)
    if not self.units in ("angstrom", "bohr"):
      raise Exception("{:s} is not a valid entry for self.units.  "
                      "Try 'bohr' or 'angstrom'.".format(self.units))
    if not self.coordinates.shape == (self.natoms, 3):
      raise Exception("Coordinate array should have shape ({:d}, 3), not {:s}."
                      .format(self.natoms, str(self.coordinates.shape)))

  def set_units(self, units):
    """Convert `self.coordinates` to different units.

    Args:
      units: A string, either 'bohr' or 'angstrom'.
    """
    if units == "angstrom" and self.units == "bohr":
      self.units  = "angstrom"
      self.coordinates  *= bohr2angstrom
    elif units == "bohr" and self.units == "angstrom":
      self.units  = "bohr"
      self.coordinates  /= bohr2angstrom
    else:
      raise Exception("{:s} is not a valid entry for self.units.  "
                      "Try 'bohr' or 'angstrom'.".format(self.units))

  def __iter__(self):
    """Iterate over atomic labels and coordinates.
    """
    for label, coordinate in zip(self.labels, self.coordinates):
      yield label, coordinate

  def __str__(self):
    geom_string = "units {:s}\n".format(self.units)
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
  print(list(iter(mol)))
