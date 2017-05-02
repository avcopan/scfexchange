import numpy as np

from . import atomdata

bohr2angstrom = 0.52917720859


class Molecule(object):
    """A class to store information about a chemical system.
  
    Attributes:
      natoms (int): The number of atoms.
      labels (`tuple` of `str`s): Atomic symbols.
      coordinates (`np.ndarray`): An `self.natoms` x 3 array of Cartesian
        coordinates corresponding to the atoms in `self.labels`.
      units (str): Either 'angstrom' or 'bohr', indicating the units of
        `self.coordinates`.
      charge (int): Total molecular charge.
      multiplicity (int): `2*S+1` where `S` is the spin-magnitude quantum number.
      nuclear_repulsion_energy (float): The nuclear repulsion energy.
      nelec (int): The number of electrons.
      ncore (int): The number of core electrons.  In the (rare) case that the
        molecule has been ionized past its valence shell, this is set to None.
      nalpha (int): The number of alpha-spin electrons.
      nbeta (int): The number of beta-spin electrons.
    """

    def __init__(self, labels, coordinates, units="angstrom", charge=0,
                 multiplicity=1):
        """Initialize this Molecule object.
        """
        self.natoms = len(labels)
        self.labels = tuple(labels)
        self.coordinates = np.array(coordinates)
        self.units = str(units.lower())
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

        if not self.units in ("angstrom", "bohr"):
            raise ValueError("{:s} is not a valid entry for self.units.  "
                             "Try 'bohr' or 'angstrom'.".format(self.units))
        if not self.coordinates.shape == (self.natoms, 3):
            raise ValueError(
                "Coordinate array should have shape ({:d}, 3), not {:s}."
                .format(self.natoms, str(self.coordinates.shape)))

        # Determine the nuclear repulsion energy.
        self.nuclear_repulsion_energy = self._get_nuclear_repulsion_energy()

        # Determine the number of electrons, based on the number of protons and the
        # total charge.
        nprot = sum(atomdata.get_charge(label) for label in self.labels)
        self.nelec = nprot - self.charge

        # Determine the number of core electrons.  If the molecule has been ionized
        # past its valence shell, leave this undefined.
        self.ncore = sum(atomdata.get_ncore(label) for label in self.labels)
        if self.nelec < self.ncore:
            self.ncore = None

        # Assuming a high-spin open-shell electronic state, so that S = M_S,
        # determine the number of alpha and beta electrons.
        nunpaired = self.multiplicity - 1
        npaired = (self.nelec - nunpaired) / 2
        self.nalpha = npaired + nunpaired
        self.nbeta = npaired

    def _get_nuclear_repulsion_energy(self):
        """Calculate the nuclear repulsion energy.
    
        Returns:
          float: The nuclear repulsion energy.
        """
        z = list(atomdata.get_charge(label) for label in self.labels)
        r = (self.coordinates if self.units == 'bohr'
             else self.coordinates / bohr2angstrom)
        nuclear_repulsion_energy = 0
        for a in range(self.natoms):
            for b in range(a):
                nuclear_repulsion_energy += z[a] * z[b] / np.linalg.norm(
                    r[a] - r[b])
        return nuclear_repulsion_energy

    def set_units(self, units):
        """Convert `self.coordinates` to different units.
    
        Args:
          units (str): Either 'bohr' or 'angstrom'.
        """
        if units == "angstrom" and self.units == "bohr":
            self.units = "angstrom"
            self.coordinates *= bohr2angstrom
        elif units == "bohr" and self.units == "angstrom":
            self.units = "bohr"
            self.coordinates /= bohr2angstrom
        elif self.units not in ("angstrom", "bohr"):
            raise ValueError("{:s} is not a valid entry for coordinate units.  "
                             "Try 'bohr' or 'angstrom'.".format(self.units))

    def __iter__(self):
        """Iterate over atomic labels and coordinates."""
        for label, coordinate in zip(self.labels, self.coordinates):
            yield label, coordinate

    def __str__(self):
        """Display the molecular geometry as a string."""
        geom_string = "units {:s}\n".format(self.units)
        geom_line_template = "{:2s} {: >15.10f} {: >15.10f} {: >15.10f}\n"
        for label, coordinate in self:
            geom_string += geom_line_template.format(label, *coordinate)
        return geom_string

    def __repr__(self):
        return self.__str__()

    @classmethod
    def build_molecule_from_xyz_string(cls, xyz_string, units="angstrom"):
        pass

    @classmethod
    def build_molecule_from_zmat_string(cls, zmat_string, units="angstrom"):
        pass


if __name__ == "__main__":
    units = "bohr"
    charge = +1
    multiplicity = 2
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000000000000, -0.143225816552, 0.000000000000],
                            [1.638036840407, 1.136548822547, -0.000000000000],
                            [-1.638036840407, 1.136548822547, -0.000000000000]])

    mol = Molecule(labels, coordinates, units=units, charge=charge,
                   multiplicity=multiplicity)
    print(list(iter(mol)))
    print(mol.nuclear_repulsion_energy)
