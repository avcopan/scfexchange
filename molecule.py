import numpy as np

from . import constants


class NuclearFramework(object):
    """A class to store information about the nuclei in a nuclei.
    
    Attributes:
        natoms (int): The number of atoms.
        labels (`tuple`): Atomic symbols.
        charges (`tuple`): Atomic charges.
        masses (`tuple`): Atomic masses.
        coordinates (`np.ndarray`): An `self.natoms` x 3 array of Cartesian
            coordinates corresponding to the atoms in `self.labels`.
        units (str): Either 'angstrom' or 'bohr', indicating the units of
            `self.coordinates`.
        nuclear_repulsion_energy (float): The nuclear repulsion energy.
    """

    def __init__(self, labels, coordinates, units="angstrom"):
        """Initialize this NuclearFramework object.
        
        Args:
            labels (`tuple`): Atomic symbols.
            coordinates (`np.ndarray`): An `self.natoms` x 3 array of Cartesian
                coordinates corresponding to the atoms in `self.labels`.
            units (str): Either 'angstrom' or 'bohr', indicating the units of
                `self.coordinates`.
        """
        self.natoms = len(labels)
        self.labels = tuple(labels)
        self.charges = tuple(constants.get_charge(lbl) for lbl in self.labels)
        self.masses = tuple(constants.get_mass(lbl) for lbl in self.labels)
        self.coordinates = np.array(coordinates)
        self.units = units.lower()
        # Determine the nuclear repulsion energy.
        self.nuclear_repulsion_energy = self._get_nuclear_repulsion_energy()
        # Make sure units have an allowed value.
        if self.units not in ("angstrom", "bohr"):
            raise ValueError("Units must be 'bohr' or 'angstrom'.")

    def _get_nuclear_repulsion_energy(self):
        """Calculate the nuclear repulsion energy.
    
        Returns:
            float: The nuclear repulsion energy.
        """
        z = list(constants.get_charge(label) for label in self.labels)
        r = (self.coordinates if self.units == 'bohr'
             else self.coordinates / constants.BOHR_TO_ANGSTROM)
        nuclear_repulsion_energy = 0
        for a in range(self.natoms):
            for b in range(a):
                nuclear_repulsion_energy += z[a] * z[b] / np.linalg.norm(
                    r[a] - r[b])
        return nuclear_repulsion_energy

    def get_center_of_mass(self):
        """Get the nuclear center of mass.
        
        Returns:
            The center of mass in the coordinate units.
        """
        m_tot = sum(self.masses)
        com = sum(m * r for m, r in zip(self.masses, self.coordinates)) / m_tot
        return com

    def get_center_of_charge(self):
        """Get the nuclear center of charge.
        
        Returns:
            The center of charge in the current coordinate units.
        """
        q_tot = sum(self.charges)
        coc = sum(q * r for q, r in zip(self.charges, self.coordinates)) / q_tot
        return coc

    def get_dipole_moment(self, origin=(0.0, 0.0, 0.0)):
        """Get the nuclear dipole moment.
        
        Args:
            origin (tuple): The point about which to compute the dipole moment.
        
        Returns:
            np.ndarray: The dipole moment vector, [mu_x, mu_y, mu_z].
        """
        o = np.array(origin)
        return sum(q * (r - o) for q, r in zip(self.charges, self.coordinates))

    def set_units(self, units):
        """Convert `self.coordinates` to different units.
    
        Args:
          units (str): Either 'bohr' or 'angstrom'.
        """
        if units == "angstrom" and self.units == "bohr":
            self.units = "angstrom"
            self.coordinates *= constants.BOHR_TO_ANGSTROM
        elif units == "bohr" and self.units == "angstrom":
            self.units = "bohr"
            self.coordinates /= constants.BOHR_TO_ANGSTROM
        elif self.units not in ("angstrom", "bohr"):
            raise ValueError("Units must be 'bohr' or 'angstrom'.")

    def __iter__(self):
        """Iterate over atomic labels and coordinates."""
        for label, coordinate in zip(self.labels, self.coordinates):
            yield label, coordinate

    def __str__(self):
        """Display the nuclear framework as a string."""
        geom_string = "units {:s}\n".format(self.units)
        geom_line_template = "{:2s} {: >15.10f} {: >15.10f} {: >15.10f}\n"
        for label, coordinate in self:
            geom_string += geom_line_template.format(label, *coordinate)
        return geom_string

    def __repr__(self):
        return self.__str__()


class Molecule(object):
    """A class to store information about a chemical system.
  
    Attributes:
        nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): The nuclear 
            framework of this nuclei.
        charge (int): Total molecular charge.
        multiplicity (int): `2*S+1` where `S` is the spin-magnitude quantum
            number.
        nelec (int): The number of electrons.
        nalpha (int): The number of alpha-spin electrons.
        nbeta (int): The number of beta-spin electrons.
    """

    def __init__(self, nuclei, charge=0, multiplicity=1):
        """Initialize this nuclei.
        
        Args:
            nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): The nuclear 
                framework of this nuclei.
            charge (int): Total molecular charge.
            multiplicity (int): `2*S+1` where `S` is the spin-magnitude quantum
                number.
        """
        self.nuclei = nuclei
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

        # Determine the number of electrons, based on the number of protons and
        # the total charge.
        nprot = sum(constants.get_charge(label) for label, _ in self.nuclei)
        self.nelec = nprot - self.charge

        # Assuming a high-spin open-shell electronic state, so that S = M_S,
        # determine the number of alpha and beta electrons.
        nunpaired = self.multiplicity - 1
        npaired = (self.nelec - nunpaired) // 2
        self.nalpha = npaired + nunpaired
        self.nbeta = npaired

    def __str__(self):
        """Display the nuclei as a string."""
        charge_str = 'charge {:d}'.format(self.charge)
        multp_str = 'multiplicity {:d}'.format(self.multiplicity)
        return '\n'.join((charge_str, multp_str, str(self.nuclei)))


if __name__ == "__main__":
    units = "bohr"
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000000000000, -0.143225816552, 0.000000000000],
                            [1.638036840407, 1.136548822547, -0.000000000000],
                            [-1.638036840407, 1.136548822547, -0.000000000000]])

    nuclei = NuclearFramework(labels, coordinates, units)
    print(nuclei)
    print(nuclei.nuclear_repulsion_energy)
    print(nuclei.get_center_of_charge())
    print(nuclei.get_center_of_mass())
    mol = Molecule(nuclei, charge=+1, multiplicity=2)
