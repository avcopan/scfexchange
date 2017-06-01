import numpy as np

from . import constants


class NuclearFramework(object):
    """A class to store information about the nuclei in a nuclei.
    
    Attributes:
        natoms (int): The number of atoms.
        labels (`tuple`): Atomic symbols.
        charges (`tuple`): Atomic charges.
        masses (`tuple`): Atomic masses.
        coordinates (`numpy.ndarray`): Atomic coordinates in Bohr.
    """

    def __init__(self, labels, coordinates):
        """Initialize this NuclearFramework object.
        
        Args:
            labels (`tuple`): Atomic symbols.
            coordinates (`numpy.ndarray`): Atomic coordinates in Bohr.
        """
        self.natoms = len(labels)
        self.labels = tuple(labels)
        self.charges = tuple(constants.get_charge(lbl) for lbl in self.labels)
        self.masses = tuple(constants.get_mass(lbl) for lbl in self.labels)
        self.coordinates = np.array(coordinates)

    def get_coordinates(self, convert_to_angstroms=False):
        """Returns the coordinate array.
        
        Args:
            convert_to_angstroms (bool): Return the coordinates in Angstroms?

        Returns:
            numpy.ndarray: The coordinates.
        """
        if not convert_to_angstroms:
            return self.coordinates
        else:
            return self.coordinates * constants.BOHR_TO_ANGSTROM

    def get_nuclear_repulsion_energy(self):
        """Calculate the nuclear repulsion energy.
    
        Returns:
            float: The nuclear repulsion energy.
        """
        z = list(constants.get_charge(label) for label in self.labels)
        r = self.coordinates
        e_nuc = 0.0
        for a in range(self.natoms):
            for b in range(a):
                e_nuc += z[a] * z[b] / np.linalg.norm(r[a] - r[b])
        return e_nuc

    def get_center_of_mass(self):
        """Get the nuclear center of mass.
        
        Returns:
            numpy.ndarray: The center of mass.
        """
        m_tot = sum(self.masses)
        r_m = sum(m * r for m, r in zip(self.masses, self.coordinates)) / m_tot
        return r_m

    def get_center_of_charge(self):
        """Get the nuclear center of charge.
        
        Returns:
            numpy.ndarray: The center of charge.
        """
        q_tot = sum(self.charges)
        r_c = sum(q * r for q, r in zip(self.charges, self.coordinates)) / q_tot
        return r_c

    def get_dipole_moment(self):
        """Get the nuclear dipole moment.
        
        Returns:
            numpy.ndarray: The dipole moment.
        """
        mu = sum(q * r for q, r in zip(self.charges, self.coordinates))
        return mu

    def __str__(self):
        """Display the nuclear framework as a string."""
        geom_line_template = "{:2s} {: >15.10f} {: >15.10f} {: >15.10f}\n"
        ret = ""
        for label, coordinate in zip(self.labels, self.coordinates):
            ret += geom_line_template.format(label, *coordinate)
        return ret

    def __repr__(self):
        return self.__str__()


class Molecule(object):
    """A class to store information about a chemical system.
  
    Attributes:
        nuclei (:obj:`scfexchange.NuclearFramework`): The nuclear framework
            of this molecule.
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
            nuclei (:obj:`scfexchange.NuclearFramework`): The nuclear framework
                of this molecule.
            charge (int): Total molecular charge.
            multiplicity (int): `2*S+1` where `S` is the spin-magnitude quantum
                number.
        """
        self.nuclei = nuclei
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

        # Determine the number of electrons as the difference between the charge
        # of the nuclear framework and the total molecular charge.
        total_nuclear_charge = sum(self.nuclei.charges)
        self.nelec = total_nuclear_charge - self.charge

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
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    print(nuclei)
    print(nuclei.get_nuclear_repulsion_energy())
    print(repr(nuclei.get_center_of_charge()))
    print(repr(nuclei.get_center_of_mass()))
    print(repr(nuclei.get_dipole_moment()))
    mol = Molecule(nuclei, charge=+1, multiplicity=2)
