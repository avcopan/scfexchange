import itertools as it

import numpy as np
from . import _constants


def coordinates_in_bohr(nuc_coords, units="bohr"):
    coords = np.array(nuc_coords)
    if not coords.ndim == 2 and coords.shape[1] == 3:
        raise ValueError("Invalid coordinate array.")
    if units.lower() == "angstrom":
        coords *= _constants.BOHR_TO_ANGSTROM
    elif units.lower() != "bohr":
        raise ValueError("Invalid 'units' argument.")
    return coords


def coordinate_string(nuc_labels, nuc_coords, units=None):
    """A string displaying the nuclear coordinates.

    Args:
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates.
        units (str): The units of `nuc_coords`, "angstrom" or "bohr".

    Returns:
        str: The coordinate string.
    """
    coord_line_template = "{:2s} {: >15.10f} {: >15.10f} {: >15.10f}"
    coord_str = "\n".join(coord_line_template.format(label, *coord)
                        for label, coord in zip(nuc_labels, nuc_coords))
    if units is not None:
        coord_str += "\nunits {:s}".format(str(units))
    return coord_str


def repulsion_energy(nuc_labels, nuc_coords, units="bohr"):
    """The Coulomb repulsion energy of some nuclei.

    Args:
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates.
        units (str): The units of `nuc_coords`, "angstrom" or "bohr".

    Returns:
        float: The nuclear repulsion energy.
    """
    if len(nuc_labels) != len(nuc_coords):
        raise ValueError("'nuc_labels' and 'nuc_coords' do not match.")
    charges = map(_constants.nuclear_charge, nuc_labels)
    coords = coordinates_in_bohr(nuc_coords, units=units)
    return sum(q1 * q2 / np.linalg.norm(r1 - r2) for (q1, r1), (q2, r2) in
               it.combinations(zip(charges, coords), r=2))


def dipole_moment(nuc_labels, nuc_coords, units="bohr"):
    """The dipole moment of some nuclei.

    Args:
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates.
        units (str): The units of `nuc_coords`, "angstrom" or "bohr".

    Returns:
        numpy.ndarray: The dipole moment.
    """
    if len(nuc_labels) != len(nuc_coords):
        raise ValueError("'nuc_labels' and 'nuc_coords' do not match.")
    charges = map(_constants.nuclear_charge, nuc_labels)
    coords = coordinates_in_bohr(nuc_coords, units=units)
    return sum(q * r for q, r in zip(charges, coords))


def center_of_charge(nuc_labels, nuc_coords, units="bohr"):
    """The center of charge of some nuclei.

    Args:
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates.
        units (str): The units of `nuc_coords`, "angstrom" or "bohr".

    Returns:
        numpy.ndarray: The center of charge.
    """
    if len(nuc_labels) != len(nuc_coords):
        raise ValueError("'nuc_labels' and 'nuc_coords' do not match.")
    charges = list(map(_constants.nuclear_charge, nuc_labels))
    coords = coordinates_in_bohr(nuc_coords, units=units)
    return sum(q * r for q, r in zip(charges, coords)) / sum(charges)


def center_of_mass(nuc_labels, nuc_coords, units="bohr"):
    """The center of mass of some nuclei.

    Args:
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates.
        units (str): The units of `nuc_coords`, "angstrom" or "bohr".

    Returns:
        numpy.ndarray: The center of mass.
    """
    if len(nuc_labels) != len(nuc_coords):
        raise ValueError("'nuc_labels' and 'nuc_coords' do not match.")
    masses = list(map(_constants.isotopic_mass, nuc_labels))
    coords = coordinates_in_bohr(nuc_coords, units=units)
    return sum(m * r for m, r in zip(masses, coords)) / sum(masses)


def _main():
    nuc_labels = ("O", "H", "H")
    nuc_coords = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                           [0.0000000000, -1.4343021349, 0.9864370414],
                           [0.0000000000, 1.4343021349, 0.9864370414]])
    print(repulsion_energy(nuc_labels, nuc_coords))
    print(dipole_moment(nuc_labels, nuc_coords))
    print(center_of_charge(nuc_labels, nuc_coords))
    print(center_of_mass(nuc_labels, nuc_coords))

if __name__ == "__main__":
    _main()
