import numpy as np
import itertools as it

from . import _constants


# Public functions
def electron_count(nuc_labels, mol_charge=0):
    """The number of electrons in a molecule.

    Args:
        nuc_labels (tuple): Atomic symbols.:
        mol_charge (int): Total molecular charge.

    Returns:
        int: The number of electrons.
    """
    nuc_charge = sum(map(_constants.nuclear_charge, nuc_labels))
    return nuc_charge - mol_charge


def electron_spin_count(nuc_labels, mol_charge=0, multp=1):
    """The number of electrons in a molecule, by spin.

    Assumes a high-spin (M_S = S) state.

    Args:
        nuc_labels (tuple): Atomic symbols.:
        mol_charge (int): Total molecular charge.
        multp (int): Electronic spin multiplicity.

    Returns:
        tuple: The number of alpha and beta electrons.
    """
    nelec = electron_count(nuc_labels, mol_charge)
    nsocc = multp - 1
    ndocc = (nelec - nsocc) // 2
    nalpha = ndocc + nsocc
    nbeta = ndocc
    return nalpha, nbeta


def nuclear_coordinates_in_bohr(nuc_coords, units="bohr"):
    ret_coords = np.array(nuc_coords)
    if not ret_coords.ndim == 2 and ret_coords.shape[1] == 3:
        raise ValueError("Invalid coordinate array.")
    if units.lower() == "angstrom":
        ret_coords *= _constants.BOHR_TO_ANGSTROM
    elif units.lower() != "bohr":
        raise ValueError("Invalid 'units' argument.")
    return ret_coords


def nuclear_coordinate_string(nuc_labels, nuc_coords, units=None):
    """A string displaying the nuclear coordinates.

    Args:
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates.
        units (str): The units of `nuc_coords`, "angstrom" or "bohr".

    Returns:
        str: The coordinate string.
    """
    coord_line_template = "{:2s} {: >15.10f} {: >15.10f} {: >15.10f}"
    ret_str = "\n".join(coord_line_template.format(label, *coord)
                        for label, coord in zip(nuc_labels, nuc_coords))
    if units is not None:
        ret_str += "\nunits {:s}".format(str(units))
    return ret_str


def nuclear_repulsion_energy(nuc_labels, nuc_coords, units="bohr"):
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
    natom = len(nuc_labels)
    r = nuclear_coordinates_in_bohr(nuc_coords, units=units)
    q = list(map(_constants.nuclear_charge, nuc_labels))
    return sum(q[i] * q[j] / np.linalg.norm(r[i] - r[j])
               for i, j in it.combinations(range(natom), r=2))


def nuclear_dipole_moment(nuc_labels, nuc_coords, units="bohr"):
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
    r = nuclear_coordinates_in_bohr(nuc_coords, units=units)
    q = list(map(_constants.nuclear_charge, nuc_labels))
    return sum(q * r for q, r in zip(q, r))


def nuclear_center_of_charge(nuc_labels, nuc_coords, units="bohr"):
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
    r = nuclear_coordinates_in_bohr(nuc_coords, units=units)
    q = list(map(_constants.nuclear_charge, nuc_labels))
    return sum(q * r for q, r in zip(q, r)) / sum(q)


def nuclear_center_of_mass(nuc_labels, nuc_coords, units="bohr"):
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
    r = nuclear_coordinates_in_bohr(nuc_coords, units=units)
    m = list(map(_constants.isotopic_mass, nuc_labels))
    return sum(m * r for m, r in zip(m, r)) / sum(m)

