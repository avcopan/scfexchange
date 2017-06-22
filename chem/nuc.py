import itertools as it

import numpy as np
from . import _constants


def repulsion_energy(atoms, centers):
    """The Coulomb repulsion energy of some nuclei.

    Args:
        atoms (tuple): Atomic symbols.
        centers (numpy.ndarray): Atomic coordinates in Bohr.

    Returns:
        float: The nuclear repulsion energy.
    """
    assert len(atoms) is len(centers)
    charges = map(_constants.nuclear_charge, atoms)
    return sum(q1 * q2 / np.linalg.norm(r1 - r2) for (q1, r1), (q2, r2) in
               it.combinations(zip(charges, centers), r=2))


def dipole_moment(atoms, centers):
    """The dipole moment of some nuclei.

    Args:
        atoms (tuple): Atomic symbols.
        centers (numpy.ndarray): Atomic coordinates in Bohr.

    Returns:
        numpy.ndarray: The dipole moment.
    """
    assert len(atoms) is len(centers)
    charges = map(_constants.nuclear_charge, atoms)
    return sum(q * r for q, r in zip(charges, centers))


def center_of_charge(atoms, centers):
    """The center of charge of some nuclei.

    Args:
        atoms (tuple): Atomic symbols.
        centers (numpy.ndarray): Atomic coordinates in Bohr.

    Returns:
        numpy.ndarray: The center of charge.
    """
    assert len(atoms) is len(centers)
    charges = list(map(_constants.nuclear_charge, atoms))
    return sum(q * r for q, r in zip(charges, centers)) / sum(charges)


def center_of_mass(atoms, centers):
    """The center of mass of some nuclei.

    Args:
        atoms (tuple): Atomic symbols.
        centers (numpy.ndarray): Atomic coordinates in Bohr.

    Returns:
        numpy.ndarray: The center of mass.
    """
    assert len(atoms) is len(centers)
    masses = list(map(_constants.isotopic_mass, atoms))
    return sum(m * r for m, r in zip(masses, centers)) / sum(masses)


def _main():
    atoms = ("O", "H", "H")
    centers = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                        [0.0000000000, -1.4343021349, 0.9864370414],
                        [0.0000000000, 1.4343021349, 0.9864370414]])
    print(repulsion_energy(atoms, centers))
    print(dipole_moment(atoms, centers))
    print(center_of_charge(atoms, centers))
    print(center_of_mass(atoms, centers))

if __name__ == "__main__":
    _main()
