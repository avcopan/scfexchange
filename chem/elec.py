from . import _constants


def count(atoms, charge=0):
    """The number of electrons in a molecule.

    Args:
        atoms (tuple): Atomic symbols.
        charge (int): Total molecular charge.

    Returns:
        int: The number of electrons.
    """
    nuc_charge = sum(map(_constants.nuclear_charge, atoms))
    return nuc_charge - charge


def count_spins(atoms, charge=0, spin=0):
    """The number of electrons in a molecule, by spin.

    Assumes a high-spin (M_S = S) state.

    Args:
        atoms (tuple): Atomic symbols.
        charge (int): Total molecular charge.
        spin (int): The total number of unpaired electrons.

    Returns:
        tuple: The number of alpha and beta electrons.
    """
    nelec = count(atoms, charge)
    if (nelec - spin) % 2 is not 0:
        raise ValueError("Inconsistent 'charge' and 'spin' arguments.")
    nbeta = (nelec - spin) // 2
    nalpha = nbeta + spin
    return nalpha, nbeta


def _main():
    atoms = ("O", "H", "H")
    print(count(atoms, charge=0))
    print(count_spins(atoms, charge=0, spin=0))

if __name__ == "__main__":
    _main()