from . import _constants


def count(nuc_labels, mol_charge=0):
    """The number of electrons in a molecule.

    Args:
        nuc_labels (tuple): Atomic symbols.
        mol_charge (int): Total molecular charge.

    Returns:
        int: The number of electrons.
    """
    nuc_charge = sum(map(_constants.nuclear_charge, nuc_labels))
    return nuc_charge - mol_charge


def count_spins(nuc_labels, mol_charge=0, nunp=0):
    """The number of electrons in a molecule, by spin.

    Assumes a high-spin (M_S = S) state.

    Args:
        nuc_labels (tuple): Atomic symbols.
        mol_charge (int): Total molecular charge.
        nunp (int): The total number of unpaired electrons.

    Returns:
        tuple: The number of alpha and beta electrons.
    """
    nelec = count(nuc_labels, mol_charge)
    if (nelec - nunp) % 2 is not 0:
        raise ValueError("Inconsistent 'mol_charge' and 'nunp' arguments.")
    npair = (nelec - nunp) // 2
    nalpha = npair + nunp
    nbeta = npair
    return nalpha, nbeta


def _main():
    nuc_labels = ("O", "H", "H")
    print(count(nuc_labels, mol_charge=0))
    print(count_spins(nuc_labels, mol_charge=0, nunp=0))

if __name__ == "__main__":
    _main()