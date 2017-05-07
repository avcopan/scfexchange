import inspect
from scfexchange.orbitals import OrbitalsInterface


def check_interface(orbitals_instance):
    # Check class documentation
    pass


if __name__ == "__main__":
    import numpy as np
    from scfexchange.pyscf_interface import Integrals

    units = "angstrom"
    charge = 0
    multiplicity = 1
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])

    mol = Molecule(labels, coordinates, units=units, charge=charge,
                   multiplicity=multiplicity)
    integrals = Integrals(mol, "cc-pvdz")
    orbitals = Orbitals(integrals,
                        freeze_core=False,
                        n_frozen_orbitals=1,
                        e_threshold=1e-14,
                        n_iterations=50,
                        restrict_spin=False)
    check_interface(orbitals)
