
def test__determinant_energy():
    import itertools as it
    import numpy as np
    from scfexchange import Nuclei, DeterminantDensity
    from scfexchange.pyscf_interface import Integrals, Orbitals

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    energies = iter([
        -74.963343795087525, -74.963343795087511, -74.654712456959146,
        -74.656730208992286
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = Orbitals(integrals, charge=charge, multiplicity=multp,
                            restrict_spin=restr)
        orbitals.solve()
        density = DeterminantDensity(orbitals)
        energy = density.get_energy()
        assert (np.isclose(energy, next(energies)))
