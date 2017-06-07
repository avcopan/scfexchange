def test__mp2_ref_energy():
    import itertools as it
    import numpy as np
    from scfexchange import Nuclei
    from scfexchange.pyscf_interface import Integrals, Orbitals
    from scfexchange.examples.mp2 import MP2Density

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
        orbitals = Orbitals(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        density = MP2Density(orbitals)
        energy = density.get_energy()
        assert (np.isclose(energy, next(energies)))


def test__mp2_total_energy():
    import numpy as np
    from scfexchange import Nuclei
    from scfexchange.pyscf_interface import Integrals, Orbitals
    from scfexchange.examples.mp2 import MP2Density

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    energies = iter([
        -74.999083376332749, -74.998983886214702, -74.684519433506409,
        -74.684441944448196
    ])
    for (charge, multp) in [(0, 1), (1, 2)]:
        orbitals = Orbitals(integrals, charge, multp, restrict_spin=False)
        orbitals.solve()
        for ncore in [0, 1]:
            orbitals.ncore = ncore
            density = MP2Density(orbitals)
            density.solve()
            energy = density.get_energy()
            assert (np.isclose(energy, next(energies)))
