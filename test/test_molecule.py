from scfexchange.molecule import NuclearFramework


def test__nuclear_framework():
    import numpy as np

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    assert (np.isclose(nuclei.get_nuclear_repulsion_energy(), 9.16714531281))
    assert (np.allclose(nuclei.get_center_of_charge(), [0., 0., 0.09750987]))
    assert (np.allclose(nuclei.get_center_of_mass(), [0., 0., -0.00036671]))
    assert (np.allclose(nuclei.get_dipole_moment(), [0., 0., 0.97509868]))
