from scfexchange.examples.phf import PerturbedHartreeFock


def test__hellmann_feynman_theorem():
    import numpy as np
    from scfexchange.pyscf_interface import Integrals
    from scfexchange.molecule import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = Integrals(nuclei, "sto-3g")
    orbitals = PerturbedHartreeFock(integrals, charge=1, multiplicity=2,
                                    restrict_spin=False)
    orbitals.solve(niter=100, e_threshold=1e-14, d_threshold=1e-12)
    d_occ = orbitals.get_mo_1e_dipole(mo_block='o,o', spin_block=None)
    mu = [np.trace(d_occ_x) for d_occ_x in d_occ]

    import scipy.misc

    def energy(field_value=0., field_component=0):
        e_field = np.zeros((3,))
        e_field[field_component] = field_value
        orbitals.solve(niter=100, e_threshold=1e-15,
                       d_threshold=1e-13, electric_field=e_field)
        return -orbitals.get_energy(electric_field=e_field)

    dedz = scipy.misc.derivative(energy, 0., dx=0.005, n=1, args=(2,),
                                 order=7)

    assert(np.isclose(dedz, mu[2]))

