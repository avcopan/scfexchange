def test__rpa_cis_spectrum():
    import numpy as np
    from scfexchange import Nuclei
    from scfexchange.pyscf_interface import Integrals, Orbitals
    from scfexchange.examples.rpa import RPA

    labels = ("O", "H", "H")
    coordinates = np.array([[0.00000000000, -0.14322581655, 0.00000000000],
                            [1.63803684041, 1.13654882255, 0.00000000000],
                            [-1.63803684041, 1.13654882255, 0.00000000000]])
    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    orbitals = Orbitals(integrals)
    orbitals.solve()
    rpa = RPA(orbitals)
    spectrum = rpa.compute_cis_spectrum()
    ref_spectrum = [
        0.2872554996, 0.2872554996, 0.2872554996, 0.3444249963, 0.3444249963,
        0.3444249963, 0.3564617587, 0.3659889948, 0.3659889948, 0.3659889948,
        0.3945137992, 0.3945137992, 0.3945137992, 0.4160717386, 0.5056282877,
        0.5142899971, 0.5142899971, 0.5142899971, 0.5551918860, 0.5630557635,
        0.5630557635, 0.5630557635, 0.6553184485, 0.9101216891, 1.1087709658,
        1.1087709658, 1.1087709658, 1.2000961331, 1.2000961331, 1.2000961331,
        1.3007851948, 1.3257620652, 19.9585264123, 19.9585264123, 19.9585264123,
        20.0109794203, 20.0113420895, 20.0113420895, 20.0113420895,
        20.0505319444
    ]
    assert (np.allclose(spectrum, ref_spectrum))
