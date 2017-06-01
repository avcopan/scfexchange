def check_interface(orbitals_instance):
    import numpy as np
    from scfexchange import Molecule, IntegralsInterface

    # Check attributes
    assert (hasattr(orbitals_instance, 'integrals'))
    assert (hasattr(orbitals_instance, 'molecule'))
    assert (hasattr(orbitals_instance, 'mo_coefficients'))
    assert (hasattr(orbitals_instance, 'spin_is_restricted'))
    assert (hasattr(orbitals_instance, 'ncore'))
    # Check attribute types
    assert (isinstance(getattr(orbitals_instance, 'integrals'),
                       IntegralsInterface))
    assert (isinstance(orbitals_instance.molecule, Molecule))
    assert (isinstance(orbitals_instance.mo_coefficients, np.ndarray))
    assert (isinstance(orbitals_instance.spin_is_restricted, bool))
    assert (isinstance(orbitals_instance.ncore, int))
    # Check array data types
    assert (orbitals_instance.mo_coefficients.dtype == np.float64)
    # Check array shapes
    norb = orbitals_instance.get_mo_count(mo_space='cov')
    assert (orbitals_instance.mo_coefficients.shape == (2, norb, norb))


def run_interface_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Build orbitals
    iterables = ([(0, 1), (1, 2)], [True, False], [0, 1])
    for (charge, multp), restr, ncore in it.product(*iterables):
        orbitals = orbitals_class(integrals, charge, multp,
                                  restrict_spin=restr, ncore=ncore)
        check_interface(orbitals)


def run_mo_counting_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    counts = iter([
        0, 5, 2, 5, 7, 7, 0, 5, 2, 5, 7, 7, 0, 10, 4, 10, 14, 14, 1, 4, 2, 5, 6,
        7, 1, 4, 2, 5, 6, 7, 2, 8, 4, 10, 12, 14, 0, 5, 2, 5, 7, 7, 0, 4, 3, 4,
        7, 7, 0, 9, 5, 9, 14, 14, 1, 4, 2, 5, 6, 7, 1, 3, 3, 4, 6, 7, 2, 7, 5,
        9, 12, 14
    ])
    iterables = ([0, 1], ['alpha', 'beta', 'spinorb'],
                 ['c', 'o', 'v', 'co', 'ov', 'cov'])
    for charge, multp in [(0, 1), (1, 2)]:
        orbitals = orbitals_class(integrals, charge, multp)
        for ncore, mo_type, mo_space in it.product(*iterables):
            orbitals.ncore = ncore
            count = orbitals.get_mo_count(mo_type, mo_space)
            assert (count == next(counts))


def run_mo_slicing_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    slices = iter([
        slice(0, 0, None), slice(0, 5, None), slice(5, 7, None),
        slice(0, 5, None), slice(0, 7, None), slice(0, 7, None),
        slice(0, 0, None), slice(0, 5, None), slice(5, 7, None),
        slice(0, 5, None), slice(0, 7, None), slice(0, 7, None),
        slice(0, 0, None), slice(0, 10, None), slice(10, 14, None),
        slice(0, 10, None), slice(0, 14, None), slice(0, 14, None),
        slice(0, 1, None), slice(1, 5, None), slice(5, 7, None),
        slice(0, 5, None), slice(1, 7, None), slice(0, 7, None),
        slice(0, 1, None), slice(1, 5, None), slice(5, 7, None),
        slice(0, 5, None), slice(1, 7, None), slice(0, 7, None),
        slice(0, 2, None), slice(2, 10, None), slice(10, 14, None),
        slice(0, 10, None), slice(2, 14, None), slice(0, 14, None),
        slice(0, 0, None), slice(0, 5, None), slice(5, 7, None),
        slice(0, 5, None), slice(0, 7, None), slice(0, 7, None),
        slice(0, 0, None), slice(0, 4, None), slice(4, 7, None),
        slice(0, 4, None), slice(0, 7, None), slice(0, 7, None),
        slice(0, 0, None), slice(0, 9, None), slice(9, 14, None),
        slice(0, 9, None), slice(0, 14, None), slice(0, 14, None),
        slice(0, 1, None), slice(1, 5, None), slice(5, 7, None),
        slice(0, 5, None), slice(1, 7, None), slice(0, 7, None),
        slice(0, 1, None), slice(1, 4, None), slice(4, 7, None),
        slice(0, 4, None), slice(1, 7, None), slice(0, 7, None),
        slice(0, 2, None), slice(2, 9, None), slice(9, 14, None),
        slice(0, 9, None), slice(2, 14, None), slice(0, 14, None)
    ])
    iterables = ([0, 1], ['alpha', 'beta', 'spinorb'],
                 ['c', 'o', 'v', 'co', 'ov', 'cov'])
    for charge, multp in [(0, 1), (1, 2)]:
        orbitals = orbitals_class(integrals, charge, multp)
        for ncore, mo_type, mo_space in it.product(*iterables):
            orbitals.ncore = ncore
            slc = orbitals.get_mo_slice(mo_type, mo_space)
            assert (slc == next(slices))


def run_mo_fock_diagonal_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    shapes = iter([
        (0,), (5,), (2,), (5,), (7,), (7,), (0,), (5,), (2,), (5,), (7,), (7,),
        (0,), (10,), (4,), (10,), (14,), (14,), (1,), (4,), (2,), (5,), (6,),
        (7,), (1,), (4,), (2,), (5,), (6,), (7,), (2,), (8,), (4,), (10,),
        (12,), (14,), (0,), (5,), (2,), (5,), (7,), (7,), (0,), (5,), (2,),
        (5,), (7,), (7,), (0,), (10,), (4,), (10,), (14,), (14,), (1,), (4,),
        (2,), (5,), (6,), (7,), (1,), (4,), (2,), (5,), (6,), (7,), (2,), (8,),
        (4,), (10,), (12,), (14,), (0,), (5,), (2,), (5,), (7,), (7,), (0,),
        (4,), (3,), (4,), (7,), (7,), (0,), (9,), (5,), (9,), (14,), (14,),
        (1,), (4,), (2,), (5,), (6,), (7,), (1,), (3,), (3,), (4,), (6,), (7,),
        (2,), (7,), (5,), (9,), (12,), (14,), (0,), (5,), (2,), (5,), (7,),
        (7,), (0,), (4,), (3,), (4,), (7,), (7,), (0,), (9,), (5,), (9,), (14,),
        (14,), (1,), (4,), (2,), (5,), (6,), (7,), (1,), (3,), (3,), (4,), (6,),
        (7,), (2,), (7,), (5,), (9,), (12,), (14,)
    ])
    norms = iter([
        0.0, 20.299946921136566, 0.9535445384200375, 20.299946921136566,
        20.322329890731346, 20.322329890731346, 0.0, 20.299946921136566,
        0.9535445384200375, 20.299946921136566, 20.322329890731346,
        20.322329890731346, 0.0, 28.708460251325288, 1.3485156185604097,
        28.708460251325288, 28.740114550492407, 28.740114550492407,
        20.242154893924866, 1.5306894692639961, 0.9535445384200375,
        20.299946921136566, 1.8034016297171234, 20.322329890731346,
        20.242154893924866, 1.5306894692639961, 0.9535445384200375,
        20.299946921136566, 1.8034016297171234, 20.322329890731346,
        28.626729982645465, 2.1647218072148178, 1.3485156185604097,
        28.708460251325288, 2.5503950431516982, 28.740114550492407, 0.0,
        20.29994692113603, 0.95354453842058662, 20.29994692113603,
        20.322329890730831, 20.322329890730831, 0.0, 20.29994692113603,
        0.95354453842058662, 20.29994692113603, 20.322329890730831,
        20.322329890730831, 0.0, 28.708460251324524, 1.3485156185611864,
        28.708460251324524, 28.740114550491679, 28.740114550491679,
        20.242154893924372, 1.5306894692633481, 0.95354453842058662,
        20.29994692113603, 1.8034016297168638, 20.322329890730831,
        20.242154893924372, 1.5306894692633464, 0.95354453842058662,
        20.29994692113603, 1.8034016297168622, 20.322329890730831,
        28.626729982644768, 2.1647218072139007, 1.3485156185611864,
        28.708460251324524, 2.55039504315133, 28.740114550491679, 0.0,
        21.205706336661116, 0.21419283417644172, 21.205706336661116,
        21.20678805965019, 21.20678805965019, 0.0, 21.13306290900756,
        0.33554919928272986, 21.13306290900756, 21.135726653730416,
        21.135726653730416, 0.0, 29.938074907261161, 0.39808521116944023,
        29.938074907261161, 29.940721450695406, 29.940721450695406,
        21.029118268369999, 2.7309643153319487, 0.21419283417644172,
        21.205706336661116, 2.7393511388336176, 21.20678805965019,
        21.003140725509979, 2.3397494696003558, 0.33554919928272986,
        21.13306290900756, 2.3636879755235096, 21.135726653730416,
        29.72130103950143, 3.596191551087295, 0.39808521116944023,
        29.938074907261161, 3.6181577504945062, 29.940721450695406, 0.0,
        21.210975962000262, 0.21400591018830339, 21.210975962000262,
        21.212055529583843, 21.212055529583843, 0.0, 21.130898192412136,
        0.33420393327111764, 21.130898192412136, 21.133540893260182,
        21.133540893260182, 0.0, 29.940279886444603, 0.39685110382788386,
        29.940279886444603, 29.942909853206462, 29.942909853206462,
        21.032772718961567, 2.7437151843912257, 0.21400591018830339,
        21.210975962000262, 2.7520486083378874, 21.212055529583843,
        21.001871289981803, 2.331579022277487, 0.33420393327111764,
        21.130898192412136, 2.3554093075595857, 21.133540893260182,
        29.722989855134962, 3.6005879728432437, 0.39685110382788386,
        29.940279886444603, 3.6223921031264448, 29.942909853206462
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'],
                  ['c', 'o', 'v', 'co', 'ov', 'cov'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type, mo_space in it.product(*iterables2):
            orbitals.ncore = ncore
            e = orbitals.get_mo_fock_diagonal(mo_type, mo_space)
            assert (e.shape == next(shapes))
            norm_ref = next(norms)
            assert (np.isclose(np.linalg.norm(e), norm_ref))


def run_mo_coefficients_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    shapes = iter([
        (7, 0), (7, 5), (7, 2), (7, 5), (7, 7), (7, 7), (7, 0), (7, 5), (7, 2),
        (7, 5), (7, 7), (7, 7), (14, 0), (14, 10), (14, 4), (14, 10), (14, 14),
        (14, 14), (7, 1), (7, 4), (7, 2), (7, 5), (7, 6), (7, 7), (7, 1),
        (7, 4), (7, 2), (7, 5), (7, 6), (7, 7), (14, 2), (14, 8), (14, 4),
        (14, 10), (14, 12), (14, 14), (7, 0), (7, 5), (7, 2), (7, 5), (7, 7),
        (7, 7), (7, 0), (7, 5), (7, 2), (7, 5), (7, 7), (7, 7), (14, 0),
        (14, 10), (14, 4), (14, 10), (14, 14), (14, 14), (7, 1), (7, 4), (7, 2),
        (7, 5), (7, 6), (7, 7), (7, 1), (7, 4), (7, 2), (7, 5), (7, 6), (7, 7),
        (14, 2), (14, 8), (14, 4), (14, 10), (14, 12), (14, 14), (7, 0), (7, 5),
        (7, 2), (7, 5), (7, 7), (7, 7), (7, 0), (7, 4), (7, 3), (7, 4), (7, 7),
        (7, 7), (14, 0), (14, 9), (14, 5), (14, 9), (14, 14), (14, 14), (7, 1),
        (7, 4), (7, 2), (7, 5), (7, 6), (7, 7), (7, 1), (7, 3), (7, 3), (7, 4),
        (7, 6), (7, 7), (14, 2), (14, 7), (14, 5), (14, 9), (14, 12), (14, 14),
        (7, 0), (7, 5), (7, 2), (7, 5), (7, 7), (7, 7), (7, 0), (7, 4), (7, 3),
        (7, 4), (7, 7), (7, 7), (14, 0), (14, 9), (14, 5), (14, 9), (14, 14),
        (14, 14), (7, 1), (7, 4), (7, 2), (7, 5), (7, 6), (7, 7), (7, 1),
        (7, 3), (7, 3), (7, 4), (7, 6), (7, 7), (14, 2), (14, 7), (14, 5),
        (14, 9), (14, 12), (14, 14)
    ])
    norms = iter([
        0.0, 2.1509557204805865, 2.2320071320697914, 2.1509557204805865,
        3.09975262707826, 3.09975262707826, 0.0, 2.1509557204805865,
        2.2320071320697914, 2.1509557204805865, 3.09975262707826,
        3.09975262707826, 0.0, 3.0419107519676367, 3.1565347574865754,
        3.0419107519676367, 4.3837122052157058, 4.3837122052157058,
        0.99453458809745499, 1.9072261178334315, 2.2320071320697914,
        2.1509557204805865, 2.9358759003330506, 3.09975262707826,
        0.99453458809745499, 1.9072261178334315, 2.2320071320697914,
        2.1509557204805865, 2.9358759003330506, 3.09975262707826,
        1.4064843027365606, 2.6972250423522253, 3.1565347574865754,
        3.0419107519676367, 4.1519555156953212, 4.3837122052157058, 0.0,
        2.1509557204795677, 2.2320071320707733, 2.1509557204795677,
        3.09975262707826, 3.09975262707826, 0.0, 2.1509557204795668,
        2.2320071320707737, 2.1509557204795668, 3.0997526270782596,
        3.0997526270782596, 0.0, 3.0419107519661961, 3.1565347574879636,
        3.0419107519661961, 4.3837122052157067, 4.3837122052157067,
        0.9945345880974491, 1.907226117832286, 2.2320071320707733,
        2.1509557204795677, 2.9358759003330523, 3.09975262707826,
        0.9945345880974491, 1.9072261178322854, 2.2320071320707737,
        2.1509557204795668, 2.9358759003330519, 3.0997526270782596,
        1.4064843027365523, 2.6972250423506048, 3.1565347574879636,
        3.0419107519661961, 4.1519555156953238, 4.3837122052157067, 0.0,
        2.1541468915132698, 2.2289274368768242, 2.1541468915132698,
        3.0997526270782596, 3.0997526270782596, 0.0, 1.9079698190003644,
        2.442973090081487, 1.9079698190003644, 3.0997526270782596,
        3.0997526270782596, 0.0, 2.8776201383143967, 3.3069978889809679,
        2.8776201383143967, 4.3837122052157058, 4.3837122052157058,
        0.99474666113694732, 1.9107139792164549, 2.2289274368768242,
        2.1541468915132698, 2.9358040515735153, 3.0997526270782596,
        0.99474666113694732, 1.6281363304014744, 2.442973090081487,
        1.9079698190003644, 2.9358040515735153, 3.0997526270782596,
        1.4067842193052242, 2.5103099053197311, 3.3069978889809679,
        2.8776201383143967, 4.1518539062051474, 4.3837122052157058, 0.0,
        2.1630680576723371, 2.2202709129644957, 2.1630680576723371,
        3.0997526270782592, 3.0997526270782592, 0.0, 1.8981355542344702,
        2.4506219142963634, 1.8981355542344702, 3.0997526270782596,
        3.0997526270782596, 0.0, 2.8778085419936077, 3.3068339380419109,
        2.8778085419936077, 4.3837122052157058, 4.3837122052157058,
        0.99448517672221726, 1.920901521526327, 2.2202709129644957,
        2.1630680576723371, 2.9358926380844297, 3.0997526270782592,
        0.99500659817928538, 1.6164406737733017, 2.4506219142963634,
        1.8981355542344702, 2.9357159635527172, 3.0997526270782596,
        1.4067831734636771, 2.5105264601734119, 3.3068339380419109,
        2.8778085419936077, 4.1518542605704036, 4.3837122052157058
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'],
                  ['c', 'o', 'v', 'co', 'ov', 'cov'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type, mo_space in it.product(*iterables2):
            orbitals.ncore = ncore
            c = orbitals.get_mo_coefficients(mo_type, mo_space)
            assert (c.shape == next(shapes))
            norm_ref = next(norms)
            assert (np.isclose(np.linalg.norm(c), norm_ref))


def run_mo_rotation_check(integrals_class, orbitals_class):
    import pytest as pt
    import numpy as np
    import scipy.linalg as spla
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    orbitals = orbitals_class(integrals)
    two = 2 * np.identity(integrals.nbf)
    sp_order = orbitals.get_spinorb_order()
    sp_two = spla.block_diag(two, two)[sp_order, :]
    iterables = (two, [two, two], sp_two)
    for rotation_matrix in iterables:
        orbitals.solve()
        norm = spla.norm(orbitals.mo_coefficients)
        orbitals.rotate(rotation_matrix)
        new_norm = spla.norm(orbitals.mo_coefficients)
        assert new_norm == 2 * norm
    with pt.raises(ValueError):
        rotation_matrix = np.ones(sp_two.shape)
        orbitals.rotate(rotation_matrix)


def run_mo_1e_kinetic_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    shapes = iter([
        (5, 5), (5, 5), (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (5, 5),
        (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (4, 4), (9, 9), (4, 4),
        (3, 3), (7, 7), (5, 5), (4, 4), (9, 9), (4, 4), (3, 3), (7, 7)
    ])
    norms = iter([
        30.879444058010858, 30.879444058011025, 43.670128585380205,
        4.498266093073906, 4.4982660930740703, 6.3615089159882565,
        30.879444058004168, 30.879444058004363, 43.67012858537089,
        4.4982660930707734, 4.498266093070713, 6.3615089159836424,
        30.936169535572919, 30.832646724875705, 43.677210185572761,
        4.6637498024308428, 3.9186835476917676, 6.0915222208101927,
        30.979616455563132, 30.783938540445401, 43.673647750032508,
        4.7321884887434562, 3.8361091324132044, 6.0917436886848897
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_1e_kinetic(mo_type, 'o,o')
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_potential_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (5, 5), (5, 5), (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (5, 5),
        (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (4, 4), (9, 9), (4, 4),
        (3, 3), (7, 7), (5, 5), (4, 4), (9, 9), (4, 4), (3, 3), (7, 7)
    ])
    norms = iter([
        65.155510544569097, 65.155510544569054, 92.143806675472803,
        18.700021522996291, 18.700021522996348, 26.445824054490245,
        65.155510544565061, 65.155510544564905, 92.14380667546709,
        18.700021522994614, 18.700021522994607, 26.445824054487844,
        65.337367922065127, 64.570065398782688, 91.860029352196307,
        19.253224336890668, 16.462331122486841, 25.331699377519836,
        65.39520742385892, 64.507649324867955, 91.857334902713831,
        19.398723766126142, 16.285996600827634, 25.328722214907962
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_1e_potential(mo_type, 'o,o')
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_dipole_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (3, 5, 5), (3, 5, 5), (3, 10, 10), (3, 4, 4), (3, 4, 4), (3, 8, 8),
        (3, 5, 5), (3, 5, 5), (3, 10, 10), (3, 4, 4), (3, 4, 4), (3, 8, 8),
        (3, 5, 5), (3, 4, 4), (3, 9, 9), (3, 4, 4), (3, 3, 3), (3, 7, 7),
        (3, 5, 5), (3, 4, 4), (3, 9, 9), (3, 4, 4), (3, 3, 3), (3, 7, 7)
    ])
    if hasattr(integrals, '_pyscf_molecule'):
        norms = iter([
            1.9082594962884663, 1.9082594962884656, 2.6986864601783584,
            1.8998982735676218, 1.8998982735676331, 2.686861905608549,
            1.9082594962894295, 1.9082594962894273, 2.698686460179712,
            1.8998982735686274, 1.8998982735686125, 2.6868619056099847,
            1.7737548834285219, 1.5141326846810295, 2.3321243906159901,
            1.7642379246354791, 1.5056412656276796, 2.319372991884463,
            1.7407542329406847, 1.5577602337013599, 2.3359884942322218,
            1.7308597270125388, 1.5496659184672159, 2.3232175648088642
        ])
    elif hasattr(integrals, '_psi4_molecule'):
        norms = iter([
            1.9082882569431838, 1.9082882569431825, 2.6987271338863628,
            1.8999511132143996, 1.8999511132143987, 2.6869366321536643,
            1.9082882569431827, 1.9082882569431829, 2.6987271338863614,
            1.8999511132144009, 1.8999511132143962, 2.6869366321536621,
            1.7737325144245595, 1.5141366424852885, 2.3321099469886697,
            1.7642412358303687, 1.5056754774187120, 2.3193977195609103,
            1.7407225575316854, 1.5577741876001878, 2.3359741954595861,
            1.7308541651322562, 1.5497093220734159, 2.3232423730375018
        ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_1e_dipole(mo_type, 'o,o')
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_fock_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (5, 5), (5, 5), (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (5, 5),
        (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (4, 4), (9, 9), (4, 4),
        (3, 3), (7, 7), (5, 5), (4, 4), (9, 9), (4, 4), (3, 3), (7, 7)
    ])
    norms = iter([
        20.299946921136566, 20.299946921136566, 28.708460251325288,
        1.5306894692639905, 1.5306894692639905, 2.1647218072148107,
        20.299946921136051, 20.299946921136055, 28.70846025132456,
        1.5306894692633697, 1.5306894692633697, 2.1647218072139323,
        21.205741163812611, 21.133097855874738, 29.938124244725049,
        2.7311207588502815, 2.3399320691633894, 3.5964291578901149,
        21.21097596200028, 21.130898192412136, 29.940279886444618,
        2.7437151843912368, 2.331579022277483, 3.6005879728432499
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_1e_fock(mo_type, 'o,o')
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_core_hamiltonian_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (5, 5), (5, 5), (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (5, 5),
        (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (4, 4), (9, 9), (4, 4),
        (3, 3), (7, 7), (5, 5), (4, 4), (9, 9), (4, 4), (3, 3), (7, 7)
    ])
    if hasattr(integrals, '_pyscf_molecule'):
        norms = iter([
            35.78911766373313, 35.789117663733109, 50.613455585417874,
            10.3484664103489, 10.34846641034895, 14.634941547277791,
            35.789117663733727, 35.789117663733741, 50.613455585418741,
            10.34846641035031, 10.348466410350357, 14.634941547279832,
            35.969825789334337, 35.16210448581905, 50.301112902069207,
            10.652105138961197, 9.1964730482843606, 14.072755963893655,
            36.005611854288176, 35.124156383542307, 50.300203246691161,
            10.70834107804119, 9.1280459103038165, 14.070884506088074
        ])
    elif hasattr(integrals, '_psi4_molecule'):
        norms = iter([
            35.788492509187442, 35.788492509187435, 50.612571483380741,
            10.347740855143851, 10.347740855143838, 14.633915457266605,
            35.788492509187414, 35.788492509187385, 50.612571483380769,
            10.347740855143847, 10.347740855143845, 14.633915457266589,
            35.969193357604652, 35.16153657709723, 50.300263669878433,
            10.651376411711716, 9.1958433113394609, 14.071792837825456,
            36.004978088129249, 35.123589972785446, 50.299354068447229,
            10.707611901771482, 9.1274167489492939, 14.069921433610016
        ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_1e_core_hamiltonian(mo_type, 'o,o',
                                                    electric_field=[0, 0, 1],
                                                    add_core_repulsion=True)
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_core_field_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (5, 5), (5, 5), (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (5, 5),
        (10, 10), (4, 4), (4, 4), (8, 8), (5, 5), (4, 4), (9, 9), (4, 4),
        (3, 3), (7, 7), (5, 5), (4, 4), (9, 9), (4, 4), (3, 3), (7, 7)
    ])
    norms = iter([
        0.0, 0.0, 0.0, 3.8960185838093566, 3.8960185838093566,
        5.5098023204808104, 0.0, 0.0, 0.0, 3.8960185838096257,
        3.8960185838096222, 5.5098023204811888, 0.0, 0.0, 0.0,
        4.0389401209706408, 3.3841013799690933, 5.2692674491522116, 0.0, 0.0,
        0.0, 4.0706002676280173, 3.3453108084212873, 5.268860497655405
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_1e_core_field(mo_type, 'o,o')
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_2e_repulsion_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (5, 5, 5, 5), (5, 5, 5, 5), (10, 10, 10, 10), (4, 4, 4, 4),
        (4, 4, 4, 4), (8, 8, 8, 8), (5, 5, 5, 5), (5, 5, 5, 5),
        (10, 10, 10, 10), (4, 4, 4, 4), (4, 4, 4, 4), (8, 8, 8, 8),
        (5, 5, 5, 5), (4, 4, 4, 4), (9, 9, 9, 9), (4, 4, 4, 4), (3, 3, 3, 3),
        (7, 7, 7, 7), (5, 5, 5, 5), (4, 4, 4, 4), (9, 9, 9, 9), (4, 4, 4, 4),
        (3, 3, 3, 3), (7, 7, 7, 7)
    ])
    norms = iter([
        6.2696641239547581, 6.2696641239547555, 12.539328247909516,
        2.8422940535191858, 2.8422940535191938, 5.6845881070383939,
        6.2696641239547359, 6.2696641239547395, 12.539328247909475,
        2.8422940535195846, 2.842294053519582, 5.684588107039164,
        6.3780048939681917, 5.8362352276244067, 12.188996204547255,
        2.9731628575794051, 2.1734195926930977, 5.121524469033937,
        6.4030784817836075, 5.8131324766771062, 12.18845750103489,
        3.0018963441591642, 2.1465213178463722, 5.1210698203967588
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_2e_repulsion(mo_type, 'o,o,o,o')
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_ao_1e_density_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (14, 14), (14, 14), (14, 14), (14, 14),
        (14, 14), (14, 14), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (14, 14), (14, 14),
        (14, 14), (14, 14), (14, 14), (14, 14), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14),
        (14, 14), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (14, 14), (14, 14), (14, 14),
        (14, 14), (14, 14), (14, 14), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (14, 14),
        (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (14, 14), (14, 14), (14, 14), (14, 14),
        (14, 14), (14, 14)
    ])
    norms = iter([
        0.0, 2.1471072366542963, 3.5264725554374987, 2.1471072366542963,
        4.2538076499375155, 4.2538076499375155, 0.0, 2.1471072366542963,
        3.5264725554374987, 2.1471072366542963, 4.2538076499375155,
        4.2538076499375155, 0.0, 3.0364681739459245, 4.9871853152362169,
        3.0364681739459245, 6.0157924702680567, 6.0157924702680567,
        0.9890990469221751, 1.8781215817791728, 3.5264725554374987,
        2.1471072366542963, 4.1223798066864976, 4.2538076499375155,
        0.9890990469221751, 1.8781215817791728, 3.5264725554374987,
        2.1471072366542963, 4.1223798066864976, 4.2538076499375155,
        1.3987972866876424, 2.6560650127377161, 4.9871853152362169,
        3.0364681739459245, 5.8299254318690226, 6.0157924702680567, 0.0,
        2.1471072366508408, 3.5264725554409364, 2.1471072366508408,
        4.2538076499375155, 4.2538076499375155, 0.0, 2.1471072366508404,
        3.5264725554409342, 2.1471072366508404, 4.2538076499375137,
        4.2538076499375137, 0.0, 3.0364681739410373, 4.9871853152410761,
        3.0364681739410373, 6.0157924702680567, 6.0157924702680567,
        0.98909904692216344, 1.8781215817753043, 3.5264725554409364,
        2.1471072366508408, 4.1223798066865038, 4.2538076499375155,
        0.98909904692216299, 1.8781215817753039, 3.5264725554409342,
        2.1471072366508404, 4.1223798066865021, 4.2538076499375137,
        1.3987972866876255, 2.6560650127322449, 4.9871853152410761,
        3.0364681739410373, 5.8299254318690297, 6.0157924702680567, 0.0,
        2.1357724012664083, 3.5156656206839156, 2.1357724012664083,
        4.2538076499375137, 4.2538076499375137, 0.0, 1.8871999761581391,
        3.6551203477394316, 1.8871999761581391, 4.2538076499375137,
        4.2538076499375137, 0.0, 2.8500960510169757, 5.0714701530145723,
        2.8500960510169757, 6.0157924702680567, 6.0157924702680567,
        0.98952091984310386, 1.86445920366094, 3.5156656206839156,
        2.1357724012664083, 4.1221892708069801, 4.2538076499375137,
        0.98952091984310386, 1.5735971918238756, 3.6551203477394316,
        1.8871999761581391, 4.1221892708069801, 4.2538076499375137,
        1.3993939050940176, 2.4397574150378092, 5.0714701530145723,
        2.8500960510169757, 5.8296559734440896, 6.0157924702680567, 0.0,
        2.1517048496893159, 3.4876400494476942, 2.1517048496893159,
        4.2538076499375146, 4.2538076499375146, 0.0, 1.8695370406572025,
        3.6815560275372317, 1.8695370406572025, 4.2538076499375137,
        4.2538076499375137, 0.0, 2.8504390024285406, 5.071241258154382,
        2.8504390024285406, 6.015792470268055, 6.015792470268055,
        0.98900076672012749, 1.8825996259943494, 3.4876400494476942,
        2.1517048496893159, 4.1224143256561732, 4.2538076499375146,
        0.99003813042041844, 1.5525903321000629, 3.6815560275372317,
        1.8695370406572025, 4.1219652412825303, 4.2538076499375137,
        1.3993920166484293, 2.440229106277656, 5.071241258154382,
        2.8504390024285406, 5.8296567071069108, 6.015792470268055
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'],
                  ['c', 'o', 'v', 'co', 'ov', 'cov'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type, mo_space in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_ao_1e_density(mo_type, mo_space)
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_ao_1e_mean_field_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (14, 14), (14, 14), (14, 14), (14, 14),
        (14, 14), (14, 14), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (14, 14), (14, 14),
        (14, 14), (14, 14), (14, 14), (14, 14), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14),
        (14, 14), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (14, 14), (14, 14), (14, 14),
        (14, 14), (14, 14), (14, 14), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (14, 14),
        (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14),
        (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7),
        (7, 7), (7, 7), (7, 7), (14, 14), (14, 14), (14, 14), (14, 14),
        (14, 14), (14, 14)
    ])
    norms = iter([
        0.0, 21.469484198565592, 7.003374862917715, 21.469484198565592,
        28.390627850074399, 28.390627850074399, 0.0, 21.469484198565592,
        7.003374862917715, 21.469484198565592, 28.390627850074399,
        28.390627850074399, 0.0, 30.362435730766318, 9.9042677135210493,
        30.362435730766318, 40.150410949862518, 40.150410949862518,
        6.9893530018851333, 14.588880507866554, 7.003374862917715,
        21.469484198565592, 21.557396723725716, 28.390627850074399,
        6.9893530018851333, 14.588880507866554, 7.003374862917715,
        21.469484198565592, 21.557396723725716, 28.390627850074399,
        9.8844378074790598, 20.63179267406537, 9.9042677135210493,
        30.362435730766318, 30.486762816150236, 40.150410949862518, 0.0,
        21.469484198566249, 7.003374862917048, 21.469484198566249,
        28.390627850074392, 28.390627850074392, 0.0, 21.469484198566246,
        7.0033748629170471, 21.469484198566246, 28.390627850074384,
        28.390627850074384, 0.0, 30.362435730767249, 9.9042677135201043,
        30.362435730767249, 40.150410949862511, 40.150410949862511,
        6.9893530018851067, 14.588880507867238, 7.003374862917048,
        21.469484198566249, 21.557396723725738, 28.390627850074392,
        6.989353001885104, 14.588880507867234, 7.0033748629170471,
        21.469484198566246, 21.557396723725738, 28.390627850074384,
        9.8844378074790207, 20.631792674066329, 9.9042677135201043,
        30.362435730767249, 30.486762816150264, 40.150410949862511, 0.0,
        19.636308590352421, 8.8619552575309868, 19.636308590352421,
        28.390627850074406, 28.390627850074406, 0.0, 20.093659536021008,
        8.4232911434694984, 20.093659536021008, 28.390627850074406,
        28.390627850074406, 0.0, 28.095191200721096, 12.226450207404142,
        28.095191200721096, 40.150410949862533, 40.150410949862533,
        6.9904567783237308, 12.737226235885704, 8.8619552575309868,
        19.636308590352421, 21.556367955398052, 28.390627850074406,
        6.9904567783237308, 13.208947689339963, 8.4232911434694984,
        20.093659536021008, 21.556367955398052, 28.390627850074406,
        9.885998783088354, 18.349747443598634, 12.226450207404142,
        28.095191200721096, 30.485307918028713, 40.150410949862533, 0.0,
        19.628195630188493, 8.867950792488557, 19.628195630188493,
        28.390627850074377, 28.390627850074377, 0.0, 20.099161130262981,
        8.4203824200116468, 20.099161130262981, 28.390627850074377,
        28.390627850074377, 0.0, 28.093457278114151, 12.228793536454836,
        28.093457278114151, 40.150410949862497, 40.150410949862497,
        6.9906840633554381, 12.730100191473223, 8.867950792488557,
        19.628195630188493, 21.556404186777964, 28.390627850074377,
        6.9902113846936489, 13.213657430740593, 8.4203824200116468,
        20.099161130262981, 21.556349108794311, 28.390627850074377,
        9.885985984025691, 18.348193196658702, 12.228793536454836,
        28.093457278114151, 30.485320210946707, 40.150410949862497
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'],
                  ['c', 'o', 'v', 'co', 'ov', 'cov'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type, mo_space in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_ao_1e_mean_field(mo_type, mo_space)
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_ao_1e_fock_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    shapes = iter([
        (7, 7), (7, 7), (14, 14), (7, 7), (7, 7), (14, 14), (7, 7), (7, 7),
        (14, 14), (7, 7), (7, 7), (14, 14)
    ])
    norms = iter([
        21.932021251272651, 21.932021251272651, 31.016561903804721,
        21.932021251272086, 21.932021251272086, 31.016561903803918,
        23.018062981238124, 22.905504625569282, 32.472963609135228,
        23.023049546734175, 22.903420413139589, 32.475028515035916
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for mo_type in ['alpha', 'beta', 'spinorb']:
            s = orbitals.get_ao_1e_fock(mo_type)
            assert (s.shape == next(shapes))
            assert (np.isclose(np.linalg.norm(s), next(norms)))


def run_hf_energy_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    energies = iter([
        -74.963343795087525, -74.963343795087511, -74.654712456959146,
        -74.656730208992286
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        energy = orbitals.get_hf_energy()
        assert (np.isclose(energy, next(energies)))


def run_core_energy_check(integrals_class, orbitals_class):
    import numpy as np
    import itertools as it
    from scfexchange import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    energies = iter([
        9.1671453128090299, -51.488166696349197, 9.1671453128090299,
        -51.488166696349211, 9.1671453128090299, -51.48869849456608,
        9.1671453128090299, -51.488650145082403
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore in [0, 1]:
            orbitals.ncore = ncore
            energy = orbitals.get_core_energy()
            assert (np.isclose(energy, next(energies)))
