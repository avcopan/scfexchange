import inspect
import numpy as np
import scipy.linalg as spla
import itertools as it
from scfexchange.integrals import IntegralsInterface
from scfexchange.orbitals import OrbitalsInterface
from scfexchange.molecule import NuclearFramework, Molecule


def check_interface(orbitals_instance):
    # Check attributes
    assert(hasattr(orbitals_instance, 'integrals'))
    assert(hasattr(orbitals_instance, 'molecule'))
    assert(hasattr(orbitals_instance, 'mo_coefficients'))
    assert(hasattr(orbitals_instance, 'mo_energies'))
    assert(hasattr(orbitals_instance, 'spin_is_restricted'))
    assert(hasattr(orbitals_instance, 'ncore'))
    # Check attribute types
    assert(isinstance(getattr(orbitals_instance, 'integrals'),
                      IntegralsInterface))
    assert(isinstance(orbitals_instance.molecule, Molecule))
    assert(isinstance(orbitals_instance.mo_coefficients, np.ndarray))
    assert(isinstance(orbitals_instance.mo_energies, np.ndarray))
    assert(isinstance(orbitals_instance.spin_is_restricted, bool))
    assert(isinstance(orbitals_instance.ncore, int))
    # Check array shapes
    norb = orbitals_instance.get_mo_count(mo_space='cov')
    assert(orbitals_instance.mo_energies.shape == (2, norb))
    assert(orbitals_instance.mo_coefficients.shape == (2, norb, norb))


def run_interface_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Build orbitals
    vars = ([(0, 1), (1, 2)], [True, False], [0, 1])
    for (charge, multp), restr, ncore in it.product(*vars):
        orbitals = orbitals_class(integrals, charge, multp,
                                  restrict_spin=restr, ncore=ncore)
        check_interface(orbitals)


def run_mo_counting_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
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
            assert(count == next(counts))


def run_mo_slicing_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
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
            assert(slc == next(slices))


def run_mo_coefficient_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Build orbitals
    orbitals = orbitals_class(integrals, charge=1, multiplicity=2,
                              restrict_spin=False, ncore=1)
    orbitals.solve()
    i = np.identity(integrals.nbf)
    norm_a = 2.93589263808
    norm_b = 2.93571596355
    norm_s = np.sqrt(norm_a ** 2 + norm_b ** 2)
    c = orbitals.get_mo_coefficients(mo_type='alpha', mo_space='ov',
                                     transformation=None)
    assert(np.isclose(np.linalg.norm(c), norm_a))
    c = orbitals.get_mo_coefficients(mo_type='alpha', mo_space='ov',
                                     transformation=i)
    assert(np.isclose(np.linalg.norm(c), norm_a))
    c = orbitals.get_mo_coefficients(mo_type='alpha', mo_space='ov',
                                     transformation=(i, i))
    assert(np.isclose(np.linalg.norm(c), norm_a))
    c = orbitals.get_mo_coefficients(mo_type='beta', mo_space='ov',
                                     transformation=None)
    assert(np.isclose(np.linalg.norm(c), norm_b))
    c = orbitals.get_mo_coefficients(mo_type='beta', mo_space='ov',
                                     transformation=i)
    assert(np.isclose(np.linalg.norm(c), norm_b))
    c = orbitals.get_mo_coefficients(mo_type='beta', mo_space='ov',
                                     transformation=(i, i))
    assert(np.isclose(np.linalg.norm(c), norm_b))
    c = orbitals.get_mo_coefficients(mo_type='spinorb', mo_space='ov',
                                     transformation=None)
    assert(np.isclose(np.linalg.norm(c), norm_s))
    c = orbitals.get_mo_coefficients(mo_type='spinorb', mo_space='ov',
                                     transformation=spla.block_diag(i, i))
    assert(np.isclose(np.linalg.norm(c), norm_s))


def run_mo_1e_kinetic_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
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
            assert(s.shape == next(shapes))
            assert(np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_potential_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
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
            assert(s.shape == next(shapes))
            assert(np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_dipole_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
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
            assert(s.shape == next(shapes))
            assert(np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_1e_core_hamiltonian_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
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
            assert(s.shape == next(shapes))
            assert(np.isclose(np.linalg.norm(s), next(norms)))


def run_mo_2e_repulsion_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
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
            assert(s.shape == next(shapes))
            assert(np.isclose(np.linalg.norm(s), next(norms)))
