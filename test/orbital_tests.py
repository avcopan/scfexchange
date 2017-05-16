import inspect
import numpy as np
import scipy.linalg as spla
import itertools as it
from scfexchange.integrals import IntegralsInterface
from scfexchange.orbitals import OrbitalsInterface
from scfexchange.molecule import NuclearFramework, Molecule


def check_interface(orbitals_instance):
    # Check class documentation
    orbitals_class = type(orbitals_instance)
    assert(orbitals_class.__doc__ == OrbitalsInterface.__doc__)
    # Check attributes
    assert(hasattr(orbitals_instance, 'integrals'))
    assert(hasattr(orbitals_instance, 'molecule'))
    assert(hasattr(orbitals_instance, 'options'))
    assert(hasattr(orbitals_instance, 'mo_energies'))
    assert(hasattr(orbitals_instance, 'mo_coefficients'))
    assert(hasattr(orbitals_instance, 'hf_energy'))
    assert(hasattr(orbitals_instance, 'n_frozen_orbitals'))
    # Check attribute types
    assert(isinstance(getattr(orbitals_instance, 'integrals'),
                      IntegralsInterface))
    assert(isinstance(orbitals_instance.molecule, Molecule))
    assert(isinstance(orbitals_instance.options, dict))
    assert(isinstance(orbitals_instance.mo_energies, np.ndarray))
    assert(isinstance(orbitals_instance.mo_coefficients, np.ndarray))
    assert(isinstance(orbitals_instance.hf_energy, float))
    assert(isinstance(orbitals_instance.n_frozen_orbitals, int))
    # Check array shapes
    norb = orbitals_instance.get_mo_count('alpha', 'cov')
    assert(orbitals_instance.mo_energies.shape == (2, norb))
    assert(orbitals_instance.mo_coefficients.shape == (2, norb, norb))
    # Check 'options' attribute
    assert(set(orbitals_instance.options.keys()) ==
           {'restrict_spin', 'n_iterations', 'e_threshold', 'd_threshold'})
    assert(isinstance(orbitals_instance.options['restrict_spin'], bool))
    assert(isinstance(orbitals_instance.options['n_iterations'], int))
    assert(isinstance(orbitals_instance.options['e_threshold'], float))
    assert(isinstance(orbitals_instance.options['d_threshold'], float))
    # Check methods
    assert(hasattr(orbitals_instance, '__init__'))
    # Check method signature
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert(inspect.signature(orbitals_class.__init__) ==
           inspect.Signature(
                parameters=[
                    inspect.Parameter('self', kind),
                    inspect.Parameter('integrals', kind),
                    inspect.Parameter('charge', kind, default=0),
                    inspect.Parameter('multiplicity', kind, default=1),
                    inspect.Parameter('restrict_spin', kind, default=True),
                    inspect.Parameter('n_iterations', kind, default=40),
                    inspect.Parameter('e_threshold', kind, default=1e-12),
                    inspect.Parameter('d_threshold', kind, default=1e-6),
                    inspect.Parameter('n_frozen_orbitals', kind, default=0)
                ]
           ))
    # Check method output
    assert(isinstance(orbitals_instance.get_mo_1e_core_field(), np.ndarray))
    assert(isinstance(orbitals_instance.get_mo_1e_kinetic(), np.ndarray))
    assert(isinstance(orbitals_instance.get_mo_1e_potential(), np.ndarray))
    assert(isinstance(orbitals_instance.get_mo_1e_dipole(), np.ndarray))
    assert(isinstance(orbitals_instance.get_mo_2e_repulsion(), np.ndarray))
    norb = orbitals_instance.get_mo_count()
    assert(orbitals_instance.get_mo_1e_core_field().shape
           == (norb, norb))
    assert(orbitals_instance.get_mo_1e_kinetic().shape
           == (norb, norb))
    assert(orbitals_instance.get_mo_1e_potential().shape
           == (norb, norb))
    assert(orbitals_instance.get_mo_1e_dipole().shape
           == (3, norb, norb))
    assert(orbitals_instance.get_mo_2e_repulsion().shape
           == (norb, norb, norb, norb))


def check_mo_slicing(orbitals_instance, mo_type, core_offset, occ_offset):
    nbf = orbitals_instance.integrals.nbf
    norb = 2 * nbf if mo_type is 'spinorb' else nbf
    assert(orbitals_instance.get_mo_slice(mo_type, mo_space='c') ==
           slice(0, core_offset, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_space='o') ==
           slice(core_offset, occ_offset, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_space='v') ==
           slice(occ_offset, norb, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_space='co') ==
           slice(0, occ_offset, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_space='ov') ==
           slice(core_offset, norb, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_space='cov') ==
           slice(0, norb, None))


def check_core_energy(orbitals_instance):
    h = orbitals_instance.get_mo_1e_core_hamiltonian(mo_type='spinorb',
                                                     mo_block='o,o',
                                                     add_core_repulsion=True)
    g = orbitals_instance.get_mo_2e_repulsion(mo_type='spinorb',
                                              mo_block='o,o,o,o',
                                              antisymmetrize=True)
    # Core energy
    e_c = orbitals_instance.get_core_energy()
    # Valence energy
    e_v = np.trace(h) + 1. / 2 * np.einsum("ijij", g)
    assert(np.isclose(e_c + e_v, orbitals_instance.hf_energy))


def check_mp2_energy(orbitals_instance, correlation_energy):
    e = orbitals_instance.get_mo_energies(mo_type='spinorb', mo_space='ov')
    g = orbitals_instance.get_mo_2e_repulsion(mo_type='spinorb',
                                              mo_block='o,o,v,v',
                                              antisymmetrize=True)
    nspocc = orbitals_instance.get_mo_count('spinorb', 'o')
    o = slice(None, nspocc)
    v = slice(nspocc, None)
    x = np.newaxis
    e_corr = (
        1. / 4 * np.sum(g * g / (
            e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))
    )
    assert(np.isclose(e_corr, correlation_energy))


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
    for (charge, multp), restr, nfrz in it.product(*vars):
        orbitals = orbitals_class(integrals, charge, multp,
                                  restrict_spin=restr, n_frozen_orbitals=nfrz)
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
        for nfrz, mo_type, mo_space in it.product(*iterables):
            orbitals.n_frozen_orbitals = nfrz
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
        for nfrz, mo_type, mo_space in it.product(*iterables):
            orbitals.n_frozen_orbitals = nfrz
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
                              restrict_spin=False, n_frozen_orbitals=1)
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


def run_mo_1e_core_field_check(integrals_class, orbitals_class):
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
        0.0, 0.0, 0.0, 3.8960185838093433, 3.8960185838093615,
        5.5098023204807953, 0.0, 0.0, 0.0, 3.8960185838096142,
        3.8960185838096231, 5.5098023204811906, 0.0, 0.0, 0.0,
        4.0389401209706373, 3.3841013799690915, 5.2692674491522125, 0.0, 0.0,
        0.0, 4.0706002676019741, 3.3453108084444305, 5.268860497655651
    ])
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        for nfrz, mo_type in it.product(*iterables2):
            orbitals.n_frozen_orbitals = nfrz
            s = orbitals.get_mo_1e_core_field(mo_type, 'o,o')
            assert(s.shape == next(shapes))
            assert(np.isclose(np.linalg.norm(s), next(norms)))


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
        for nfrz, mo_type in it.product(*iterables2):
            orbitals.n_frozen_orbitals = nfrz
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
        for nfrz, mo_type in it.product(*iterables2):
            orbitals.n_frozen_orbitals = nfrz
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
        for nfrz, mo_type in it.product(*iterables2):
            orbitals.n_frozen_orbitals = nfrz
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
        for nfrz, mo_type in it.product(*iterables2):
            orbitals.n_frozen_orbitals = nfrz
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
        for nfrz, mo_type in it.product(*iterables2):
            orbitals.n_frozen_orbitals = nfrz
            s = orbitals.get_mo_2e_repulsion(mo_type, 'o,o,o,o')
            assert(s.shape == next(shapes))
            assert(np.isclose(np.linalg.norm(s), next(norms)))


def run_core_energy_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Build orbitals
    iterables1 = ([(0, 1), (1, 2)], [True, False])
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = orbitals_class(integrals, charge, multp, restrict_spin=restr)
        for nfrz in [0, 1]:
            orbitals.n_frozen_orbitals = nfrz
            check_core_energy(orbitals)


def run_mp2_energy_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Build orbitals
    orbitals1 = orbitals_class(integrals, charge=0, multiplicity=1,
                               restrict_spin=False, n_frozen_orbitals=0)
    orbitals2 = orbitals_class(integrals, charge=0, multiplicity=1,
                               restrict_spin=False, n_frozen_orbitals=1)
    orbitals3 = orbitals_class(integrals, charge=0, multiplicity=1,
                               restrict_spin=True, n_frozen_orbitals=0)
    orbitals4 = orbitals_class(integrals, charge=0, multiplicity=1,
                               restrict_spin=True, n_frozen_orbitals=1)
    orbitals5 = orbitals_class(integrals, charge=1, multiplicity=2,
                               restrict_spin=False, n_frozen_orbitals=0)
    orbitals6 = orbitals_class(integrals, charge=1, multiplicity=2,
                               restrict_spin=False, n_frozen_orbitals=1)
    # Test the mp2 energy
    check_mp2_energy(orbitals1, -0.0357395812441)
    check_mp2_energy(orbitals2, -0.0356400911261)
    check_mp2_energy(orbitals3, -0.0357395812441)
    check_mp2_energy(orbitals4, -0.0356400911261)
    check_mp2_energy(orbitals5, -0.0277892244936)
    check_mp2_energy(orbitals6, -0.0277117354355)


def run_dipole_moment_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    integrals = integrals_class(nuclei, "sto-3g")
    orbitals1 = orbitals_class(integrals, charge=0, multiplicity=1,
                               restrict_spin=False)
    orbitals2 = orbitals_class(integrals, charge=1, multiplicity=2,
                               restrict_spin=False)
    # PySCF and Psi4 give different answers for the dipole, so I'm testing them
    # separately for now.
    if hasattr(integrals, '_pyscf_molecule'):
        dipole_ref_1 = [0.0, 0.0, -0.29749417]
        dipole_ref_2 = [0.0, 0.0, +0.09273273]
    elif hasattr(integrals, '_psi4_molecule'):
        dipole_ref_1 = [0.0, 0.0, -0.30116127]
        dipole_ref_2 = [0.0, 0.0,  0.08943234]
    dipole_ints_1 = orbitals1.get_mo_1e_dipole('spinorb', 'o,o')
    dipole_ints_2 = orbitals2.get_mo_1e_dipole('spinorb', 'o,o')
    mu_1 = [np.trace(component) for component in dipole_ints_1]
    mu_2 = [np.trace(component) for component in dipole_ints_2]
    assert(np.allclose(mu_1, dipole_ref_1))
    assert(np.allclose(mu_2, dipole_ref_2))
