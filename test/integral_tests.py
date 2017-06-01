def check_interface(integrals_instance):
    import inspect
    import numpy
    from scfexchange import NuclearFramework, IntegralsInterface

    # Check attributes
    assert (hasattr(integrals_instance, 'nuclei'))
    assert (hasattr(integrals_instance, 'basis_label'))
    assert (hasattr(integrals_instance, 'nbf'))
    # Check attribute types
    assert (isinstance(getattr(integrals_instance, 'nuclei'), NuclearFramework))
    assert (isinstance(getattr(integrals_instance, 'basis_label'), str))
    assert (isinstance(getattr(integrals_instance, 'nbf'), int))
    # Check methods
    assert (hasattr(integrals_instance, '__init__'))
    assert (hasattr(integrals_instance, 'get_ao_1e_overlap'))
    assert (hasattr(integrals_instance, 'get_ao_1e_potential'))
    assert (hasattr(integrals_instance, 'get_ao_1e_kinetic'))
    assert (hasattr(integrals_instance, 'get_ao_1e_dipole'))
    assert (hasattr(integrals_instance, 'get_ao_2e_repulsion'))
    # Check method signature
    integrals_class = type(integrals_instance)
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert (inspect.signature(integrals_class.__init__) ==
            inspect.Signature(
                parameters=[
                    inspect.Parameter('self', kind),
                    inspect.Parameter('nuclei', kind),
                    inspect.Parameter('basis_label', kind)
                ]
            ))
    assert (inspect.signature(integrals_class.get_ao_1e_overlap) ==
            inspect.signature(IntegralsInterface.get_ao_1e_overlap))
    assert (inspect.signature(integrals_class.get_ao_1e_kinetic) ==
            inspect.signature(IntegralsInterface.get_ao_1e_kinetic))
    assert (inspect.signature(integrals_class.get_ao_1e_potential) ==
            inspect.signature(IntegralsInterface.get_ao_1e_potential))
    assert (inspect.signature(integrals_class.get_ao_1e_dipole) ==
            inspect.signature(IntegralsInterface.get_ao_1e_dipole))
    assert (inspect.signature(integrals_class.get_ao_2e_repulsion) ==
            inspect.signature(IntegralsInterface.get_ao_2e_repulsion))
    # Check method output
    assert (isinstance(integrals_instance.get_ao_1e_overlap(), numpy.ndarray))
    assert (isinstance(integrals_instance.get_ao_1e_kinetic(), numpy.ndarray))
    assert (isinstance(integrals_instance.get_ao_1e_potential(), numpy.ndarray))
    assert (isinstance(integrals_instance.get_ao_1e_dipole(), numpy.ndarray))
    assert (isinstance(integrals_instance.get_ao_2e_repulsion(), numpy.ndarray))
    nbf = integrals_instance.nbf
    assert (integrals_instance.get_ao_1e_overlap().shape
            == (nbf, nbf))
    assert (integrals_instance.get_ao_1e_kinetic().shape
            == (nbf, nbf))
    assert (integrals_instance.get_ao_1e_potential().shape
            == (nbf, nbf))
    assert (integrals_instance.get_ao_1e_dipole().shape
            == (3, nbf, nbf))
    assert (integrals_instance.get_ao_2e_repulsion().shape
            == (nbf, nbf, nbf, nbf))


def run_interface_check(integrals_class):
    import numpy
    from scfexchange import NuclearFramework
    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "cc-pvdz")
    # Test the integrals interface
    check_interface(integrals)


def run_ao_1e_overlap_check(integrals_class):
    import numpy
    from scfexchange import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    s = integrals.get_ao_1e_overlap()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 2.95961615642))
    assert (hasattr(integrals, '_ao_1e_overlap'))
    integrals._ao_1e_overlap[:, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_1e_overlap()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_1e_overlap(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 2.95961615642))


def run_ao_1e_kinetic_check(integrals_class):
    import numpy
    from scfexchange import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    s = integrals.get_ao_1e_kinetic()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 29.3703412473))
    assert (hasattr(integrals, '_ao_1e_kinetic'))
    integrals._ao_1e_kinetic[:, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_1e_kinetic()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_1e_kinetic(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 29.3703412473))


def run_ao_1e_potential_check(integrals_class):
    import numpy
    from scfexchange import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    s = integrals.get_ao_1e_potential()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 67.1181391119))
    assert (hasattr(integrals, '_ao_1e_potential'))
    integrals._ao_1e_potential[:, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_1e_potential()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_1e_potential(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 67.1181391119))


def run_ao_1e_dipole_check(integrals_class):
    import numpy
    from scfexchange import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    if hasattr(integrals, '_pyscf_molecule'):
        norm = 3.36216114637
    elif hasattr(integrals, '_psi4_molecule'):
        norm = 3.36241054669
    s = integrals.get_ao_1e_dipole()
    assert (s.shape == (3, 7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))
    assert (hasattr(integrals, '_ao_1e_dipole'))
    integrals._ao_1e_dipole[:, :, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_1e_dipole()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_1e_dipole(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))


def run_ao_1e_core_hamiltonian_check(integrals_class):
    import numpy
    from scfexchange import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    s = integrals.get_ao_1e_core_hamiltonian(electric_field=[0, 0, 1])
    assert (s.shape == (7, 7))
    if hasattr(integrals, '_pyscf_molecule'):
        norm = 39.7722869121
    elif hasattr(integrals, '_psi4_molecule'):
        norm = 39.771471762
    assert (numpy.isclose(numpy.linalg.norm(s), norm))


def run_ao_2e_repulsion_check(integrals_class):
    import numpy
    from scfexchange import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    s = integrals.get_ao_2e_repulsion()
    assert (s.shape == (7, 7, 7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 8.15009229415))
    assert (hasattr(integrals, '_ao_2e_repulsion'))
    integrals._ao_2e_repulsion[:, :, :, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_2e_repulsion()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_2e_repulsion(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 8.15009229415))
