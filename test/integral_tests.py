def check_interface(integrals_instance):
    import inspect
    import numpy
    from scfexchange import IntegralsInterface

    # Check attributes
    assert (hasattr(integrals_instance, 'basis_label'))
    assert (hasattr(integrals_instance, 'nuc_labels'))
    assert (hasattr(integrals_instance, 'nuc_coords'))
    assert (hasattr(integrals_instance, 'nbf'))
    # Check attribute types
    assert (isinstance(getattr(integrals_instance, 'basis_label'), str))
    assert (isinstance(getattr(integrals_instance, 'nuc_labels'), tuple))
    assert (isinstance(getattr(integrals_instance, 'nuc_coords'),
                       numpy.ndarray))
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
    print(inspect.signature(integrals_class.__init__))
    assert (inspect.signature(integrals_class.__init__) ==
            inspect.Signature(
                parameters=[
                    inspect.Parameter('self', kind),
                    inspect.Parameter('basis_label', kind),
                    inspect.Parameter('nuc_labels', kind),
                    inspect.Parameter('nuc_coords', kind),
                    inspect.Parameter('units', kind, default="bohr")
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


def run_test__interface(integrals_class):
    import numpy

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    integrals = integrals_class("sto-3g", labels, coordinates)
    check_interface(integrals)


def run_test__get_ao_1e_overlap(integrals_class):
    import numpy

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    integrals = integrals_class("sto-3g", labels, coordinates)
    s = integrals.get_ao_1e_overlap()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 2.95961615642))
    assert (hasattr(integrals, '_ao_1e_overlap'))
    integrals._ao_1e_overlap[:, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_1e_overlap()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_1e_overlap(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 2.95961615642))


def run_test__get_ao_1e_kinetic(integrals_class):
    import numpy

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    integrals = integrals_class("sto-3g", labels, coordinates)
    s = integrals.get_ao_1e_kinetic()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 29.3703412473))
    assert (hasattr(integrals, '_ao_1e_kinetic'))
    integrals._ao_1e_kinetic[:, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_1e_kinetic()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_1e_kinetic(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 29.3703412473))


def run_test__get_ao_1e_potential(integrals_class):
    import numpy

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    integrals = integrals_class("sto-3g", labels, coordinates)
    s = integrals.get_ao_1e_potential()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 67.1181391119))
    assert (hasattr(integrals, '_ao_1e_potential'))
    integrals._ao_1e_potential[:, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_1e_potential()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_1e_potential(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 67.1181391119))


def run_test__get_ao_1e_dipole(integrals_class):
    import numpy

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    integrals = integrals_class("sto-3g", labels, coordinates)
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


def run_test__get_ao_1e_core_hamiltonian(integrals_class):
    import numpy

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    integrals = integrals_class("sto-3g", labels, coordinates)
    s = integrals.get_ao_1e_core_hamiltonian(electric_field=[0, 0, 1])
    assert (s.shape == (7, 7))
    if hasattr(integrals, '_pyscf_molecule'):
        norm = 39.7722869121
    elif hasattr(integrals, '_psi4_molecule'):
        norm = 39.771471762
    assert (numpy.isclose(numpy.linalg.norm(s), norm))


def run_test__get_ao_2e_repulsion(integrals_class):
    import numpy

    labels = ("O", "H", "H")
    coordinates = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    integrals = integrals_class("sto-3g", labels, coordinates)
    s = integrals.get_ao_2e_repulsion()
    assert (s.shape == (7, 7, 7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 8.15009229415))
    assert (hasattr(integrals, '_ao_2e_repulsion'))
    integrals._ao_2e_repulsion[:, :, :, :] = numpy.zeros(s.shape)
    s = integrals.get_ao_2e_repulsion()
    assert (numpy.linalg.norm(s) == 0.0)
    s = integrals.get_ao_2e_repulsion(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 8.15009229415))
