def check_interface(aoints_instance):
    import inspect
    import numpy
    from scfexchange import AOIntegralsInterface

    # Check attributes
    assert (hasattr(aoints_instance, 'basis_label'))
    assert (hasattr(aoints_instance, 'nuc_labels'))
    assert (hasattr(aoints_instance, 'nuc_coords'))
    assert (hasattr(aoints_instance, 'nbf'))
    # Check attribute types
    assert (isinstance(getattr(aoints_instance, 'basis_label'), str))
    assert (isinstance(getattr(aoints_instance, 'nuc_labels'), tuple))
    assert (isinstance(getattr(aoints_instance, 'nuc_coords'),
                       numpy.ndarray))
    assert (isinstance(getattr(aoints_instance, 'nbf'), int))
    # Check methods
    assert (hasattr(aoints_instance, '__init__'))
    assert (hasattr(aoints_instance, 'overlap'))
    assert (hasattr(aoints_instance, 'potential'))
    assert (hasattr(aoints_instance, 'kinetic'))
    assert (hasattr(aoints_instance, 'dipole'))
    assert (hasattr(aoints_instance, 'electron_repulsion'))
    # Check method signature
    aoints_cls = type(aoints_instance)
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    print(inspect.signature(aoints_cls.__init__))
    assert (inspect.signature(aoints_cls.__init__) ==
            inspect.Signature(
                parameters=[
                    inspect.Parameter('self', kind),
                    inspect.Parameter('basis_label', kind),
                    inspect.Parameter('nuc_labels', kind),
                    inspect.Parameter('nuc_coords', kind),
                    inspect.Parameter('units', kind, default="bohr")
                ]
            ))
    assert (inspect.signature(aoints_cls.overlap) ==
            inspect.signature(AOIntegralsInterface.overlap))
    assert (inspect.signature(aoints_cls.kinetic) ==
            inspect.signature(AOIntegralsInterface.kinetic))
    assert (inspect.signature(aoints_cls.potential) ==
            inspect.signature(AOIntegralsInterface.potential))
    assert (inspect.signature(aoints_cls.dipole) ==
            inspect.signature(AOIntegralsInterface.dipole))
    assert (inspect.signature(aoints_cls.electron_repulsion) ==
            inspect.signature(AOIntegralsInterface.electron_repulsion))
    # Check method output
    assert (isinstance(aoints_instance.overlap(), numpy.ndarray))
    assert (isinstance(aoints_instance.kinetic(), numpy.ndarray))
    assert (isinstance(aoints_instance.potential(), numpy.ndarray))
    assert (isinstance(aoints_instance.dipole(), numpy.ndarray))
    assert (isinstance(aoints_instance.electron_repulsion(), numpy.ndarray))
    nbf = aoints_instance.nbf
    assert (aoints_instance.overlap().shape
            == (nbf, nbf))
    assert (aoints_instance.kinetic().shape
            == (nbf, nbf))
    assert (aoints_instance.potential().shape
            == (nbf, nbf))
    assert (aoints_instance.dipole().shape
            == (3, nbf, nbf))
    assert (aoints_instance.electron_repulsion().shape
            == (nbf, nbf, nbf, nbf))


def run_test__interface(aoints_cls):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = aoints_cls("sto-3g", nuc_labels, nuc_coords)
    check_interface(aoints)


def run_test__overlap(aoints_cls):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = aoints_cls("sto-3g", nuc_labels, nuc_coords)
    s = aoints.overlap()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 2.95961615642))
    assert (hasattr(aoints, '_ao_1e_overlap'))
    aoints._ao_1e_overlap[:, :] = numpy.zeros(s.shape)
    s = aoints.overlap()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.overlap(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 2.95961615642))


def run_test__kinetic(aoints_cls):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = aoints_cls("sto-3g", nuc_labels, nuc_coords)
    s = aoints.kinetic()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 29.3703412473))
    assert (hasattr(aoints, '_ao_1e_kinetic'))
    aoints._ao_1e_kinetic[:, :] = numpy.zeros(s.shape)
    s = aoints.kinetic()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.kinetic(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 29.3703412473))


def run_test__potential(aoints_cls):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = aoints_cls("sto-3g", nuc_labels, nuc_coords)
    s = aoints.potential()
    assert (s.shape == (7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 67.1181391119))
    assert (hasattr(aoints, '_ao_1e_potential'))
    aoints._ao_1e_potential[:, :] = numpy.zeros(s.shape)
    s = aoints.potential()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.potential(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 67.1181391119))


def run_test__dipole(aoints_cls):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = aoints_cls("sto-3g", nuc_labels, nuc_coords)
    if hasattr(aoints, '_pyscf_molecule'):
        norm = 3.36216114637
    elif hasattr(aoints, '_psi4_molecule'):
        norm = 3.36241054669
    s = aoints.dipole()
    assert (s.shape == (3, 7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))
    assert (hasattr(aoints, '_ao_1e_dipole'))
    aoints._ao_1e_dipole[:, :, :] = numpy.zeros(s.shape)
    s = aoints.dipole()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.dipole(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))


def run_test__electron_repulsion(aoints_cls):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = aoints_cls("sto-3g", nuc_labels, nuc_coords)
    s = aoints.electron_repulsion()
    assert (s.shape == (7, 7, 7, 7))
    assert (numpy.isclose(numpy.linalg.norm(s), 8.15009229415))
    assert (hasattr(aoints, '_ao_2e_repulsion'))
    aoints._ao_2e_repulsion[:, :, :, :] = numpy.zeros(s.shape)
    s = aoints.electron_repulsion()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.electron_repulsion(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), 8.15009229415))


def run_test__core_hamiltonian(aoints_cls):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                               [0.0000000000, -1.4343021349, 0.9864370414],
                               [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = aoints_cls("sto-3g", nuc_labels, nuc_coords)
    s = aoints.core_hamiltonian(electric_field=[0, 0, 1])
    assert (s.shape == (7, 7))
    if hasattr(aoints, '_pyscf_molecule'):
        norm = 39.7722869121
    elif hasattr(aoints, '_psi4_molecule'):
        norm = 39.771471762
    assert (numpy.isclose(numpy.linalg.norm(s), norm))
