def check_interface(aoints_instance):
    import inspect
    import numpy
    from scfexchange.ao import AOIntegralsInterface

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


def run_test__interface(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    check_interface(aoints)


def run_test__overlap(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    norm = 2.95961615642
    spinorb_norm = numpy.sqrt(2 * norm ** 2)
    nbf = aoints.nbf

    # Test default
    s = aoints.overlap()
    assert (s.shape == (nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    assert (hasattr(aoints, '_overlap'))
    aoints._overlap[:] = numpy.zeros((nbf, nbf))
    s = aoints.overlap()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.overlap(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.overlap(use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._overlap[:] = numpy.zeros((nbf, nbf))
    s = aoints.overlap(use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.overlap(use_spinorbs=True, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))


def run_test__kinetic(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    norm = 29.3703412473
    spinorb_norm = numpy.sqrt(2 * norm ** 2)
    nbf = aoints.nbf

    # Test default
    s = aoints.kinetic()
    assert (s.shape == (nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    assert (hasattr(aoints, '_kinetic'))
    aoints._kinetic[:] = numpy.zeros((nbf, nbf))
    s = aoints.kinetic()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.kinetic(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.kinetic(use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._kinetic[:] = numpy.zeros((nbf, nbf))
    s = aoints.kinetic(use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.kinetic(use_spinorbs=True, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))


def run_test__potential(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    norm = 67.1181391119
    spinorb_norm = numpy.sqrt(2 * norm ** 2)
    nbf = aoints.nbf

    # Test default
    s = aoints.potential()
    assert (s.shape == (nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    assert (hasattr(aoints, '_potential'))
    aoints._potential[:] = numpy.zeros((nbf, nbf))
    s = aoints.potential()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.potential(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.potential(use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._potential[:] = numpy.zeros((nbf, nbf))
    s = aoints.potential(use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.potential(use_spinorbs=True, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))


def run_test__dipole(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    if hasattr(aoints, '_pyscf_molecule'):
        norm = 3.36216114637
    elif hasattr(aoints, '_psi4_molecule'):
        norm = 3.36241054669
    spinorb_norm = numpy.sqrt(2 * norm ** 2)
    nbf = aoints.nbf

    # Test default
    s = aoints.dipole()
    assert (s.shape == (3, nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    assert (hasattr(aoints, '_dipole'))
    aoints._dipole[:] = numpy.zeros((3, nbf, nbf))
    s = aoints.dipole()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.dipole(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.dipole(use_spinorbs=True)
    assert (s.shape == (3, 2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._dipole[:] = numpy.zeros((3, nbf, nbf))
    s = aoints.dipole(use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.dipole(use_spinorbs=True, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))


def run_test__electron_repulsion(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    norm = 8.15009229415
    asym_norm = 7.0473787105
    spinorb_norm = numpy.sqrt(4 * norm ** 2)
    asym_spinorb_norm = numpy.sqrt(2 * asym_norm ** 2 + 4 * norm ** 2)
    nbf = aoints.nbf

    # Test default
    s = aoints.electron_repulsion()
    assert (s.shape == (nbf, nbf, nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    assert (hasattr(aoints, '_electron_repulsion'))
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.electron_repulsion()
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.electron_repulsion(recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.electron_repulsion(use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf, 2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.electron_repulsion(use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.electron_repulsion(use_spinorbs=True, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test antisymmetrize
    s = aoints.electron_repulsion(antisymmetrize=True)
    assert (s.shape == (nbf, nbf, nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), asym_norm))

    # Test antisymmetrize recompute
    assert (hasattr(aoints, '_electron_repulsion'))
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.electron_repulsion(antisymmetrize=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.electron_repulsion(antisymmetrize=True, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), asym_norm))

    # Test antisymmetrize spin-orbital
    s = aoints.electron_repulsion(antisymmetrize=True, use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf, 2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), asym_spinorb_norm))

    # Test antisymmetrize spin-orbital recompute
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.electron_repulsion(antisymmetrize=True, use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.electron_repulsion(antisymmetrize=True, use_spinorbs=True,
                                  recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), asym_spinorb_norm))


def run_test__core_hamiltonian(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    if hasattr(aoints, '_pyscf_molecule'):
        norm = 39.7722869121
    elif hasattr(aoints, '_psi4_molecule'):
        norm = 39.771471762
    spinorb_norm = numpy.sqrt(2 * norm ** 2)
    nbf = aoints.nbf

    # Test default
    s = aoints.core_hamiltonian(electric_field=[0., 0., 1.])
    assert (s.shape == (nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    aoints._kinetic[:] = numpy.zeros((nbf, nbf))
    aoints._potential[:] = numpy.zeros((nbf, nbf))
    aoints._dipole[:] = numpy.zeros((nbf, nbf))
    s = aoints.core_hamiltonian(electric_field=[0., 0., 1.])
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.core_hamiltonian(electric_field=[0., 0., 1.], recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.core_hamiltonian(electric_field=[0., 0., 1.], use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._kinetic[:] = numpy.zeros((nbf, nbf))
    aoints._potential[:] = numpy.zeros((nbf, nbf))
    aoints._dipole[:] = numpy.zeros((nbf, nbf))
    s = aoints.core_hamiltonian(electric_field=[0., 0., 1.], use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.core_hamiltonian(electric_field=[0., 0., 1.], use_spinorbs=True,
                                recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))


def run_test__mean_field(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    norm = 28.0934572781
    spinorb_norm = norm
    nbf = aoints.nbf
    mo_coeffs = interface.hf_mo_coefficients(aoints, charge=1, multp=2,
                                             restricted=False)
    alpha_coeffs = mo_coeffs[0, :, :5]
    beta_coeffs = mo_coeffs[1, :, :4]

    # Test default
    s = aoints.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs)
    assert (s.shape == (2, nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs,
                          use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs,
                          use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.mean_field(alpha_coeffs, beta_coeffs=beta_coeffs,
                          use_spinorbs=True, recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))


def run_test__fock(interface):
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = interface.AOIntegrals("sto-3g", nuc_labels, nuc_coords)

    if hasattr(aoints, '_pyscf_molecule'):
        norm = 32.5544682349
    elif hasattr(aoints, '_psi4_molecule'):
        norm = 32.5537247982
    spinorb_norm = norm
    nbf = aoints.nbf
    mo_coeffs = interface.hf_mo_coefficients(aoints, charge=1, multp=2,
                                             restricted=False)
    alpha_coeffs = mo_coeffs[0, :, :5]
    beta_coeffs = mo_coeffs[1, :, :4]

    # Test default
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=[0., 0., 1.])
    assert (s.shape == (2, nbf, nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test recompute
    aoints._kinetic[:] = numpy.zeros((nbf, nbf))
    aoints._potential[:] = numpy.zeros((nbf, nbf))
    aoints._dipole[:] = numpy.zeros((nbf, nbf))
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=[0., 0., 1.])
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=[0., 0., 1.], recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), norm))

    # Test spin-orbital
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=[0., 0., 1.], use_spinorbs=True)
    assert (s.shape == (2 * nbf, 2 * nbf))
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))

    # Test spin-orbital recompute
    aoints._kinetic[:] = numpy.zeros((nbf, nbf))
    aoints._potential[:] = numpy.zeros((nbf, nbf))
    aoints._dipole[:] = numpy.zeros((nbf, nbf))
    aoints._electron_repulsion[:] = numpy.zeros((nbf, nbf, nbf, nbf))
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=[0., 0., 1.], use_spinorbs=True)
    assert (numpy.linalg.norm(s) == 0.0)
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=[0., 0., 1.], use_spinorbs=True,
                    recompute=True)
    assert (numpy.isclose(numpy.linalg.norm(s), spinorb_norm))
