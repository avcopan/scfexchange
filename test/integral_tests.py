import inspect
import numpy as np
from scfexchange.integrals import IntegralsInterface
from scfexchange.molecule import NuclearFramework


def check_interface(integrals_instance):
    # Check class documentation
    integrals_class = type(integrals_instance)
    assert(integrals_class.__doc__ == IntegralsInterface.__doc__)
    # Check attributes
    assert(hasattr(integrals_instance, 'nuclei'))
    assert(hasattr(integrals_instance, 'basis_label'))
    assert(hasattr(integrals_instance, 'nbf'))
    # Check attribute types
    assert(isinstance(getattr(integrals_instance, 'nuclei'), NuclearFramework))
    assert(isinstance(getattr(integrals_instance, 'basis_label'), str))
    assert(isinstance(getattr(integrals_instance, 'nbf'), int))
    # Check methods
    assert(hasattr(integrals_instance, '__init__'))
    assert(hasattr(integrals_instance, 'get_ao_1e_overlap'))
    assert(hasattr(integrals_instance, 'get_ao_1e_potential'))
    assert(hasattr(integrals_instance, 'get_ao_1e_kinetic'))
    assert(hasattr(integrals_instance, 'get_ao_2e_repulsion'))
    # Check method documentation
    assert(integrals_class.__init__.__doc__
           == """Initialize Integrals object.
    
        Args:
            nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): Specifies the
                positions of the atomic centers.
            basis_label (str): What basis set to use.
        """)
    assert(integrals_instance.get_ao_1e_overlap.__doc__
           == IntegralsInterface.get_ao_1e_overlap.__doc__)
    assert(integrals_instance.get_ao_1e_kinetic.__doc__
           == IntegralsInterface.get_ao_1e_kinetic.__doc__)
    assert(integrals_instance.get_ao_1e_potential.__doc__
           == IntegralsInterface.get_ao_1e_potential.__doc__)
    assert(integrals_instance.get_ao_2e_repulsion.__doc__
           == IntegralsInterface.get_ao_2e_repulsion.__doc__)
    # Check method signature
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert(inspect.signature(integrals_class.__init__) ==
           inspect.Signature(
                parameters=[
                    inspect.Parameter('self', kind),
                    inspect.Parameter('nuclei', kind),
                    inspect.Parameter('basis_label', kind)
                ]
           ))
    assert(inspect.signature(integrals_class.get_ao_1e_overlap) ==
           inspect.signature(IntegralsInterface.get_ao_1e_overlap))
    assert(inspect.signature(integrals_class.get_ao_1e_kinetic) ==
           inspect.signature(IntegralsInterface.get_ao_1e_kinetic))
    assert(inspect.signature(integrals_class.get_ao_1e_potential) ==
           inspect.signature(IntegralsInterface.get_ao_1e_potential))
    assert(inspect.signature(integrals_class.get_ao_2e_repulsion) ==
           inspect.signature(IntegralsInterface.get_ao_2e_repulsion))
    # Check method output
    assert(isinstance(integrals_instance.get_ao_1e_overlap(), np.ndarray))
    assert(isinstance(integrals_instance.get_ao_1e_kinetic(), np.ndarray))
    assert(isinstance(integrals_instance.get_ao_1e_potential(), np.ndarray))
    assert(isinstance(integrals_instance.get_ao_2e_repulsion(), np.ndarray))
    assert(integrals_instance.get_ao_1e_overlap().shape
           == (integrals_instance.nbf,) * 2)
    assert(integrals_instance.get_ao_1e_kinetic().shape
           == (integrals_instance.nbf,) * 2)
    assert(integrals_instance.get_ao_1e_potential().shape
           == (integrals_instance.nbf,) * 2)
    assert(integrals_instance.get_ao_2e_repulsion().shape
           == (integrals_instance.nbf,) * 4)


def check_ao_1e_overlap(integrals_instance, shape, norm):
    s = integrals_instance.get_ao_1e_overlap(save=True)
    assert(s.shape == shape)
    assert(np.isclose(np.linalg.norm(s), norm))
    assert(hasattr(integrals_instance, '_ao_1e_overlap'))
    integrals_instance._ao_1e_overlap[:, :] = np.zeros(s.shape)
    s_new = integrals_instance.get_ao_1e_overlap()
    assert(np.linalg.norm(s_new) == 0.0)


def check_ao_1e_kinetic(integrals_instance, shape, norm):
    s = integrals_instance.get_ao_1e_kinetic(save=True)
    assert(s.shape == shape)
    assert(np.isclose(np.linalg.norm(s), norm))
    assert(hasattr(integrals_instance, '_ao_1e_kinetic'))
    integrals_instance._ao_1e_kinetic[:, :] = np.zeros(s.shape)
    s_new = integrals_instance.get_ao_1e_kinetic()
    assert(np.linalg.norm(s_new) == 0.0)


def check_ao_1e_potential(integrals_instance, shape, norm):
    s = integrals_instance.get_ao_1e_potential(save=True)
    assert(s.shape == shape)
    assert(np.isclose(np.linalg.norm(s), norm))
    assert(hasattr(integrals_instance, '_ao_1e_potential'))
    integrals_instance._ao_1e_potential[:, :] = np.zeros(s.shape)
    s_new = integrals_instance.get_ao_1e_potential()
    assert(np.linalg.norm(s_new) == 0.0)


def check_ao_1e_dipole(integrals_instance, shape, norm):
    s = integrals_instance.get_ao_1e_dipole(save=True)
    assert(s.shape == shape)
    assert(np.isclose(np.linalg.norm(s), norm))
    assert(hasattr(integrals_instance, '_ao_1e_dipole'))
    integrals_instance._ao_1e_dipole[:, :, :] = np.zeros(s.shape)
    s_new = integrals_instance.get_ao_1e_dipole()
    assert(np.linalg.norm(s_new) == 0.0)


def check_ao_2e_repulsion(integrals_instance, shape, norm):
    s = integrals_instance.get_ao_2e_repulsion(save=True)
    assert(s.shape == shape)
    assert(np.isclose(np.linalg.norm(s), norm))
    assert(hasattr(integrals_instance, '_ao_2e_chem_repulsion'))
    integrals_instance._ao_2e_chem_repulsion[:, :, :, :] = np.zeros(s.shape)
    s_new = integrals_instance.get_ao_2e_repulsion()
    assert(np.linalg.norm(s_new) == 0.0)


def run_interface_check(integrals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "cc-pvdz")
    # Test the integrals interface
    check_interface(integrals)


def run_ao_1e_overlap_check(integrals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    check_ao_1e_overlap(integrals, (7, 7), 2.95961615642)


def run_ao_1e_kinetic_check(integrals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    check_ao_1e_kinetic(integrals, (7, 7), 29.3703412473)


def run_ao_1e_potential_check(integrals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    check_ao_1e_potential(integrals, (7, 7), 67.1181391119)


def run_ao_1e_dipole_check(integrals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    if hasattr(integrals, '_pyscf_molecule'):
        norm = 3.36216114637
    elif hasattr(integrals, '_psi4_molecule'):
        norm = 3.36241054669
    check_ao_1e_dipole(integrals, (3, 7, 7), norm)


def run_ao_2e_repulsion_check(integrals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "sto-3g")
    # Test the integrals interface
    check_ao_2e_repulsion(integrals, (7, 7, 7, 7), 8.15009229415)
