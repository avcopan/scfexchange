import inspect
import numpy as np
from scfexchange.integrals import IntegralsInterface
from scfexchange.orbitals import OrbitalsInterface


def check_interface(orbitals_instance):
    # Check class documentation
    orbitals_class = type(orbitals_instance)
    assert(orbitals_class.__doc__ == OrbitalsInterface.__doc__)
    # Check attributes
    assert(hasattr(orbitals_instance, 'integrals'))
    assert(hasattr(orbitals_instance, 'options'))
    assert(hasattr(orbitals_instance, 'nfrz'))
    assert(hasattr(orbitals_instance, 'norb'))
    assert(hasattr(orbitals_instance, 'naocc'))
    assert(hasattr(orbitals_instance, 'nbocc'))
    assert(hasattr(orbitals_instance, 'mo_energies'))
    assert(hasattr(orbitals_instance, 'mo_coefficients'))
    assert(hasattr(orbitals_instance, 'mso_energies'))
    assert(hasattr(orbitals_instance, 'mso_coefficients'))
    assert(hasattr(orbitals_instance, 'core_energy'))
    assert(hasattr(orbitals_instance, 'hf_energy'))
    # Check attribute types
    assert(isinstance(getattr(orbitals_instance, 'integrals'),
                      IntegralsInterface))
    assert(isinstance(getattr(orbitals_instance, 'options'), dict))
    assert(isinstance(getattr(orbitals_instance, 'nfrz'), int))
    assert(isinstance(getattr(orbitals_instance, 'norb'), int))
    assert(isinstance(getattr(orbitals_instance, 'naocc'), int))
    assert(isinstance(getattr(orbitals_instance, 'nbocc'), int))
    assert(isinstance(getattr(orbitals_instance, 'core_energy'), float))
    assert(isinstance(getattr(orbitals_instance, 'hf_energy'), float))
    assert(isinstance(getattr(orbitals_instance, 'mo_energies'),
                      np.ndarray))
    assert(isinstance(getattr(orbitals_instance, 'mo_coefficients'),
                      np.ndarray))
    assert(isinstance(getattr(orbitals_instance, 'mso_energies'),
                      np.ndarray))
    assert(isinstance(getattr(orbitals_instance, 'mso_coefficients'),
                      np.ndarray))
    # Check 'options' attribute
    assert(set(orbitals_instance.options.keys()) ==
           {'restrict_spin', 'n_iterations', 'e_threshold', 'd_threshold',
            'freeze_core', 'n_frozen_orbitals'})
    assert(isinstance(orbitals_instance.options['restrict_spin'], bool))
    assert(isinstance(orbitals_instance.options['n_iterations'], int))
    assert(isinstance(orbitals_instance.options['e_threshold'], float))
    assert(isinstance(orbitals_instance.options['d_threshold'], float))
    assert(isinstance(orbitals_instance.options['freeze_core'], bool))
    assert(isinstance(orbitals_instance.options['n_frozen_orbitals'], int))
    # Check attributes that are arrays
    norb = orbitals_instance.norb + orbitals_instance.nfrz
    assert(orbitals_instance.mo_energies.shape == (2, norb))
    assert(orbitals_instance.mo_coefficients.shape == (2, norb, norb))
    assert(orbitals_instance.mso_energies.shape == (2 * norb,))
    assert(orbitals_instance.mso_coefficients.shape == (2 * norb, 2 * norb))
    # Check methods
    assert(hasattr(orbitals_instance, '__init__'))
    # Check method documentation
    assert(orbitals_class.__init__.__doc__
           == """Initialize Orbitals object.
        
        Args:
            integrals (:obj:`scfexchange.integrals.IntegralsInterface`): The
                atomic-orbital integrals object.
            restrict_spin: Spin-restrict the orbitals?
            n_iterations: Maximum number of Hartree-Fock iterations allowed
                before the orbitals are considered unconverged.
            e_threshold: Energy convergence threshold.
            d_threshold: Density convergence threshold, based on the norm of the
                orbital gradient
            freeze_core: Freeze the core orbitals?
            n_frozen_orbitals: How many core orbitals should be set to `frozen`.
        """)
    # Check method signature
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert(inspect.signature(orbitals_class.__init__) ==
           inspect.Signature(
                parameters=[
                    inspect.Parameter('self', kind),
                    inspect.Parameter('integrals', kind),
                    inspect.Parameter('restrict_spin', kind, default=True),
                    inspect.Parameter('n_iterations', kind, default=40),
                    inspect.Parameter('e_threshold', kind, default=1e-12),
                    inspect.Parameter('d_threshold', kind, default=1e-6),
                    inspect.Parameter('freeze_core', kind, default=False),
                    inspect.Parameter('n_frozen_orbitals', kind, default=0)
                ]
           ))


def check_mo_slicing(orbitals_instance, mo_type, core_offset, occ_offset):
    assert(orbitals_instance.get_mo_slice(mo_type, mo_block='c') ==
           slice(None, core_offset, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_block='o') ==
           slice(core_offset, occ_offset, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_block='v') ==
           slice(occ_offset, None, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_block='co') ==
           slice(None, occ_offset, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_block='ov') ==
           slice(core_offset, None, None))
    assert(orbitals_instance.get_mo_slice(mo_type, mo_block='cov') ==
           slice(None, None, None))


def check_core_energy(orbitals_instance, core_energy, valence_energy,
                      core_valence_energy):
    h = (
        +
        orbitals_instance.get_mo_1e_kinetic(mo_type='spinor', mo_block='o,o')
        +
        orbitals_instance.get_mo_1e_potential(mo_type='spinor', mo_block='o,o')
    )
    v = orbitals_instance.get_mo_1e_core_field(mo_type='spinor', mo_block='o,o')
    g = orbitals_instance.get_mo_2e_repulsion(mo_type='spinor',
                                              mo_block='o,o,o,o',
                                              antisymmetrize=True)
    e_c = orbitals_instance.core_energy
    e_v = np.trace(h) + 1. / 2 * np.einsum("ijij", g)
    e_cv = np.trace(v)
    e_hf = orbitals_instance.hf_energy
    assert(np.isclose(e_c, core_energy))
    assert(np.isclose(e_v, valence_energy))
    assert(np.isclose(e_cv, core_valence_energy))
    assert(np.isclose(e_hf, core_energy + valence_energy + core_valence_energy))


def check_mp2_energy(orbitals_instance, correlation_energy):
    e = orbitals_instance.get_mo_energies(mo_type='spinor', mo_block='ov')
    g = orbitals_instance.get_mo_2e_repulsion(mo_type='spinor',
                                              mo_block='o,o,v,v',
                                              antisymmetrize=True)
    nspocc = orbitals_instance.naocc + orbitals_instance.nbocc
    o = slice(None, nspocc)
    v = slice(nspocc, None)
    x = np.newaxis
    e_corr = (
        1. / 4 * np.sum(g * g / (
            e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))
    )
    assert(np.isclose(e_corr, correlation_energy))


def run_interface_check(integrals_class, orbitals_class):
    import numpy as np
    from scfexchange import Molecule
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])
    mol1 = Molecule(labels, coordinates, charge=0, multiplicity=1)
    mol2 = Molecule(labels, coordinates, charge=1, multiplicity=2)
    # Build integrals
    ints1 = integrals_class(mol1, "cc-pvdz")
    ints2 = integrals_class(mol2, "cc-pvdz")
    # Build orbitals
    orbs1 = orbitals_class(ints1, restrict_spin=True, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs2 = orbitals_class(ints2, restrict_spin=False, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs3 = orbitals_class(ints1, restrict_spin=True, freeze_core=True,
                           n_frozen_orbitals=1)
    orbs4 = orbitals_class(ints2, restrict_spin=False, freeze_core=True,
                           n_frozen_orbitals=1)
    # Test the orbitals interface
    check_interface(orbs1)
    check_interface(orbs2)
    check_interface(orbs3)
    check_interface(orbs4)


def run_mo_slicing_check(integrals_class, orbitals_class):
    import numpy as np
    from scfexchange import Molecule
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])
    mol1 = Molecule(labels, coordinates, charge=0, multiplicity=1)
    mol2 = Molecule(labels, coordinates, charge=1, multiplicity=2)
    # Build integrals
    ints1 = integrals_class(mol1, "cc-pvdz")
    ints2 = integrals_class(mol2, "cc-pvdz")
    # Build orbitals
    orbs1 = orbitals_class(ints1, restrict_spin=True, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs2 = orbitals_class(ints2, restrict_spin=False, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs3 = orbitals_class(ints1, restrict_spin=True, freeze_core=True,
                           n_frozen_orbitals=1)
    orbs4 = orbitals_class(ints2, restrict_spin=False, freeze_core=True,
                           n_frozen_orbitals=1)
    # Test the orbitals interface
    check_mo_slicing(orbs1, 'spinor', 0, 10)
    check_mo_slicing(orbs1, 'alpha', 0, 5)
    check_mo_slicing(orbs1, 'beta', 0, 5)
    check_mo_slicing(orbs2, 'spinor', 0, 9)
    check_mo_slicing(orbs2, 'alpha', 0, 5)
    check_mo_slicing(orbs2, 'beta', 0, 4)
    check_mo_slicing(orbs3, 'spinor', 2, 10)
    check_mo_slicing(orbs3, 'alpha', 1, 5)
    check_mo_slicing(orbs3, 'beta', 1, 5)
    check_mo_slicing(orbs4, 'spinor', 2, 9)
    check_mo_slicing(orbs4, 'alpha', 1, 5)
    check_mo_slicing(orbs4, 'beta', 1, 4)


def run_core_energy_check(integrals_class, orbitals_class):
    import numpy as np
    from scfexchange import Molecule
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])
    mol1 = Molecule(labels, coordinates, charge=0, multiplicity=1)
    mol2 = Molecule(labels, coordinates, charge=1, multiplicity=2)
    # Build integrals
    ints1 = integrals_class(mol1, "cc-pvdz")
    ints2 = integrals_class(mol2, "cc-pvdz")
    # Build orbitals
    orbs1 = orbitals_class(ints1, restrict_spin=True, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs2 = orbitals_class(ints2, restrict_spin=False, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs3 = orbitals_class(ints1, restrict_spin=True, freeze_core=True,
                           n_frozen_orbitals=1)
    orbs4 = orbitals_class(ints2, restrict_spin=False, freeze_core=True,
                           n_frozen_orbitals=1)
    # Test the orbitals interface
    check_core_energy(orbs1, 9.16714531316, -85.1937928046, 0.0)
    check_core_energy(orbs2, 9.16714531316, -84.7992709454, 0.0)
    check_core_energy(orbs3, -52.1425516992, -39.3520210647, 15.4679252725)
    check_core_energy(orbs4, -52.1426191971, -37.8242618135, 14.3347553783)


def run_mp2_energy_check(integrals_class, orbitals_class):
    import numpy as np
    from scfexchange import Molecule
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])
    mol1 = Molecule(labels, coordinates, charge=0, multiplicity=1)
    mol2 = Molecule(labels, coordinates, charge=1, multiplicity=2)
    # Build integrals
    ints1 = integrals_class(mol1, "cc-pvdz")
    ints2 = integrals_class(mol2, "cc-pvdz")
    # Build orbitals
    orbs1 = orbitals_class(ints1, restrict_spin=True, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs2 = orbitals_class(ints2, restrict_spin=False, freeze_core=False,
                           n_frozen_orbitals=0)
    orbs3 = orbitals_class(ints1, restrict_spin=True, freeze_core=True,
                           n_frozen_orbitals=1)
    orbs4 = orbitals_class(ints2, restrict_spin=False, freeze_core=True,
                           n_frozen_orbitals=1)
    # Test the orbitals interface
    check_mp2_energy(orbs1, -0.204165905785)
    check_mp2_energy(orbs2, -0.15335969536)
    check_mp2_energy(orbs3, -0.201833176653)
    check_mp2_energy(orbs4, -0.151178068917)

