import inspect
import numpy as np
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
    assert(hasattr(orbitals_instance, 'nfrz'))
    assert(hasattr(orbitals_instance, 'norb'))
    assert(hasattr(orbitals_instance, 'naocc'))
    assert(hasattr(orbitals_instance, 'nbocc'))
    assert(hasattr(orbitals_instance, 'core_energy'))
    assert(hasattr(orbitals_instance, 'hf_energy'))
    # Check attribute types
    assert(isinstance(getattr(orbitals_instance, 'integrals'),
                      IntegralsInterface))
    assert(isinstance(getattr(orbitals_instance, 'molecule'), Molecule))
    assert(isinstance(getattr(orbitals_instance, 'options'), dict))
    assert(isinstance(getattr(orbitals_instance, 'nfrz'), int))
    assert(isinstance(getattr(orbitals_instance, 'norb'), int))
    assert(isinstance(getattr(orbitals_instance, 'naocc'), int))
    assert(isinstance(getattr(orbitals_instance, 'nbocc'), int))
    assert(isinstance(getattr(orbitals_instance, 'core_energy'), float))
    assert(isinstance(getattr(orbitals_instance, 'hf_energy'), float))
    # Check 'options' attribute
    assert(set(orbitals_instance.options.keys()) ==
           {'restrict_spin', 'n_iterations', 'e_threshold', 'd_threshold'})
    assert(isinstance(orbitals_instance.options['restrict_spin'], bool))
    assert(isinstance(orbitals_instance.options['n_iterations'], int))
    assert(isinstance(orbitals_instance.options['e_threshold'], float))
    assert(isinstance(orbitals_instance.options['d_threshold'], float))
    # Check methods
    assert(hasattr(orbitals_instance, '__init__'))
    # Check method documentation
    assert(orbitals_class.__init__.__doc__
           == """Initialize Orbitals object.
        
        Args:
            integrals (:obj:`scfexchange.integrals.IntegralsInterface`): The
                atomic-orbital integrals object.
            charge (int): Total molecular charge.
            multiplicity (int): Spin multiplicity.
            restrict_spin (bool): Spin-restrict the orbitals?
            n_iterations (int): Maximum number of Hartree-Fock iterations 
                allowed before the orbitals are considered unconverged.
            e_threshold (float): Energy convergence threshold.
            d_threshold (float): Density convergence threshold, based on the 
                norm of the orbital gradient
            n_frozen_orbitals (int): How many core orbitals should be set to 
                `frozen`.
        """)
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
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "cc-pvdz")
    # Build orbitals
    vars = ([(0, 1), (1, 2)], [True, False], [0, 1])
    for (charge, multp), restr, nfrz in it.product(*vars):
        orbitals = orbitals_class(integrals, charge, multp,
                                  restrict_spin=restr, n_frozen_orbitals=nfrz)
        check_interface(orbitals)


def run_core_energy_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "cc-pvdz")
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
    orbitals7 = orbitals_class(integrals, charge=1, multiplicity=2,
                               restrict_spin=True, n_frozen_orbitals=0)
    orbitals8 = orbitals_class(integrals, charge=1, multiplicity=2,
                               restrict_spin=True, n_frozen_orbitals=1)
    # Test the core energy
    check_core_energy(orbitals1,  9.16714531316, -85.1937928452,           0.0)
    check_core_energy(orbitals2, -52.1425517089, -39.3520211260, 15.4679253029)
    check_core_energy(orbitals3,  9.16714531316, -85.1937928452,           0.0)
    check_core_energy(orbitals4, -52.1425517089, -39.3520211260, 15.4679253029)
    check_core_energy(orbitals5,  9.16714531316, -84.7992709854,           0.0)
    check_core_energy(orbitals6, -52.1426192068, -37.8242618465, 14.3347553810)
    check_core_energy(orbitals7,  9.16714531316, -84.7947487279,           0.0)
    check_core_energy(orbitals8, -52.1427130957, -37.8177520225, 14.3328617035)


def run_mp2_energy_check(integrals_class, orbitals_class):
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = integrals_class(nuclei, "cc-pvdz")
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
    check_mp2_energy(orbitals1, -0.204165905228)
    check_mp2_energy(orbitals2, -0.201833176089)
    check_mp2_energy(orbitals3, -0.204165905228)
    check_mp2_energy(orbitals4, -0.201833176089)
    check_mp2_energy(orbitals5, -0.153359695366)
    check_mp2_energy(orbitals6, -0.151178068749)

