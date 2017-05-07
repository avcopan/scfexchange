import inspect
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


if __name__ == "__main__":
    import numpy as np
    from scfexchange.molecule import Molecule
    from scfexchange.psi4_interface import Integrals, Orbitals

    units = "angstrom"
    charge = 0
    multiplicity = 1
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])

    mol = Molecule(labels, coordinates, units=units, charge=charge,
                   multiplicity=multiplicity)
    integrals = Integrals(mol, "cc-pvdz")
    orbitals = Orbitals(integrals,
                        freeze_core=False,
                        n_frozen_orbitals=1,
                        e_threshold=1e-14,
                        n_iterations=50,
                        restrict_spin=False)
    check_interface(orbitals)
