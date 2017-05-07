import inspect
from scfexchange.integrals import IntegralsInterface
from scfexchange.molecule import Molecule


def check_interface(integrals_instance):
    # Check class documentation
    integrals_class = type(integrals_instance)
    assert(integrals_class.__doc__ == IntegralsInterface.__doc__)
    # Check attributes
    assert(hasattr(integrals_instance, 'basis_label'))
    assert(hasattr(integrals_instance, 'molecule'))
    assert(hasattr(integrals_instance, 'nbf'))
    # Check attribute types
    assert(isinstance(getattr(integrals_instance, 'basis_label'), str))
    assert(isinstance(getattr(integrals_instance, 'molecule'), Molecule))
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
          molecule (:obj:`scfexchange.molecule.Molecule`): The molecule.
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
    assert(inspect.signature(integrals_class.__init__) ==
           inspect.Signature(
                parameters=[
                    inspect.Parameter('self',
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter('molecule',
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter('basis_label',
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
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


if __name__ == "__main__":
    import numpy as np
    from scfexchange.psi4_interface import Integrals

    units = "angstrom"
    charge = 1
    multiplicity = 2
    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])

    mol = Molecule(labels, coordinates, units=units, charge=charge,
                   multiplicity=multiplicity)
    integrals = Integrals(mol, "cc-pvdz")
    check_interface(integrals)
