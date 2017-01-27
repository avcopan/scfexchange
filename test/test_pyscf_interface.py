import numpy  as np
from scfexchange import Molecule
from scfexchange.pyscf_interface import Integrals, Orbitals

units = "angstrom"
charge = 1
multiplicity = 2
labels = ("O", "H", "H")
coordinates = np.array([[0.000,  0.000, -0.066],
                        [0.000, -0.759,  0.522],
                        [0.000,  0.759,  0.522]])

mol = Molecule(labels, coordinates, units = units, charge = charge,
               multiplicity = multiplicity)
integrals = Integrals(mol, "cc-pvdz")
orbital_options = {
  'freeze_core': False,
  'n_frozen_orbitals': 1,
  'e_threshold': 1e-14,
  'n_iterations': 50,
  'restrict_spin': False
}
orbitals = Orbitals(integrals, **orbital_options)

def test__get_mo_slice():
  assert(orbitals.get_mo_slice(mo_type = 'spinor', mo_block = 'c'  ) == slice(None, 2, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'spinor', mo_block = 'o'  ) == slice(2, 9, None)      )
  assert(orbitals.get_mo_slice(mo_type = 'spinor', mo_block = 'v'  ) == slice(9, None, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'spinor', mo_block = 'co' ) == slice(None, 9, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'spinor', mo_block = 'ov' ) == slice(2, None, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'spinor', mo_block = 'cov') == slice(None, None, None))
  assert(orbitals.get_mo_slice(mo_type = 'alpha', mo_block = 'c'   ) == slice(None, 1, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'alpha', mo_block = 'o'   ) == slice(1, 5, None)      )
  assert(orbitals.get_mo_slice(mo_type = 'alpha', mo_block = 'v'   ) == slice(5, None, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'alpha', mo_block = 'co'  ) == slice(None, 5, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'alpha', mo_block = 'ov'  ) == slice(1, None, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'alpha', mo_block = 'cov' ) == slice(None, None, None))
  assert(orbitals.get_mo_slice(mo_type = 'beta', mo_block = 'c'    ) == slice(None, 1, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'beta', mo_block = 'o'    ) == slice(1, 4, None)      )
  assert(orbitals.get_mo_slice(mo_type = 'beta', mo_block = 'v'    ) == slice(4, None, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'beta', mo_block = 'co'   ) == slice(None, 4, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'beta', mo_block = 'ov'   ) == slice(1, None, None)   )
  assert(orbitals.get_mo_slice(mo_type = 'beta', mo_block = 'cov'  ) == slice(None, None, None))

def test__get_mo_energies():
  assert(orbitals.get_mo_energies(mo_type = 'spinor', mo_block = 'c'  ).shape == (2,) )
  assert(orbitals.get_mo_energies(mo_type = 'spinor', mo_block = 'o'  ).shape == (7,) )
  assert(orbitals.get_mo_energies(mo_type = 'spinor', mo_block = 'v'  ).shape == (39,))
  assert(orbitals.get_mo_energies(mo_type = 'spinor', mo_block = 'co' ).shape == (9,) )
  assert(orbitals.get_mo_energies(mo_type = 'spinor', mo_block = 'ov' ).shape == (46,))
  assert(orbitals.get_mo_energies(mo_type = 'spinor', mo_block = 'cov').shape == (48,))
  assert(orbitals.get_mo_energies(mo_type = 'alpha', mo_block = 'c'   ).shape == (1,) )
  assert(orbitals.get_mo_energies(mo_type = 'alpha', mo_block = 'o'   ).shape == (4,) )
  assert(orbitals.get_mo_energies(mo_type = 'alpha', mo_block = 'v'   ).shape == (19,))
  assert(orbitals.get_mo_energies(mo_type = 'alpha', mo_block = 'co'  ).shape == (5,) )
  assert(orbitals.get_mo_energies(mo_type = 'alpha', mo_block = 'ov'  ).shape == (23,))
  assert(orbitals.get_mo_energies(mo_type = 'alpha', mo_block = 'cov' ).shape == (24,))
  assert(orbitals.get_mo_energies(mo_type = 'beta', mo_block = 'c'    ).shape == (1,) )
  assert(orbitals.get_mo_energies(mo_type = 'beta', mo_block = 'o'    ).shape == (3,) )
  assert(orbitals.get_mo_energies(mo_type = 'beta', mo_block = 'v'    ).shape == (20,))
  assert(orbitals.get_mo_energies(mo_type = 'beta', mo_block = 'co'   ).shape == (4,) )
  assert(orbitals.get_mo_energies(mo_type = 'beta', mo_block = 'ov'   ).shape == (23,))
  assert(orbitals.get_mo_energies(mo_type = 'beta', mo_block = 'cov'  ).shape == (24,))

def test__get_mo_coefficients():
  assert(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'c'  ).shape == (48, 2) )
  assert(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'o'  ).shape == (48, 7) )
  assert(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'v'  ).shape == (48, 39))
  assert(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'co' ).shape == (48, 9) )
  assert(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'ov' ).shape == (48, 46))
  assert(orbitals.get_mo_coefficients(mo_type = 'spinor', mo_block = 'cov').shape == (48, 48))
  assert(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'c'   ).shape == (24, 1) )
  assert(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'o'   ).shape == (24, 4) )
  assert(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'v'   ).shape == (24, 19))
  assert(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'co'  ).shape == (24, 5) )
  assert(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'ov'  ).shape == (24, 23))
  assert(orbitals.get_mo_coefficients(mo_type = 'alpha', mo_block = 'cov' ).shape == (24, 24))
  assert(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'c'    ).shape == (24, 1) )
  assert(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'o'    ).shape == (24, 3) )
  assert(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'v'    ).shape == (24, 20))
  assert(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'co'   ).shape == (24, 4) )
  assert(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'ov'   ).shape == (24, 23))
  assert(orbitals.get_mo_coefficients(mo_type = 'beta', mo_block = 'cov'  ).shape == (24, 24))

