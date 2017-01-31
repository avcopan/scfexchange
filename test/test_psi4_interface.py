import numpy  as np
from scfexchange import Molecule
from scfexchange.psi4_interface import Integrals, Orbitals

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

def test__frozen_core():
  h = orbitals.get_mo_1e_kinetic(mo_type = 'spinor', mo_block = ('o', 'o')) + \
      orbitals.get_mo_1e_potential(mo_type = 'spinor', mo_block = ('o', 'o'))
  v = orbitals.get_mo_1e_core_field(mo_type = 'spinor', mo_block = ('o', 'o'))
  g = orbitals.get_mo_2e_repulsion(mo_type = 'spinor', mo_block = ('o', 'o', 'o', 'o'))
  g = g - g.transpose((0, 2, 1, 3))
  core_energy = orbitals.core_energy
  valence_energy = np.trace(h) + 1./2 * np.einsum("ijij", g)
  core_valence_energy = np.trace(v)
  total_energy = valence_energy + core_energy + core_valence_energy

  assert(np.allclose(total_energy,         orbitals.hf_energy, rtol=1e-09, atol=1e-10))
  assert(np.allclose(core_energy,         -52.142619197068512, rtol=1e-09, atol=1e-10))
  assert(np.allclose(valence_energy,      -37.824261813733628, rtol=1e-09, atol=1e-10))
  assert(np.allclose(core_valence_energy,  14.334755378526848, rtol=1e-09, atol=1e-10))
  assert(np.allclose(total_energy,        -75.632125632275290, rtol=1e-09, atol=1e-10))
