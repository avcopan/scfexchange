import psi4.core
import scipy.linalg as spla
import numpy as np

from .integrals import IntegralsInterface
from .orbitals import OrbitalsInterface
from .util import with_doc


class Integrals(IntegralsInterface):
    """Molecular integrals.
    
    Attributes:
      basis_label (str): The basis set label (e.g. 'sto-3g').
      molecule: Together with `self.basis_label`, this specifies the atomic
        orbitals entereing the integral computation.
      nbf (int): The number of basis functions.
    """

    def __init__(self, molecule, basis_label):
        """Initialize Integrals object.
    
        Args:
          molecule (:obj:`scfexchange.molecule.Molecule`): The molecule.
          basis_label (str): What basis set to use.
        """
        molstr = str(molecule)
        self._psi4_molecule = psi4.core.Molecule.create_molecule_from_string(
            molstr)
        self._psi4_molecule.set_molecular_charge(molecule.charge)
        self._psi4_molecule.set_multiplicity(molecule.multiplicity)
        self._psi4_molecule.reset_point_group("c1")
        self._psi4_molecule.update_geometry()
        basisset = psi4.core.BasisSet.build(self._psi4_molecule, "BASIS",
                                            basis_label)
        self._mints_helper = psi4.core.MintsHelper(basisset)
        self.molecule = molecule
        self.basis_label = basis_label
        self.nbf = int(self._mints_helper.nbf())

    def get_ao_1e_overlap(self, integrate_spin=True, save=False):
        """Compute overlap integrals for the atomic orbital basis.
    
        Args:
          integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
          save (bool): Save the computed array for later use?
    
        Returns:
          A nbf x nbf array of overlap integrals,
          < mu(1) | nu(1) >.
        """
        def compute(): return np.array(self._mints_helper.ao_overlap())
        return self._compute_ao_1e('overlap', compute, integrate_spin, save)

    def get_ao_1e_potential(self, integrate_spin=True, save=False):
        """Compute nuclear potential operator in the atomic orbital basis.
    
        Args:
          integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
          save (bool): Save the computed array for later use?
    
        Returns:
          A nbf x nbf array of nuclear potential operator integrals,
          < mu(1) | sum_A Z_A / r_1A | nu(1) >.
        """
        def compute(): return np.array(self._mints_helper.ao_potential())
        return self._compute_ao_1e('potential', compute, integrate_spin, save)

    def get_ao_1e_kinetic(self, integrate_spin=True, save=False):
        """Compute kinetic energy operator in the atomic orbital basis.
    
        Args:
          integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
          save (bool): Save the computed array for later use?
    
        Returns:
          A nbf x nbf array of kinetic energy operator integrals,
          < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
        """
        def compute(): return np.array(self._mints_helper.ao_kinetic())
        return self._compute_ao_1e('kinetic', compute, integrate_spin, save)

    def get_ao_2e_repulsion(self, integrate_spin=True, save=False,
                            antisymmetrize=False):
        """Compute electron-repulsion operator in the atomic orbital basis.
    
        Args:
          integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
          save (bool): Save the computed array for later use?
          antisymmetrize (bool): Antisymmetrize the repulsion integrals?
    
        Returns:
          A nbf x nbf x nbf x nbf array of electron
          repulsion operator integrals,
          < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
        """
        def compute(): return np.array(self._mints_helper.ao_eri())
        return self._compute_ao_2e('repulsion', compute, integrate_spin, save,
                                   antisymmetrize)


class Orbitals(OrbitalsInterface):
    __doc__ = """**OrbitalsInterface.__doc__**

{:s}

**Orbitals.__doc__**

Interface for accessing Psi4 molecular orbitals.

  Attributes:
    _psi4_hf (:obj:`psi4.core.HF`): Used to access Psi4 orbitals.

  """.format(OrbitalsInterface.__doc__)

    def __init__(self, integrals, **options):
        """Initialize Orbitals object (Psi4 interface).
    
        Args:
          integrals (:obj:`scfexchange.pyscf_interface.Integrals`): AO integrals.
          restrict_spin (:obj:`bool`, optional): Use spin-restricted orbitals?
          n_iterations (:obj:`int`, optional): Maximum number of iterations allowed
            before considering the orbitals unconverged.
          e_threshold (:obj:`float`, optional): Energy convergence threshold.
          d_threshold (:obj:`float`, optional): Density convergence threshold, based
            on the norm of the orbital gradient.
          freeze_core (:obj:`bool`, optional): Whether or not to cut core orbitals
            out of the MO coefficients.
          n_frozen_orbitals (:obj:`int`, optional): How many core orbitals to cut
            out from the MO coefficients.
        """

        if not isinstance(integrals, Integrals):
            raise ValueError(
                "Please use an integrals object from this interface.")
        self.integrals = integrals
        self.options = self._process_options(options)
        # Determine the orbital counts (total, frozen, and occupied)
        self.nfrz, self.norb, (self.naocc, self.nbocc) = self._count_orbitals()
        # Build Psi4 HF object and compute the energy.
        wfn = psi4.core.Wavefunction.build(integrals._psi4_molecule,
                                           integrals.basis_label)
        sf, _ = psi4.driver.dft_functional.build_superfunctional("HF")
        psi4.core.set_global_option("guess", "gwh")
        psi4.core.set_global_option("e_convergence",
                                    self.options['e_threshold'])
        psi4.core.set_global_option("d_convergence",
                                    self.options['d_threshold'])
        psi4.core.set_global_option("maxiter", self.options['n_iterations'])
        if self.options['restrict_spin']:
            if integrals.molecule.multiplicity is 1:
                psi4.core.set_global_option("reference", "RHF")
                self._psi4_hf = psi4.core.RHF(wfn, sf)
            else:
                psi4.core.set_global_option("reference", "ROHF")
                self._psi4_hf = psi4.core.ROHF(wfn, sf)
        else:
            psi4.core.set_global_option("reference", "UHF")
            self._psi4_hf = psi4.core.UHF(wfn, sf)
        self.hf_energy = self._psi4_hf.compute_energy()
        # Get MO energies and coefficients and put them in the right format
        mo_alpha_energies = self._psi4_hf.epsilon_a().to_array()
        mo_beta_energies = self._psi4_hf.epsilon_b().to_array()
        mo_alpha_coeffs = np.array(self._psi4_hf.Ca())
        mo_beta_coeffs = np.array(self._psi4_hf.Cb())
        self.mo_energies = np.array([mo_alpha_energies, mo_beta_energies])
        self.mo_coefficients = np.array([mo_alpha_coeffs, mo_beta_coeffs])
        # Build spin-orbital energy and coefficient arrays, sorted by orbital energy
        mso_energies = np.concatenate(self.mo_energies)
        mso_coefficients = spla.block_diag(*self.mo_coefficients)
        sorting_indices = mso_energies.argsort()
        self.mso_energies = mso_energies[sorting_indices]
        self.mso_coefficients = mso_coefficients[:, sorting_indices]
        # Get the core field and energy
        self.core_energy = self._compute_core_energy()


if __name__ == "__main__":
    import numpy as np
    from . import Molecule

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
    orbital_options = {
        'freeze_core': False,
        'n_frozen_orbitals': 1,
        'e_threshold': 1e-14,
        'n_iterations': 50,
        'restrict_spin': False
    }
    orbitals = Orbitals(integrals, **orbital_options)
    core_energy = orbitals.core_energy
    h = orbitals.get_mo_1e_kinetic(mo_type='spinor', mo_block='o,o') + \
        orbitals.get_mo_1e_potential(mo_type='spinor', mo_block='o,o')
    g = orbitals.get_mo_2e_repulsion(mo_type='spinor', mo_block='o,o,o,o')
    g = g - g.transpose((0, 2, 1, 3))
    valence_energy = np.trace(h) + 1. / 2 * np.einsum("ijij", g)
    v = orbitals.get_mo_1e_core_field(mo_type='spinor', mo_block='o,o')
    core_valence_energy = np.trace(v)
    total_energy = valence_energy + core_energy + core_valence_energy
    print("Core energy:            {:20.15f}".format(core_energy))
    print("Valence energy:         {:20.15f}".format(valence_energy))
    print("C-V interaction energy: {:20.15f}".format(core_valence_energy))
    print("Total energy:           {:20.15f}".format(total_energy))
    print("Total energy:           {:20.15f}".format(orbitals.hf_energy))
    print(np.allclose(total_energy, orbitals.hf_energy, rtol=1e-09, atol=1e-10))
    e = orbitals.get_mo_energies(mo_type='spinor', mo_block='ov')
    g = orbitals.get_mo_2e_repulsion(mo_type='spinor', mo_block='o,o,v,v',
                                     antisymmetrize=True)
    nspocc = orbitals.naocc + orbitals.nbocc
    o = slice(None, nspocc)
    v = slice(nspocc, None)
    x = np.newaxis
    correlation_energy = (
        1. / 4 * np.sum(g * g / (
        e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))
    )
    print("Correlation energy:     {:20.15f}".format(correlation_energy))
    import inspect
    print(repr(inspect.signature(integrals.get_ao_1e_kinetic)))
