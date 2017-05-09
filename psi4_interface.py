import psi4.core
import scipy.linalg as spla
import numpy as np

from .integrals import IntegralsInterface
from .orbitals import OrbitalsInterface
from .molecule import Molecule


class Integrals(IntegralsInterface):
    """Molecular integrals.
    
    Attributes:
        nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): Specifies the
            positions of the atomic centers.
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nbf (int): The number of basis functions.
    """

    def __init__(self, nuclei, basis_label):
        """Initialize Integrals object.
    
        Args:
            nuclei (:obj:`scfexchange.nuclei.NuclearFramework`): Specifies the
                positions of the atomic centers.
            basis_label (str): What basis set to use.
        """
        self._psi4_molecule = (
            psi4.core.Molecule.create_molecule_from_string(str(nuclei))
        )
        self._psi4_molecule.reset_point_group("c1")
        self._psi4_molecule.update_geometry()
        basisset = psi4.core.BasisSet.build(self._psi4_molecule, "BASIS",
                                            basis_label)
        self._mints_helper = psi4.core.MintsHelper(basisset)
        self.nuclei = nuclei
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
    """Molecular orbitals.
    
    Attributes:
        integrals (:obj:`scfexchange.integrals.Integrals`): Contributions to the
            Hamiltonian operator, in the molecular orbital basis.
        molecule (:obj:`scfexchange.nuclei.Molecule`): A Molecule object
            specifying the molecular charge and multiplicity
        options (dict): A dictionary of options, by keyword argument.
        nfrz (int): The number of frozen (spatial) orbitals.  This can be set
            with the option 'n_frozen_orbitals'.  Alternatively, if
            'freeze_core' is True and the number of frozen orbitals is not set,
            this defaults to the number of core orbitals, as determined by the
            nuclei object.
        norb (int): The total number of non-frozen (spatial) orbitals.  That is,
            the number of basis functions minus the number of frozen orbitals.
        naocc (int): The number of occupied non-frozen alpha orbitals.
        nbocc (int): The number of occupied non-frozen beta orbitals.
        core_energy (float): Hartree-Fock energy of the frozen core, including
            nuclear repulsion energy.
        hf_energy (float): The total Hartree-Fock energy.
    """

    def __init__(self, integrals, charge=0, multiplicity=1, restrict_spin=True,
                 n_iterations=40, e_threshold=1e-12, d_threshold=1e-6,
                 n_frozen_orbitals=0):
        """Initialize Orbitals object.
        
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
        """
        if not isinstance(integrals, Integrals):
            raise ValueError(
                "Please use an integrals object from this interface.")
        integrals._psi4_molecule.set_molecular_charge(charge)
        integrals._psi4_molecule.set_multiplicity(multiplicity)
        self.integrals = integrals
        self.options = {
            'restrict_spin': restrict_spin,
            'n_iterations': n_iterations,
            'e_threshold': e_threshold,
            'd_threshold': d_threshold,
        }
        self.molecule = Molecule(self.integrals.nuclei, charge, multiplicity)
        # Determine the orbital counts (total, frozen, and occupied)
        self.nfrz = n_frozen_orbitals
        self.naocc = self.molecule.nalpha - self.nfrz
        self.nbocc = self.molecule.nbeta - self.nfrz
        self.norb = self.integrals.nbf - self.nfrz
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
            if multiplicity is 1:
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
        self._mo_energies = np.array([mo_alpha_energies, mo_beta_energies])
        self._mo_coefficients = np.array([mo_alpha_coeffs, mo_beta_coeffs])
        # Build spin-orbital energy and coefficient arrays, sorted by energy
        mso_energies = np.concatenate(self._mo_energies)
        mso_coefficients = spla.block_diag(*self._mo_coefficients)
        sorting_indices = mso_energies.argsort()
        self._mso_energies = mso_energies[sorting_indices]
        self._mso_coefficients = mso_coefficients[:, sorting_indices]
        # Get the core field and energy
        self.core_energy = self._compute_core_energy()


if __name__ == "__main__":
    import numpy as np
    from .molecule import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])

    nuclei = NuclearFramework(labels, coordinates, units="angstrom")
    integrals = Integrals(nuclei, "sto-3g")
    s = integrals.get_ao_1e_overlap(integrate_spin=False, save=True)
    g = integrals.get_ao_2e_repulsion(integrate_spin=False, save=True,
                                      antisymmetrize=True)
    print(g.shape)
    orbitals = Orbitals(integrals, charge=+1, multiplicity=2)
    t = orbitals.get_mo_1e_kinetic(mo_type='spinor', save=True)
    g = orbitals.get_mo_2e_repulsion(mo_type='spinor', save=True,
                                     antisymmetrize=True)

