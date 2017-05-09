import pyscf
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
        self._pyscf_molecule = pyscf.gto.Mole(atom=list(iter(nuclei)),
                                              unit=nuclei.units,
                                              basis=basis_label)
        self._pyscf_molecule.build()

        self.nuclei = nuclei
        self.basis_label = basis_label
        self.nbf = int(self._pyscf_molecule.nao_nr())

    def set_gauge_origin(self, origin=(0, 0, 0)):
        """Set the gauge origin for integral computations.

        Args:
             origin: A 3-tuples specifying the gauge origin for the gauge
                 integrals.
        """
        self._pyscf_molecule.set_common_orig(origin)

    def get_ao_1e_overlap(self, integrate_spin=True, save=False):
        """Compute overlap integrals for the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A nbf x nbf array of overlap integrals,
            < mu(1) | nu(1) >.
        """
        def compute_ints():
            return self._pyscf_molecule.intor('cint1e_ovlp_sph')
        return self._compute_ao_1e('overlap', compute_ints, integrate_spin,
                                   save)

    def get_ao_1e_kinetic(self, integrate_spin=True, save=False):
        """Compute kinetic energy operator in the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A nbf x nbf array of kinetic energy operator integrals,
            < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >.
        """
        def compute(): return self._pyscf_molecule.intor('cint1e_kin_sph')
        return self._compute_ao_1e('kinetic', compute, integrate_spin, save)

    def get_ao_1e_potential(self, integrate_spin=True, save=False):
        """Compute nuclear potential operator in the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A nbf x nbf array of nuclear potential operator integrals,
            < mu(1) | sum_A Z_A / r_1A | nu(1) >.
        """
        def compute(): return self._pyscf_molecule.intor('cint1e_nuc_sph')
        return self._compute_ao_1e('potential', compute, integrate_spin, save)

    def get_ao_1e_dipole(self, integrate_spin=True, save=False):
        """Compute the dipole operator in the atomic orbital basis.
        
        Args:
            integrate_spin (bool): Use spatial orbitals?
            save (bool): Save the computed array for later use?
    
        Returns:
            A 3 x nbf x nbf array of dipole operator integrals,
            < mu(1) | [x, y, z] | nu(1) >
        """
        def compute(): return self._pyscf_molecule.intor('cint1e_r_sph', comp=3)
        return self._compute_ao_1e('dipole', compute, integrate_spin, save,
                                   ncomp=3)

    def get_ao_1e_dipole(self, integrate_spin=True, save=False):
        """Compute the dipole operator in the atomic orbital basis.
        
        Args:
          integrate_spin (bool): Use spatial orbitals instead of spin-orbitals?
          save (bool): Save the computed array for later use?
    
        Returns:
          A 3 x nbf x nbf array of dipole operator integrals,
          < mu(1) | r | nu(1) >.
        """
        mu = self._pyscf_molecule.intor('cint1e_r_sph', comp=3)
        print(mu)
        print(mu.shape)
        print(self._pyscf_molecule.atom_coords())

    def get_ao_2e_repulsion(self, integrate_spin=True, save=False,
                            antisymmetrize=False):
        """Compute electron-repulsion operator in the atomic orbital basis.
    
        Args:
            integrate_spin (bool): Use spatial orbitals?
            save (bool): Save the computed array for later use?
            antisymmetrize (bool): Antisymmetrize the repulsion integrals?
    
        Returns:
            A nbf x nbf x nbf x nbf array of electron
            repulsion operator integrals,
            < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >.
        """
        def compute():
            shape = (self.nbf, self.nbf, self.nbf, self.nbf)
            return self._pyscf_molecule.intor('cint2e_sph').reshape(shape)
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
        integrals._pyscf_molecule.build(charge=charge, spin=multiplicity-1)
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
        # Build PySCF HF object and compute the energy.
        if self.options['restrict_spin']:
            self._pyscf_hf = pyscf.scf.RHF(integrals._pyscf_molecule)
        else:
            self._pyscf_hf = pyscf.scf.UHF(integrals._pyscf_molecule)
        self._pyscf_hf.conv_tol = self.options['e_threshold']
        self._pyscf_hf.conv_tol_grad = self.options['d_threshold']
        self._pyscf_hf.max_cycle = self.options['n_iterations']
        self._pyscf_hf.kernel()
        self.hf_energy = self._pyscf_hf.e_tot
        self._mo_energies = self._pyscf_hf.mo_energy
        self._mo_coefficients = self._pyscf_hf.mo_coeff
        if self.options['restrict_spin']:
            self._mo_energies = np.array([self._mo_energies] * 2)
            self._mo_coefficients = np.array([self._mo_coefficients] * 2)
        # Build spin-orbital energy and coefficient arrays
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
    from . import constants

    labels = ("O", "H", "H")
    coordinates = np.array([[0.000, 0.000, -0.066],
                            [0.000, -0.759, 0.522],
                            [0.000, 0.759, 0.522]])

    nuclei = NuclearFramework(labels, coordinates, units="angstrom")
    integrals = Integrals(nuclei, "cc-pvdz")
    print(np.linalg.norm(integrals.get_ao_1e_kinetic()))
    '''
    coc = nuclei.get_center_of_charge()
    integrals.set_gauge_origin((0, 0, 0))
    mu = integrals.get_ao_1e_dipole()
    print(mu.shape)
    print(np.linalg.norm(mu))
    '''
