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
        self._pyscf_molecule = pyscf.gto.Mole(atom=str(nuclei),
                                              unit="bohr",
                                              basis=basis_label)
        self._pyscf_molecule.build()

        self.nuclei = nuclei
        self.basis_label = basis_label
        self.nbf = int(self._pyscf_molecule.nao_nr())

    def get_ao_1e_overlap(self, use_spinorbs=False, recompute=False):
        """Get the overlap integrals.
       
        Returns the overlap matrix of the atomic-orbital basis, <mu(1)|nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        def integrate(): return self._pyscf_molecule.intor('cint1e_ovlp_sph')
        s = self._compute_ao_1e('overlap', integrate, use_spinorbs, recompute)
        return s

    def get_ao_1e_kinetic(self, use_spinorbs=False, recompute=False):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the atomic-orbital basis, <mu(1)| - 1 / 2 * nabla_1^2 |nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        def integrate(): return self._pyscf_molecule.intor('cint1e_kin_sph')
        t = self._compute_ao_1e('kinetic', integrate, use_spinorbs, recompute)
        return t

    def get_ao_1e_potential(self, use_spinorbs=False, recompute=False):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        atomic-orbital basis, <mu(1)| sum_A Z_A / ||r_1 - r_A|| |nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        def integrate(): return self._pyscf_molecule.intor('cint1e_nuc_sph')
        v = self._compute_ao_1e('potential', integrate, use_spinorbs, recompute)
        return v

    def get_ao_1e_dipole(self, use_spinorbs=False, recompute=False):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        atomic-orbital basis, <mu(1)| [-x, -y, -z] |nu(1)>.
        
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            np.ndarray: The integrals.
        """
        def integrate():
            return -self._pyscf_molecule.intor('cint1e_r_sph', comp=3)
        d = self._compute_ao_1e('dipole', integrate, use_spinorbs, recompute,
                                ncomp=3)
        return d

    def get_ao_2e_repulsion(self, use_spinorbs=False, recompute=False,
                            antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        atomic-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            antisymmetrize (bool): Antisymmetrize the integral tensor?
    
        Returns:
            np.ndarray: The integrals.
        """
        def integrate():
            shape = (self.nbf, self.nbf, self.nbf, self.nbf)
            g_chem = self._pyscf_molecule.intor('cint2e_sph').reshape(shape)
            return g_chem.transpose((0, 2, 1, 3))
        g = self._compute_ao_2e('repulsion', integrate, use_spinorbs, recompute)
        if antisymmetrize:
            g = g - g.transpose((0, 1, 3, 2))
        return g


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
    import itertools as it
    from .molecule import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000,  0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349,  0.9864370414],
                            [0.0000000000,  1.4343021349,  0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    # Build integrals
    integrals = Integrals(nuclei, "sto-3g")
    # Test the integrals interface
    orbitals = Orbitals(integrals, charge=1, multiplicity=2,
                        restrict_spin=False, n_frozen_orbitals=1)
    nbf = integrals.nbf
    i = np.identity(nbf)
    norm_a = np.linalg.norm(orbitals._mo_coefficients[0,:,1:])
    norm_b = np.linalg.norm(orbitals._mo_coefficients[1,:,1:])
    print(norm_a)
    print(norm_b)
    norm_s = np.sqrt(norm_a ** 2 + norm_b ** 2)
    c = orbitals.get_mo_coefficients(mo_type='alpha', mo_space='ov',
                                     transformation=None)
    assert(np.linalg.norm(c) == norm_a)
    c = orbitals.get_mo_coefficients(mo_type='alpha', mo_space='ov',
                                     transformation=i)
    assert(np.linalg.norm(c) == norm_a)
    c = orbitals.get_mo_coefficients(mo_type='alpha', mo_space='ov',
                                     transformation=(i, i))
    assert(np.linalg.norm(c) == norm_a)
    c = orbitals.get_mo_coefficients(mo_type='beta', mo_space='ov',
                                     transformation=None)
    assert(np.linalg.norm(c) == norm_b)
    c = orbitals.get_mo_coefficients(mo_type='beta', mo_space='ov',
                                     transformation=i)
    assert(np.linalg.norm(c) == norm_b)
    c = orbitals.get_mo_coefficients(mo_type='beta', mo_space='ov',
                                     transformation=(i, i))
    assert(np.linalg.norm(c) == norm_b)
    c = orbitals.get_mo_coefficients(mo_type='spinorb', mo_space='ov',
                                     transformation=None)
    assert(np.linalg.norm(c) == norm_s)
    c = orbitals.get_mo_coefficients(mo_type='spinorb', mo_space='ov',
                                     transformation=spla.block_diag(i, i))
    assert(np.linalg.norm(c) == norm_s)


    '''
    shapes = []
    norms = []
    vars = ([(0, 1), (1, 2)], [True, False], [0, 1], ['alpha', 'beta', 'spinorb'])
    for (charge, multp), restr, nfrz, mo_type in it.product(*vars):
        orbitals = Orbitals(integrals, charge, multp,
                                  restrict_spin=restr, n_frozen_orbitals=nfrz)
        s = orbitals.get_mo_2e_repulsion(mo_type, mo_block='o,o,o,o')
        shapes.append(s.shape)
        norms.append(spla.norm(s))
    print(shapes)
    print(norms)
    '''
