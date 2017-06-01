import numpy as np
import pyscf

from .integrals import IntegralsInterface
from .orbitals import OrbitalsInterface


class Integrals(IntegralsInterface):
    """Molecular integrals (PySCF).
    
    Attributes:
        nuclei (:obj:`scfexchange.NuclearFramework`): Specifies the positions
            of the atomic centers.
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nbf (int): The number of basis functions.
    """

    def __init__(self, nuclei, basis_label):
        """Initialize Integrals object.
    
        Args:
            nuclei (:obj:`scfexchange.NuclearFramework`): Specifies the
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
            numpy.ndarray: The integrals.
        """

        def integrate(): return self._pyscf_molecule.intor('cint1e_ovlp_sph')

        s = self._get_ints('1e_overlap', integrate, use_spinorbs, recompute)
        return s

    def get_ao_1e_kinetic(self, use_spinorbs=False, recompute=False):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the atomic-orbital basis, <mu(1)| - 1 / 2 * nabla_1^2 |nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate(): return self._pyscf_molecule.intor('cint1e_kin_sph')

        t = self._get_ints('1e_kinetic', integrate, use_spinorbs, recompute)
        return t

    def get_ao_1e_potential(self, use_spinorbs=False, recompute=False):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        atomic-orbital basis, <mu(1)| sum_A Z_A / ||r_1 - r_A|| |nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate(): return self._pyscf_molecule.intor('cint1e_nuc_sph')

        v = self._get_ints('1e_potential', integrate, use_spinorbs, recompute)
        return v

    def get_ao_1e_dipole(self, use_spinorbs=False, recompute=False):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        atomic-orbital basis, <mu(1)| [-x, -y, -z] |nu(1)>.
        
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate():
            return -self._pyscf_molecule.intor('cint1e_r_sph', comp=3)

        d = self._get_ints('1e_dipole', integrate, use_spinorbs, recompute,
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
            numpy.ndarray: The integrals.
        """

        def integrate():
            shape = (self.nbf, self.nbf, self.nbf, self.nbf)
            g_chem = self._pyscf_molecule.intor('cint2e_sph').reshape(shape)
            return g_chem.transpose((0, 2, 1, 3))

        g = self._get_ints('2e_repulsion', integrate, use_spinorbs, recompute)
        if antisymmetrize:
            g = g - g.transpose((0, 1, 3, 2))
        return g


class Orbitals(OrbitalsInterface):
    """Molecular orbitals (PySCF).
    
    Attributes:
        integrals (:obj:`scfexchange.IntegralsInterface`): The integrals.
        molecule (:obj:`scfexchange.Molecule`): A Molecule object specifying
            the total molecular charge and spin multiplicity of the system.
        mo_coefficients (numpy.ndarray): The orbital expansion coefficients.
        spin_is_restricted (bool): Are the orbital spin-restricted?
        ncore (int): The number of low-energy orbitals assigned to the core
            orbital space.
    """

    def solve(self, niter=40, e_threshold=1e-12, d_threshold=1e-6):
        """Solve for Hartree-Fock orbitals with PySCF.
        
        The orbitals are stored in `self.mo_coefficients`.
        
        Args:
            niter (int): Maximum number of iterations allowed.
            e_threshold (float): Energy convergence threshold.
            d_threshold (float): Density convergence threshold.
        """
        if not isinstance(self.integrals, Integrals):
            raise ValueError("Requires integrals object from the PySCF "
                             "interface.")
        charge = self.molecule.charge
        spin = self.molecule.multiplicity - 1
        self.integrals._pyscf_molecule.build(charge=charge, spin=spin)
        if self.spin_is_restricted:
            pyscf_hf = pyscf.scf.RHF(self.integrals._pyscf_molecule)
        else:
            pyscf_hf = pyscf.scf.UHF(self.integrals._pyscf_molecule)
        pyscf_hf.conv_tol = e_threshold
        pyscf_hf.conv_tol_grad = d_threshold
        pyscf_hf.max_cycle = niter
        pyscf_hf.kernel()
        mo_energies = pyscf_hf.mo_energy
        self.mo_coefficients = pyscf_hf.mo_coeff
        if self.spin_is_restricted:
            mo_energies = np.array([mo_energies] * 2)
            self.mo_coefficients = np.array([self.mo_coefficients] * 2)
        self.spinorb_order = np.argsort(np.concatenate(mo_energies))


if __name__ == "__main__":
    import scipy.linalg as spla
    from .molecule import NuclearFramework

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = NuclearFramework(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    orbitals = Orbitals(integrals, charge=1, multiplicity=2,
                        restrict_spin=False)
    orbitals.solve()
    spinorb_order = orbitals.get_spinorb_order()
    print(spinorb_order)
    c = spla.block_diag(*orbitals.mo_coefficients)
    print(c.round(1))
    print(c[:, spinorb_order].round(1))
