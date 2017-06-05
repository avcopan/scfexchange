import numpy as np
import psi4.core

from .integrals import IntegralsInterface
from .orbitals import OrbitalsInterface


class Integrals(IntegralsInterface):
    """Molecular integrals (Psi4).
    
    Attributes:
        nuclei (:obj:`scfexchange.Nuclei`): The nuclei on which the basis
            functions are centered
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nbf (int): The number of basis functions.
    """

    def __init__(self, nuclei, basis_label):
        """Initialize Integrals object.
    
        Args:
            nuclei (:obj:`scfexchange.Nuclei`): The nuclei on which the basis
                functions are centered
            basis_label (str): What basis set to use.
        """
        self.nuclei = nuclei
        self.basis_label = basis_label
        s = '\n'.join(["units bohr", str(nuclei)])
        self._psi4_molecule = psi4.core.Molecule.create_molecule_from_string(s)
        self._psi4_molecule.reset_point_group("c1")
        self._psi4_molecule.update_geometry()
        self._mints_helper = self._get_mints_helper()
        self.nbf = int(self._mints_helper.nbf())

    def _get_mints_helper(self):
        basis, _ = psi4.core.BasisSet.build(self._psi4_molecule, "BASIS",
                                            self.basis_label)
        return psi4.core.MintsHelper(basis)

    def get_ao_1e_overlap(self, use_spinorbs=False, recompute=False):
        """Get the overlap integrals.
       
        Returns the overlap matrix of the atomic-orbital basis, <mu(1)|nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate(): return np.array(self._mints_helper.ao_overlap())

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

        def integrate(): return np.array(self._mints_helper.ao_kinetic())

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

        def integrate(): return np.array(self._mints_helper.ao_potential())

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
            comps = self._mints_helper.ao_dipole()
            return np.array([np.array(comp) for comp in comps])

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
            g_chem = np.array(self._mints_helper.ao_eri())
            return g_chem.transpose((0, 2, 1, 3))

        g = self._get_ints('2e_repulsion', integrate, use_spinorbs, recompute)
        if antisymmetrize:
            g = g - g.transpose((0, 1, 3, 2))
        return g


class Orbitals(OrbitalsInterface):
    """Molecular orbitals (Psi4).
    
    Attributes:
        integrals (:obj:`scfexchange.IntegralsInterface`): The integrals.
        molecule (:obj:`scfexchange.Molecule`): A Molecule object specifying
            the total molecular charge and spin multiplicity of the system.
        mo_coefficients (numpy.ndarray): The orbital expansion coefficients.
        spin_is_restricted (bool): Are the orbital spin-restricted?
        ncore (int): The number of low-energy orbitals assigned to the core
            orbital space.
    """

    def solve(self, niter=40, e_threshold=1e-12, d_threshold=1e-6,
              guess="auto"):
        """Solve for Hartree-Fock orbitals with Psi4.

        The orbitals are stored in `self.mo_coefficients`.

        Args:
            niter (int): Maximum number of iterations allowed.
            e_threshold (float): Energy convergence threshold.
            d_threshold (float): Density convergence threshold.
            guess (str): The starting guess to be used by Psi4.  Possible 
                values include 'auto', 'core', 'gwh', 'sad', and 'read'.
        """
        if not isinstance(self.integrals, Integrals):
            raise ValueError("Requires integrals object from the Psi4 "
                             "interface.")
        charge = self.molecule.charge
        multp = self.molecule.multiplicity
        self.integrals._psi4_molecule.set_molecular_charge(charge)
        self.integrals._psi4_molecule.set_multiplicity(multp)
        wfn = psi4.core.Wavefunction.build(self.integrals._psi4_molecule,
                                           self.integrals.basis_label)
        sf, _ = psi4.driver.dft_funcs.build_superfunctional("HF", False)
        psi4.core.set_global_option("guess", guess)
        psi4.core.set_global_option("e_convergence", e_threshold)
        psi4.core.set_global_option("d_convergence", d_threshold)
        psi4.core.set_global_option("maxiter", niter)
        if self.spin_is_restricted:
            if multp is 1:
                psi4.core.set_global_option("reference", "RHF")
                psi4_hf = psi4.core.RHF(wfn, sf)
            else:
                psi4.core.set_global_option("reference", "ROHF")
                psi4_hf = psi4.core.ROHF(wfn, sf)
        else:
            psi4.core.set_global_option("reference", "UHF")
            psi4_hf = psi4.core.UHF(wfn, sf)
        psi4_hf.compute_energy()
        mo_alpha_coeffs = np.array(psi4_hf.Ca())
        mo_beta_coeffs = np.array(psi4_hf.Cb())
        self.mo_coefficients = np.array([mo_alpha_coeffs, mo_beta_coeffs])


if __name__ == "__main__":
    import itertools as it
    from .molecule import Nuclei

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")

    iterables1 = ([(0, 1), (1, 2)], [True, False])
    iterables2 = ([0, 1], ['alpha', 'beta', 'spinorb'])
    norms = []
    shapes = []
    for (charge, multp), restr in it.product(*iterables1):
        orbitals = Orbitals(integrals, charge, multp, restrict_spin=restr)
        orbitals.solve()
        for ncore, mo_type in it.product(*iterables2):
            orbitals.ncore = ncore
            s = orbitals.get_mo_1e_kinetic('o,o', mo_type)
            norms.append(np.linalg.norm(s))
            shapes.append(s.shape)
    print(shapes)
    print(norms)
