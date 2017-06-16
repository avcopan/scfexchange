import numpy as np
import psi4.core

from .ao import AOIntegralsInterface
from .molecule import nuclear_coordinate_string, nuclear_coordinates_in_bohr


# Functions
def hf_mo_coefficients(aoints, charge=0, multp=1, restricted=False, niter=100,
                       e_threshold=1e-12, d_threshold=1e-6, guess="auto"):
    if not isinstance(aoints, AOIntegrals):
        raise ValueError("Please use an aoints object from the PySCF "
                         "interface.")
    aoints._psi4_molecule.set_molecular_charge(charge)
    aoints._psi4_molecule.set_multiplicity(multp)
    wfn = psi4.core.Wavefunction.build(aoints._psi4_molecule,
                                       aoints.basis_label)
    sf, _ = psi4.driver.dft_funcs.build_superfunctional("HF", False)
    psi4.core.set_global_option("guess", guess)
    psi4.core.set_global_option("e_convergence", e_threshold)
    psi4.core.set_global_option("d_convergence", d_threshold)
    psi4.core.set_global_option("maxiter", niter)
    if restricted:
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
    ac = np.array(psi4_hf.Ca())
    bc = np.array(psi4_hf.Cb())
    mo_coefficients = np.array([ac, bc])
    return mo_coefficients


# Classes
class AOIntegrals(AOIntegralsInterface):
    """Molecular integrals (Psi4).
    
    Attributes:
        basis_label (str): The basis set label (e.g. 'sto-3g').
        nuc_labels (tuple): Atomic symbols.
        nuc_coords (numpy.ndarray): Atomic coordinates in Bohr.
        nbf (int): The number of basis functions.
    """

    def __init__(self, basis_label, nuc_labels, nuc_coords, units="bohr"):
        """Initialize AOIntegrals object.

        Args:
            basis_label (str): What basis set to use.
            nuc_labels (tuple): Atomic symbols.
            nuc_coords (numpy.ndarray): Atomic coordinates.
            units (str): The units of `nuc_coords`, "angstrom" or "bohr".
        """
        s = nuclear_coordinate_string(nuc_labels, nuc_coords, units)
        self._psi4_molecule = psi4.core.Molecule.create_molecule_from_string(s)
        self._psi4_molecule.reset_point_group("c1")
        self._psi4_molecule.update_geometry()
        basis, _ = psi4.core.BasisSet.build(self._psi4_molecule, "BASIS",
                                            basis_label)
        self._mints_helper = psi4.core.MintsHelper(basis)

        self.nuc_labels = tuple(nuc_labels)
        self.nuc_coords = nuclear_coordinates_in_bohr(nuc_coords, units)
        self.basis_label = str(basis_label)
        self.basis_label = basis_label
        self.nbf = int(self._mints_helper.nbf())

    def overlap(self, spinorb=False, recompute=False):
        """Get the overlap integrals.
       
        Returns the overlap matrix of the atomic-orbital basis, <mu(1)|nu(1)>.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate(): return np.array(self._mints_helper.ao_overlap())

        s = self._compute('_overlap', integrate, spinorb, recompute)
        return s

    def kinetic(self, spinorb=False, recompute=False):
        """Get the kinetic energy integrals.
        
        Returns the representation of the electron kinetic-energy operator in
        the atomic-orbital basis, <mu(1)| - 1 / 2 * nabla_1^2 |nu(1)>.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate(): return np.array(self._mints_helper.ao_kinetic())

        t = self._compute('_kinetic', integrate, spinorb, recompute)
        return t

    def potential(self, spinorb=False, recompute=False):
        """Get the potential energy integrals.

        Returns the representation of the nuclear potential operator in the
        atomic-orbital basis, <mu(1)| sum_A Z_A / ||r_1 - r_A|| |nu(1)>.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate(): return np.array(self._mints_helper.ao_potential())

        v = self._compute('_potential', integrate, spinorb, recompute)
        return v

    def dipole(self, spinorb=False, recompute=False):
        """Get the dipole integrals.

        Returns the representation of the electric dipole operator in the
        atomic-orbital basis, <mu(1)| [-x, -y, -z] |nu(1)>.
        
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate():
            comps = self._mints_helper.ao_dipole()
            return np.array([np.array(comp) for comp in comps])

        m = self._compute('_dipole', integrate, spinorb, recompute,
                          multicomp=True)
        return m

    def electron_repulsion(self, spinorb=False, recompute=False,
                           antisymmetrize=False):
        """Get the electron-repulsion integrals.

        Returns the representation of the electron repulsion operator in the 
        atomic-orbital basis, <mu(1) nu(2)| 1 / ||r_1 - r_2|| |rh(1) si(2)>.
        Note that these are returned in physicist's notation.
    
        Args:
            spinorb (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
            antisymmetrize (bool): Antisymmetrize the integral tensor?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate():
            g_chem = np.array(self._mints_helper.ao_eri())
            return g_chem.transpose((0, 2, 1, 3))

        g = self._compute('_electron_repulsion', integrate, spinorb, recompute)
        if antisymmetrize:
            g = g - g.transpose((0, 1, 3, 2))
        return g


def _main():
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = hf_mo_coefficients(aoints, charge=1, multp=2,
                                   restricted=False)
    alpha_coeffs = mo_coeffs[0, :, :5]
    beta_coeffs = mo_coeffs[1, :, :4]

    # Test default
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=(0., 0., 1.))
    print(numpy.linalg.norm(s))

if __name__ == "__main__":
    _main()

