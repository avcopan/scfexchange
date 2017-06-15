import numpy as np
import pyscf

from .ao import AOIntegralsInterface
from .molecule import nuclear_coordinates_in_bohr


class AOIntegrals(AOIntegralsInterface):
    """Molecular integrals (PySCF).
    
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
        atom_arg = list(zip(nuc_labels, nuc_coords))
        self._pyscf_molecule = pyscf.gto.Mole(atom=atom_arg,
                                              unit=units,
                                              basis=basis_label)
        self._pyscf_molecule.build()

        self.nuc_labels = tuple(nuc_labels)
        self.nuc_coords = nuclear_coordinates_in_bohr(nuc_coords, units)
        self.basis_label = str(basis_label)
        self.nbf = int(self._pyscf_molecule.nao_nr())

    def overlap(self, use_spinorbs=False, recompute=False):
        """Get the overlap integrals.
       
        Returns the overlap matrix of the atomic-orbital basis, <mu(1)|nu(1)>.
    
        Args:
            use_spinorbs (bool): Return the integrals in the spin-orbital basis?
            recompute (bool): Recompute the integrals, if we already have them?
    
        Returns:
            numpy.ndarray: The integrals.
        """

        def integrate(): return self._pyscf_molecule.intor('cint1e_ovlp_sph')

        s = self._compute('_overlap', integrate, use_spinorbs, recompute)
        return s

    def kinetic(self, use_spinorbs=False, recompute=False):
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

        t = self._compute('_kinetic', integrate, use_spinorbs, recompute)
        return t

    def potential(self, use_spinorbs=False, recompute=False):
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

        v = self._compute('_potential', integrate, use_spinorbs, recompute)
        return v

    def dipole(self, use_spinorbs=False, recompute=False):
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

        d = self._compute('_dipole', integrate, use_spinorbs, recompute,
                          ncomp=3)
        return d

    def electron_repulsion(self, use_spinorbs=False, recompute=False,
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

        g = self._compute('_electron_repulsion', integrate, use_spinorbs,
                          recompute)
        if antisymmetrize:
            g = g - g.transpose((0, 1, 3, 2))
        return g


def hf_mo_coefficients(aoints, charge=0, multp=1, restrict_spin=False,
                       niter=100, e_threshold=1e-12, d_threshold=1e-6):
    if not isinstance(aoints, AOIntegrals):
        raise ValueError("Please use an aoints object from the PySCF "
                         "interface.")
    aoints._pyscf_molecule.build(charge=charge, spin=multp - 1)
    if restrict_spin:
        pyscf_hf = pyscf.scf.RHF(aoints._pyscf_molecule)
    else:
        pyscf_hf = pyscf.scf.UHF(aoints._pyscf_molecule)
    pyscf_hf.conv_tol = e_threshold
    pyscf_hf.conv_tol_grad = d_threshold
    pyscf_hf.max_cycle = niter
    pyscf_hf.kernel()
    mo_coefficients = pyscf_hf.mo_coeff
    if restrict_spin:
        mo_coefficients = np.array([mo_coefficients] * 2)
    return mo_coefficients


if __name__ == "__main__":
    import numpy

    nuc_labels = ("O", "H", "H")
    nuc_coords = numpy.array([[0.0000000000, 0.0000000000, -0.1247219248],
                              [0.0000000000, -1.4343021349, 0.9864370414],
                              [0.0000000000, 1.4343021349, 0.9864370414]])
    aoints = AOIntegrals("sto-3g", nuc_labels, nuc_coords)
    mo_coeffs = hf_mo_coefficients(aoints, charge=1, multp=2,
                                   restrict_spin=False)
    alpha_coeffs = mo_coeffs[0, :, :5]
    beta_coeffs = mo_coeffs[1, :, :4]

    # Test default
    s = aoints.fock(alpha_coeffs, beta_coeffs=beta_coeffs,
                    electric_field=[0., 0., 1.])
    print(numpy.linalg.norm(s))
