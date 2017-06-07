import numpy as np
import tensorutils as tu
import scipy.linalg as spla
from scfexchange.density import CorrelatedDensityInterface


class MP2Density(CorrelatedDensityInterface):

    def __init__(self, orbitals):
        naocc = orbitals.get_mo_count(mo_space='o', spin='a')
        navir = orbitals.get_mo_count(mo_space='v', spin='a')
        nbocc = orbitals.get_mo_count(mo_space='o', spin='b')
        nbvir = orbitals.get_mo_count(mo_space='v', spin='b')
        self.aat2 = np.zeros((naocc, naocc, navir, navir))
        self.abt2 = np.zeros((naocc, nbocc, navir, nbvir))
        self.bbt2 = np.zeros((nbocc, nbocc, nbvir, nbvir))
        CorrelatedDensityInterface.__init__(self, orbitals)

    def solve(self):
        aag = self.orbitals.get_mo_2e_repulsion(mo_block='o,o,v,v',
                                                spin_sector='a,a',
                                                antisymmetrize=True)
        abg = self.orbitals.get_mo_2e_repulsion(mo_block='o,o,v,v',
                                                spin_sector='a,b',
                                                antisymmetrize=True)
        bbg = self.orbitals.get_mo_2e_repulsion(mo_block='o,o,v,v',
                                                spin_sector='b,b',
                                                antisymmetrize=True)
        aeocc = self.orbitals.get_mo_fock_diagonal(mo_space='o', spin='a')
        aevir = self.orbitals.get_mo_fock_diagonal(mo_space='v', spin='a')
        beocc = self.orbitals.get_mo_fock_diagonal(mo_space='o', spin='b')
        bevir = self.orbitals.get_mo_fock_diagonal(mo_space='v', spin='b')
        x = np.newaxis
        self.aat2 = aag / (+ aeocc[:, x, x, x] + aeocc[x, :, x, x]
                           - aevir[x, x, :, x] - aevir[x, x, x, :])
        self.abt2 = abg / (+ aeocc[:, x, x, x] + beocc[x, :, x, x]
                           - aevir[x, x, :, x] - bevir[x, x, x, :])
        self.bbt2 = bbg / (+ beocc[:, x, x, x] + beocc[x, :, x, x]
                           - bevir[x, x, :, x] - bevir[x, x, x, :])

    def get_2e_amplitudes(self, spin_sector='s,s'):
        if spin_sector == 'a,a':
            return self.aat2
        elif spin_sector == 'a,b':
            return self.abt2
        elif spin_sector == 'b,b':
            return self.bbt2
        elif spin_sector == 's,s':
            nocc = orbitals.get_mo_count(mo_space='o', spin='s')
            nvir = orbitals.get_mo_count(mo_space='v', spin='s')
            t2 = np.zeros((nocc, nocc, nvir, nvir))
            # Finish this later
            raise NotImplementedError
        else:
            raise ValueError("Invalid 'spin_sector' argument.")

    def get_1e_moment_correction(self, mo_block='ov,ov', spin_sector='s'):
        """Get the correlation correction to the one-particle density moments.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        norb = self.orbitals.get_mo_count(mo_space='cov', spin=spin_sector)
        aat2 = self.get_2e_amplitudes(spin_sector='a,a')
        abt2 = self.get_2e_amplitudes(spin_sector='a,b')
        bbt2 = self.get_2e_amplitudes(spin_sector='b,b')
        o = self.orbitals.get_mo_slice(mo_space='o', spin=spin_sector)
        v = self.orbitals.get_mo_slice(mo_space='v', spin=spin_sector)

        if spin_sector == 'a':
            tau1 = np.zeros((norb, norb))
            tau1[o, o] = - (1. / 2 * tu.einsum('jkab,ikab->ij', aat2, aat2)
                            + tu.einsum('jKaB,iKaB->ij', abt2, abt2))
            tau1[v, v] = + (1. / 2 * tu.einsum('ijac,ijbc->ab', aat2, aat2)
                            + tu.einsum('iJaC,iJbC->ab', abt2, abt2))
        elif spin_sector == 'b':
            tau1 = np.zeros((norb, norb))
            tau1[o, o] = - (1. / 2 * tu.einsum('JKAB,IKAB->IJ', bbt2, bbt2)
                            + tu.einsum('kJbA,kIbA->IJ', abt2, abt2))
            tau1[v, v] = + (1. / 2 * tu.einsum('IJAC,IJBC->AB', bbt2, bbt2)
                            + tu.einsum('jIcA,jIcB->AB', abt2, abt2))
        elif spin_sector == 's':
            atau1 = self.get_1e_moment_correction(mo_block='cov,cov',
                                                  spin_sector='a')
            btau1 = self.get_1e_moment_correction(mo_block='cov,cov',
                                                  spin_sector='b')
            tau1 = spla.block_diag(atau1, btau1)
            spinorb_order = self.orbitals.get_spinorb_order()
            tau1 = tau1[:, spinorb_order]
            tau1 = tau1[spinorb_order, :]

        space_keys, spin_keys = self.orbitals.get_block_keys(mo_block,
                                                             spin_sector)
        slices = tuple(self.orbitals.get_mo_slice(mo_space, spin)
                       for mo_space, spin in zip(space_keys, spin_keys))
        return tau1[slices]

    def get_2e_moment_correction(self, mo_block='ov,ov,ov,ov',
                                 spin_sector='s,s'):
        """Get the correlation correction to the two-particle density moments.

        Args:
            mo_block (str): A comma-separated list of characters specifying the
                MO space block.  Each MO space is identified by a contiguous
                combination of 'c' (core), 'o' (occupied), and 'v' (virtual).
            spin_sector (str): The requested spin sector (diagonal spin block).
                Spins are 'a' (alpha), 'b' (beta), or 's' (spin-orbital).

        Returns:
            numpy.ndarray: The moment array.
        """
        s0, s1 = spin_sector.split(',')
        norb = self.orbitals.get_mo_count('cov', spin=s0)
        tau2 = np.zeros((norb, norb, norb, norb))
        s0o = self.orbitals.get_mo_slice(mo_space='o', spin=s0)
        s1o = self.orbitals.get_mo_slice(mo_space='o', spin=s1)
        s0v = self.orbitals.get_mo_slice(mo_space='v', spin=s0)
        s1v = self.orbitals.get_mo_slice(mo_space='v', spin=s1)
        t2 = self.get_2e_amplitudes(spin_sector)
        tau2[s0o, s1o, s0v, s1v] = t2
        tau2[s0v, s1v, s0o, s1o] = t2.transpose((2, 3, 0, 1))

        space_keys, spin_keys = self.orbitals.get_block_keys(mo_block,
                                                             spin_sector)
        slices = tuple(self.orbitals.get_mo_slice(mo_space, spin)
                       for mo_space, spin in zip(space_keys, spin_keys))
        return tau2[slices]


if __name__ == "__main__":
    from scfexchange import Nuclei
    from scfexchange.pyscf_interface import Integrals, Orbitals

    labels = ("O", "H", "H")
    coordinates = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                            [0.0000000000, -1.4343021349, 0.9864370414],
                            [0.0000000000, 1.4343021349, 0.9864370414]])
    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    energies = []
    for (charge, multp) in [(0, 1), (1, 2)]:
        orbitals = Orbitals(integrals, charge, multp, restrict_spin=False)
        orbitals.solve()
        for ncore in [0, 1]:
            orbitals.ncore = ncore
            density = MP2Density(orbitals)
            density.solve()
            energy = density.get_energy()
            print(energy)
            energies.append(energy)

    print(energies)
