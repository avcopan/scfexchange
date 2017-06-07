import numpy as np
import tensorutils as tu
import scipy.linalg as spla


class RPA(object):

    def __init__(self, orbitals):
        self.orbitals = orbitals

    def compute_a_matrix(self):
        gvovo = self.orbitals.get_mo_2e_repulsion(mo_block='v,o,v,o',
                                                  spin_sector='s,s',
                                                  antisymmetrize=True)
        foo = self.orbitals.get_mo_1e_fock(mo_block='o,o')
        fvv = self.orbitals.get_mo_1e_fock(mo_block='v,v')
        nocc = self.orbitals.get_mo_count(mo_space='o')
        nvir = self.orbitals.get_mo_count(mo_space='v')
        ioo = np.identity(nocc)
        ivv = np.identity(nvir)
        a_array = (
            +
                tu.einsum('ab,ij->iajb', fvv, ioo)
            -
                tu.einsum('ji,ab->iajb', foo, ivv)
            -
                tu.einsum('ajbi->iajb', gvovo)
        )
        a_matrix = a_array.reshape((nocc * nvir, nocc * nvir))
        return a_matrix

    def compute_b_matrix(self):
        goovv = self.orbitals.get_mo_2e_repulsion(mo_block='o,o,v,v',
                                                  spin_sector='s,s',
                                                  antisymmetrize=True)
        nocc = self.orbitals.get_mo_count(mo_space='o')
        nvir = self.orbitals.get_mo_count(mo_space='v')
        b_array = tu.einsum('ijab->iajb', goovv)
        b_matrix = b_array.reshape((nocc * nvir, nocc * nvir))
        return b_matrix

    def compute_cis_spectrum(self):
        a = self.compute_a_matrix()
        spectrum = spla.eigvalsh(a)
        return spectrum

    def compute_rpa_spectrum(self):
        a = self.compute_a_matrix()
        b = self.compute_b_matrix()
        h = (a + b).dot(a - b)
        spectrum = np.sqrt(spla.eigvals(h).real)
        spectrum.sort()
        return spectrum


if __name__ == "__main__":
    from scfexchange import Nuclei
    from scfexchange.pyscf_interface import Integrals, Orbitals

    labels = ("O", "H", "H")
    coordinates = np.array([[ 0.000000000000, -0.143225816552,  0.000000000000],
                            [ 1.638036840407,  1.136548822547, -0.000000000000],
                            [-1.638036840407,  1.136548822547, -0.000000000000]])

    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    orbitals = Orbitals(integrals)
    orbitals.solve()
    rpa = RPA(orbitals)
    spectrum = rpa.compute_rpa_spectrum()
    print(spectrum)

