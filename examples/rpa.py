import numpy as np
import tensorutils as tu
import scipy.linalg as spla


class RPA(object):

    def __init__(self, orbitals):
        self.orbitals = orbitals

    def a_matrix(self):
        gvovo = self.orbitals.electron_repulsion(mo_block='v,o,v,o',
                                                 spin_sector='s,s',
                                                 antisymmetrize=True)
        foo = self.orbitals.fock(mo_block='o,o')
        fvv = self.orbitals.fock(mo_block='v,v')
        nocc = self.orbitals.mo_count(mo_space='o')
        nvir = self.orbitals.mo_count(mo_space='v')
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

    def b_matrix(self):
        goovv = self.orbitals.electron_repulsion(mo_block='o,o,v,v',
                                                 spin_sector='s,s',
                                                 antisymmetrize=True)
        nocc = self.orbitals.mo_count(mo_space='o')
        nvir = self.orbitals.mo_count(mo_space='v')
        b_array = tu.einsum('ijab->iajb', goovv)
        b_matrix = b_array.reshape((nocc * nvir, nocc * nvir))
        return b_matrix

    def dipole_gradient_matrix(self):
        nocc = self.orbitals.mo_count(mo_space='o')
        nvir = self.orbitals.mo_count(mo_space='v')
        d_ints = self.orbitals.dipole(mo_block='o,v',
                                      spin_sector='s')
        d_array = d_ints.transpose((1, 2, 0))
        d_matrix = d_array.reshape((nocc * nvir, 3))
        return d_matrix

    def cis_spectrum(self):
        a = self.a_matrix()
        spectrum = spla.eigvalsh(a)
        return spectrum

    def rpa_spectrum(self):
        a = self.a_matrix()
        b = self.b_matrix()
        h = (a + b).dot(a - b)
        spectrum = np.sqrt(spla.eigvals(h).real)
        spectrum.sort()
        return spectrum

    def dipole_polarizability_tensor(self):
        a = self.a_matrix()
        b = self.b_matrix()
        d = self.dipole_gradient_matrix()
        e = np.bmat([[a, b], [b, a]]).view(np.ndarray)
        t = np.bmat([[d], [d]]).view(np.ndarray)
        r = spla.solve(e, t, sym_pos=True)
        return t.T.dot(r)


def _main():
    from scfexchange.mo import MOIntegrals
    from scfexchange.pyscf_interface import AOIntegrals, hf_mo_coefficients

    labels = ("O", "H", "H")
    coordinates = np.array([[ 0.000000000000, -0.143225816552,  0.000000000000],
                            [ 1.638036840407,  1.136548822547, -0.000000000000],
                            [-1.638036840407,  1.136548822547, -0.000000000000]])

    integrals = AOIntegrals("sto-3g", labels, coordinates)
    mo_coefficients = hf_mo_coefficients(integrals, charge=0, multp=1)
    orbitals = MOIntegrals(integrals, mo_coefficients, naocc=5, nbocc=5)
    rpa = RPA(orbitals)
    alpha = rpa.dipole_polarizability_tensor()
    print(alpha)

if __name__ == "__main__":
    _main()
