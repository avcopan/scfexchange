import numpy as np
import tensorutils as tu
import scipy.linalg as spla


class RPA(object):

    def __init__(self, orbitals):
        self.orbitals = orbitals

    def get_a_matrix(self):
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

    def get_b_matrix(self):
        goovv = self.orbitals.get_mo_2e_repulsion(mo_block='o,o,v,v',
                                                  spin_sector='s,s',
                                                  antisymmetrize=True)
        nocc = self.orbitals.get_mo_count(mo_space='o')
        nvir = self.orbitals.get_mo_count(mo_space='v')
        b_array = tu.einsum('ijab->iajb', goovv)
        b_matrix = b_array.reshape((nocc * nvir, nocc * nvir))
        return b_matrix

    def get_dipole_gradient_matrix(self):
        nocc = self.orbitals.get_mo_count(mo_space='o')
        nvir = self.orbitals.get_mo_count(mo_space='v')
        d_ints = self.orbitals.get_mo_1e_dipole(mo_block='o,v',
                                                spin_sector='s')
        d_array = d_ints.transpose((1, 2, 0))
        d_matrix = d_array.reshape((nocc * nvir, 3))
        return d_matrix

    def get_cis_spectrum(self):
        a = self.get_a_matrix()
        spectrum = spla.eigvalsh(a)
        return spectrum

    def get_rpa_spectrum(self):
        a = self.get_a_matrix()
        b = self.get_b_matrix()
        h = (a + b).dot(a - b)
        spectrum = np.sqrt(spla.eigvals(h).real)
        spectrum.sort()
        return spectrum

    def get_dipole_polarizability_tensor(self):
        a = self.get_a_matrix()
        b = self.get_b_matrix()
        d = self.get_dipole_gradient_matrix()
        e = np.bmat([[a, b], [b, a]]).view(np.ndarray)
        t = np.bmat([[d], [d]]).view(np.ndarray)
        r = spla.solve(e, t, sym_pos=True)
        return t.T.dot(r)


if __name__ == "__main__":
    from scfexchange import Nuclei
    from scfexchange.pyscf_interface import Integrals
    from scfexchange.examples.phf import PerturbedHartreeFock

    labels = ("O", "H", "H")
    coordinates = np.array([[ 0.000000000000, -0.143225816552,  0.000000000000],
                            [ 1.638036840407,  1.136548822547, -0.000000000000],
                            [-1.638036840407,  1.136548822547, -0.000000000000]])

    nuclei = Nuclei(labels, coordinates)
    integrals = Integrals(nuclei, "sto-3g")
    phf = PerturbedHartreeFock(integrals)
    phf.solve(niter=300, e_threshold=1e-14, d_threshold=1e-13)
    rpa = RPA(phf)
    alpha = rpa.get_dipole_polarizability_tensor()

    import numdifftools as ndt

    energy_fn = phf.get_energy_field_function(niter=300, e_threshold=1e-14,
                                              d_threshold=1e-13)
    hess_diag_fn = ndt.Hessdiag(energy_fn, step=0.005, order=12,
                                method='central', full_output=True)
    hess_fn = ndt.Hessian(energy_fn, step=0.005, order=12, method='central',
                          full_output=True)
    hess_diag, hess_diag_results = hess_diag_fn(np.r_[0., 0., 0.])
    hess, hess_results = hess_fn(np.r_[0., 0., 0.])

    # This is the correct answer:
    print(-alpha.diagonal())  # ->[-7.93556221 -3.06821077 -0.05038621]

    # ndt.Hessdiag gives the correct answer, but seems to use a different
    # stepping scheme from the one I requested:
    print(hess_diag)          # ->[-7.93556221 -3.06821077 -0.05038621]

    # ndt.Hessian gives an incorrect answer, but seems to use the stepping
    # scheme I requested:
    print(hess.diagonal())    # ->[-7.93496262 -3.06834544 -0.05039031]

    # This is the output string from ndt.Hessdiag:
    print(hess_diag_results)
    # ->info(error_estimate=array([ 0.66617107,  0.66617107,  0.66617107])
    #        final_step=array([ 0.0524288,  0.0524288,  0.0524288])
    #        index=array([0, 1, 2]))

    # This is the output string from ndt.Hessian:
    print(hess_results)
    # ->info(error_estimate=array([[ 0.06353102,  0.06353102,  0.06353102],
    #                              [ 0.06353102,  0.06353102,  0.06353102],
    #                              [ 0.06353102,  0.06353102,  0.06353102]]),
    #        final_step=array([[ 0.005,  0.005,  0.005],
    #                          [ 0.005,  0.005,  0.005],
    #                          [ 0.005,  0.005,  0.005]]),
    #        index=array([0, 1, 2, 3, 4, 5, 6, 7, 8]) )
