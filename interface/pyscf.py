import math
import numpy as np
import pyscf


# Public
def overlap(basis, atoms, centers):
    molecule = _pyscf_molecule(basis, atoms, centers)
    return molecule.intor('cint1e_ovlp_sph')


def kinetic(basis, atoms, centers):
    molecule = _pyscf_molecule(basis, atoms, centers)
    return molecule.intor('cint1e_kin_sph')


def potential(basis, atoms, centers):
    molecule = _pyscf_molecule(basis, atoms, centers)
    return molecule.intor('cint1e_nuc_sph')


def dipole(basis, atoms, centers):
    molecule = _pyscf_molecule(basis, atoms, centers)
    return -molecule.intor('cint1e_r_sph', comp=3)


def electron_repulsion(basis, atoms, centers):
    molecule = _pyscf_molecule(basis, atoms, centers)
    matrix = molecule.intor('cint2e_sph')
    nbf = int(math.sqrt(matrix.shape[0]))
    return matrix.reshape((nbf, nbf, nbf, nbf)).transpose((0, 2, 1, 3))


def hf_mo_coefficients(basis, atoms, centers, charge=0, spin=0,
                       restricted=False, niter=100, e_threshold=1e-12,
                       d_threshold=1e-6):
    molecule = _pyscf_molecule(basis, atoms, centers, charge=charge, spin=spin)
    if restricted:
        hf = pyscf.scf.RHF(molecule)
    else:
        hf = pyscf.scf.UHF(molecule)
    hf.max_cycle = niter
    hf.conv_tol = e_threshold
    hf.conv_tol_grad = d_threshold
    hf.kernel()
    mo_coefficients = hf.mo_coeff
    if restricted:
        mo_coefficients = np.array([mo_coefficients] * 2)
    return mo_coefficients


# Private
def _pyscf_molecule(basis, atoms, centers, charge=0, spin=0):
    molecule = pyscf.gto.Mole(atom=zip(atoms, centers),
                              unit="bohr",
                              basis=basis,
                              charge=charge,
                              spin=spin)
    molecule.build()
    return molecule


def _main():
    import scfexchange as scfx

    atoms = ("O", "H", "H")
    centers = np.array([[0.0000000000, 0.0000000000, -0.1247219248],
                        [0.0000000000, -1.4343021349, 0.9864370414],
                        [0.0000000000, 1.4343021349, 0.9864370414]])
    charge = spin = 1

    t = kinetic("sto-3g", atoms, centers)
    v = potential("sto-3g", atoms, centers)
    g = electron_repulsion("sto-3g", atoms, centers)
    ac, bc = hf_mo_coefficients("sto-3g", atoms, centers, charge=charge,
                                spin=spin)
    na, nb = scfx.chem.elec.count_spins(atoms, charge=charge, spin=spin)
    ad = scfx.hf.density(ac, slc=slice(na))
    bd = scfx.hf.density(bc, slc=slice(nb))

    e_elec = scfx.hf.energy(t + v, g, ad, bd)
    e_nuc = scfx.chem.nuc.repulsion_energy(atoms, centers)
    e_tot = e_nuc + e_elec
    print(e_tot)

if __name__ == "__main__":
    _main()
