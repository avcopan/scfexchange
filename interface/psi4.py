import numpy as np
import itertools as it
import psi4.core


# Public
def overlap(basis, atoms, centers):
    helper = _psi4_helper(basis, atoms, centers)
    return np.array(helper.ao_overlap())


def kinetic(basis, atoms, centers):
    helper = _psi4_helper(basis, atoms, centers)
    return np.array(helper.ao_kinetic())


def potential(basis, atoms, centers):
    helper = _psi4_helper(basis, atoms, centers)
    return np.array(helper.ao_potential())


def dipole(basis, atoms, centers):
    helper = _psi4_helper(basis, atoms, centers)
    return np.array(list(map(np.array, helper.ao_dipole())))


def electron_repulsion(basis, atoms, centers):
    helper = _psi4_helper(basis, atoms, centers)
    return np.array(helper.ao_eri()).transpose((0, 2, 1, 3))


def hf_mo_coefficients(basis, atoms, centers, charge=0, spin=0,
                       restricted=False, niter=100, e_threshold=1e-12,
                       d_threshold=1e-6):
    molecule = _psi4_molecule(atoms, centers, charge=charge, spin=spin)
    wavefunction = psi4.core.Wavefunction.build(molecule, basis)
    superfunctional, _ = psi4.driver.dft_funcs.build_superfunctional("HF",
                                                                     False)
    psi4.core.set_global_option("e_convergence", e_threshold)
    psi4.core.set_global_option("d_convergence", d_threshold)
    psi4.core.set_global_option("maxiter", niter)
    if restricted:
        if spin is 0:
            psi4.core.set_global_option("reference", "RHF")
            hf = psi4.core.RHF(wavefunction, superfunctional)
        else:
            psi4.core.set_global_option("reference", "ROHF")
            hf = psi4.core.ROHF(wavefunction, superfunctional)
    else:
        psi4.core.set_global_option("reference", "UHF")
        hf = psi4.core.UHF(wavefunction, superfunctional)
    hf.compute_energy()
    ac = np.array(hf.Ca())
    bc = np.array(hf.Cb())
    return np.array([ac, bc])


# Private
def _psi4_molecule(atoms, centers, charge=0, spin=0):
    format_fn = '{0:s} {1[0]:.20f} {1[1]:.20f} {1[2]:.20f}'.format
    molecule_string = '\n'.join(it.starmap(format_fn, zip(atoms, centers)))
    geom_string = '\n'.join([molecule_string, "units bohr"])
    molecule = psi4.core.Molecule.create_molecule_from_string(geom_string)
    molecule.set_molecular_charge(charge)
    molecule.set_multiplicity(spin + 1)
    molecule.reset_point_group("c1")
    molecule.update_geometry()
    return molecule


def _psi4_helper(basis, atoms, centers, charge=0, spin=0):
    molecule = _psi4_molecule(atoms, centers, charge=charge, spin=spin)
    basis, _ = psi4.core.BasisSet.build(molecule, "BASIS", basis)
    return psi4.core.MintsHelper(basis)


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
