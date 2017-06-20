def test__mains():
    import importlib

    names = ('chem._constants', 'chem.elec', 'chem.nuc', 'examples.puhf',
             'examples.rpa', 'ao', 'mo', 'pyscf_interface', 'psi4_interface')
    mod_names = ('.'.join(['scfexchange', name]) for name in names)

    for mod_name in mod_names:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, '_main'):
            print(mod)
            mod._main()


