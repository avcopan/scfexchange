def test__mains():
    import importlib

    names = ('_constants', 'chem', 'ao', 'mo', 'psi4_interface',
             'pyscf_interface', 'examples.puhf', 'examples.rpa')
    mod_names = ('.'.join(['scfexchange', name]) for name in names)

    for mod_name in mod_names:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, '_main'):
            print(mod)
            mod._main()


