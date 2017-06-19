def test__mains():
    import importlib

    names = ('ao', 'mo', 'constants', 'chem', 'psi4_interface',
             'pyscf_interface', 'examples.rpa', 'examples.puhf')
    module_names = ('.'.join(['scfexchange', name]) for name in names)

    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, '_main'):
            print(module)
            module._main()


if __name__ == "__main__":
    test__mains()
