import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('pybees', parent_package, top_path)

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage('bees_algorithm')
    config.add_subpackage('bees_algorithm/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')

    # add the test directory
    config.add_subpackage('tests')    

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())