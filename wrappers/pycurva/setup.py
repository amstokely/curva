import os.path

from setuptools import find_packages
from setuptools import setup

path = os.path.relpath(os.path.dirname(__file__))

setup(
    author='Andy Stokely',
    email='amstokely@ucsd.edu',
    name='pycurva',
    install_requires=[],
    platforms=['Linux',
               'Unix', ],
    python_requires="<=3.9",
    py_modules=[path + "/pycurva/pycurva"],
    packages=find_packages() + [''],
    zip_safe=False,
    package_data={
        '': [
            path + '/pycurva/_pycurva.so'
        ]
    },
)