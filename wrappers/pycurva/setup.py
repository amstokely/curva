
from setuptools import find_packages
from setuptools import setup

setup(
    author='Andy Stokely',
    email='amstokely@ucsd.edu',
    name='pycurva',
    install_requires=[],
    platforms=['Linux',
               'Unix', ],
    python_requires="<=3.9",
    py_modules=["pycurva/pycurva"],
    packages=find_packages() + [''],
    zip_safe=False,
    package_data={
        '': [
            'pycurva/_pycurva.so'
        ]
    },
)

