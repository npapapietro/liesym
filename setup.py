from setuptools import find_packages, setup
from setuptools.command.install import install as InstallCommand
from os import path
import subprocess
import sys

setup_requires = [
    'setuptools-rust>=0.11.6',
    'twine>=3.4.2',
    "pytest-runner"
]
try:
    from setuptools_rust import RustExtension, Binding
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + setup_requires)
    from setuptools_rust import RustExtension, Binding

install_requires = [
    'numpy>=1.16',
    'sympy>=1.5',
    'symengine>=0.7.2'
]

dev_install_requires = install_requires + [
    "autopep8>=1.5.6",
    "jupyter>=1.0.0",
]

doc_install_requires = install_requires + [
    "Sphinx>=4.1.1",
    "groundwork-sphinx-theme>=1.1.1",
    "numpydoc>=1.1.0",
    "pydata_sphinx_theme>=0.5.2",
    "sphinx-math-dollar>=1.2"
]

test_requires = install_requires + [
    "pytest>=6.2.3",
]


class InstallDocs(InstallCommand):
    """ Customized setuptools install command which uses pip. """

    def run(self, *args, **kwargs):
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + doc_install_requires)


class InstallDev(InstallCommand):
    """ Customized setuptools install command which uses pip. """

    def run(self, *args, **kwargs):
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dev_install_requires)


class InstallTest(InstallCommand):
    """ Customized setuptools install command which uses pip. """

    def run(self, *args, **kwargs):
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + test_requires)


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
         

setup(
    name='liesym',
    version='0.5.3',
    description='Sympy Lie Algebra extensions powered by rust.',
    rust_extensions=[RustExtension(
        'liesym.liesym',
        './Cargo.toml',
        binding=Binding.PyO3,
        debug=False
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    python_requires=">=3.7",
    packages=find_packages(),
    test_requires=test_requires,
    zip_safe=False,
    include_package_data=True,
    author="Nathan Papapietro <npapapietro95@gmail.com>",
    url="https://github.com/npapapietro/liesym",
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmdclass={
        "setupdev": InstallDev,
        "setupdocs": InstallDocs,
        "setuptests": InstallTest,
    }
)