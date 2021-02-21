from setuptools import find_packages, setup
from setuptools_rust import RustExtension

setup_requires = ['setuptools-rust>=0.11.6', 'pytest-runner']
install_requires = ['numpy', 'sympy']

setup(
    name='liesym',
    version='1.0.0',
    description='Sympy Lie Algebra extensions powered by rust.',
    rust_extensions=[RustExtension(
        'liesym.liesym',
        './Cargo.toml',
        debug=False
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    test_requires=["pytest"]
)