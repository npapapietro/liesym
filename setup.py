from setuptools import find_packages, setup
from setuptools_rust import RustExtension

setup_requires = ['setuptools-rust>=0.11.6', 'pytest-runner']
install_requires = [
    'numpy>=1.19.1',
    'sympy>=1.7.1'
]

setup(
    name='liesym',
    version='0.1.0',
    description='Sympy Lie Algebra extensions powered by rust.',
    rust_extensions=[RustExtension(
        'liesym.liesym',
        './Cargo.toml',
        debug=False
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    packages=find_packages(include=["liesym"]),
    zip_safe=False,
    include_package_data=True,
    test_requires=["pytest"],
    author="Nathan Papapietro <npapapietro95@gmail.com>",
    url="https://github.com/npapapietro/liesym",
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ]
)
