from setuptools import find_packages, setup
from setuptools_rust import RustExtension, Binding
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup_requires = [
    'setuptools-rust>=0.11.6', 
    'pytest-runner'
]

install_requires = [
    'numpy>=1.16',
    'sympy>=1.5',
]

setup(
    name='liesym',
    version='0.4.1',
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
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
