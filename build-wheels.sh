#!/bin/bash
set -ex

curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
export PATH="$HOME/.cargo/bin:$PATH"

for PYBIN in /opt/python/cp{38,39,310}*/bin; do
    "${PYBIN}/pip" install -U wheel maturin
    "${PYBIN}/python" -m maturin build -i "${PYBIN}/python" --release --manylinux 2014
done

for whl in target/wheels/*.whl; do
    auditwheel repair "${whl}"
done

for PYBIN in /opt/python/cp{310}*/bin; do
    "${PYBIN}/python" -m maturin upload -r testpypi
done