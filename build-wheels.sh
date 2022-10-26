#!/bin/bash
set -ex

curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
export PATH="$HOME/.cargo/bin:$PATH"

cd /io

for PYBIN in /opt/python/cp{38,39,310}*/bin; do
    "${PYBIN}/pip" install -U wheel maturin
    "${PYBIN}/python" -m build
done

for whl in target/wheels/*.whl; do
    auditwheel repair "${whl}"
done
