FROM npapapietro/pythonbundles:py38rust

# ENV RUSTUP_HOME=/usr/local/rustup \
#     CARGO_HOME=/usr/local/cargo \
#     PATH=/usr/local/cargo/bin:$PATH

# RUN set -eux; \
#     \
#     url="https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"; \
#     wget "$url"; \
#     chmod +x rustup-init; \
#     ./rustup-init -y --no-modify-path --default-toolchain stable; \
#     rm rustup-init; \
#     chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
#     rustup --version; \
#     cargo --version; \
#     rustc --version;

WORKDIR /app
COPY . .
RUN pip install -U pip
RUN pip install -U setuptools setuptools-rust twine wheel numpy pytest auditwheel maturin
ENV MATURIN_PASSWORD=pypi-AgENdGVzdC5weXBpLm9yZwIkZDZkZGNhMDMtMmM0Mi00Nzg4LTg0MGItOTM0MDliZjkxM2UxAAIleyJwZXJtaXNzaW9ucyI6ICJ1c2VyIiwgInZlcnNpb24iOiAxfQAABiDpCyQTXGg8Bqlq9AicNfB3RAcwfAUHYmx2BK7hruofoA
RUN maturin publish --no-sdist -u __token__ -r https://test.pypi.org/legacy/

# RUN maturin publish --no-sdist -u __token__ -r https://test.pypi.org/legacy/
# RUN cargo test --no-default-features

# RUN poetry run maturin develop --release
