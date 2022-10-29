FROM quay.io/pypa/manylinux2014_x86_64:latest
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
ENV PATH="$HOME/.cargo/bin:$PATH"

WORKDIR /app
COPY . .
