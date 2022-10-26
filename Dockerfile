FROM python:3.10
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
ENV PATH="$HOME/.cargo/bin:$PATH"

WORKDIR /app

RUN pip install -U pip
COPY . .
