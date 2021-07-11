FROM python:3.8

RUN pip install -U pip
COPY . .
RUN python setup.py bdist_wheel