FROM texlive/texlive:latest

RUN apt-get update && apt install -y imagemagick

WORKDIR /app

COPY picgen.py picgen.py

CMD [ "python", "picgen.py" ]