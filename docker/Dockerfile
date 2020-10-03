FROM python:3.8.6-buster as builder

# Build specific tag with
# pip install git+https://github.com/ANTsX/ANTsPy.git@tag 
#
# Maybe also reduce container size with
# pip install git+https://github.com/ANTsX/ANTsPy.git --install-option="--novtk"

RUN apt-get update && \
    apt-get install -y cmake=3.13.4-1 && \
    python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install wheel && \
    pip install git+https://github.com/ANTsX/ANTsPy.git && \
    git clone https://github.com/ANTsX/ANTsPy.git /opt/ANTsPy

FROM python:3.8.5-slim

COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

# Make a user and add data to the expected location
RUN useradd --create-home antspyuser && \
    mkdir /home/antspyuser/.antspy

COPY --from=builder /opt/ANTsPy/data/* /home/antspyuser/.antspy/
    
WORKDIR /home/antspyuser
USER antspyuser

ENTRYPOINT ["python"]

