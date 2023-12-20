# This Dockerfile supports amd64,arm64,ppc64le
# Note: QEMU emulated ppc64le build might take ~6 hours

# Use conda to resolve dependencies cross-platform
FROM debian:bookworm-slim as builder

# install libpng to system for cross-architecture support
# https://github.com/ANTsX/ANTs/issues/1069#issuecomment-681131938
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      bash \
      build-essential \
      ca-certificates \
      git \
      libpng-dev \
      wget

# install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-1-Linux-$(uname -m).sh \
    && /bin/bash Miniconda3-py310_23.11.0-1-Linux-$(uname -m).sh -b -p /opt/conda \
    && rm Miniconda3-py310_23.11.0-1-Linux-$(uname -m).sh
ENV PATH=/opt/conda/bin:$PATH

# install cmake binary using conda for multi-arch support
# apt install fails because libssl1.0.0 is not available for newer Debian
RUN conda update -c defaults conda
RUN conda install -c conda-forge cmake

WORKDIR /usr/local/src
COPY environment.yml .
RUN conda env update -n base
COPY . .

# number of parallel make jobs
ARG j=2
RUN pip --no-cache-dir -v install .

# run tests
RUN bash tests/run_tests.sh

# optimize layers
FROM debian:bookworm-slim
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH