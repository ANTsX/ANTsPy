# This Dockerfile supports amd64,arm64,ppc64le
# Note: QEMU emulated ppc64le build might take ~6 hours

# Use conda to resolve dependencies cross-platform
FROM debian:bookworm as builder

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

WORKDIR /usr/local/src

COPY environment.yml .

# Activate the base environment and update it
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda info && \
    conda config --show-sources && \
    echo "Updating conda" && \
    conda env update -n base && \
    echo "installing cmake" && \
    conda install -c conda-forge cmake

COPY . .

# number of parallel make jobs
ARG j=2
RUN . /opt/conda/etc/profile.d/conda.sh && \
    pip --no-cache-dir -v install .

# run tests
RUN bash tests/run_tests.sh

# optimize layers
FROM debian:bookworm-slim
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
