# This Dockerfile supports amd64,arm64,ppc64le
# Note: QEMU emulated ppc64le build might take ~6 hours

# Use mamba to resolve dependencies cross-platform
FROM debian:trixie AS builder

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

# install miniforge3, which includes mamba
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -O miniforge.sh \
    && /bin/bash miniforge.sh -b -p /opt/conda \
    && rm miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /usr/local/src

COPY environment.yml .

# Update the base environment with mamba
RUN mamba info && \
    conda config --show-sources && \
    echo "Updating base environment" && \
    mamba env update -n base -f environment.yml && \
    echo "installing cmake" && \
    mamba install -y -n base cmake && \
    mamba clean -afy

COPY . .

# number of parallel make jobs
ARG j=2
RUN . /opt/conda/etc/profile.d/conda.sh && \
    pip --no-cache-dir -v install .

# run tests
RUN bash tests/run_tests.sh

# optimize layers
FROM debian:trixie-slim
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
