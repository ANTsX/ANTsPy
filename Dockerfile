# This Dockerfile supports amd64,ppc64le
# Note: QEMU emulated ppc64le build might take ~6 hours

# Use conda to resolve dependencies cross-platform
FROM continuumio/miniconda3:4.11.0 as builder

# install libpng to system for cross-architecture support
# https://github.com/ANTsX/ANTs/issues/1069#issuecomment-681131938
# Also install kitware key and get recent cmake
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      build-essential \
      ca-certificates \
      curl \
      git \
      gnupg \
      libpng-dev \
      software-properties-common 

# Install cmake from binary
# apt install fails because libssl1.0.0 is not available for newer Debian
# Download verification stuff from https://cmake.org/install/
ARG CMAKE_VERSION=3.23.1
RUN curl -OL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-SHA-256.txt && \
    curl -OL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    sha256sum -c --ignore-missing cmake-${CMAKE_VERSION}-SHA-256.txt && \
    curl -OL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-SHA-256.txt.asc && \
    gpg --keyserver hkps://keyserver.ubuntu.com --recv-keys C6C265324BBEBDC350B513D02D2CEF1034921684 && \
    gpg --verify cmake-${CMAKE_VERSION}-SHA-256.txt.asc cmake-${CMAKE_VERSION}-SHA-256.txt
    
RUN mkdir /opt/cmake && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/opt/cmake

ENV PATH=/opt/cmake/bin:${PATH}

WORKDIR /usr/local/src
COPY environment.yml .
RUN conda env update -n base
COPY . .
# number of parallel make jobs
ARG j=4
RUN pip --no-cache-dir -v install .

# optimize layers
FROM debian:bullseye-slim
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
