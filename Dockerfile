# This Dockerfile supports amd64,arm64,ppc64le
# Note: QEMU emulated ppc64le build might take ~6 hours

# Use conda to resolve dependencies cross-platform
FROM continuumio/miniconda3:23.5.2-0 as builder

# install libpng to system for cross-architecture support
# https://github.com/ANTsX/ANTs/issues/1069#issuecomment-681131938
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      bash \
      build-essential \
      git \
      libpng-dev

# install cmake binary using conda for multi-arch support
# apt install fails because libssl1.0.0 is not available for newer Debian
RUN conda install -c anaconda cmake

WORKDIR /usr/local/src
COPY environment.yml .
RUN conda env update -n base
COPY . .

# number of parallel make jobs
ARG j=20
RUN pip --no-cache-dir -v install .

# run tests
RUN bash tests/run_tests.sh

# optimize layers
FROM debian:bullseye-20230919-slim
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH