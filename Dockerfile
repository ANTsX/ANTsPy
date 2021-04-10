# This Dockerfile supports amd64,ppc64le
# Note: QEMU emulated ppc64le build might take ~6 hours

# Use conda to resolve dependencies cross-platform
FROM fnndsc/conda:4.9.2 as builder

# install libpng to system for cross-architecture support
# https://github.com/ANTsX/ANTs/issues/1069#issuecomment-681131938
RUN apt-get update && apt-get install -y build-essential git cmake libpng-dev

WORKDIR /usr/local/src
COPY environment.yml .
RUN conda env update -n base
COPY . .
ARG j=4
RUN pip --no-cache-dir install .

# optimize layers
FROM debian:buster
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
