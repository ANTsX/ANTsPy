# use conda to resolve dependencies cross-platform
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

FROM debian:buster
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
