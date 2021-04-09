# use conda to resolve dependencies cross-platform
# TODO: pin dependency versions
FROM fnndsc/conda:4.9.2 as base

FROM base as builder
RUN apt-get update && apt-get install -y build-essential git cmake

WORKDIR /usr/local/src
COPY environment.yml .
RUN conda env update -n base
COPY . .
RUN pip --no-cache-dir install .

FROM base

COPY --from=builder /opt/conda /opt/conda
