## Installing ANTsPy from PyPI

The easiest way to install ANTsPy is from PyPI with pip:

```bash
pip install antspyx
```

Because of space limitations, we are only able to retain antspyx >= 0.4.2 on PyPI.
Starting with 0.5.3, refactoring of the C++ bindings with nanobind produces wheels that
are about 20x smaller, meaning we will be able to keep many more releases on PyPI going
forward.

## Github releases

Starting from 0.3.8, wheels are available for supported platforms on the [releases
page](https://github.com/ANTsX/ANTsPy/releases).


## Development wheels

Non-release commits have wheels built automatically, which are available for download for a limited period.
Look under the [Actions tab](https://github.com/ANTsX/ANTsPy/actions). A limited subset of wheels are [built
nightly](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels_faster.yml), a longer
workflow is [built weekly](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels.yml)
and covers more platforms and python versions.

Wheels are built locally like this:

```
rm -r -f build/ antspy.egg-info/ dist/
python3 setup.py sdist bdist_wheel
pipx run twine upload dist/*
```

## Docker images

Available on [Docker Hub](https://hub.docker.com/repository/docker/antsx/antspy). The
development version is pushed nightly, and release versions is pushed on release creation.
As with PyPI, space limitations may mean we have to remove older images in the future.


## Compiling from source

```bash
git clone https://github.com/ANTsX/ANTsPy.git
cd ANTsPy
pip install .
```
or
```bash
pip install git+https://github.com/ANTsX/ANTsPy.git
```
or install a specific version `${antspyx_version}`:

```bash
pip install git+https://github.com/ANTsX/ANTsPy.git@${antspyx_version}
```

where `antspyx_version` is a tag or commit hash.


## Other notes on compilation

In some cases, you may need some other libraries if they are not already installed eg if cmake says something about
a missing png library or a missing `Python.h` file.

```
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install gfortran
sudo apt-get install libpng-dev
sudo apt-get install python3-dev  # for python3.x installs
```

### Build documentation

```
cd docs
sphinx-apidoc -o source/ ../
make html
```

### Method 4: Development Installation

If you want to develop code for ANTsPy, you should install the project as follows and
then refer to the [contributor's guide](CONTRIBUTING.md) for notes on project structure
and how to add code.

```bash
git clone https://github.com/ANTsX/ANTsPy.git
cd ANTsPy
pip install -e .
```

This will make an "editable" installation of ANTsPy, meaning that you can modify the
Python code without having to reinstall the package. This is useful for Python development
but will not recompile C++ code (of ANTsPy or ANTs or ITK).


## Docker Installation

Tested on Ubuntu Linux (arch amd64), a standard docker command will build ANTsPy

```
cd ANTsPy
docker build -t antspy:latest .
```

For cross-platform builds with buildx, the following is suggested by @jennydaman:

```
# enable QEMU emulation for targeted foreign architectures
docker run --rm --privileged aptman/qus -s -- -p ppc64le aarch64
# enable advanced buildx features such as multi-platform support
docker buildx create --name moc_builder --use

# build the container image, using 6 make jobs
# (2 concurrent builds each using 6 jobs, recommended to have 12 CPU cores)
# targeting the platforms amd64 and ppc64le
# updating the base image first
# and finally pushing the result to a container registry
docker buildx build --pull --build-arg j=6 -t dockeruser/antspy:latest --platform linux/amd64,linux/ppc64le,linux/arm64 --push .

# optional clean-up
docker buildx rm
docker run --rm --privileged aptman/qus -- -r
```
