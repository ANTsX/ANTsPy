## Installing specific versions

We cannot store the entire history of releases because storage space on `pip` is limited. If you need an older release, you can check the [Github Releases page](https://github.com/ANTsX/ANTsPy/releases) or
build from source.

which will attempt to build from source (requires a machine with developer tools).

## Recent wheels

Non-release commits have wheels built automatically, which are available for download for a limited period.
Look under the [Actions tab](https://github.com/ANTsX/ANTsPy/actions). Then click on the commit for the software version you want.
Recent commits will have wheels stored as "artifacts".

Wheels are built locally like this:

```
rm -r -f build/ antspymm.egg-info/ dist/
python3 setup.py sdist bdist_wheel
pipx run twine upload dist/*
```

## Docker images

Available on [Docker Hub](https://hub.docker.com/repository/docker/antsx/antspy). To build
ANTsPy docker images, see the (installation tutorial)(https://github.com/ANTsX/ANTsPy/blob/master/tutorials/InstallingANTsPy.md#docker-installation).

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

## Installation methods

### Method 1: Pre-Compiled Binaries (preferred)

The fastest method is to install the pre-compiled binaries for the latest
stable, weekly release (takes ~1 min):

If you have MacOS:

```bash
pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.8/antspyx-0.1.8-cp37-cp37m-macosx_10_14_x86_64.whl
```

If you have Linux:

```bash
pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.2.0/antspyx-0.2.0-cp37-cp37m-linux_x86_64.whl
```

---

### Method 2: Github Master Branch

If you want the latest code, you can install directly from source (takes ~45 min):

```bash
pip install git+https://github.com/ANTsX/ANTsPy.git
```

with an option to specify the branch or particular release by `@v0.1.6` on the end of the path.

---

### Method 3: PyPI Source Distribution

If this doesn't work, you should install the latest stable source release from PyPI (takes ~45 min):

```bash
pip install -v antspy
```

ANTsPy will by default install with VTK in order to use the visualization functions such as
`ants.surf` and `ants.vol`. If you dont want VTK support, use the following:

```bash
git clone https://github.com/ANTsX/ANTsPy.git
cd ANTsPy
python setup.py install --novtk
```

### Method 4: Development Installation

If you want to develop code for ANTsPy, you should install the project as follows and
then refer to the [contributor's guide](CONTRIBUTING.md) for notes on project structure
and how to add code.

```bash
git clone https://github.com/ANTsX/ANTsPy.git
cd ANTsPy
python setup.py install
```

ANTsPy is known to install on MacOS, Ubuntu, and CentOS - all with Python3.6. It does not
currently work on Python 2.7, but we're planning on adding support.

## CentOS Installation

To install ANTsPy on CentOS (tested on "7") with virtual environment,
follow these commands:

background:

- follow python3.6 installation from [here](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7)
- create a virtual environment.
- clone `ANTsPy`

then call

```
sudo python3.6 setup.py develop
```

To use the toolkit, you then need to install dependencies:

```
 pip3.6 install numpy
 pip3.6 install pandas
 pip3.6 install pillow
 pip3.6 install webcolors
 pip3.6 install plotly
 pip3.6 install matplotlib
 sudo yum --enablerepo=ius-archive install python36u-tkinter
```

after this, you may try to run examples such as the following:

```
help( ants.vol )
help( ants.sparse_decom2 )
```

---

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
