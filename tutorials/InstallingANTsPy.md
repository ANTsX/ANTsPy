
## Installing ANTsPy

### Method 1: Pre-Compiled Binaries (preferred)
The fastest method is to install the pre-compiled binaries for the latest 
stable, weekly release (takes ~1 min):

If you have MacOS:
```bash
pip install https://github.com/ANTsX/ANTsPy/releases/download/Weekly/antspy-0.1.4a0-cp36-cp36m-macosx_10_7_x86_64.whl
```

If you have Linux:
```bash
pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
```

------------------------------------------------------------------------------
### Method 2: Github Master Branch
If you want the latest code, you can install directly from source (takes ~45 min):

```bash
pip install git+https://github.com/ANTsX/ANTsPy.git
```
with an option to specify the branch or particular release by `@v0.1.6` on the end of the path.

------------------------------------------------------------------------------

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

* follow python3.6 installation from [here](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7)
* create a virtual environment.
* clone `ANTsPy`

then call

```
sudo python3.6 setup.py develop
```

To use the toolkit, you then need to install dependencies:

```
 pip3.6 install numpy
 pip3.6 install pandas
 pip3.6 install pillow
 pip3.6 install sklearn
 pip3.6 install scikit-image
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

------------------------------------------------------------------------------
