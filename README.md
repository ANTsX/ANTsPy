
# Advanced Normalization Tools in Python

## Installation

To install, run the following:
```bash
git clone https://github.com/ncullen93/ANTsPy.git
cd ANTsPy-master
python setup.py develop
```

By default, ANTsPy will search for an existing ITK installation. If that is not
found, it will install it for you. If you want to use 3D visualization tools
such as `ants.Surf` or `ants.Vol`, you need to install VTK on your own right now.