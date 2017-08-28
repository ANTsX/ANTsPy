
# Advanced Normalization Tools in Python

What is this, a center for ANTs? <br>
How can we expect to teach children to analyze brain images, if they can't
even fit in the building? <br>
It needs to be at least... three times bigger than this. <br>

## Installation

To install, run the following:
```bash
git clone https://github.com/ANTsConsortium/ANTsPy.git
cd ANTsPy-master
python setup.py develop
```

By default, ANTsPy will search for an existing ITK installation. If that is not
found, it will install it for you. If you want to use 3D visualization tools
such as `ants.Surf` or `ants.Vol`, you need to install VTK on your own right now.