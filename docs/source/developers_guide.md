
# ANTsPy Developers Guide

So, you want to contribute to ANTsPy - great! This guide tells you everything
you need to know to set up an efficient workflow and get started writing code.

## Installing ANTsPy Development Environment

To start developing, you need to install a development copy of ANTsPy.
This is straight-forward, just run the following commands:

```bash
git clone https://github.com/ANTsX/ANTsPy.git
cd ANTsPy
python setup.py develop
```

The above commands will take ~45 min, so go ride your bike around the block or
something until it finishes. With the development environment, you get the following:

- if you change PURE PYTHON code only, just restart the python kernel and import
  as normal - the code will be automatically updated.
- if you change any C++ code, just go back to the main directory and run
  `python setup.py develop` - only the changed code will be re-compiled, so it will
  be quick.


## What happens when I install ANTsPy?

When you run `python setup.py install` or `python setup.py develop` to install
ANTsPy from source, the following steps happen. Refer here if you want to change
any part of the install process.

1. The `setup.py` file is run. The entire build runs from here.
2. We check for a local ITK copy by seeing if the environment variable $ITK_DIR is set.
2. If there is a local ITK build found, move on. Otherwise, clone the ITK repo
and make it by running the script `configure_ITK.sh`. This does NOT run `make install`,
so you will not now have a local ITK build.. that could be changed.
3. After ITK is built, the `configure_ANTsPy.sh` script is run. 
This clones the core ANTs repo and copies all of the source files 
into the `ants/lib/` directory. Note that ANTs core is not actually built on its own directly.
Then, a copy of `pybind11` is cloned and put into the `ants/lib/` directory - 
which is how we can actually wrap C++ code in Python. Finally, the example data/images
which ship with ANTsPy (in the `data/` directory) are moved to a folder in the user's
home directory (`~/.antspy/`). 
4. Then, the ANTsPy library is built by running the cmake command in the `ants/lib/`
directory and referring to the `ants/lib/CMakeLists.txt` file. This builds all of the
shared object libraries.
5. Finally, the library is installed as any other python package. To see what happens
there (depends on whether `develop` or `install` was run), refer to the official
python documentation. 


## Software Architecture

This section describes the general architecture of the package.

- The python code has the following modules:
    - `core` : contains all core object class definitions such as `ANTsImage`, `ANTsTransform`,
        and `ANTsImageToImageMetric`. Also contains all IO related to these objects.
    - `learn` : contains any code related to Statistical Learning, such as `sparse_decom2`.
    - `lib` : contains any C++ wrapping code. Any file which has code which does not 
        directly wrap ANTs functions are prefaced with `LOCAL_`. 
        Any code which directly wraps ANTs functions are prefaced with `WRAP_`.
    - `registration` : contains all code related to image registration.
    - `segmentation` : contains all code related to image segmentation.
    - `utils` : contains all code which perform random functions or functions which are
        not easily categorized elsewhere.
    - `viz` : contains any visualization code.


## How do I wrap an ANTs core function?

Wrapping an ANTs function is ridiculously easy since pybind11 implicitly casts between
python and C++ standard types, allowing you to directly interface with C++ code. 
Here's an example:

Say we want to wrap the `Atropos` function. We create the following file called 
`WRAP_Atropos.cxx` in the `ants/lib/` directory:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/Atropos.h"

namespace py = pybind11;

int Atropos( std::vector<std::string> instring )
{
    return ants::Atropos(instring, NULL);
}

PYBIND11_MODULE(Atropos, m)
{
  m.def("Atropos", &Atropos);
}
```

The `WRAP_function.cxx` file is the SAME for every ANTs function - simply exchange the 
word "Atropos" with whatever the function name is.

Next, we add the following two lines in the `ants/lib/CMakeLists.txt` file:

```
pybind11_add_module(Atropos antscore/Atropos.cxx WRAP_Atropos.cxx)
target_link_libraries(Atropos PRIVATE ${ITK_LIBRARIES} antsUtilities)
```

Again, these two lines are the same for every function with only the function named changed.

Finally, we add the following line in the `ants/lib/__init__.py` file:

```
from .Atropos import *
```

That's it! Now you have access to the Atropos function by calling `ants.lib.Atropos(...)`. 
Of course, we always add some python wrapper code instead of directly calling the library
function. The general workflow for wrapping a library calls involves the following steps:

- write a wrapper python function (e.g. `def atropos(...)`)
- build up a list or dictionary of string argument names as in ANTs
- pass those raw arguments through the function `utils._int_antsProcessArguments(args)`
- pass those processed arguments into the library call (e.g. `lib.Atropos(processed_args)`).



