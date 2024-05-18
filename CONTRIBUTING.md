# Guide to contributing to ANTsPy

ANTsPy is a Python library that primarily wraps the ANTs C++ library, but also contains a lot of custom C++ and Python code. There are a decent amount of moving parts to get to familiar with before being able to contribute, but we've done our best to make the process easy.

This guide tells you everything you need to know about ANTsPy to add or change any code in the project. The guide is composed of the following sections.

- Project structure
- Setting up a dev environment
- Wrapping ANTs functions
- Adding C++ / ITK code
- Adding Python code
- Running tests

The first two sections and the last section should be read by everyone, but the other sections can be skipped depending on your goal.

<br />

## Project structure

The ANTsPy project consists of multiple folders which are listed and explained here.

- **.github/** : contains all GitHub actions
- **ants/** : contains the Python code for the library
- **data/** : contains any data (images) included with the installed library
- **docs/** : contains the structure for building the library documentation
- **scripts/** : contains scripts to build / clone ITK and ANTs during installation
- **src/** : contains the C++ code for the library
- **tests/** : contains all tests
- **tutorials/** contains all .md and .ipynb tutorials

If you are adding code to the library, the three folders you'll care about most are `ants/` (to add Python code), `src/` (to add C++ code), and `tests/` (to add tests).

### Nanobind

The C++ code is wrapped using [nanobind](https://nanobind.readthedocs.io/en/latest/). It is basically an updated version of pybind11 that makes it easy to call C++ functions from Python. Having a basic understanding of nanobind can help in some scenarios, but it's not strictly necessary.

The `CMakeLists.txt` file and the `src/main.cpp` file contains most of the information for determining how nanobind wraps and builds the C++ files in the project.

### Scikit-build

The library is built using [scikit-build](https://scikit-build.readthedocs.io/en/latest/), which is a modern alternative to `setup.py` files for projects that include C++ code.

The `pyproject.toml` file is the central location for steering the build process. If you need to change the way the library is built, that's the best place to start.

<br />

## Setting up a dev environment

To start developing, you need to build a development copy of ANTsPy. This process is the same as developing for any python package.

```bash
git clone https://github.com/ANTsX/ANTsPy.git
cd ANTsPy
python -m pip install -v -e .
```

Notice the `-v` flag to have a verbose output so you can follow the build process that can take 30 - 45 minutes. Then there is also the `-e` flag that will build the library in such a way that any changes to the Python code will be automatically detected when you restart your python terminal without having to build the package again.

Any changes to C++ code will require you to run that last line (`python -m pip install -v -e .`) again to rebuild the compiled libraries. However, it should not take more than a couple of minutes if you've only made minor changes or additions.

### What happens when you install ANTsPy

When you run `python -m pip install .` or `python -m pip install -e .` to install antspy from source, the CMakeLists.txt file is run. Refer there if you want to change any part of the install process. Briefly, it performs the following steps:

1. ITK is cloned and built from the `scripts/configure_ITK.sh` file.
2. ANTs is cloned from `scripts/configure_ANTs.sh`
3. The C++ files from the `src` directory are used to build the antspy library files
4. The antspy python package is built as normal

<br />

## Wrapping ANTs functions

Wrapping an ANTs function is easy since nanobind implicitly casts between python and C++ standard types, allowing you to directly interface with C++ code. Here's an example:

Say we want to wrap the `Atropos` function. We create the following file called
`WRAP_Atropos.cxx` in the `src/` directory:

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/Atropos.h"

namespace nb = nanobind;
using namespace nb::literals;

int Atropos( std::vector<std::string> instring )
{
    return ants::Atropos(instring, NULL);
}

void wrap_Atropos(nb::module_ &m)
{
  m.def("Atropos", &Atropos);
}
```

The `WRAP_Atropos.cxx` file is the same for every ANTs function - simply exchange the word "Atropos" with whatever the function name is and include the correct file.

Next, we add the following two lines to the top of the `src/main.cpp` file:

```c++
#include "WRAP_Atropos.cxx"
void wrap_Atropos(nb::module_ &);
```

Then, we add the following line inside the `NB_MODULE(lib, m) { ... }` call in the same file:

```c++
wrap_Atropos(m);
```

Rebuilding the package should make the `lib.Atropos` function available for you. However, remember that lib functions should never be called directly by users, so you have to add the python wrapper code to process the arguments and call this underlying lib function.

The general workflow for wrapping a library calls involves the following steps:

- write a wrapper python function (e.g. `def atropos(...)`)
- build up a list or dictionary of string argument names as in ANTs
- pass those raw arguments through the function `myargs = process_arguments(args)`
- get the library function by calling `libfn = get_lib_fn('Atropos')`
- pass those processed arguments into the library function (e.g. `libfn(myargs)`).

<br />

## Adding C++ / ITK code

You can write any kind of custom code to process images in ANTsPy. The ANTsImage class holds a pointer to the underlying ITK object in the in the property `self.pointer`.

To go from a C++ ANTsImage to an ITK image, pass in an `AntsImage<ImageType>` argument (`image.pointer` in python) and call `.ptr` to access the ITK image.

```cpp
#include "antsImage.h"

template <typename ImageType>
ImageType::Pointer getITKImage( AntsImage<ImageType> & antsImage )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = antsImage.ptr;
    return itkImage
}
```

Now, say you wrapped this code and wanted to call it from python. You wouldn't pass
the Python ANTsImage object directly, you would pass in the `self.pointer` attribute which
contains the ITK image pointer.

### Example - getOrigin

Let's do a full example where we get the origin of a Python AntsImage from the underlying ITK image.

We would create the following file `src/getOrigin.cxx`:

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "itkImage.h" // any ITK or other includes

#include "antsImage.h" // needed for casting to & from ANTsImage<->ITKImage

// all functions accepted ANTsImage types must be templated
template <typename ImageType>
std::vector getOrigin( AntsImage<ImageType> & antsImage )
{
    // cast to ITK image as shown above
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = antsImage.ptr;

    // do everything else as normal with ITK Image
    typename ImageType::PointType origin = image->GetOrigin();
    unsigned int ndim = ImageType::GetImageDimension();

    std::vector originlist;
    for (int i = 0; i < ndim; i++)
    {
        originlist.append( origin[i] );
    }

    return originlist;
}

// wrap function above with all possible types
void getOrigin(nb::module_ &m)
{
    m.def("getOrigin", &getOrigin<itk::Image<unsigned char,2>>);
    m.def("getOrigin", &getOrigin<itk::Image<unsigned char,3>>);
    m.def("getOrigin", &getOrigin<itk::Image<float,2>>);
    m.def("getOrigin", &getOrigin<itk::Image<float,3>>);
    // ...
}

```

Now we add the following lines in `src/main.cpp` :

```c++
#include "getOrigin.cxx"
void getOrigin(nb::module_ &);
```

And add the following line to the same file inside the `NB_MODULE(lib, m) { ... }` call:

```
getOrigin(m);
```

Finally, we create a wrapper function in python file `get_origin.py`. Notice that the `lib.getOrigin` is overloaded so that it can automatically infer the ITK ImageType.

```python

from ants.decorators import image_method
from ants.internal import get_lib_fn

@image_method
def get_origin(image):
    libfn = get_lib_fn('getOrigin')
    origin = libfn(image.pointer)

    return tuple(origin)
```

And that's it! More details about how to write Python code for ANTsPy is presented below. For other return types, consult the nanobind docs. However, most C++ types will be automatically converted to the corresponding Python types - both arguments and return values.

### Wrapping an ITK image

In the previous section, we saw how easy it is to cast from AntsImage to ITK Image by calling `antsImage.ptr`. It is also easy to go the other way and wrap an ITK image as an AntsImage.

Here is an example:

```cpp
#include "itkImage.h" // include any other ITK imports as normal
#include "antsImage.h"

template <typename ImageType>
AntsImage<ImageType> someFunction( AntsImage<ImageType> & antsImage )
{
    // cast from ANTsImage to ITK Image
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = antsImage.ptr;

    // do some stuff on ITK image
    // ...

    // wrap ITK Image in AntsImage struct
    AntsImage<ImageType> outImage = { itkImage };

    return outImage;
}
```

If the function doesnt return the same image type, you need two template arguments:

```cpp
#include "itkImage.h" // include any other ITK imports as normal
#include "antsImage.h"

template <typename InImageType, typename OutImageType>
AntsImage<OutImageType> someFunction( AntsImage<InImageType> antsImage )
{
    // cast from ANTsImage to ITK Image
    typedef typename InImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = antsImage.ptr;

    // do some stuff on ITK image
    // ...

    // wrap ITK Image in AntsImage struct
    AntsImage<OutImageType> outImage = { itkImage };

    return outImage;
}
```

<br />

## Adding Python code

If you want to add custom Python code that calls other ANTsPy functions or the wrapped code, there are a few things to know. The `label_clusters` function provides a good example of how to do so.

```python
import ants
from ants.internal import get_lib_fn, process_arguments
from ants.decorators import image_method

@image_method
def label_clusters(image, min_cluster_size=50, min_thresh=1e-6, max_thresh=1, fully_connected=False):
    """
    This will give a unique ID to each connected
    component 1 through N of size > min_cluster_size
    """
    clust = ants.threshold_image(image, min_thresh, max_thresh)
    args = [image.dimension, clust, clust, min_cluster_size, int(fully_connected)]
    processed_args = process_arguments(args)
    libfn = get_lib_fn('LabelClustersUniquely')
    libfn(processed_args)
    return clust
```

First, notice the imports at the top. You generally need three imports. First, you need to import the library so that all other internal functions (such as `ants.threshold_image`) are available.

```python
import ants
```

Next, you need import a few functions from `ants.internal` that let you get a function from the compiled C++ library (`get_lib_fn`) and that let you combined arguments into the format ANTs expects (`process_arguments`). Note that `process_arguments` is only needed if you are called a wrapped ANTs function.

```python
from ants.internal import get_lib_fn, process_arguments
```

Finally, you should import `image_method` from `ants.decorators`. This decorator lets you attach a function to the ANTsImage class so that the function can be chained to the image. This is why you can call `image.dosomething()` instead of only `ants.dosomething(image)`.

```python
from ants.decorators import image_method
```

With those three imports, you can call any internal Python function or any C++ function (wrapped or custom).

<br />

## Running tests

Whenever you add or change code in a meaningful way, you should add tests. All tests can be executed by running the following command from the main project directory:

```bash
sh ./tests/run_tests.sh
```
Refer to that file for adding tests. We use the `unittest` module for creating tests, which
generally have the following structure:

```python
class TestMyFunction(unittest.TestCase):

    def setUp(self):
        # add whatever code here to set up before all the tests
        # examples include loading a bunch of test images
        pass

    def tearDown(self):
        # add whatever code here to tear down after all the tests
        pass

    def test_function1(self):
        # add whatever here
        # use self.assertTrue(...), self.assertEqual(...),
        # nptest.assert_close(...), etc for tests
        pass

    def test_function2(self):
        # add whatever here
        # use self.assertTrue(...), self.assertEqual(...),
        # nptest.assert_close(...), etc for tests
        pass

    ...
```

Tests are actually carried out through assertion statements such as `self.assertTrue(...)`
and `self.assertEqual(...)`.
