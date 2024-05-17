# ANTsPy Contributors Guide

This guide tells you everything you need to know to set up the development workflow that will allow you to contribute code to antspy.

## Installing ANTsPy Development Environment

To start developing, you need to install a development copy of ANTsPy. This follows the same steps as developing any python package.

```bash
git clone https://github.com/ANTsX/ANTsPy.git
cd ANTsPy
python -m pip install -v -e .
```

Notice the `-v` flag to have a verbose output so you can follow the build process that can take ~45 minutes. Then there is also the `-e` flag that will build the antspy package in such a way that any changes to the python code will be automatically detected when you restart your python terminal without having to build the package again.

Any changes to C++ code will require you to run that last line (`python -m pip install -v -e .`) again to rebuild the compiled libraries.

## What happens when I install ANTsPy?

When you run `python -m pip install .` or `python -m pip install -e .` to install antspy from source, the CMakeLists.txt file is run. Refer there if you want to change any part of the install process. Briefly, it performs the following steps:

1. ITK is cloned and built from the `scripts/configure_ITK.sh` file.
2. ANTs is cloned from `scripts/configure_ANTs.sh`
3. The C++ files from the `src` directory are used to build the antspy library files
4. The antspy python package is built as normal

## Wrapping core ANTs functions

Wrapping an ANTs function is easy since pybind11 implicitly casts between python and C++ standard types, allowing you to directly interface with C++ code. Here's an example:

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

The `WRAP_function.cxx` file is the SAME for every ANTs function - simply exchange the word "Atropos" with whatever the function name is.

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
- pass those raw arguments through the function `process_arguments(args)`
- pass those processed arguments into the library call (e.g. `lib.Atropos(processed_args)`).

## Writing custom code for antspy

You can write any kind of custom code to process antspy images. The underlying image is ITK so the AntsImage class holds a pointer to the underlying ITK object in the in the property `self.pointer`.

To go from a C++ ANTsImage class to an ITK image, pass in an `AntsImage<ImageType>` argument (`image.pointer` in python) and call `.ptr` to access the ITK image.

```cpp
#include "antsImage.h"

template <typename ImageType>
ImageType::Pointer getITKImage( AntsImage<ImageType> antsImage )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = antsImage.ptr;
    return itkImage
}
```

Now, say you wrapped this code and wanted to call it from python. You wouldn't pass
the Python ANTsImage object directly, you would pass in the `self.pointer` attribute which
contains the ITK image pointer.

### Example 1 - getOrigin

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
std::vector getOrigin( AntsImage<ImageType> antsImage )
{
    // cast to ITK image as shown above
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = antsImage.ptr;

    // do everything else as normal with ITK Imaeg
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

from ants import lib # use relative import e.g. "from .. import lib" in package code

def get_origin(img):
    idim = img.dimension
    ptype = img.pixeltype

    # call function - NOTE how we pass in `img.pointer`, not `img` directly
    origin = lib.getOrigin(img.pointer)

    # return as tuple
    return tuple(origin)
```

And that's it! For more other return types, you should refer to the nanobind docs.

## Wrapping an ITK image for antspy

In the previous section, we saw how easy it is to cast from AntsImage to ITK Image by calling `antsImage.ptr`. It is also easy to go the other way and wrap an ITK image as an AntsImage.

Here is an example:

```cpp
#include "itkImage.h" // include any other ITK imports as normal
#include "antsImage.h"

template <typename ImageType>
AntsImage<ImageType> someFunction( AntsImage<ImageType> antsImage )
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

## Running Tests

All tests can be executed by running the following command from the main directory:

```bash
sh ./tests/run_tests.sh
```

Similarly, code coverage can be calculated by adding a flag to the above command:

```bash
sh ./tests/run_tests.sh -c
```

This will create an html folder in the `tests` directory with detailed coverage information.
In order to publish this information to coveralls, we run the following (given that the
.coveralls.yml file is in the tests directory with the correct repo token):

```bash
./tests/run_tests.sh -c
cd tests
coveralls
```

We use https://coveralls.io/ for hosting our coverage information.

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
