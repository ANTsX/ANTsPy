# ANTsPy Contributors Guide

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

The above commands will take ~1 hour, so go ride your bike around the block or
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
2. We check for a local VTK copy by seeing if the environment variable $VTK_DIR is set.
3. If there is a local VTK build found, move on. Otherwise, clone the VTK repo
and make it by running the script `configure_VTK.sh`. This does NOT run `make install`.
4. We check for a local ITK copy by seeing if the environment variable $ITK_DIR is set.
5. If there is a local ITK build found, move on. Otherwise, clone the ITK repo
and make it by running the script `configure_ITK.sh`. This does NOT run `make install`.
6. After VTK and ITK are built, the `configure_ANTsPy.sh` script is run. 
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
    - `contrib` : contains any experimental code or code which significantly differs from ANTsR.
    - `core` : contains all core object class definitions and all IO related to these objects.
    - `learn` : contains any code related to Statistical Learning, such as `sparse_decom2`.
    - `lib` : contains any C++ wrapping code. Any file which has code which does not 
        directly wrap ANTs functions are prefaced with "LOCAL_". 
        Any code which directly wraps ANTs functions are prefaced with "WRAP_".
    - `registration` : contains all code related to image registration.
    - `segmentation` : contains all code related to image segmentation.
    - `utils` : contains all code which perform utility functions or functions which are
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

## How do I go from an ANTsImage to an ITK Image?

First off, the python-based ANTsImage class holds a pointer to the underlying
ITK object in the property `self.pointer`.

To go from a C++ ANTsImage class to an ITK image, use the following code:

```cpp
#include "LOCAL_antsImage.h"

template <typename ImageType>
ImageType::Pointer getITKImage( py::capsule antsImage )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = as<ImageType>( antsImage );
    return itkImage
}
```

Now, say you wrapped this code and wanted to call it from python. You wouldn't pass
the Python ANTsImage object directly, you would pass in the `self.pointer` attribute which
contains the ITK image pointer.

### Example 1 - getOrigin

Let's do a full example where we get the origin of a Python ANTsImage from the 
underlying ITK image.

We would create the following file `ants/lib/LOCAL_getOrigin.cxx`:

```cpp
#include <pybind11/pybind11.h> // needed for wrapping
#include <pybind11/stl.h> // needed for casting from std::vector to python::list

#include "itkImage.h" // any ITK or other includes

#include "LOCAL_antsImage.h" // needed for casting to & from ANTsImage<->ITKImage

// all functions accepted ANTsImage types must be templated
template <typename ImageType>
std::vector getOrigin( py::capsule antsImage )
{
    // cast to ITK image as shown above
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = as<ImageType>( antsImage );

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
PYBIND11_MODULE(getOrigin, m)
{
    m.def("getOriginUC2", &getOrigin<itk::Image<unsigned char,2>>);
    m.def("getOriginUC3", &getOrigin<itk::Image<unsigned char,3>>);
    m.def("getOriginF2", &getOrigin<itk::Image<float,2>>);
    m.def("getOriginF3", &getOrigin<itk::Image<float,3>>);
}

```

Now we add the following lines in `ants/lib/CMakeLists.txt` :

```
pybind11_add_module(getOrigin LOCAL_getOrigin.cxx)
target_link_libraries(getOrigin PRIVATE ${ITK_LIBRARIES})
```

And add the following line in `ants/lib/__init__.py`:

```
from .getOrigin import *
```

Finally, we create a wrapper function in python file `get_origin.py`:

```python

from ants import lib # use relative import e.g. "from .. import lib" in package code

# dictionary for storing template-wrapped library function
_get_origin_dict = {
    2: {
        'unsigned char': 'getOriginUC2',
        'float': 'getOriginF2'
    },
    3: {
        'unsigned char': 'getOriginUC3',
        'float': 'getOriginF3'
    }
}

def get_origin(img):
    idim = img.dimension
    ptype = img.pixeltype

    # get the template function corresponding to image dimension and pixeltype
    _get_origin_fn = lib.__dict__[_get_origin_dict[idim][ptype]]

    # call function - NOTE how we pass in `img.pointer`, not `img` directly
    origin = _get_origin_fn(img.pointer)

    # return as tuple
    return tuple(origin)
```

And that's it! For more other return types, you should refer to the pybind11 docs.

## How do I go from an ITK Image to an ANTsImage?

In the previous section, we saw how easy it is to cast from ANTsImage to ITK Image 
using the `as<ImageType>( antsImage )` function. The same is true for going the other way.

Example:

```cpp
#include "itkImage.h" // include any other ITK imports as normal
#include "LOCAL_antsImage.h"

template <typename ImageType>
py::capsule someFunction( py::capsule antsImage )
{
    // cast from ANTsImage to ITK Image
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = as<ImageType>( antsImage );
    
    // do some stuff on ITK image
    // ...

    // cast from ITK Image to ANTsImage
    py::capsule newAntsImage;
    newAntsImage = wrap<ImageType>( itkImage );

    return newAntsImage;
}
```

If the function doesnt return the same image type, you need two template arguments:

```cpp
#include "itkImage.h" // include any other ITK imports as normal
#include "LOCAL_antsImage.h"

template <typename InImageType, typename OutImageType>
py::capsule someFunction( py::capsule antsImage )
{
    // cast from ANTsImage to ITK Image of InImageType
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = as<InImageType>( antsImage );
    
    // do some stuff on ITK image
    // ...

    // cast from ITK Image to ANTsImage of OutImageType
    py::capsule newAntsImage;
    newAntsImage = wrap<OutImageType>( itkImage );

    return newAntsImage;
}
```

So you see we use  `as<ImageType>( antsImage )` to go from ANTsImage to ITK Image
and `wrap<ImageType>( itkImagePointer)` to go from ITK Image to ANTsImage

### Example 2 - Cloning an Image

In this example, I will show how to clone an ANTsImage directly in ITK.
First, we create the C++ file `ants/lib/LOCAL_antsImageClone.cxx`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "itkImage.h"
#include "itkImageFileWriter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template<typename InImageType, typename OutImageType>
py::capsule antsImageClone( py::capsule antsImage )
{
  // ---------------------------------------------
  // cast from ANTsImage to ITK Image
  typedef typename InImageType::Pointer InImagePointerType;
  InImagePointerType in_image = as< InImageType >( antsImage );

  // ---------------------------------------------
  // do stuff on ITK Image ...
  typename OutImageType::Pointer out_image = OutImageType::New() ;
  out_image->SetRegions( in_image->GetLargestPossibleRegion() ) ;
  out_image->SetSpacing( in_image->GetSpacing() ) ;
  out_image->SetOrigin( in_image->GetOrigin() ) ;
  out_image->SetDirection( in_image->GetDirection() );
  //out_image->CopyInformation( in_image );
  out_image->AllocateInitialized() ;

  itk::ImageRegionConstIterator< InImageType > in_iterator( in_image , in_image->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< OutImageType > out_iterator( out_image , out_image->GetLargestPossibleRegion() ) ;
  for( in_iterator.GoToBegin() , out_iterator.GoToBegin() ; !in_iterator.IsAtEnd() ; ++in_iterator , ++out_iterator )
    {
    out_iterator.Set( static_cast< typename OutImageType::PixelType >( in_iterator.Get() ) ) ;
    }
  // ---------------------------------------------
  // cast from ITK image to ANTsImage and return that image
  return wrap< OutImageType >( out_image );
}

// wrap this function for possible image types
// this is annoying, but saves on time and eliminates chances for bugs in code
// NOTE: you need the random `m` there.
PYBIND11_MODULE(imageCloneModule, m)
{
  m.def("antsImageCloneUC2UC2", antsImageClone<itk::Image<unsigned char, 2>, itk::Image<unsigned char, 2>>);
  m.def("antsImageCloneUI2UI2", antsImageClone<itk::Image<unsigned int, 2>, itk::Image<unsigned int, 2>>);
  m.def("antsImageCloneF2F2", antsImageClone<itk::Image<float, 2>, itk::Image<float, 2>>);
  // ... and so on ...
}
```

Note above how we have to explicilty wrap every combination of image input and output types. This
is annoying but ultimately saves on runtime and most importantly reduces possibilities for bugs
by eliminates the need for massive IF-ELSE statements determining the type of the image 
in the c++ code.

Next, we add the build lines to `ants/lib/CMakeLists.txt`:

```
pybind11_add_module(imageCloneModule LOCAL_antsImageClone.cxx)
target_link_libraries(imageCloneModule PRIVATE ${ITK_LIBRARIES})
```

And add the following line in `ants/lib/__init__.py`:

```
from .imageCloneModule import *
```

Finally, we add the Python wrapping function:

```python
from ants import lib
from ants.core import ants_image as iio

_image_clone_dict = {
    2: {
        'unsigned char': {
            'unsigned char': 'antsImageCloneUCUC2'
        },
        'unsigned int': {
            'unsigned int': 'antsImageCloneUI2UI2'
        },
        'float': {
            'float': 'antsImageCloneF2F2'
        }
    }
}

def image_clone(img1, pixeltype=None):
    idim = img1.dimension
    ptype1 = img1.pixeltype
    
    if pixeltype is None:
        ptype2 = img1.pixeltype
    else:
        ptype2 = pixeltype

    _image_clone_fn = lib.__dict__[_image_clone_dict[idim][ptype1][ptype2]]
    
    # this function returns a C++ ANTsImage object
    cloned_img_ptr = _image_clone_fn(img1.pointer)

    # we need to wrap the C++ ANTsImage object into a Python ANTsImage object
    cloned_ants_image = iio.ANTsImage(pixeltype=ptype1, dimension=idim,
                                      components=img1.components, pointer=cloned_img_ptr)
    return cloned_ants_image
```

And that's it!


## Running Tests

All tests can be executed by running the following command from the main directory:

```bash
./tests/run_tests.sh
```

Similarly, code coverage can be calculated by adding a flag to the above command:

```bash
./tests/run_tests.sh -c
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





