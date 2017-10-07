# Using C++ ITK code in Python

The beauty of ANTsPy is the ease at which users can incorporate (i.e. wrap)
their existing C++ code into Python while simultaneously leveraging the
IO, visualization, and greater python inter-operability provided by ANTsPy. 
This short tutorial demonstrates how you can wrap your ITK C++ functions 
in 15 minutes or less.

# Problem Definition

I have a function that takes a 2D ITK image and scales it by some factor in 
each direction. I want to use this function in Python, but don't want to go 
about 1) learning a new framework for wrapping my C++ code, 2) learning how 
to build ITK and my python code at the same time, 3) building an entire 
set of IO and plotting tools just for using my function. 

Here is my original code in a file called "scaleImage.cxx":

```cpp
#include "itkImage.h"
#include "itkScaleTransform.h"
#include "itkResampleImageFilter.h"

template <typename ImageType>
ImageType::Pointer scaleImage( ImageType::Pointer myImage, float scale1, float scale2 )
{  
    typedef itk::ScaleTransform<double, 2> TransformType;
    typename TransformType::Pointer scaleTransform = TransformType::New();
    itk::FixedArray<float, 2> scale;
    scale[0] = scale1; // newWidth/oldWidth
    scale[1] = scale2;
    scaleTransform->SetScale( scale );

    itk::Point<float,2> center;
    center[0] = myImage->GetLargestPossibleRegion().GetSize()[0]/2;
    center[1] = myImage->GetLargestPossibleRegion().GetSize()[1]/2;

    scaleTransform->SetCenter( center );

    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
    typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetTransform( scaleTransform );
    resampleFilter->SetInput( myImage );
    resampleFilter->SetSize( myImage->GetLargestPossibleRegion().GetSize() );
    resampleFilter->Update();

    typename ImageType::Pointer resampledImage = resampleFilter->GetOutput();
    return resampledImage;
}
```

# Problem Solution

## 1. Functions take `py::capsule` types as input+output instead of `ImageType::Pointer` types

The first thing to do is to convert the Input and Output argument types from `ImageType::Pointer` to
`py::capsule`. This needs to be done *ONLY* for input and output types.

```cpp
#include "itkImage.h"
#include "itkScaleTransform.h"
#include "itkResampleImageFilter.h"

template <typename ImageType>
py::capsule scaleImage( py::capsule myImage, float scale1, float scale2 )
{  
    typedef itk::ScaleTransform<double, 2> TransformType;
    typename TransformType::Pointer scaleTransform = TransformType::New();
    itk::FixedArray<float, 2> scale;
    scale[0] = scale1; // newWidth/oldWidth
    scale[1] = scale2;
    scaleTransform->SetScale( scale );

    itk::Point<float,2> center;
    center[0] = myImage->GetLargestPossibleRegion().GetSize()[0]/2;
    center[1] = myImage->GetLargestPossibleRegion().GetSize()[1]/2;

    scaleTransform->SetCenter( center );

    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
    typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetTransform( scaleTransform );
    resampleFilter->SetInput( myImage );
    resampleFilter->SetSize( myImage->GetLargestPossibleRegion().GetSize() );
    resampleFilter->Update();

    typename ImageType::Pointer resampledImage = resampleFilter->GetOutput();
    return resampledImage;
}
```

## 2. Input `py::capsule` types need to be un-wrapped. Output `py::capsule` types need to be wrapped.

Next, we add our simple functions to un-wrap a `py::capsule` into an ITK image pointer, and
we add our simple function to wrap an ITK image pointer into a `py::capsule`. NOTE how we 
add another header `LOCAL_antsImage.h`

```cpp
#include "itkImage.h"
#include "itkScaleTransform.h"
#include "itkResampleImageFilter.h"

// NEED THIS INCLUDE FOR WRAPPING/UNWRAPPING
#include "LOCAL_antsImage.h"

template <typename ImageType>
py::capsule scaleImage( py::capsule myAntsImage, float scale1, float scale2 )
{  
    // UN-WRAP THE CAPSULE
    ImageType::Pointer myImage = as<ImageType>( myAntsImage );

    typedef itk::ScaleTransform<double, 2> TransformType;
    typename TransformType::Pointer scaleTransform = TransformType::New();
    itk::FixedArray<float, 2> scale;
    scale[0] = scale1; // newWidth/oldWidth
    scale[1] = scale2;
    scaleTransform->SetScale( scale );

    itk::Point<float,2> center;
    center[0] = myImage->GetLargestPossibleRegion().GetSize()[0]/2;
    center[1] = myImage->GetLargestPossibleRegion().GetSize()[1]/2;

    scaleTransform->SetCenter( center );

    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
    typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetTransform( scaleTransform );
    resampleFilter->SetInput( myImage );
    resampleFilter->SetSize( myImage->GetLargestPossibleRegion().GetSize() );
    resampleFilter->Update();

    typename ImageType::Pointer resampledImage = resampleFilter->GetOutput();

    // WRAP THE ITK IMAGE 
    py::capsule resampledCapsule = wrap<ImageType>( resampledImage );
    return resampledCapsule
}
```

## 3. Functions need to be declared for `pybind11` 

Now, in the file, we need to declare the function for `pybind11` to wrap it in python. 
This is a simple process, but can be tedious since every image type needs to be declared
explicitly. You can get around this by having a wrapper function which takes in strings
declaring the precision and dimension (as in `ANTsR`), but we opt out of that in order
to increase code readability and maintainability.

After declaring your function, your code will look like this:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "itkImage.h"
#include "itkScaleTransform.h"
#include "itkResampleImageFilter.h"

// NEED THIS INCLUDE FOR WRAPPING/UNWRAPPING
#include "LOCAL_antsImage.h"

// NEED THIS FOR PYBIND11
namespace py = pybind11;

template <typename ImageType>
py::capsule scaleImage2D( py::capsule myAntsImage, float scale1, float scale2 )
{  
    // UN-WRAP THE CAPSULE
    typename ImageType::Pointer myImage = as<ImageType>( myAntsImage );

    typedef itk::ScaleTransform<double, 2> TransformType;
    typename TransformType::Pointer scaleTransform = TransformType::New();
    itk::FixedArray<float, 2> scale;
    scale[0] = scale1; // newWidth/oldWidth
    scale[1] = scale2;
    scaleTransform->SetScale( scale );

    itk::Point<float,2> center;
    center[0] = myImage->GetLargestPossibleRegion().GetSize()[0]/2;
    center[1] = myImage->GetLargestPossibleRegion().GetSize()[1]/2;

    scaleTransform->SetCenter( center );

    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
    typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetTransform( scaleTransform );
    resampleFilter->SetInput( myImage );
    resampleFilter->SetSize( myImage->GetLargestPossibleRegion().GetSize() );
    resampleFilter->Update();

    typename ImageType::Pointer resampledImage = resampleFilter->GetOutput();

    // WRAP THE ITK IMAGE 
    py::capsule resampledCapsule = wrap<ImageType>( resampledImage );
    return resampledCapsule;
}

// DECLARE OUR FUNCTION FOR APPROPRIATE TYPES 
PYBIND11_MODULE(scaleImageModule, m)
{
    m.def("scaleImage2D", &scaleImage2D<itk::Image<float,2>>);
}
```

That's all! The wrap declaration process is second nature after a while. Notice how
we named our module `scaleImageModule` - this is what our python module will also be called.

## 4. Functions need to be built in `CMakeLists.txt` and imported in `__init__.py`

Next (almost done), we need to actually build our function during the build process by adding
it to the CMakeLists.txt file in the `antspy/ants/lib/` directory. We only need to add two
lines:

```
pybind11_add_module(scaleImageModule scaleImage.cxx)
target_link_libraries(scaleImageModule PRIVATE ${ITK_LIBRARIES})
```

That's it! The code will now be built when the user runs `python setup.py develop` or
`python setup.py install`. However, we have one last step to make sure it's imported into
the package namespace - we need to import it in the `__init__.py` file found in the
`antspy/ants/lib` directory. Simply add this line:

```python
from .scaleImageModule import *
```

## 5. Wrapped C++ functions need a Python interface

Now, we have full access to our C++ function in the python/ANTsPY namespace. We can use
this function directly if we want, but for users it's better to have a small python 
interface. This is very easy to implement, and mostly involves argument checking and such.

```python
# relative imports in the package
from .. import utils, core
# could use the following if not in the package:
# from ants import utils, core

def scale_image2d(image, scale1, scale2):
    """
    Scale an 2D image by a specified factor in each dimension.

    Arguments
    ---------
    image : ANTsImage type
        image to be scaled

    scale1 : float
        scale factor in x direction

    scale 2 : float
        scale factor in y direction

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> img_scaled = ants.scale_image2d(img, 0.8, 0.8)
    >>> ants.plot(img)
    >>> ants.plot(img_scaled)
    """
    image = image.clone('float')

    # get function from its name
    lib_fn = utils.get_lib_fn('scaleImage2D')

    # apply the function to my image
    # remember this function returns a py::capsule
    scaled_img_ptr = lib_fn(image.pointer, scale1, scale2)

    # wrap the py::capsule back in ANTsImage class
    scaled_img = core.ANTsImage(pixeltype=image.pixeltype,
                                dimension=image.dimension,
                                components=image.components,
                                pointer=scaled_img_ptr)
    return scaled_img
```

We also need to import this function in the ANTsPy namespace. If we put the above
python code in a file called `scaled_image.py` in the `antspy/ants/utils` directory, 
then we would add the following line to the `__init__.py` file in the 
`antspy/ants/utils` directory:

```python
from .scale_image import *
```

Finally, we rebuild the package by going to the main directory and running setup.py again:
```
python setup.py develop # (or python setup.py install)
```

Now we can use this function quite easily:

```python
import ants
img = ants.image_read(ants.get_data('r16'))
ants.plot(img)
scaled_img = ants.scale_image2d(img, 1.2, 1.2)
ants.plot(scaled_img)
```

And that's it - a painless 15 minute process that gives you access to all of the ANTsPy functionality
such as IO, visualization, numpy-conversion, etc.





