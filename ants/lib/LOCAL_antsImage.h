#ifndef __ANTSPYIMAGE_H
#define __ANTSPYIMAGE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "itkImageIOBase.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkPyBuffer.h"
#include "itkVectorImage.h"
#include "itkChangeInformationImageFilter.h"

#include "itkMath.h"
#include "itkPyVnl.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix_fixed.hxx"
#include "vnl/vnl_transpose.h"
#include "vnl/algo/vnl_matrix_inverse.h"
#include "vnl/vnl_matrix.h"
#include "vnl/algo/vnl_determinant.h"

namespace py = pybind11;

/*
template <typename ImageType>
class ANTsImage {
public:
    typedef ImageType itkImageType;
    // standard image properties which can't be directly changed
    std::string pixeltype;
    std::string dtype;
    unsigned int dimension;
    unsigned int components;
    //py::tuple shape;
    py::array _ndarr; // needed for creating image from numpy array

    // VERY important - holds pointer to itk::Image
    py::capsule pointer;

    // physical image properties which can be directly changed

    // SHAPE (can be changed through resampling)
    py::tuple getShape();

    // ORIGIN
    py::tuple getOrigin();
    void setOrigin( std::vector<double> );

    // SPACING
    py::tuple getSpacing();
    void setSpacing( std::vector<double> );

    // DIRECTION
    py::array getDirection();
    void setDirection( py::array );

    // convert to numpy array
    py::array numpy();
    void toFile( std::string );

};


// gets associated ITK pixel type as string
std::string getPixelTypeStringFromDummy( unsigned char dummyval ) { return "unsigned char"; }
std::string getPixelTypeStringFromDummy( unsigned int dummyval ) { return "unsigned int"; }
std::string getPixelTypeStringFromDummy( float dummyval ) { return "float"; }
std::string getPixelTypeStringFromDummy( double dummyval ) { return "double"; }

std::string getPixelTypeStringFromDummy( itk::VariableLengthVector<unsigned char> dummyval ) { return "unsigned char"; }
std::string getPixelTypeStringFromDummy( itk::VariableLengthVector<unsigned int> dummyval ) { return "unsigned int"; }
std::string getPixelTypeStringFromDummy( itk::VariableLengthVector<float>dummyval ) { return "float"; }
std::string getPixelTypeStringFromDummy( itk::VariableLengthVector<double> dummyval ) { return "double"; }

std::string getPixelTypeStringFromDummy( itk::RGBPixel<unsigned char> dummyval ) { return "unsigned char"; }

// gets associated NUMPY data type as string
std::string getDataTypeStringFromDummy( unsigned char dummyval ) { return "uint8"; }
std::string getDataTypeStringFromDummy( unsigned int dummyval ) { return "uint32"; }
std::string getDataTypeStringFromDummy( float dummyval ) { return "float32"; }
std::string getDataTypeStringFromDummy( double dummyval ) { return "float64"; }
std::string getDataTypeStringFromDummy( itk::VariableLengthVector<unsigned char> dummyval ) { return "uint8"; }
std::string getDataTypeStringFromDummy( itk::VariableLengthVector<unsigned int> dummyval ) { return "uint32"; }
std::string getDataTypeStringFromDummy( itk::VariableLengthVector<float> dummyval ) { return "float32"; }
std::string getDataTypeStringFromDummy( itk::VariableLengthVector<double> dummyval ) { return "float64"; }
std::string getDataTypeStringFromDummy( itk::RGBPixel<unsigned char> dummyval ) { return "uint8"; }


template <typename ImageType>
py::capsule wrap( const typename ImageType::Pointer &image )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType * ptr = new ImagePointerType( image );

    std::string ptype;
    std::string dtype;

    typename ImageType::PixelType dummyval;
    ptype = getPixelTypeStringFromDummy(dummyval);
    dtype = getDataTypeStringFromDummy(dummyval);
    unsigned int ndim = ImageType::GetImageDimension();

    ANTsImage<ImageType>      antsimage;
    antsimage.pixeltype     = ptype;
    antsimage.dtype         = dtype;
    antsimage.dimension     = ndim;
    antsimage.components    = image->GetNumberOfComponentsPerPixel();
    antsimage.pointer       = py::capsule(ptr, "itk::Image::Pointer");

    return antsimage;
}
*/

template <typename ImageType>
void capsuleDestructor( void * f )
{
    //std::cout << "calling capsule destructor" << std::endl;
    typename ImageType::Pointer * foo  = reinterpret_cast<typename ImageType::Pointer *>( f );
    *foo = ITK_NULLPTR;
}

// converts an ITK image pointer to a py::capsule
template <typename ImageType>
py::capsule wrap( const typename ImageType::Pointer &image )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType * ptr = new ImagePointerType( image );
    return py::capsule(ptr, capsuleDestructor<ImageType>);
}

/*
// converts a py::capsule to an ITK image pointer
template <typename ImageType>
typename ImageType::Pointer as( py::capsule pointer )
{
    return static_cast<typename ImageType::Pointer>(pointer);
}
*/
template <typename ImageType>
typename ImageType::Pointer as( void * ptr )
{
    //void *ptr = image.pointer;
    typename ImageType::Pointer * real  = static_cast<typename ImageType::Pointer *>(ptr); // static_cast or reinterpret_cast ??
    return *real;
}


template <typename ImageType>
void toFile( py::capsule & myPointer, std::string filename )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType image = ImageType::New();

    image = as<ImageType>( myPointer );

    typedef itk::ImageFileWriter< ImageType > ImageWriterType ;
    typename ImageWriterType::Pointer image_writer = ImageWriterType::New() ;
    image_writer->SetFileName( filename.c_str() ) ;
    image_writer->SetInput( image );
    image_writer->Update();
}


template <typename ImageType>
py::tuple getShapeHelper( typename ImageType::Pointer image )
{
    unsigned int ndim = ImageType::GetImageDimension();
    image->UpdateOutputInformation();
    typename ImageType::SizeType shape = image->GetBufferedRegion().GetSize();
    py::list shapelist;
    for (int i = 0; i < ndim; i++)
    {
        shapelist.append( shape[i] );
    }
    return shapelist;
}

template <typename ImageType>
py::tuple getShape( py::capsule & myPointer )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    return getShapeHelper<ImageType>( itkImage );
}


template <typename ImageType>
py::tuple getOriginHelper( typename ImageType::Pointer image )
{
    typename ImageType::PointType origin = image->GetOrigin();
    unsigned int ndim = ImageType::GetImageDimension();

    py::list originlist;
    for (int i = 0; i < ndim; i++)
    {
        originlist.append( origin[i] );
    }

    return originlist;
}

template <typename ImageType>
py::tuple getOrigin( py::capsule & myPointer )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    return getOriginHelper<ImageType>( itkImage );
}

template <typename ImageType>
void setOriginHelper( typename ImageType::Pointer &itkImage, std::vector<double> new_origin)
{
    unsigned int nvals = new_origin.size();
    typename ImageType::PointType origin = itkImage->GetOrigin();
    for (int i = 0; i < nvals; i++)
    {
        origin[i] = new_origin[i];
    }
    itkImage->SetOrigin( origin );
}

template <typename ImageType>
void setOrigin( py::capsule & myPointer, std::vector<double> new_origin )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    setOriginHelper<ImageType>( itkImage, new_origin );
}



template <typename ImageType>
py::array getDirectionHelper( typename ImageType::Pointer image )
{
    typedef typename ImageType::DirectionType ImageDirectionType;
    ImageDirectionType direction = image->GetDirection();

    typedef typename ImageDirectionType::InternalMatrixType DirectionInternalMatrixType;
    DirectionInternalMatrixType fixed_matrix = direction.GetVnlMatrix();

    vnl_matrix<double> vnlmat1 = fixed_matrix.as_matrix();

    const unsigned int ndim = ImageType::SizeType::GetSizeDimension();

    vnl_matrix<double> * vnlmat2 = new vnl_matrix<double>;
    vnlmat2->set_size(ndim, ndim);
    vnlmat2->fill(0);
    for (int i = 0; i < ndim; i++)
    {
        for (int j = 0; j < ndim; j++)
        {
            vnlmat2->put(i, j, vnlmat1(i,j));
        }
    }

    typedef itk::PyVnl<double> PyVnlType;
    PyObject * mymatrix = PyVnlType::_GetArrayViewFromVnlMatrix( vnlmat2 );
    py::array myarray = py::reinterpret_borrow<py::object>( mymatrix );
    std::string viewtype( "float64" );
    py::array myarrayview = myarray.view( viewtype ).reshape({ndim, ndim});

    return myarrayview;
}

template <typename ImageType>
py::array getDirection( py::capsule & myPointer )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    return getDirectionHelper<ImageType>( itkImage );
}


template <typename ImageType>
void setDirectionHelper( typename ImageType::Pointer &itkImage, py::array new_direction)
{
    // using FilterType = itk::ChangeInformationImageFilter<ImageType>;
    // typename FilterType::Pointer filter = FilterType::New();
    // filter->SetInput( itkImage );
    // filter->ChangeDirectionOn();

    PyObject * new_dir_obj = new_direction.ptr();

    py::tuple shapetuple = py::make_tuple( new_direction.shape(0), new_direction.shape(1) );
    PyObject * new_dir_shape = shapetuple.ptr();

    typedef itk::PyVnl<double> PyVnlType;
    vnl_matrix<double> new_matrix = PyVnlType::_GetVnlMatrixViewFromArray( new_dir_obj, new_dir_shape );

    typename ImageType::DirectionType new_matrix2 = itkImage->GetDirection( );
    for ( py::ssize_t i = 0; i < new_direction.shape(0); i++ )
      for ( py::ssize_t j = 0; j < new_direction.shape(1); j++ ) {
        new_matrix2(i,j) = new_matrix(i,j);
      }
    itkImage->SetDirection( new_matrix2 );
}

template <typename ImageType>
void setDirection( py::capsule & myPointer, py::array new_direction )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    setDirectionHelper<ImageType>( itkImage, new_direction );
}

template <typename ImageType>
void setSpacingHelper( typename ImageType::Pointer &itkImage, std::vector<double> new_spacing)
{
    unsigned int nvals = new_spacing.size();
    typename ImageType::SpacingType spacing = itkImage->GetSpacing();
    for (int i = 0; i < nvals; i++)
    {
        spacing[i] = new_spacing[i];
    }
    itkImage->SetSpacing( spacing );
}

template <typename ImageType>
void setSpacing( py::capsule & myPointer, std::vector<double> new_spacing )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    setSpacingHelper<ImageType>( itkImage, new_spacing );
}

template <typename ImageType>
py::tuple getSpacingHelper( typename ImageType::Pointer image )
{
    typename ImageType::SpacingType spacing = image->GetSpacing();
    unsigned int ndim = ImageType::GetImageDimension();

    py::list spacinglist;
    for (int i = 0; i < ndim; i++)
    {
        spacinglist.append( spacing[i] );
    }

    return spacinglist;
}

template <typename ImageType>
py::tuple getSpacing( py::capsule & myPointer )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    return getSpacingHelper<ImageType>( itkImage );
}

extern std::string ptrstr(py::capsule c);


#endif
