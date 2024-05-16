#ifndef __ANTSPYIMAGE_H
#define __ANTSPYIMAGE_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

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

namespace nb = nanobind;
using namespace nb::literals;


template <typename ImageType>
typename ImageType::Pointer as( void * ptr )
{
    typename ImageType::Pointer * real  = static_cast<typename ImageType::Pointer *>(ptr); // static_cast or reinterpret_cast ??
    return *real;
}

template <typename ImageType>
void * wrap( const typename ImageType::Pointer &image )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType * ptr = new ImagePointerType( image );
    return ptr;
}

template <typename ImageType>
typename ImageType::Pointer asImage( void * ptr ) {
    typename ImageType::Pointer itkImage = ImageType::New();
    itkImage = as<ImageType>( ptr );
    return itkImage;
}


template <typename ImageType> 
struct AntsImage {
    typename ImageType::Pointer ptr;
};


template <typename ImageType>
void toFile( AntsImage<ImageType> & myPointer, std::string filename )
{
    typename ImageType::Pointer image = myPointer.ptr;

    typedef itk::ImageFileWriter< ImageType > ImageWriterType ;
    typename ImageWriterType::Pointer image_writer = ImageWriterType::New() ;
    image_writer->SetFileName( filename.c_str() ) ;
    image_writer->SetInput( image );
    image_writer->Update();
}


template <typename ImageType>
std::list<int> getShape( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    unsigned int ndim = ImageType::GetImageDimension();
    image->UpdateOutputInformation();
    typename ImageType::SizeType shape = image->GetBufferedRegion().GetSize();
    std::list<int> shapelist;
    for (int i = 0; i < ndim; i++)
    {
        shapelist.push_back( shape[i] );
    }
    return shapelist;
}


template <typename ImageType>
int getComponents( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    return image->GetNumberOfComponentsPerPixel();
}

template <typename ImageType>
std::vector<double> getOrigin( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typename ImageType::PointType origin = image->GetOrigin();
    unsigned int ndim = ImageType::GetImageDimension();

    std::vector<double> originlist;
    for (int i = 0; i < ndim; i++)
    {
        originlist.push_back( origin[i] );
    }

    return originlist;
}

template <typename ImageType>
void setOrigin( AntsImage<ImageType> & myPointer, std::vector<double> new_origin)
{
    typename ImageType::Pointer itkImage = myPointer.ptr;
    unsigned int nvals = new_origin.size();
    typename ImageType::PointType origin = itkImage->GetOrigin();
    for (int i = 0; i < nvals; i++)
    {
        origin[i] = new_origin[i];
    }
    itkImage->SetOrigin( origin );
}


template <typename ImageType>
std::vector<double> getDirection( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typedef typename ImageType::DirectionType ImageDirectionType;
    ImageDirectionType direction = image->GetDirection();

    typedef typename ImageDirectionType::InternalMatrixType DirectionInternalMatrixType;
    DirectionInternalMatrixType fixed_matrix = direction.GetVnlMatrix();

    vnl_matrix<double> vnlmat1 = fixed_matrix.as_matrix();

    const unsigned int ndim = ImageType::SizeType::GetSizeDimension();

    std::vector<double> dvec;

    for (int i = 0; i < ndim; i++)
    {
        for (int j = 0; j < ndim; j++)
        {
            dvec.push_back(vnlmat1(i,j));
        }
    }
    return dvec;

}


template <typename ImageType>
void setDirection( AntsImage<ImageType> & myPointer, std::vector<std::vector<double>> new_direction)
{

    typename ImageType::Pointer itkImage = myPointer.ptr;

    typename ImageType::DirectionType new_matrix2 = itkImage->GetDirection( );
    for ( std::size_t i = 0; i < new_direction.size(); i++ )
      for ( std::size_t j = 0; j < new_direction[0].size(); j++ ) {
        new_matrix2(i,j) = new_direction[i][j];
      }
    itkImage->SetDirection( new_matrix2 );
}

template <typename ImageType>
void setSpacing( AntsImage<ImageType> & myPointer, std::vector<double> new_spacing)
{
    typename ImageType::Pointer itkImage = myPointer.ptr;
    unsigned int nvals = new_spacing.size();
    typename ImageType::SpacingType spacing = itkImage->GetSpacing();

    for (int i = 0; i < nvals; i++)
    {
        spacing[i] = new_spacing[i];
    }
    itkImage->SetSpacing( spacing );
}

template <typename ImageType>
std::vector<double> getSpacing( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typename ImageType::SpacingType spacing = image->GetSpacing();
    unsigned int ndim = ImageType::GetImageDimension();

    std::vector<double> spacinglist;
    for (int i = 0; i < ndim; i++)
    {
        spacinglist.push_back( spacing[i] );
    }

    return spacinglist;
}

/*
This function resets the region of an image to index from zero if needed. This
keeps the voxel indices in the numpy matrix consistent with the ITK image, and
also keeps the origin of physical space of the consistent with how it will be
saved as NIFTI.
*/
template <typename ImageType>
static void FixNonZeroIndex( typename ImageType::Pointer img )
{
    assert(img);

    typename ImageType::RegionType r = img->GetLargestPossibleRegion();
    typename ImageType::IndexType idx = r.GetIndex();

    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i)
    {
        // if any index is non-zero, reset the origin and region
        if ( idx[i] != 0 )
        {
            typename ImageType::PointType o;
            img->TransformIndexToPhysicalPoint( idx, o );
            img->SetOrigin( o );

            idx.Fill( 0 );
            r.SetIndex( idx );
            img->SetRegions( r );

            return;
        }
    }
}

#endif
