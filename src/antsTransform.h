

#ifndef __ANTSPYTRANSFORM_H
#define __ANTSPYTRANSFORM_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>


#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>

#include "itkMacro.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkVector.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/vnl_vector_ref.h"
#include "itkTransform.h"
#include "itkTransformBase.h"
#include "itkAffineTransform.h"
#include "itkAffineTransform.h"
#include "itkCenteredAffineTransform.h"
#include "itkEuler2DTransform.h"
#include "itkEuler3DTransform.h"
#include "itkRigid2DTransform.h"
#include "itkRigid3DTransform.h"
#include "itkCenteredRigid2DTransform.h"
#include "itkCenteredEuler3DTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkCenteredSimilarity2DTransform.h"
#include "itkSimilarity3DTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFileReader.h"
#include "itkCompositeTransform.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkDisplacementFieldTransform.h"
#include "itkConstantBoundaryCondition.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkGaussianInterpolateImageFunction.h"
#include "itkInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkWindowedSincInterpolateImageFunction.h"
#include "itkLabelImageGaussianInterpolateImageFunction.h"
#include "itkTransformFileWriter.h"
#include "itkTransformFactory.h"

#include "itkMacro.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkVector.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/vnl_vector_ref.h"
#include "itkTransform.h"
#include "itkAffineTransform.h"

#include "antscore/antsUtilities.h"
#include "itkAffineTransform.h"
#include "antsImage.h"
#include "register_transforms.h"

namespace nb = nanobind;
using namespace nb::literals;


template <typename TransformType> 
struct AntsTransform {
    typename TransformType::Pointer ptr;
};


// --------------------------------------------------------------


template <typename TransformType>
std::vector<float> getTransformParameters( AntsTransform<TransformType> & myTx)
{
    typename TransformType::Pointer itkTransform = myTx.ptr;

    std::vector<float> parameterslist;
    for (unsigned int i = 0; i < itkTransform->GetNumberOfParameters(); i++ )
    {
        parameterslist.push_back( itkTransform->GetParameters()[i] );
    }
    return parameterslist;
}


template <typename TransformType>
void setTransformParameters( AntsTransform<TransformType> & myTx, std::vector<float> new_parameters )
{
    typename TransformType::Pointer itkTransform = myTx.ptr;

    typename TransformType::ParametersType itkParameters;
    itkParameters.SetSize( itkTransform->GetNumberOfParameters() );

    for (unsigned int i = 0; i < itkTransform->GetNumberOfParameters(); i++)
    {
        itkParameters[i] = new_parameters[i];
    }

    itkTransform->SetParameters( itkParameters );
}

// --------------------------------------------------------------

template <typename TransformType>
std::vector<float> getTransformFixedParameters( AntsTransform<TransformType> & myTx )
{
    typename TransformType::Pointer itkTransform = myTx.ptr;

    std::vector<float> parameterslist;
    for (unsigned int i = 0; i < itkTransform->GetNumberOfFixedParameters(); i++ )
    {
        parameterslist.push_back( itkTransform->GetFixedParameters()[i] );
    }
    return parameterslist;
}


template <typename TransformType>
void setTransformFixedParameters( AntsTransform<TransformType>& myTx, std::vector<float> new_parameters )
{
    typename TransformType::Pointer itkTransform = myTx.ptr;

    typename TransformType::FixedParametersType itkParameters;
    itkParameters.SetSize( itkTransform->GetNumberOfFixedParameters() );

    for (unsigned int i = 0; i < itkTransform->GetNumberOfFixedParameters(); i++)
    {
        itkParameters[i] = new_parameters[i];
    }

    itkTransform->SetFixedParameters( itkParameters );
}

// --------------------------------------------------------------

template <typename TransformType>
std::vector< float > transformPoint( AntsTransform<TransformType> & myTx, std::vector< double > inPoint )
{
    typedef typename TransformType::Pointer          TransformPointerType;
    typedef typename TransformType::InputPointType   InputPointType;
    typedef typename TransformType::OutputPointType  OutputPointType;

    TransformPointerType itkTransform = myTx.ptr;

    InputPointType inItkPoint;
    for (unsigned int i = 0; i < InputPointType::PointDimension; i++)
    {
        inItkPoint[i] = inPoint[i];
    }

    OutputPointType outItkPoint = itkTransform->TransformPoint( inItkPoint );

    std::vector< float > outPoint( OutputPointType::PointDimension );
    for (unsigned int i = 0; i < OutputPointType::PointDimension; i++)
    {
        outPoint[i] = outItkPoint[i];
    }

    return outPoint;
}

template <typename TransformType>
std::vector< float > transformVector( AntsTransform<TransformType> myTx, std::vector< float > inVector )
{
    typedef typename TransformType::Pointer          TransformPointerType;
    typedef typename TransformType::InputVectorType   InputVectorType;
    typedef typename TransformType::OutputVectorType  OutputVectorType;

    TransformPointerType itkTransform = myTx.ptr; 

    InputVectorType inItkVector;
    for (unsigned int i = 0; i < InputVectorType::Dimension; i++)
    {
        inItkVector[i] = inVector[i];
    }

    OutputVectorType outItkVector = itkTransform->TransformVector( inItkVector );

    std::vector< float > outVector( OutputVectorType::Dimension );
    for (unsigned int i = 0; i < OutputVectorType::Dimension; i++)
    {
        outVector[i] = outItkVector[i];
    }

    return outVector;
}

template <typename TransformType, typename ReturnTransformType>
AntsTransform<ReturnTransformType> inverseTransform( AntsTransform<TransformType> & myTx )
{
    typedef typename TransformType::Pointer  TransformPointerType;
    TransformPointerType itkTransform = myTx.ptr; 

    if ( !itkTransform->IsLinear() )
    {
        throw std::invalid_argument("Only linear transforms may be inverted here");
    }

    TransformPointerType inverse = itkTransform->GetInverseTransform();
    AntsTransform<ReturnTransformType> outTransform = { inverse };
    return outTransform;
}

template <typename TransformType, typename ImageType>
AntsImage<ImageType> transformImage( AntsTransform<TransformType> & myTx, 
                            AntsImage<ImageType> & image, 
                            AntsImage<ImageType> & ref, std::string interp)
{
    typedef typename TransformType::Pointer          TransformPointerType;

    const unsigned int Dimension = TransformType::InputSpaceDimension;

    TransformPointerType transform = myTx.ptr;

    typedef typename TransformType::ParametersValueType              PrecisionType;

    //typedef itk::Image<PixelType,TransformType::InputSpaceDimension> ImageType;
    typedef typename ImageType::Pointer                              ImagePointerType;

    // Use base for reference image so we can ignore it's pixeltype
    typedef itk::ImageBase<TransformType::InputSpaceDimension> ImageBaseType;
    typedef typename ImageBaseType::Pointer                    ImageBasePointerType;

    ImagePointerType inputImage = image.ptr;
    ImagePointerType refImage = ref.ptr;

    typedef itk::ResampleImageFilter<ImageType,ImageType,PrecisionType,PrecisionType> FilterType;
    typename FilterType::Pointer filter = FilterType::New();

    typedef itk::InterpolateImageFunction<ImageType, PrecisionType> InterpolatorType;
    typename InterpolatorType::Pointer interpolator = nullptr;

  if( interp == "linear" )
    {
    typedef itk::LinearInterpolateImageFunction<ImageType, PrecisionType> LinearInterpolatorType;
    typename LinearInterpolatorType::Pointer linearInterpolator = LinearInterpolatorType::New();
    interpolator = linearInterpolator;
    }
  else if( interp == "nearestneighbor" )
    {
    typedef itk::NearestNeighborInterpolateImageFunction<ImageType, PrecisionType> NearestNeighborInterpolatorType;
    typename NearestNeighborInterpolatorType::Pointer nearestNeighborInterpolator = NearestNeighborInterpolatorType::New();
    interpolator = nearestNeighborInterpolator;
    }
  else if( interp == "bspline" )
    {
    typedef itk::BSplineInterpolateImageFunction<ImageType, PrecisionType> BSplineInterpolatorType;
    typename BSplineInterpolatorType::Pointer bSplineInterpolator = BSplineInterpolatorType::New();
    interpolator = bSplineInterpolator;
    }
  else if( interp == "gaussian" )
    {
    typedef itk::GaussianInterpolateImageFunction<ImageType, PrecisionType> GaussianInterpolatorType;
    typename GaussianInterpolatorType::Pointer gaussianInterpolator = GaussianInterpolatorType::New();
    double sigma[Dimension];
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      sigma[d] = inputImage->GetSpacing()[d];
      }
    double alpha = 1.0;
    gaussianInterpolator->SetParameters( sigma, alpha );
    interpolator = gaussianInterpolator;
    }
  else if( interp ==  "cosinewindowedsinc" )
    {
    typedef itk::WindowedSincInterpolateImageFunction
                 <ImageType, 3, itk::Function::CosineWindowFunction<3, PrecisionType, PrecisionType>, itk::ConstantBoundaryCondition< ImageType >, PrecisionType> CosineInterpolatorType;
    typename CosineInterpolatorType::Pointer cosineInterpolator = CosineInterpolatorType::New();
    interpolator = cosineInterpolator;
    }
  else if( interp == "hammingwindowedsinc" )
    {
    typedef itk::WindowedSincInterpolateImageFunction
                 <ImageType, 3, itk::Function::HammingWindowFunction<3, PrecisionType, PrecisionType >, itk::ConstantBoundaryCondition< ImageType >, PrecisionType> HammingInterpolatorType;
    typename HammingInterpolatorType::Pointer hammingInterpolator = HammingInterpolatorType::New();
    interpolator = hammingInterpolator;
    }
  else if( interp == "lanczoswindowedsinc" )
    {
    typedef itk::WindowedSincInterpolateImageFunction
                 <ImageType, 3, itk::Function::LanczosWindowFunction<3, PrecisionType, PrecisionType>, itk::ConstantBoundaryCondition< ImageType >, PrecisionType > LanczosInterpolatorType;
    typename LanczosInterpolatorType::Pointer lanczosInterpolator = LanczosInterpolatorType::New();
    interpolator = lanczosInterpolator;
    }
  else if( interp == "blackmanwindowedsinc" )
    {
    typedef itk::WindowedSincInterpolateImageFunction
                 <ImageType, 3, itk::Function::BlackmanWindowFunction<3, PrecisionType, PrecisionType>, itk::ConstantBoundaryCondition< ImageType >, PrecisionType > BlackmanInterpolatorType;
    typename BlackmanInterpolatorType::Pointer blackmanInterpolator = BlackmanInterpolatorType::New();
    interpolator = blackmanInterpolator;
    }
  else if( interp == "welchwindowedsinc" )
    {
    typedef itk::WindowedSincInterpolateImageFunction
                 <ImageType, 3, itk::Function::WelchWindowFunction<3, PrecisionType, PrecisionType>, itk::ConstantBoundaryCondition< ImageType >, PrecisionType > WelchInterpolatorType;
    typename WelchInterpolatorType::Pointer welchInterpolator = WelchInterpolatorType::New();
    interpolator = welchInterpolator;
    }
  else if( interp == "multilabel" )
    {
    const unsigned int NVectorComponents = 1;
    typedef ants::VectorPixelCompare<PrecisionType, NVectorComponents> CompareType;
    typedef typename itk::LabelImageGaussianInterpolateImageFunction<ImageType, PrecisionType,
        CompareType> MultiLabelInterpolatorType;
    typename MultiLabelInterpolatorType::Pointer multiLabelInterpolator = MultiLabelInterpolatorType::New();
    double sigma[Dimension];
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      sigma[d] = inputImage->GetSpacing()[d];
      }
    double alpha = 4.0;
    multiLabelInterpolator->SetParameters( sigma, alpha );
    interpolator = multiLabelInterpolator;
    }

    filter->SetInput( inputImage );
    filter->SetSize( refImage->GetLargestPossibleRegion().GetSize() );
    filter->SetOutputSpacing( refImage->GetSpacing() );
    filter->SetOutputOrigin( refImage->GetOrigin() );
    filter->SetOutputDirection( refImage->GetDirection() );
    filter->SetInterpolator( interpolator );

    filter->SetTransform( transform );
    filter->Update();

    ImagePointerType filterOutput = filter->GetOutput();

    AntsImage<ImageType> outImage = { filterOutput };
    return outImage;
}


// need nb:list instead of std::vector<AntsTransform<TransformBaseType>> in order
// to support a mix of standard itk transform and itk displacementfieldtransform types
template <typename TransformBaseType, typename PrecisionType, unsigned int Dimension>
AntsTransform<TransformBaseType> composeTransforms( nb::list tformlist, 
                                                    std::string precision, 
                                                    unsigned int dimension)
{
    typedef typename itk::DisplacementFieldTransform<PrecisionType, Dimension> DisplacementTransformType;
    typedef typename DisplacementTransformType::Pointer  DisplacementTransformPointerType;
    typedef typename TransformBaseType::Pointer  TransformBasePointerType;
    typedef typename itk::CompositeTransform<PrecisionType, Dimension> CompositeTransformType;

    typename CompositeTransformType::Pointer comp_transform = CompositeTransformType::New();

    for ( nb::handle_t<AntsTransform<TransformBaseType>> h: tformlist )
    {
        PyObject * a_py = h.ptr(); 
        AntsTransform<TransformBaseType> mytx;
        bool res = nb::try_cast<AntsTransform<TransformBaseType> &>(h, mytx);
        if (res == false) {
            // failed cast means its a displacement field transform
            AntsTransform<DisplacementTransformType> &mytx = nb::cast<AntsTransform<DisplacementTransformType> &>(h);
            comp_transform->AddTransform( mytx.ptr );
        } else {
            comp_transform->AddTransform( mytx.ptr );
        }
    }
    AntsTransform<TransformBaseType> outTransform = { comp_transform.GetPointer() };
    return outTransform;
}

template <typename TransformBaseType, class PrecisionType, unsigned int Dimension>
AntsTransform<TransformBaseType> readTransform( std::string filename, unsigned int dimension, std::string precision )
{
    register_transforms();
    
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    typedef typename itk::CompositeTransform<PrecisionType, Dimension> CompositeTransformType;

    typedef itk::TransformFileReaderTemplate<PrecisionType> TransformReaderType;
    typedef typename TransformReaderType::TransformListType TransformListType;

    typename TransformReaderType::Pointer reader = TransformReaderType::New();
    reader->SetFileName( filename );
    reader->Update();

    const typename TransformReaderType::TransformListType * transformList = reader->GetTransformList();

    TransformBasePointerType transform;

    if ( transformList->size() > 1 )
    {
        typename CompositeTransformType::Pointer comp_transform = CompositeTransformType::New();
        typedef typename TransformListType::const_iterator TransformIteratorType;
        for (TransformIteratorType i = transformList->begin(); i != transformList->end(); ++i)
        {
            comp_transform->AddTransform( dynamic_cast<TransformBaseType *>( i->GetPointer()) );
        }
        transform = dynamic_cast<TransformBaseType *>(comp_transform.GetPointer());
        AntsTransform<TransformBaseType> outTransform = { transform };
        return outTransform;
    }
    else
    {
        transform = dynamic_cast<TransformBaseType *>( transformList->front().GetPointer() );
        AntsTransform<TransformBaseType> outTransform = { transform };
        return outTransform;
    }

    AntsTransform<TransformBaseType> outTransform = { transform };
    return outTransform;
}


template <typename TransformType>
void writeTransform( AntsTransform<TransformType> & transform, std::string filename )
{
    typedef typename TransformType::Pointer          TransformPointerType;
    TransformPointerType itkTransform = transform.ptr;
    typedef itk::TransformFileWriter TransformWriterType;
    typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
    transformWriter->SetInput( itkTransform );
    transformWriter->SetFileName( filename.c_str() );
    transformWriter->Update();
}



// ------------------------------------------------------------------

template< typename TransformBaseType, class PrecisionType, unsigned int Dimension >
AntsTransform<TransformBaseType> matrixOffset(  std::string type, std::string precision, unsigned int dimension,
                           std::vector<std::vector<float> > matrix,
                           std::vector<float> offset,
                           std::vector<float> center,
                           std::vector<float> translation,
                           std::vector<float> parameters,
                           std::vector<float> fixedparameters)
{

    typedef itk::MatrixOffsetTransformBase< PrecisionType, Dimension, Dimension> MatrixOffsetBaseType;
    typedef typename MatrixOffsetBaseType::Pointer                               MatrixOffsetBasePointerType;
    typedef typename TransformBaseType::Pointer                                  TransformBasePointerType;

    MatrixOffsetBasePointerType matrixOffset = nullptr;

    // Initialize transform by type
    if ( type == "AffineTransform" )
    {
        typedef itk::AffineTransform<PrecisionType,Dimension> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "CenteredAffineTransform" )
    {
        typedef itk::CenteredAffineTransform<PrecisionType,Dimension> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "Euler3DTransform" )
    {
        typedef itk::Euler3DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "Euler2DTransform" )
    {
        typedef itk::Euler2DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "QuaternionRigidTransform" )
    {
        typedef itk::QuaternionRigidTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "Rigid2DTransform" )
    {
        typedef itk::Rigid2DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "Rigid3DTransform" )
    {
        typedef itk::Rigid3DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "CenteredEuler3DTransform" )
    {
        typedef itk::CenteredEuler3DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "CenteredRigid2DTransform" )
    {
        typedef itk::CenteredRigid2DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "Similarity3DTransform" )
    {
        typedef itk::Similarity3DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "Similarity2DTransform" )
    {
        typedef itk::Similarity2DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else if ( type == "CenteredSimilarity2DTransform" )
    {
        typedef itk::CenteredSimilarity2DTransform<PrecisionType> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }
    else
    {
        typedef itk::AffineTransform<PrecisionType,Dimension> TransformType;
        auto transformPointer = TransformType::New();
        matrixOffset = dynamic_cast<MatrixOffsetBaseType*>( transformPointer.GetPointer() );
    }

    matrixOffset->SetIdentity();

    if (matrix.size() > 1)
    {
        typename MatrixOffsetBaseType::MatrixType itkMatrix;
        for ( unsigned int i=0; i<dimension; i++)
            for ( unsigned int j=0; j<dimension; j++)
            {
                itkMatrix(i,j) = matrix[i][j];
            }
        matrixOffset->SetMatrix( itkMatrix );
    }

    if (translation.size() > 0)
    {
        typename MatrixOffsetBaseType::OutputVectorType itkTranslation;
        for ( unsigned int i=0; i<dimension; i++)
        {
            itkTranslation[i] = translation[i];
        }
        matrixOffset->SetTranslation( itkTranslation );
    }

    if (offset.size() > 0)
    {
        typename MatrixOffsetBaseType::OutputVectorType itkOffset;
        for ( unsigned int i=0; i<dimension; i++)
        {
            itkOffset[i] = offset[i];
        }
        matrixOffset->SetOffset( itkOffset );
    }

    if (center.size() > 0)
    {
        typename MatrixOffsetBaseType::InputPointType itkCenter;
        for ( unsigned int i=0; i<dimension; i++)
        {
            itkCenter[i] = center[i];
        }
        matrixOffset->SetCenter( itkCenter );
    }

    if (parameters.size() > 0)
    {
        typename MatrixOffsetBaseType::ParametersType itkParameters;
        itkParameters.SetSize( matrixOffset->GetNumberOfParameters() );
        for ( unsigned int i=0; i<matrixOffset->GetNumberOfParameters(); i++)
        {
            itkParameters[i] = parameters[i];
        }
        matrixOffset->SetParameters( itkParameters );
    }

    if (fixedparameters.size() > 0)
    {
        typename MatrixOffsetBaseType::FixedParametersType itkFixedParameters;
        itkFixedParameters.SetSize( matrixOffset->GetNumberOfFixedParameters() );
        for ( unsigned int i=0; i<matrixOffset->GetNumberOfFixedParameters(); i++)
        {
            itkFixedParameters[i] = fixedparameters[i];
        }
        matrixOffset->SetFixedParameters( itkFixedParameters );
    }

    TransformBasePointerType itkTransform = dynamic_cast<TransformBaseType*>( matrixOffset.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { itkTransform };

    return outTransform;
}


#endif
