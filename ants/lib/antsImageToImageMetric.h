
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <vector>
#include <string>

#include "antsUtilities.h"
#include "itkDisplacementFieldTransform.h"
#include "itkImageToImageMetricv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkANTSNeighborhoodCorrelationImageToImageMetricv4.h"
#include "itkDemonsImageToImageMetricv4.h"
#include "itkJointHistogramMutualInformationImageToImageMetricv4.h"
#include "itkImageMaskSpatialObject.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageRandomConstIteratorWithIndex.h"

#include "antsImage.h"

namespace py = pybind11;

template <typename MetricType>
class ANTsImageToImageMetric {
public:
    typedef MetricType itkMetricType;
    std::string precision;
    unsigned int dimension;
    std::string metrictype;
    unsigned int isVector;
    py::capsule pointer;
};


template <typename MetricType>
ANTsImageToImageMetric<MetricType> wrap_metric( const typename MetricType::Pointer & itkMetric )
{
    typedef typename MetricType::Pointer MetricPointerType;
    MetricPointerType * ptr = new MetricPointerType( itkMetric );


    ANTsImageToImageMetric<MetricType> antsMetric;
    antsMetric.precision         = "float";
    antsMetric.dimension         = MetricType::FixedImageDimension;
    antsMetric.metrictype        = itkMetric->GetNameOfClass();
    antsMetric.isVector          = 0;
    antsMetric.pointer           = py::capsule(ptr, "itk::ImageToImageMetricv4::Pointer");

    return antsMetric;
}

template <typename MetricType>
typename MetricType::Pointer as_metric( ANTsImageToImageMetric<MetricType> & metric )
{
    void *ptr = metric.pointer;
    typename MetricType::Pointer * real  = reinterpret_cast<typename MetricType::Pointer *>(ptr); // static_cast or reinterpret_cast ??

    return *real;
}


template <typename MetricType, unsigned int Dimension>
ANTsImageToImageMetric<MetricType> new_ants_metric( std::string precision, unsigned int dimension, std::string metrictype )
{
    //if ( metricmetrictype == "MeanSquares" )
    typedef itk::MeanSquaresImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> SpecificMetricType;
    typename SpecificMetricType::Pointer metricPointer = SpecificMetricType::New();

    typedef itk::ImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> MetricBaseType;
    typedef typename MetricBaseType::Pointer  MetricBasePointerType;

    MetricBasePointerType basePointer = dynamic_cast<MetricBaseType *>( metricPointer.GetPointer() );

    return wrap_metric< MetricType >( basePointer );
}

template< typename MetricBaseType, unsigned int Dimension >
ANTsImageToImageMetric< MetricBaseType > create_ants_metric(std::string pixeltype, 
                                                            unsigned int dimension, 
                                                            std::string metrictype, 
                                                            unsigned int isVector, 
                                                            ANTsImage<itk::Image<float,Dimension>> fixed_img, 
                                                            ANTsImage<itk::Image<float,Dimension>> moving_img )
{
  typedef itk::Image<float, Dimension> ImageType;
  typedef typename ImageType::Pointer  ImagePointerType;

  ImagePointerType fixed = as< ImageType >( fixed_img );
  ImagePointerType moving = as< ImageType >( moving_img );

    typedef typename MetricBaseType::Pointer  MetricBasePointerType;

  //supportedTypes = c("MeanSquares", "MattesMutualInformation", "ANTSNeighborhoodCorrelation", "Correlation", "Demons", "JointHistogramMutualInformation")
  if ( metrictype == "MeanSquares" ) {
    typedef itk::MeanSquaresImageToImageMetricv4<ImageType,ImageType> MetricType;
    typename MetricType::Pointer metric = MetricType::New();
    metric->SetFixedImage( fixed );
    metric->SetMovingImage( moving );

    MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
    return( wrap_metric< MetricBaseType >( baseMetric ) );
  }
  else if ( metrictype == "Correlation" ) {
    typedef itk::CorrelationImageToImageMetricv4<ImageType,ImageType> MetricType;
    typename MetricType::Pointer metric = MetricType::New();
    metric->SetFixedImage( fixed );
    metric->SetMovingImage( moving );
    MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
    return( wrap_metric< MetricBaseType >( baseMetric ) );
  }
  else if ( metrictype == "ANTSNeighborhoodCorrelation" ) {
    typedef itk::ANTSNeighborhoodCorrelationImageToImageMetricv4<ImageType,ImageType> MetricType;
    typename MetricType::Pointer metric = MetricType::New();
    metric->SetFixedImage( fixed );
    metric->SetMovingImage( moving );
    MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
    return( wrap_metric< MetricBaseType >( baseMetric ) );
  }
  else if ( metrictype == "Demons" ) {
    typedef itk::DemonsImageToImageMetricv4<ImageType,ImageType> MetricType;
    typename MetricType::Pointer metric = MetricType::New();
    metric->SetFixedImage( fixed );
    metric->SetMovingImage( moving );

    /* This block needed to prevent exception on call to Initialize() */
    typedef itk::DisplacementFieldTransform<typename MetricType::InternalComputationValueType,
      ImageType::ImageDimension> TransformType;
    typedef typename TransformType::DisplacementFieldType     DisplacementFieldType;

    typename DisplacementFieldType::Pointer itkField = DisplacementFieldType::New();
    itkField->SetRegions( moving->GetLargestPossibleRegion() );
    itkField->SetSpacing( moving->GetSpacing() );
    itkField->SetOrigin( moving->GetOrigin() );
    itkField->SetDirection( moving->GetDirection() );
    itkField->Allocate();
    //itkField->FillBuffer(0);

    typename TransformType::Pointer idTransform = TransformType::New();
    idTransform->SetDisplacementField( itkField );

    metric->SetMovingTransform( idTransform );
    MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
    return( wrap_metric< MetricBaseType >( baseMetric ) );
    }
  else if ( metrictype == "MattesMutualInformation" ) {
    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType,ImageType> MetricType;
    typename MetricType::Pointer metric = MetricType::New();
    metric->SetFixedImage( fixed );
    metric->SetMovingImage( moving );
    MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
    return( wrap_metric< MetricBaseType >( baseMetric ) );

  }
  else if ( metrictype == "JointHistogramMutualInformation" ) {
    typedef itk::JointHistogramMutualInformationImageToImageMetricv4<ImageType,ImageType> MetricType;
    typename MetricType::Pointer metric = MetricType::New();
    metric->SetFixedImage( fixed );
    metric->SetMovingImage( moving );
    MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
    return( wrap_metric< MetricBaseType >( baseMetric ) );
  }

  typedef itk::JointHistogramMutualInformationImageToImageMetricv4<ImageType,ImageType> MetricType;
  typename MetricType::Pointer metric = MetricType::New();
    MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
    return( wrap_metric< MetricBaseType >( baseMetric ) );
}




