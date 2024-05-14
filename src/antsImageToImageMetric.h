
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

#include "antscore/antsUtilities.h"

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

namespace nb = nanobind;
using namespace nb::literals;

template <typename MetricType>
class ANTsImageToImageMetric {
public:
    typedef MetricType itkMetricType;
    std::string precision;
    unsigned int dimension;
    std::string metrictype;
    unsigned int isVector;
    nb::capsule pointer;

    template <typename MyImageType>
    void setFixedImage( AntsImage<MyImageType> &, bool);

    template <typename MyImageType>
    void setMovingImage(  AntsImage<MyImageType> &, bool);

    void setSampling(std::string, float);
    void initialize();
    float getValue();

};




template <typename MetricType>
ANTsImageToImageMetric<MetricType> wrap_metric( const typename MetricType::Pointer & itkMetric )
{
    typedef typename MetricType::Pointer MetricPointerType;
    MetricPointerType* ptr = new MetricPointerType( itkMetric );


    ANTsImageToImageMetric<MetricType> antsMetric;
    antsMetric.precision         = "float";
    antsMetric.dimension         = MetricType::FixedImageDimension;
    antsMetric.metrictype        = itkMetric->GetNameOfClass();
    antsMetric.isVector          = 0;
    antsMetric.pointer           = nb::capsule(ptr);

    return antsMetric;
}

template <typename MetricType>
typename MetricType::Pointer as_metric( ANTsImageToImageMetric<MetricType> & metric )
{
    void *ptr = metric.pointer.data();
    typename MetricType::Pointer * real  = reinterpret_cast<typename MetricType::Pointer *>(ptr); // static_cast or reinterpret_cast ??

    return *real;
}


template <typename MetricType>
template <typename MyImageType>
void ANTsImageToImageMetric< MetricType >::setFixedImage(  AntsImage<MyImageType> & antsImage,
                                                          bool isMask )
{
  typedef typename MetricType::FixedImageType   ImageType;
  typedef typename ImageType::Pointer           ImagePointerType;
  typedef typename MetricType::Pointer          MetricPointerType;
  typedef itk::ImageMaskSpatialObject<ImageType::ImageDimension>  ImageMaskSpatialObjectType;
  typedef typename ImageMaskSpatialObjectType::ImageType          MaskImageType;

  MetricPointerType metric = as_metric< MetricType >( *this );
  ImagePointerType image = antsImage.ptr;

  if ( isMask ) {
    typename ImageMaskSpatialObjectType::Pointer mask = ImageMaskSpatialObjectType::New();
    typedef itk::CastImageFilter<ImageType,MaskImageType> CastFilterType;
    typename CastFilterType::Pointer cast = CastFilterType::New();
    cast->SetInput( image );
    cast->Update();
    mask->SetImage( cast->GetOutput() );
    metric->SetMovingImageMask(mask);
  }
  else {
    metric->SetFixedImage(image);
  }
}


template <typename MetricType>
template <typename MyImageType>
void ANTsImageToImageMetric< MetricType >::setMovingImage(  AntsImage<MyImageType> & antsImage,
                                                          bool isMask )
{
  typedef typename MetricType::MovingImageType  ImageType;
  typedef typename ImageType::Pointer           ImagePointerType;
  typedef typename MetricType::Pointer          MetricPointerType;
  typedef itk::ImageMaskSpatialObject<ImageType::ImageDimension>  ImageMaskSpatialObjectType;
  typedef typename ImageMaskSpatialObjectType::ImageType          MaskImageType;


  MetricPointerType metric = as_metric< MetricType >( *this );
  ImagePointerType image = antsImage.ptr;

  if ( isMask ) {
    typename ImageMaskSpatialObjectType::Pointer mask = ImageMaskSpatialObjectType::New();
    typedef itk::CastImageFilter<ImageType,MaskImageType> CastFilterType;
    typename CastFilterType::Pointer cast = CastFilterType::New();
    cast->SetInput( image );
    cast->Update();
    mask->SetImage( cast->GetOutput() );
    metric->SetMovingImageMask(mask);
  }
  else {
    metric->SetMovingImage(image);
  }
}

template< class MetricType >
float ANTsImageToImageMetric< MetricType >::getValue()
{
  //Rcpp::Rcout << "antsrImageToImageMetric_GetValue<MetricType>()" << std::endl;
  typedef typename MetricType::Pointer          MetricPointerType;
  MetricPointerType metric = as_metric< MetricType >( *this );
  return metric->GetValue();
}

template< class MetricType >
void ANTsImageToImageMetric< MetricType >::setSampling( std::string strategy, float percentage )
{
  typedef typename MetricType::MovingImageType  ImageType;
  typedef typename MetricType::Pointer          MetricPointerType;

  typedef typename MetricType::FixedSampledPointSetType MetricSamplePointSetType;
  typedef typename MetricSamplePointSetType::PointType  SamplePointType;

  typename MetricSamplePointSetType::Pointer samplePointSet = MetricSamplePointSetType::New();
  samplePointSet->Initialize();

  MetricPointerType metric = as_metric< MetricType >( *this );

  typedef typename itk::Statistics::MersenneTwisterRandomVariateGenerator RandomizerType;
  typename RandomizerType::Pointer randomizer = RandomizerType::New();
  randomizer->SetSeed( 1234 );

  const typename ImageType::SpacingType oneThirdVirtualSpacing = metric->GetFixedImage()->GetSpacing() / 3.0;
  unsigned long index = 0;

  if ( strategy == "regular" )
    {
    const unsigned long sampleCount = static_cast<unsigned long>( std::ceil( 1.0 / percentage ) );
    unsigned long count = sampleCount; //Start at sampleCount to keep behavior backwards identical, using first element.
    itk::ImageRegionConstIteratorWithIndex<ImageType> It( metric->GetFixedImage(), metric->GetFixedImage()->GetRequestedRegion() );
    for( It.GoToBegin(); !It.IsAtEnd(); ++It )
      {
      if( count == sampleCount )
        {
        count = 0; //Reset counter
        SamplePointType point;
        metric->GetFixedImage()->TransformIndexToPhysicalPoint( It.GetIndex(), point );

        // randomly perturb the point within a voxel (approximately)
        for( unsigned int d = 0; d < ImageType::ImageDimension; d++ )
          {
          point[d] += randomizer->GetNormalVariate() * oneThirdVirtualSpacing[d];
          }
        if( !metric->GetFixedImageMask() || metric->GetFixedImageMask()->IsInsideInWorldSpace( point ) )
          {
          samplePointSet->SetPoint( index, point );
          ++index;
          }
        }
      ++count;
      }
      metric->SetFixedSampledPointSet( samplePointSet );
      metric->SetUseSampledPointSet( true );
    }
    else if (strategy == "random")
      {
      const unsigned long totalVirtualDomainVoxels = metric->GetFixedImage()->GetRequestedRegion().GetNumberOfPixels();
      const unsigned long sampleCount = static_cast<unsigned long>( static_cast<float>( totalVirtualDomainVoxels ) * percentage );
      itk::ImageRandomConstIteratorWithIndex<ImageType> ItR( metric->GetFixedImage(), metric->GetFixedImage()->GetRequestedRegion() );
      ItR.SetNumberOfSamples( sampleCount );
      for( ItR.GoToBegin(); !ItR.IsAtEnd(); ++ItR )
        {
        SamplePointType point;
        metric->GetFixedImage()->TransformIndexToPhysicalPoint( ItR.GetIndex(), point );

        // randomly perturb the point within a voxel (approximately)
        for ( unsigned int d = 0; d < ImageType::ImageDimension; d++ )
          {
          point[d] += randomizer->GetNormalVariate() * oneThirdVirtualSpacing[d];
          }
        if( !metric->GetFixedImageMask() || metric->GetFixedImageMask()->IsInsideInWorldSpace( point ) )
          {
          samplePointSet->SetPoint( index, point );
          ++index;
          }
        }
        metric->SetFixedSampledPointSet( samplePointSet );
        metric->SetUseSampledPointSet( true );
      }
}

template< class MetricType >
void ANTsImageToImageMetric< MetricType >::initialize()
{
  typedef typename MetricType::Pointer          MetricPointerType;
  MetricPointerType metric = as_metric< MetricType >( *this );
  metric->Initialize();
}

template <typename MetricType, unsigned int Dimension>
ANTsImageToImageMetric<MetricType> new_ants_metric( std::string precision, unsigned int dimension, std::string metrictype )
{
    if ( metrictype == "MeanSquares" )
    {
      typedef itk::Image<float, Dimension> ImageType;
      typedef itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType> SpecificMetricType;
      typename SpecificMetricType::Pointer metricPointer = SpecificMetricType::New();

      typedef itk::ImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> MetricBaseType;
      typedef typename MetricBaseType::Pointer  MetricBasePointerType;

      MetricBasePointerType basePointer = dynamic_cast<MetricBaseType *>( metricPointer.GetPointer() );
      return wrap_metric< MetricType >( basePointer );
    }
    else if ( metrictype == "Correlation" )
    {
      typedef itk::Image<float, Dimension> ImageType;
      typedef itk::CorrelationImageToImageMetricv4<ImageType,ImageType> SpecificMetricType;
      typename SpecificMetricType::Pointer metricPointer = SpecificMetricType::New();

      typedef itk::ImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> MetricBaseType;
      typedef typename MetricBaseType::Pointer  MetricBasePointerType;

      MetricBasePointerType basePointer = dynamic_cast<MetricBaseType *>( metricPointer.GetPointer() );
      return wrap_metric< MetricType >( basePointer );
    }
    else if ( metrictype == "ANTSNeighborhoodCorrelation" )
    {
      typedef itk::Image<float, Dimension> ImageType;
      typedef itk::ANTSNeighborhoodCorrelationImageToImageMetricv4<ImageType,ImageType> SpecificMetricType;
      typename SpecificMetricType::Pointer metricPointer = SpecificMetricType::New();

      typedef itk::ImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> MetricBaseType;
      typedef typename MetricBaseType::Pointer  MetricBasePointerType;

      MetricBasePointerType basePointer = dynamic_cast<MetricBaseType *>( metricPointer.GetPointer() );
      return wrap_metric< MetricType >( basePointer );
    }
    else if ( metrictype == "Demons" )
    {
      typedef itk::Image<float, Dimension> ImageType;
      typedef itk::DemonsImageToImageMetricv4<ImageType,ImageType> SpecificMetricType;
      typename SpecificMetricType::Pointer metricPointer = SpecificMetricType::New();

      typedef itk::ImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> MetricBaseType;
      typedef typename MetricBaseType::Pointer  MetricBasePointerType;

      MetricBasePointerType basePointer = dynamic_cast<MetricBaseType *>( metricPointer.GetPointer() );
      return wrap_metric< MetricType >( basePointer );
    }
    else if ( metrictype == "JointHistogramMutualInformation" )
    {
      typedef itk::Image<float, Dimension> ImageType;
      typedef itk::JointHistogramMutualInformationImageToImageMetricv4<ImageType,ImageType> SpecificMetricType;
      typename SpecificMetricType::Pointer metricPointer = SpecificMetricType::New();
      typedef itk::ImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> MetricBaseType;
      typedef typename MetricBaseType::Pointer  MetricBasePointerType;

      MetricBasePointerType basePointer = dynamic_cast<MetricBaseType *>( metricPointer.GetPointer() );
      return wrap_metric< MetricType >( basePointer );
    }
    else if ( metrictype == "MattesMutualInformation" )
    {
      typedef itk::Image<float, Dimension> ImageType;
      typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType,ImageType> SpecificMetricType;
      typename SpecificMetricType::Pointer metricPointer = SpecificMetricType::New();
      typedef itk::ImageToImageMetricv4<itk::Image<float, Dimension>,itk::Image<float,Dimension>> MetricBaseType;
      typedef typename MetricBaseType::Pointer  MetricBasePointerType;

      MetricBasePointerType basePointer = dynamic_cast<MetricBaseType *>( metricPointer.GetPointer() );
      return wrap_metric< MetricType >( basePointer );
    }
    // should never reach this
    typedef itk::Image<float, Dimension> ImageType;
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
                                                            AntsImage<itk::Image<float, Dimension>> & fixed_img,
                                                            AntsImage<itk::Image<float, Dimension>> & moving_img )
{
  typedef itk::Image<float, Dimension> ImageType;
  typedef typename ImageType::Pointer  ImagePointerType;

  ImagePointerType fixed = fixed_img.ptr;
  ImagePointerType moving = moving_img.ptr;

    typedef typename MetricBaseType::Pointer  MetricBasePointerType;

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
    itkField->AllocateInitialized();

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

  // python code should prevent us getting here by checking for known metric types
  std::cerr << "Unsupported metric type requested: " << metrictype << std::endl;
  std::cerr << "Returning JointHistogramMutualInformation metric" << std::endl;

  typedef itk::JointHistogramMutualInformationImageToImageMetricv4<ImageType,ImageType> MetricType;
  typename MetricType::Pointer metric = MetricType::New();
  metric->SetFixedImage( fixed );
  metric->SetMovingImage( moving );
  MetricBasePointerType baseMetric = dynamic_cast<MetricBaseType *>( metric.GetPointer() );
  return( wrap_metric< MetricBaseType >( baseMetric ) );

}
