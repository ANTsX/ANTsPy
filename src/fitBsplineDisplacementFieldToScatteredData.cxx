
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkDisplacementFieldToBSplineImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template<unsigned int Dimension>
AntsImage<itk::VectorImage<float, Dimension>> fitBsplineVectorImageToScatteredDataHelper(
  std::vector<std::vector<double>> displacementOrigins,
  std::vector<std::vector<double>> displacements,
  std::vector<double> displacementWeights,
  std::vector<double> origin,
  std::vector<double> spacing,
  std::vector<unsigned int> size,
  std::vector<std::vector<double>> direction,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder,
  bool enforceStationaryBoundary,
  bool estimateInverse,
  bool rasterizePoints )
{
  using RealType = float;

  using ANTsFieldType = itk::VectorImage<RealType, Dimension>;
  using ANTsFieldPointerType = typename ANTsFieldType::Pointer;

  using VectorType = itk::Vector<RealType, Dimension>;
  using PointSetType = itk::PointSet<VectorType, Dimension>;

  using ITKFieldType = itk::Image<VectorType, Dimension>;
  using ITKFieldPointerType = typename ITKFieldType::Pointer;
  using IteratorType = itk::ImageRegionIteratorWithIndex<ITKFieldType>;

  using BSplineFilterType = itk::DisplacementFieldToBSplineImageFilter<ITKFieldType, PointSetType>;
  using WeightsContainerType = typename BSplineFilterType::WeightsContainerType;
  using WeightImageType = typename BSplineFilterType::RealImageType;
  using WeightImagePointerType = typename WeightImageType::Pointer;

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();

  ////////////////////////////
  //
  //  Define the output B-spline field domain
  //

  auto originP = origin;//.unchecked<1>();
  auto spacingP = spacing;//.unchecked<1>();
  auto sizeP = size;//.unchecked<1>();
  auto directionP = direction;//.unchecked<2>();

  if( originP.size() == 0 || sizeP.size() == 0 || spacingP.size() == 0 || directionP.size() == 0 )
    {
    throw std::invalid_argument( "one or more b-spline domain definitions are not specified." );
    }
  else
    {
    typename ITKFieldType::PointType fieldOrigin;
    typename ITKFieldType::SpacingType fieldSpacing;
    typename ITKFieldType::SizeType fieldSize;
    typename ITKFieldType::DirectionType fieldDirection;

    for( unsigned int d = 0; d < Dimension; d++ )
      {
      fieldOrigin[d] = originP[d];
      fieldSpacing[d] = spacingP[d];
      fieldSize[d] = sizeP[d];
      for( unsigned int e = 0; e < Dimension; e++ )
        {
        fieldDirection(d, e) = directionP[d][e];
        }
      }
    bsplineFilter->SetBSplineDomain( fieldOrigin, fieldSpacing, fieldSize, fieldDirection );
    }

  ////////////////////////////
  //
  //  Add the inputs (if they are specified)
  //

  auto displacementOriginsP = displacementOrigins;//.unchecked<2>();
  auto displacementsP = displacements;//.unchecked<2>();

  typename PointSetType::Pointer pointSet = PointSetType::New();
  pointSet->Initialize();
  typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

  unsigned int numberOfPoints = displacementsP.size();

  if( rasterizePoints )
    {
    // First, distribute the weights and displacements to an image the same size as the b-spline domain.

    WeightImagePointerType weightImage = WeightImageType::New();
    weightImage->SetOrigin( bsplineFilter->GetBSplineDomainOrigin() );
    weightImage->SetSpacing( bsplineFilter->GetBSplineDomainSpacing() );
    weightImage->SetDirection( bsplineFilter->GetBSplineDomainDirection() );
    weightImage->SetRegions( bsplineFilter->GetBSplineDomainSize() );
    weightImage->AllocateInitialized();

    WeightImagePointerType countImage = WeightImageType::New();
    countImage->SetOrigin( bsplineFilter->GetBSplineDomainOrigin() );
    countImage->SetSpacing( bsplineFilter->GetBSplineDomainSpacing() );
    countImage->SetDirection( bsplineFilter->GetBSplineDomainDirection() );
    countImage->SetRegions( bsplineFilter->GetBSplineDomainSize() );
    countImage->AllocateInitialized();

    ITKFieldPointerType rasterizedField = ITKFieldType::New();
    rasterizedField->SetOrigin( bsplineFilter->GetBSplineDomainOrigin() );
    rasterizedField->SetSpacing( bsplineFilter->GetBSplineDomainSpacing() );
    rasterizedField->SetDirection( bsplineFilter->GetBSplineDomainDirection() );
    rasterizedField->SetRegions( bsplineFilter->GetBSplineDomainSize() );
    rasterizedField->AllocateInitialized();

    for( unsigned int n = 0; n < numberOfPoints; n++ )
      {
      typename ITKFieldType::PointType imagePoint;
      VectorType imageDisplacement;
      for( unsigned int d = 0; d < Dimension; d++ )
        {
        imagePoint[d] = displacementOriginsP[n][d];//displacementOriginsP(n, d);
        imageDisplacement[d] = displacementsP[n][d];
        }
      typename ITKFieldType::IndexType imageIndex =
        weightImage->TransformPhysicalPointToIndex( imagePoint );
      weightImage->SetPixel( imageIndex, displacementWeights[n] + weightImage->GetPixel( imageIndex ) );
      rasterizedField->SetPixel( imageIndex, imageDisplacement * displacementWeights[n] + rasterizedField->GetPixel( imageIndex ) );
      countImage->SetPixel( imageIndex, 1.0 + countImage->GetPixel( imageIndex ) );
      }

    // Second, iterate through the weight image and pull those indices/points which have non-zero weights.

    unsigned count = 0;

    typename itk::ImageRegionIteratorWithIndex<WeightImageType>
      ItC( countImage, countImage->GetLargestPossibleRegion() );
    for( ItC.GoToBegin(); ! ItC.IsAtEnd(); ++ItC )
      {
      if( ItC.Get() > 0.0 )
        {
        typename ITKFieldType::PointType imagePoint;
        weightImage->TransformIndexToPhysicalPoint( ItC.GetIndex(), imagePoint );
        typename PointSetType::PointType point;
        point.CastFrom( imagePoint );
        pointSet->SetPoint( count, point );
        RealType sumWeight = weightImage->GetPixel( ItC.GetIndex() );
        RealType weight = sumWeight / ItC.Get();
        VectorType imageDisplacement = rasterizedField->GetPixel( ItC.GetIndex() ) / sumWeight;
        pointSet->SetPointData( count, imageDisplacement );
        weights->InsertElement( count, weight );
        count++;
        }
      }
    }
  else
    {
    for( unsigned int n = 0; n < numberOfPoints; n++ )
      {
      typename PointSetType::PointType point;
      for( unsigned int d = 0; d < Dimension; d++ )
        {
        point[d] = displacementOriginsP[n][d];
        }
      pointSet->SetPoint( n, point );

      VectorType data( 0.0 );
      for( unsigned int d = 0; d < Dimension; d++ )
        {
        data[d] = displacementsP[n][d];
        }
      pointSet->SetPointData( n, data );
      weights->InsertElement( n, displacementWeights[n] );
      }
    }
  bsplineFilter->SetPointSet( pointSet );
  bsplineFilter->SetPointSetConfidenceWeights( weights );

  typename BSplineFilterType::ArrayType ncps;
  typename BSplineFilterType::ArrayType isClosed;

  for( unsigned int d = 0; d < Dimension; d++ )
    {
    ncps[d] = numberOfControlPoints[d];
    }

  bsplineFilter->SetNumberOfControlPoints( ncps );
  bsplineFilter->SetSplineOrder( splineOrder );
  bsplineFilter->SetNumberOfFittingLevels( numberOfFittingLevels );
  bsplineFilter->SetEnforceStationaryBoundary( enforceStationaryBoundary );
  bsplineFilter->SetEstimateInverse( estimateInverse );
  bsplineFilter->Update();

  //////////////////////////
  //
  //  Now convert back to vector image type.
  //

  ANTsFieldPointerType antsField = ANTsFieldType::New();
  antsField->CopyInformation( bsplineFilter->GetOutput() );
  antsField->SetRegions( bsplineFilter->GetOutput()->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->AllocateInitialized();

  IteratorType It( bsplineFilter->GetOutput(),
    bsplineFilter->GetOutput()->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    VectorType data = It.Value();

    typename ANTsFieldType::PixelType antsVector( Dimension );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      antsVector[d] = data[d];
      }
    antsField->SetPixel( It.GetIndex(), antsVector );
    }

  AntsImage<ANTsFieldType> out_ants_image = { antsField };
  return out_ants_image;
}

void local_fitBsplineDisplacementFieldToScatteredData(nb::module_ &m)
{
  m.def("fitBsplineDisplacementFieldToScatteredDataD2", &fitBsplineVectorImageToScatteredDataHelper<2>);
  m.def("fitBsplineDisplacementFieldToScatteredDataD3", &fitBsplineVectorImageToScatteredDataHelper<3>);
}

