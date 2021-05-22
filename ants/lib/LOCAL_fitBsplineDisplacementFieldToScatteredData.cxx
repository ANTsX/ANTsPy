
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkDisplacementFieldToBSplineImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template<unsigned int Dimension>
py::capsule fitBsplineVectorImageToScatteredDataHelper(
  py::array_t<double> displacementOrigins,
  py::array_t<double> displacements,
  std::vector<double> displacementWeights,
  py::array_t<double> origin,
  py::array_t<double> spacing,
  py::array_t<unsigned int> size,
  py::array_t<double> direction,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder,
  bool enforceStationaryBoundary,
  bool estimateInverse )
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

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();

  ////////////////////////////
  //
  //  Add the inputs (if they are specified)
  //

  auto displacementOriginsP = displacementOrigins.unchecked<2>();
  auto displacementsP = displacements.unchecked<2>();

  unsigned int numberOfPoints = displacementsP.shape(0);

  if( numberOfPoints > 0 )
    {
    typename PointSetType::Pointer pointSet = PointSetType::New();
    pointSet->Initialize();
    typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

    unsigned int numberOfPoints = displacementsP.shape(0);

    for( unsigned int n = 0; n < numberOfPoints; n++ )
      {
      typename PointSetType::PointType point;
      for( unsigned int d = 0; d < Dimension; d++ )
        {
        point[d] = displacementOriginsP(n, d);
        }
      pointSet->SetPoint( n, point );

      VectorType data( 0.0 );
      for( unsigned int d = 0; d < Dimension; d++ )
        {
        data[d] = displacementsP(n, d);
        }
      pointSet->SetPointData( n, data );

      weights->InsertElement( n, displacementWeights[n] );
      }
    bsplineFilter->SetPointSet( pointSet );
    bsplineFilter->SetPointSetConfidenceWeights( weights );
    }

  ////////////////////////////
  //
  //  Define the output B-spline field domain
  //

  auto originP = origin.unchecked<1>();
  auto spacingP = spacing.unchecked<1>();
  auto sizeP = size.unchecked<1>();
  auto directionP = direction.unchecked<2>();

  if( originP.shape(0) == 0 || sizeP.shape(0) == 0 || spacingP.shape(0) == 0 || directionP.shape(0) == 0 )
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
      fieldOrigin[d] = originP(d);
      fieldSpacing[d] = spacingP(d);
      fieldSize[d] = sizeP(d);
      for( unsigned int e = 0; e < Dimension; e++ )
        {
        fieldDirection(d, e) = directionP(d, e);
        }
      }
    bsplineFilter->SetBSplineDomain( fieldOrigin, fieldSpacing, fieldSize, fieldDirection );
    }

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
  antsField->Allocate();

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

  return wrap< ANTsFieldType >( antsField );
}

PYBIND11_MODULE(fitBsplineDisplacementFieldToScatteredData, m)
{
  m.def("fitBsplineDisplacementFieldToScatteredDataD2", &fitBsplineVectorImageToScatteredDataHelper<2>);
  m.def("fitBsplineDisplacementFieldToScatteredDataD3", &fitBsplineVectorImageToScatteredDataHelper<3>);
}

