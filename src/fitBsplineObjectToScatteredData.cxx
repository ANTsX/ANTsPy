
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
#include "itkBSplineScatteredDataPointSetToImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template<unsigned int DataDimension>
std::vector<std::vector<double>> fitBsplineCurveHelper(
  std::vector<std::vector<double>> scatteredData,
  std::vector<std::vector<double>> parametricData,
  std::vector<double> dataWeights,
  std::vector<double> parametricDomainOrigin,
  std::vector<double> parametricDomainSpacing,
  std::vector<unsigned int> parametricDomainSize,
  std::vector<bool> isParametricDimensionClosed,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder )
{
  const unsigned int ParametricDimension = 1;

  using RealType = float;
  using ScatteredDataType = itk::Vector<RealType, DataDimension>;
  using PointSetType = itk::PointSet<ScatteredDataType, ParametricDimension>;
  using OutputImageType = itk::Image<ScatteredDataType, ParametricDimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<OutputImageType>;

  using BSplineFilterType = itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, OutputImageType>;
  using WeightsContainerType = typename BSplineFilterType::WeightsContainerType;

  typename PointSetType::Pointer pointSet = PointSetType::New();
  pointSet->Initialize();
  typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

  auto scatteredDataR = scatteredData;//.unchecked<2>();
  auto parametricDataR = parametricData;//.unchecked<2>();

  unsigned int numberOfPoints = scatteredDataR.size();

  for( unsigned int n = 0; n < numberOfPoints; n++ )
    {
    typename PointSetType::PointType point;
    for( unsigned int d = 0; d < ParametricDimension; d++ )
      {
      point[d] = parametricDataR[n][d];
      }
    pointSet->SetPoint( n, point );

    ScatteredDataType data( 0.0 );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      data[d] = scatteredDataR[n][d];
      }
    pointSet->SetPointData( n, data );

    weights->InsertElement( n, dataWeights[n] );
    }

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();
  bsplineFilter->SetInput( pointSet );
  bsplineFilter->SetPointWeights( weights );
  bsplineFilter->SetGenerateOutputImage( true );

  typename OutputImageType::PointType origin;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::SizeType size;
  typename BSplineFilterType::ArrayType ncps;
  typename BSplineFilterType::ArrayType isClosed;

  for( unsigned int d = 0; d < ParametricDimension; d++ )
    {
    origin[d] = parametricDomainOrigin[d];
    spacing[d] = parametricDomainSpacing[d];
    size[d] = parametricDomainSize[d];
    ncps[d] = numberOfControlPoints[d];
    isClosed[d] = static_cast<bool>( isParametricDimensionClosed[d] );
    }
  bsplineFilter->SetOrigin( origin );
  bsplineFilter->SetSpacing( spacing );
  bsplineFilter->SetSize( size );
  bsplineFilter->SetNumberOfControlPoints( ncps );
  bsplineFilter->SetSplineOrder( splineOrder );
  bsplineFilter->SetNumberOfLevels( numberOfFittingLevels );
  bsplineFilter->SetCloseDimension( isClosed );
  bsplineFilter->Update();

  //////////////////////////
  //
  //  Only difference between the Curve, Image, and Object function
  //  is the return type.
  //

  std::vector<std::vector<double>> bsplineCurve(parametricDomainSize[0], std::vector<double>(DataDimension) );
  auto bsplineCurveR = bsplineCurve;//.mutable_unchecked<2>();

  unsigned int count = 0;
  IteratorType It( bsplineFilter->GetOutput(),
    bsplineFilter->GetOutput()->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    ScatteredDataType data = It.Value();
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      bsplineCurveR[count][d] = data[d];
      }
    count++;
    }

  return bsplineCurveR;
}

template<unsigned int ParametricDimension>
AntsImage<itk::Image<float, ParametricDimension>> fitBsplineImageHelper(
  std::vector<std::vector<double>> scatteredData,
  std::vector<std::vector<double>> parametricData,
  std::vector<double> dataWeights,
  std::vector<double> parametricDomainOrigin,
  std::vector<double> parametricDomainSpacing,
  std::vector<unsigned int> parametricDomainSize,
  std::vector<bool> isParametricDimensionClosed,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder )
{
  const unsigned int DataDimension = 1;

  using RealType = float;
  using ScatteredDataType = itk::Vector<RealType, DataDimension>;
  using PointSetType = itk::PointSet<ScatteredDataType, ParametricDimension>;
  using OutputImageType = itk::Image<ScatteredDataType, ParametricDimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<OutputImageType>;

  using BSplineFilterType = itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, OutputImageType>;
  using WeightsContainerType = typename BSplineFilterType::WeightsContainerType;

  typename PointSetType::Pointer pointSet = PointSetType::New();
  pointSet->Initialize();
  typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

  auto scatteredDataR = scatteredData;//.unchecked<2>();
  auto parametricDataR = parametricData;//.unchecked<2>();

  unsigned int numberOfPoints = scatteredDataR.size();

  for( unsigned int n = 0; n < numberOfPoints; n++ )
    {
    typename PointSetType::PointType point;
    for( unsigned int d = 0; d < ParametricDimension; d++ )
      {
      point[d] = parametricDataR[n][d];
      }
    pointSet->SetPoint( n, point );

    ScatteredDataType data( 0.0 );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      data[d] = scatteredDataR[n][d];
      }
    pointSet->SetPointData( n, data );

    weights->InsertElement( n, dataWeights[n] );
    }

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();
  bsplineFilter->SetInput( pointSet );
  bsplineFilter->SetPointWeights( weights );
  bsplineFilter->SetGenerateOutputImage( true );

  typename OutputImageType::PointType origin;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::SizeType size;
  typename BSplineFilterType::ArrayType ncps;
  typename BSplineFilterType::ArrayType isClosed;

  for( unsigned int d = 0; d < ParametricDimension; d++ )
    {
    origin[d] = parametricDomainOrigin[d];
    spacing[d] = parametricDomainSpacing[d];
    size[d] = parametricDomainSize[d];
    ncps[d] = numberOfControlPoints[d];
    isClosed[d] = static_cast<bool>( isParametricDimensionClosed[d] );
    }
  bsplineFilter->SetOrigin( origin );
  bsplineFilter->SetSpacing( spacing );
  bsplineFilter->SetSize( size );
  bsplineFilter->SetNumberOfControlPoints( ncps );
  bsplineFilter->SetSplineOrder( splineOrder );
  bsplineFilter->SetNumberOfLevels( numberOfFittingLevels );
  bsplineFilter->SetCloseDimension( isClosed );
  bsplineFilter->Update();

  //////////////////////////
  //
  //  Only difference between the Curve, Image, and Object function
  //  is the return type.
  //

  using ScalarImageType = itk::Image<RealType, ParametricDimension>;
  using SelectionFilterType = itk::VectorIndexSelectionCastImageFilter<OutputImageType, ScalarImageType>;
  typename SelectionFilterType::Pointer selectionFilter = SelectionFilterType::New();
  selectionFilter->SetIndex( 0 );
  selectionFilter->SetInput( bsplineFilter->GetOutput() );
  selectionFilter->Update();

  AntsImage<ScalarImageType> out_ants_image = { selectionFilter->GetOutput() };
  return out_ants_image;
}

template<unsigned int ParametricDimension, unsigned int DataDimension>
AntsImage<itk::VectorImage<float, ParametricDimension>> fitBsplineVectorImageHelper(
  std::vector<std::vector<double>> scatteredData,
  std::vector<std::vector<double>> parametricData,
  std::vector<double> dataWeights,
  std::vector<double> parametricDomainOrigin,
  std::vector<double> parametricDomainSpacing,
  std::vector<unsigned int> parametricDomainSize,
  std::vector<bool> isParametricDimensionClosed,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder )
{
  using RealType = float;
  using ScatteredDataType = itk::Vector<RealType, DataDimension>;
  using PointSetType = itk::PointSet<ScatteredDataType, ParametricDimension>;
  using OutputImageType = itk::Image<ScatteredDataType, ParametricDimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<OutputImageType>;

  using BSplineFilterType = itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, OutputImageType>;
  using WeightsContainerType = typename BSplineFilterType::WeightsContainerType;

  typename PointSetType::Pointer pointSet = PointSetType::New();
  pointSet->Initialize();
  typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

  auto scatteredDataR = scatteredData;//.unchecked<2>();
  auto parametricDataR = parametricData;//.unchecked<2>();

  unsigned int numberOfPoints = scatteredDataR.size();

  for( unsigned int n = 0; n < numberOfPoints; n++ )
    {
    typename PointSetType::PointType point;
    for( unsigned int d = 0; d < ParametricDimension; d++ )
      {
      point[d] = parametricDataR[n][d];
      }
    pointSet->SetPoint( n, point );

    ScatteredDataType data( 0.0 );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      data[d] = scatteredDataR[n][d];
      }
    pointSet->SetPointData( n, data );

    weights->InsertElement( n, dataWeights[n] );
    }

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();
  bsplineFilter->SetInput( pointSet );
  bsplineFilter->SetPointWeights( weights );
  bsplineFilter->SetGenerateOutputImage( true );

  typename OutputImageType::PointType origin;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::SizeType size;
  typename BSplineFilterType::ArrayType ncps;
  typename BSplineFilterType::ArrayType isClosed;

  for( unsigned int d = 0; d < ParametricDimension; d++ )
    {
    origin[d] = parametricDomainOrigin[d];
    spacing[d] = parametricDomainSpacing[d];
    size[d] = parametricDomainSize[d];
    ncps[d] = numberOfControlPoints[d];
    isClosed[d] = static_cast<bool>( isParametricDimensionClosed[d] );
    }
  bsplineFilter->SetOrigin( origin );
  bsplineFilter->SetSpacing( spacing );
  bsplineFilter->SetSize( size );
  bsplineFilter->SetNumberOfControlPoints( ncps );
  bsplineFilter->SetSplineOrder( splineOrder );
  bsplineFilter->SetNumberOfLevels( numberOfFittingLevels );
  bsplineFilter->SetCloseDimension( isClosed );
  bsplineFilter->Update();

  //////////////////////////
  //
  //  Only difference between the Curve, Image, and Object function
  //  is the return type.
  //

  using VectorImageType = itk::VectorImage<RealType, ParametricDimension>;
  using VectorImagePointerType = typename VectorImageType::Pointer;

  typename VectorImageType::DirectionType direction;
  direction.SetIdentity();

  VectorImagePointerType antsField = VectorImageType::New();
  antsField->SetOrigin( origin );
  antsField->SetRegions( size );
  antsField->SetSpacing( spacing );
  antsField->SetVectorLength( DataDimension );
  antsField->SetDirection( direction );
  antsField->AllocateInitialized();

  IteratorType It( bsplineFilter->GetOutput(),
    bsplineFilter->GetOutput()->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    ScatteredDataType data = It.Value();

    typename VectorImageType::PixelType antsVector( DataDimension );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      antsVector[d] = data[d];
      }
    antsField->SetPixel( It.GetIndex(), antsVector );
    }

  AntsImage<VectorImageType> out_ants_image = { antsField };
  return out_ants_image;
}

void local_fitBsplineObjectToScatteredData(nb::module_ &m)
{
  m.def("fitBsplineObjectToScatteredDataP1D1", &fitBsplineCurveHelper<1>);
  m.def("fitBsplineObjectToScatteredDataP1D2", &fitBsplineCurveHelper<2>);
  m.def("fitBsplineObjectToScatteredDataP1D3", &fitBsplineCurveHelper<3>);
  m.def("fitBsplineObjectToScatteredDataP1D4", &fitBsplineCurveHelper<4>);

  m.def("fitBsplineObjectToScatteredDataP2D1", &fitBsplineImageHelper<2>);
  m.def("fitBsplineObjectToScatteredDataP3D1", &fitBsplineImageHelper<3>);
  m.def("fitBsplineObjectToScatteredDataP4D1", &fitBsplineImageHelper<4>);

  m.def("fitBsplineObjectToScatteredDataP2D2", &fitBsplineVectorImageHelper<2, 2>);
  m.def("fitBsplineObjectToScatteredDataP3D3", &fitBsplineVectorImageHelper<3, 3>);

  m.def("fitBsplineObjectToScatteredDataP3D2", &fitBsplineVectorImageHelper<3, 2>);
  m.def("fitBsplineObjectToScatteredDataP4D3", &fitBsplineVectorImageHelper<4, 3>);
}
