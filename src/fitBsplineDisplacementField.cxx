
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "itkPointSet.h"
#include "itkDisplacementFieldToBSplineImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;

template<unsigned int Dimension>
AntsImage<itk::VectorImage<float, Dimension>> fitBsplineVectorImageHelper(
  AntsImage<itk::VectorImage<float, Dimension>> & displacementField,
  AntsImage<itk::Image<float, Dimension>> & displacementFieldWeightImage,
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

  ANTsFieldPointerType inputANTsField = displacementField.ptr;

  typename ITKFieldType::PointType fieldOrigin;
  typename ITKFieldType::SpacingType fieldSpacing;
  typename ITKFieldType::SizeType fieldSize;
  typename ITKFieldType::DirectionType fieldDirection;

  for( unsigned int d = 0; d < Dimension; d++ )
    {
    fieldOrigin[d] = inputANTsField->GetOrigin()[d];
    fieldSpacing[d] = inputANTsField->GetSpacing()[d];
    fieldSize[d] = inputANTsField->GetRequestedRegion().GetSize()[d];
    for( unsigned int e = 0; e < Dimension; e++ )
      {
      fieldDirection(d, e) = inputANTsField->GetDirection()(d, e);
      }
    }

  ITKFieldPointerType inputITKField = ITKFieldType::New();
  inputITKField->SetOrigin( fieldOrigin );
  inputITKField->SetRegions( fieldSize );
  inputITKField->SetSpacing( fieldSpacing );
  inputITKField->SetDirection( fieldDirection );
  inputITKField->AllocateInitialized();

  IteratorType It( inputITKField, inputITKField->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    VectorType vector;

    typename ANTsFieldType::PixelType antsVector = inputANTsField ->GetPixel( It.GetIndex() );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      vector[d] = antsVector[d];
      }
    It.Set( vector );
    }
  bsplineFilter->SetDisplacementField( inputITKField );

  using WeightImageType = typename BSplineFilterType::RealImageType;
  using WeightImagePointerType = typename WeightImageType::Pointer;

  using InputWeightImageType = itk::Image<RealType, Dimension>;
  using WeightCastFilterType = itk::CastImageFilter<InputWeightImageType, WeightImageType>;
  typename WeightCastFilterType::Pointer weightCastFilter = WeightCastFilterType::New();
  weightCastFilter->SetInput( displacementFieldWeightImage.ptr );
  weightCastFilter->Update();
  WeightImagePointerType weightImage = weightCastFilter->GetOutput();
  bsplineFilter->SetConfidenceImage( weightImage );

  auto displacementOriginsP = displacementOrigins;
  auto displacementsP = displacements;

  unsigned int numberOfPoints = displacementsP.size();

  if( numberOfPoints > 0 )
    {
    typename PointSetType::Pointer pointSet = PointSetType::New();
    pointSet->Initialize();
    typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

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
    bsplineFilter->SetPointSet( pointSet );
    bsplineFilter->SetPointSetConfidenceWeights( weights );
    }

  ////////////////////////////
  //
  //  Define the output B-spline field domain
  //

  auto originP = origin;
  auto spacingP = spacing;
  auto sizeP = size;
  auto directionP = direction;

  if( originP.size() == 0 || sizeP.size() == 0 || spacingP.size() == 0 || directionP.size() == 0 )
    {
    bsplineFilter->SetUseInputFieldToDefineTheBSplineDomain( true );
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

  IteratorType ItB( bsplineFilter->GetOutput(),
    bsplineFilter->GetOutput()->GetRequestedRegion() );
  for( ItB.GoToBegin(); !ItB.IsAtEnd(); ++ItB )
    {
    VectorType data = ItB.Value();

    typename ANTsFieldType::PixelType antsVector( Dimension );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      antsVector[d] = data[d];
      }
    antsField->SetPixel( ItB.GetIndex(), antsVector );
    }

  AntsImage<ANTsFieldType> out_ants_image = { antsField };
  return out_ants_image;
}

void local_fitBsplineDisplacementField(nb::module_ &m)
{
  m.def("fitBsplineDisplacementFieldD2", &fitBsplineVectorImageHelper<2>);
  m.def("fitBsplineDisplacementFieldD3", &fitBsplineVectorImageHelper<3>);
}
