
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkThinPlateSplineKernelTransform.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template<unsigned int Dimension>
py::capsule fitThinPlateSplineVectorImageToScatteredDataHelper(
  py::array_t<double> displacementOrigins,
  py::array_t<double> displacements,
  py::array_t<double> origin,
  py::array_t<double> spacing,
  py::array_t<unsigned int> size,
  py::array_t<double> direction
 )
{
  using RealType = float;

  using ANTsFieldType = itk::VectorImage<RealType, Dimension>;
  using ANTsFieldPointerType = typename ANTsFieldType::Pointer;

  using VectorType = itk::Vector<RealType, Dimension>;

  using ITKFieldType = itk::Image<VectorType, Dimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<ITKFieldType>;

  using CoordinateRepType = float;
  using TransformType = itk::ThinPlateSplineKernelTransform<CoordinateRepType, Dimension>;
  using PointType = itk::Point<CoordinateRepType, Dimension>;
  using PointSetType = typename TransformType::PointSetType;

  auto tps = TransformType::New();

  ////////////////////////////
  //
  //  Define the output thin-plate spline field domain
  //

  auto field = ITKFieldType::New();

  auto originP = origin.unchecked<1>();
  auto spacingP = spacing.unchecked<1>();
  auto sizeP = size.unchecked<1>();
  auto directionP = direction.unchecked<2>();

  if( originP.shape(0) == 0 || sizeP.shape(0) == 0 || spacingP.shape(0) == 0 || directionP.shape(0) == 0 )
    {
    throw std::invalid_argument( "Thin-plate spline domain is not specified." );
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
    field->SetRegions( fieldSize );
    field->SetOrigin( fieldOrigin );
    field->SetSpacing( fieldSpacing );
    field->SetDirection( fieldDirection );
    field->AllocateInitialized();
    }

  auto sourceLandmarks = PointSetType::New();
  auto targetLandmarks = PointSetType::New();
  typename PointSetType::PointsContainer::Pointer sourceLandmarkContainer = sourceLandmarks->GetPoints();
  typename PointSetType::PointsContainer::Pointer targetLandmarkContainer = targetLandmarks->GetPoints();

  PointType sourcePoint;
  PointType targetPoint;

  auto displacementOriginsP = displacementOrigins.unchecked<2>();
  auto displacementsP = displacements.unchecked<2>();
  unsigned int numberOfPoints = displacementsP.shape(0);

  for( unsigned int n = 0; n < numberOfPoints; n++ )
    {
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      sourcePoint[d] = displacementOriginsP(n, d);
      targetPoint[d] = displacementOriginsP(n, d) + displacementsP(n, d);
      }
    sourceLandmarkContainer->InsertElement( n, sourcePoint );
    targetLandmarkContainer->InsertElement( n, targetPoint );
    }

  tps->SetSourceLandmarks( sourceLandmarks );
  tps->SetTargetLandmarks( targetLandmarks );
  tps->ComputeWMatrix();

  //////////////////////////
  //
  //  Now convert back to vector image type.
  //

  ANTsFieldPointerType antsField = ANTsFieldType::New();
  antsField->CopyInformation( field );
  antsField->SetRegions( field->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->AllocateInitialized();

  typename TransformType::InputPointType  source;
  typename TransformType::OutputPointType target;

  IteratorType It( field, field->GetLargestPossibleRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    field->TransformIndexToPhysicalPoint( It.GetIndex(), source );
    target = tps->TransformPoint( source );

    typename ANTsFieldType::PixelType antsVector( Dimension );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      antsVector[d] = target[d] - source[d];
      }
    antsField->SetPixel( It.GetIndex(), antsVector );
    }

  return wrap< ANTsFieldType >( antsField );
}

PYBIND11_MODULE(fitThinPlateSplineDisplacementFieldToScatteredData, m)
{
  m.def("fitThinPlateSplineDisplacementFieldToScatteredDataD2", &fitThinPlateSplineVectorImageToScatteredDataHelper<2>);
  m.def("fitThinPlateSplineDisplacementFieldToScatteredDataD3", &fitThinPlateSplineVectorImageToScatteredDataHelper<3>);
}

