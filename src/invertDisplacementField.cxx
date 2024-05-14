
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
#include "itkVector.h"
#include "itkInvertDisplacementFieldImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <unsigned int Dimension>
AntsImage<itk::VectorImage<float, Dimension>> invertDisplacementField( AntsImage<itk::VectorImage<float, Dimension>> & antsDisplacementField,
                                     AntsImage<itk::VectorImage<float, Dimension>> & antsInverseFieldInitialEstimate,
                                     unsigned int maximumNumberOfIterations,
                                     float meanErrorToleranceThreshold,
                                     float maxErrorToleranceThreshold,
                                     bool enforceBoundaryCondition )
{
  using RealType = float;

  using ANTsFieldType = itk::VectorImage<RealType, Dimension>;
  using ANTsFieldPointerType = typename ANTsFieldType::Pointer;

  using VectorType = itk::Vector<RealType, Dimension>;

  using ITKFieldType = itk::Image<VectorType, Dimension>;
  using ITKFieldPointerType = typename ITKFieldType::Pointer;

  using IteratorType = itk::ImageRegionIteratorWithIndex<ITKFieldType>;
  using ConstIteratorType = itk::ImageRegionConstIteratorWithIndex<ITKFieldType>;

  ANTsFieldPointerType inputDisplacementField = antsDisplacementField.ptr;
  ANTsFieldPointerType inputInverseFieldInitialEstimate = antsInverseFieldInitialEstimate.ptr;

  typename ITKFieldType::PointType fieldOrigin;
  typename ITKFieldType::SpacingType fieldSpacing;
  typename ITKFieldType::SizeType fieldSize;
  typename ITKFieldType::DirectionType fieldDirection;

  for( unsigned int d = 0; d < Dimension; d++ )
    {
    fieldOrigin[d] = inputDisplacementField->GetOrigin()[d];
    fieldSpacing[d] = inputDisplacementField->GetSpacing()[d];
    fieldSize[d] = inputDisplacementField->GetRequestedRegion().GetSize()[d];
    for( unsigned int e = 0; e < Dimension; e++ )
      {
      fieldDirection(d, e) = inputDisplacementField->GetDirection()(d, e);
      }
    }

  ITKFieldPointerType inputITKDisplacementField = ITKFieldType::New();
  inputITKDisplacementField->SetOrigin( fieldOrigin );
  inputITKDisplacementField->SetRegions( fieldSize );
  inputITKDisplacementField->SetSpacing( fieldSpacing );
  inputITKDisplacementField->SetDirection( fieldDirection );
  inputITKDisplacementField->AllocateInitialized();

  ITKFieldPointerType inputITKInverseFieldInitialEstimate = ITKFieldType::New();
  inputITKInverseFieldInitialEstimate->SetOrigin( fieldOrigin );
  inputITKInverseFieldInitialEstimate->SetRegions( fieldSize );
  inputITKInverseFieldInitialEstimate->SetSpacing( fieldSpacing );
  inputITKInverseFieldInitialEstimate->SetDirection( fieldDirection );
  inputITKInverseFieldInitialEstimate->AllocateInitialized();

  IteratorType ItF( inputITKDisplacementField, inputITKDisplacementField->GetRequestedRegion() );
  IteratorType ItE( inputITKInverseFieldInitialEstimate, inputITKInverseFieldInitialEstimate->GetRequestedRegion() );
  for( ItF.GoToBegin(), ItE.GoToBegin(); !ItF.IsAtEnd(); ++ItF, ++ItE )
    {
    VectorType vectorF;
    VectorType vectorE;

    typename ANTsFieldType::PixelType antsVectorF = inputDisplacementField->GetPixel( ItF.GetIndex() );
    typename ANTsFieldType::PixelType antsVectorE = inputInverseFieldInitialEstimate->GetPixel( ItE.GetIndex() );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      vectorF[d] = antsVectorF[d];
      vectorE[d] = antsVectorE[d];
      }
    ItF.Set( vectorF );
    ItE.Set( vectorE );
    }

  using InverterType = itk::InvertDisplacementFieldImageFilter<ITKFieldType>;
  typename InverterType::Pointer inverter = InverterType::New();

  inverter->SetInput( inputITKDisplacementField );
  inverter->SetInverseFieldInitialEstimate( inputITKInverseFieldInitialEstimate );
  inverter->SetMaximumNumberOfIterations( maximumNumberOfIterations );
  inverter->SetMeanErrorToleranceThreshold( meanErrorToleranceThreshold );
  inverter->SetMaxErrorToleranceThreshold( maxErrorToleranceThreshold );
  inverter->SetEnforceBoundaryCondition( enforceBoundaryCondition );
  inverter->Update();

  //////////////////////////
  //
  //  Now convert back to vector image type.
  //

  ANTsFieldPointerType antsField = ANTsFieldType::New();
  antsField->CopyInformation( inverter->GetOutput() );
  antsField->SetRegions( inverter->GetOutput()->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->AllocateInitialized();

  ConstIteratorType ItI( inverter->GetOutput(),
    inverter->GetOutput()->GetRequestedRegion() );
  for( ItI.GoToBegin(), ItF.GoToBegin(); !ItI.IsAtEnd(); ++ItI, ++ItF )
    {
    VectorType data = ItI.Value();

    typename ANTsFieldType::PixelType antsVector( Dimension );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      antsVector[d] = data[d];
      }
    antsField->SetPixel( ItI.GetIndex(), antsVector );
    }

  AntsImage<ANTsFieldType> out_ants_image = { antsField };
  return out_ants_image;
}

void local_invertDisplacementField(nb::module_ &m)
{
  m.def("invertDisplacementFieldD2", &invertDisplacementField<2>);
  m.def("invertDisplacementFieldD3", &invertDisplacementField<3>);
}

