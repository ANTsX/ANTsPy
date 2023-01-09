
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkVector.h"
#include "itkComposeDisplacementFieldsImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template <unsigned int Dimension>
py::capsule composeDisplacementFields( py::capsule & antsDisplacementField,
                                       py::capsule & antsWarpingField )
{
  using RealType = float;

  using ANTsFieldType = itk::VectorImage<RealType, Dimension>;
  using ANTsFieldPointerType = typename ANTsFieldType::Pointer;

  using VectorType = itk::Vector<RealType, Dimension>;

  using ITKFieldType = itk::Image<VectorType, Dimension>;
  using ITKFieldPointerType = typename ITKFieldType::Pointer;

  using IteratorType = itk::ImageRegionIteratorWithIndex<ITKFieldType>;
  using ConstIteratorType = itk::ImageRegionConstIteratorWithIndex<ITKFieldType>;

  ANTsFieldPointerType inputDisplacementField = as<ANTsFieldType>( antsDisplacementField );
  ANTsFieldPointerType inputWarpingField = as<ANTsFieldType>( antsWarpingField );

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
  inputITKDisplacementField->Allocate();

  ITKFieldPointerType inputITKWarpingField = ITKFieldType::New();
  inputITKWarpingField->SetOrigin( fieldOrigin );
  inputITKWarpingField->SetRegions( fieldSize );
  inputITKWarpingField->SetSpacing( fieldSpacing );
  inputITKWarpingField->SetDirection( fieldDirection );
  inputITKWarpingField->Allocate();

  IteratorType ItF( inputITKDisplacementField, inputITKDisplacementField->GetRequestedRegion() );
  IteratorType ItE( inputITKWarpingField, inputITKWarpingField->GetRequestedRegion() );
  for( ItF.GoToBegin(), ItE.GoToBegin(); !ItF.IsAtEnd(); ++ItF, ++ItE )
    {
    VectorType vectorF;
    VectorType vectorE;

    typename ANTsFieldType::PixelType antsVectorF = inputDisplacementField->GetPixel( ItF.GetIndex() );
    typename ANTsFieldType::PixelType antsVectorE = inputWarpingField->GetPixel( ItE.GetIndex() );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      vectorF[d] = antsVectorF[d];
      vectorE[d] = antsVectorE[d];
      }
    ItF.Set( vectorF );
    ItE.Set( vectorE );
    }

  using ComposerType = itk::ComposeDisplacementFieldsImageFilter<ITKFieldType>;
  typename ComposerType::Pointer composer = ComposerType::New();

  composer->SetDisplacementField( inputITKDisplacementField );
  composer->SetWarpingField( inputITKWarpingField );
  composer->Update();

  //////////////////////////
  //
  //  Now convert back to vector image type.
  //

  ANTsFieldPointerType antsField = ANTsFieldType::New();
  antsField->CopyInformation( composer->GetOutput() );
  antsField->SetRegions( composer->GetOutput()->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->Allocate();

  ConstIteratorType ItI( composer->GetOutput(),
    composer->GetOutput()->GetRequestedRegion() );
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

  return wrap< ANTsFieldType >( antsField );
}

PYBIND11_MODULE(composeDisplacementFields, m)
{
  m.def("composeDisplacementFieldsD2", &composeDisplacementFields<2>);
  m.def("composeDisplacementFieldsD3", &composeDisplacementFields<3>);
}

