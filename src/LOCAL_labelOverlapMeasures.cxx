
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkLabelOverlapMeasuresImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;
using namespace py::literals;

template<class PrecisionType, unsigned int ImageDimension>
py::dict labelOverlapMeasures( py::capsule & antsSourceImage,
                                  py::capsule & antsTargetImage )
{
  using ImageType = itk::Image<PrecisionType, ImageDimension>;
  using ImagePointerType = typename ImageType::Pointer;

  ImagePointerType itkSourceImage = as< ImageType >( antsSourceImage );
  ImagePointerType itkTargetImage = as< ImageType >( antsTargetImage );

  using FilterType = itk::LabelOverlapMeasuresImageFilter<ImageType>;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetSourceImage( itkSourceImage );
  filter->SetTargetImage( itkTargetImage );
  filter->Update();

  typename FilterType::MapType labelMap = filter->GetLabelSetMeasures();

  // Sort the labels

  std::vector<PrecisionType> allLabels;
  allLabels.clear();
  for( typename FilterType::MapType::const_iterator it = labelMap.begin();
       it != labelMap.end(); ++it )
    {
    if( (*it).first == 0 )
      {
      continue;
      }

    const int label = (*it).first;
    allLabels.push_back( label );
    }
  std::sort( allLabels.begin(), allLabels.end() );


  // Now put the results in an Rcpp data frame

  unsigned int vectorLength = 1 + allLabels.size();

  std::vector<PrecisionType> labels( vectorLength );
  std::vector<double> totalOrTargetOverlap( vectorLength );
  std::vector<double> unionOverlap( vectorLength );
  std::vector<double> meanOverlap( vectorLength );
  std::vector<double> volumeSimilarity( vectorLength );
  std::vector<double> falseNegativeError( vectorLength );
  std::vector<double> falsePositiveError( vectorLength );

  // We'll replace label '0' with "All" in the R wrapper.
  labels[0] = itk::NumericTraits<PrecisionType>::Zero;
  totalOrTargetOverlap[0] = filter->GetTotalOverlap();
  unionOverlap[0] = filter->GetUnionOverlap();
  meanOverlap[0] = filter->GetMeanOverlap();
  volumeSimilarity[0] = filter->GetVolumeSimilarity();
  falseNegativeError[0] = filter->GetFalseNegativeError();
  falsePositiveError[0] = filter->GetFalsePositiveError();

  unsigned int i = 1;
  typename std::vector<PrecisionType>::const_iterator itL = allLabels.begin();
  for( itL = allLabels.begin(); itL != allLabels.end(); ++itL )
    {
    labels[i] = *itL;
    totalOrTargetOverlap[i] = filter->GetTargetOverlap( *itL );
    unionOverlap[i] = filter->GetUnionOverlap( *itL );
    meanOverlap[i] = filter->GetMeanOverlap( *itL );
    volumeSimilarity[i] = filter->GetVolumeSimilarity( *itL );
    falseNegativeError[i] = filter->GetFalseNegativeError( *itL );
    falsePositiveError[i] = filter->GetFalsePositiveError( *itL );
    i++;
    }

  py::dict labelOverlapMeasures = py::dict( "Label"_a=labels,
                                            "TotalOrTargetOverlap"_a=totalOrTargetOverlap,
                                            "UnionOverlap"_a=unionOverlap,
                                            "MeanOverlap"_a=meanOverlap,
                                            "VolumeSimilarity"_a=volumeSimilarity,
                                            "FalseNegativeError"_a=falseNegativeError,
                                            "FalsePositiveError"_a=falsePositiveError );

  return (labelOverlapMeasures);
}

PYBIND11_MODULE(labelOverlapMeasures, m)
{
  m.def("labelOverlapMeasures2D", &labelOverlapMeasures<unsigned int, 2>);
  m.def("labelOverlapMeasures3D", &labelOverlapMeasures<unsigned int, 3>);
  m.def("labelOverlapMeasures4D", &labelOverlapMeasures<unsigned int, 4>);
}
