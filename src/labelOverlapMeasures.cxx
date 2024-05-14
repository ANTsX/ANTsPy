
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
#include "itkLabelOverlapMeasuresImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template<class PrecisionType, unsigned int ImageDimension>
nb::dict labelOverlapMeasures( AntsImage<itk::Image<PrecisionType, ImageDimension>> &  antsSourceImage,
                                AntsImage<itk::Image<PrecisionType, ImageDimension>> &  antsTargetImage )
{
  using ImageType = itk::Image<PrecisionType, ImageDimension>;
  using ImagePointerType = typename ImageType::Pointer;

  typename ImageType::Pointer itkSourceImage = antsSourceImage.ptr;
  typename ImageType::Pointer itkTargetImage = antsTargetImage.ptr;

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

  nb::dict labelOverlapMeasures;
  labelOverlapMeasures["Label"] = labels;
  labelOverlapMeasures["TotalOrTargetOverlap"] = totalOrTargetOverlap;
  labelOverlapMeasures["UnionOverlap"] = unionOverlap;
  labelOverlapMeasures["MeanOverlap"] = meanOverlap;
  labelOverlapMeasures["VolumeSimilarity"] = volumeSimilarity;
  labelOverlapMeasures["FalseNegativeError"] = falseNegativeError;
  labelOverlapMeasures["FalsePositiveError"] = falsePositiveError;

  return labelOverlapMeasures;
}

void local_labelOverlapMeasures(nb::module_ &m)
{
  m.def("labelOverlapMeasures2D", &labelOverlapMeasures<unsigned int, 2>);
  m.def("labelOverlapMeasures3D", &labelOverlapMeasures<unsigned int, 3>);
  m.def("labelOverlapMeasures4D", &labelOverlapMeasures<unsigned int, 4>);
}
