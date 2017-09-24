
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "itkImage.h"
#include "itkLabelStatisticsImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;
using namespace py::literals;

template< unsigned int Dimension >
py::dict labelStatsHelper(
  typename itk::Image< float, Dimension >::Pointer image,
  typename itk::Image< unsigned int, Dimension>::Pointer labelImage)
{
  typedef float PixelType;
  typedef itk::Image< PixelType, Dimension > ImageType;
  typedef unsigned int LabelType;
  typedef typename ImageType::PointType PointType;
  typedef itk::Image< LabelType, Dimension > LabelImageType;
  typedef itk::ImageRegionIteratorWithIndex<LabelImageType>                    Iterator;
  typedef itk::LabelStatisticsImageFilter< ImageType, LabelImageType >
    LabelStatisticsImageFilterType;
  typename LabelStatisticsImageFilterType::Pointer labelStatisticsImageFilter =
    LabelStatisticsImageFilterType::New();
  labelStatisticsImageFilter->SetInput( image );
  labelStatisticsImageFilter->SetLabelInput( labelImage );
  labelStatisticsImageFilter->Update();

  typedef typename LabelStatisticsImageFilterType::ValidLabelValuesContainerType
    ValidLabelValuesType;

  long nlabs = labelStatisticsImageFilter->GetNumberOfLabels();

  std::vector<double> labelvals(nlabs);
  std::vector<double> means(nlabs);
  std::vector<double> mins(nlabs);
  std::vector<double> maxes(nlabs);
  std::vector<double> variances(nlabs);
  std::vector<double> counts(nlabs);
  std::vector<double> volumes(nlabs);
  std::vector<double> x(nlabs);
  std::vector<double> y(nlabs);
  std::vector<double> z(nlabs);
  std::vector<double> t(nlabs);
  std::vector<double> mass(nlabs,0.0);

  typename ImageType::SpacingType spacing = image->GetSpacing();
  float voxelVolume = 1.0;
  for (unsigned int ii = 0; ii < spacing.Size(); ii++)
  {
    voxelVolume *= spacing[ii];
  }

  std::map<LabelType, LabelType> RoiList;

  LabelType ii = 0; // counter for label values
  for (typename ValidLabelValuesType::const_iterator
         labelIterator  = labelStatisticsImageFilter->GetValidLabelValues().begin();
         labelIterator != labelStatisticsImageFilter->GetValidLabelValues().end();
         ++labelIterator)
  {
    if ( labelStatisticsImageFilter->HasLabel(*labelIterator) )
    {
      LabelType labelValue = *labelIterator;
      labelvals[ii] = labelValue;
      means[ii]     = labelStatisticsImageFilter->GetMean(labelValue);
      mins[ii]      = labelStatisticsImageFilter->GetMinimum(labelValue);
      maxes[ii]     = labelStatisticsImageFilter->GetMaximum(labelValue);
      variances[ii] = labelStatisticsImageFilter->GetVariance(labelValue);
      counts[ii]    = labelStatisticsImageFilter->GetCount(labelValue);
      volumes[ii]   = labelStatisticsImageFilter->GetCount(labelValue) * voxelVolume;
      RoiList[ labelValue ] = ii;
    }
    ++ii;
  }

  Iterator It( labelImage, labelImage->GetLargestPossibleRegion() );
  std::vector<PointType> comvec;
  for ( unsigned int i = 0; i < nlabs; i++ )
    {
    typename ImageType::PointType myCenterOfMass;
    myCenterOfMass.Fill(0);
    comvec.push_back( myCenterOfMass );
    }
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    LabelType label = static_cast<LabelType>( It.Get() );
    if(  label > 0  )
      {
      typename ImageType::PointType point;
      image->TransformIndexToPhysicalPoint( It.GetIndex(), point );
      for( unsigned int i = 0; i < spacing.Size(); i++ )
        {
        comvec[  RoiList[ label ] ][i] += point[i];
        }
      mass[  RoiList[ label ] ] += image->GetPixel( It.GetIndex() );
      }
    }
  for ( unsigned int labelcount = 0; labelcount < comvec.size(); labelcount++ )
    {
    for ( unsigned int k = 0; k < Dimension; k++ )
      {
      comvec[ labelcount ][k] = comvec[ labelcount ][k] / counts[labelcount];
      }
    x[labelcount]=comvec[ labelcount ][0];
    y[labelcount]=comvec[ labelcount ][1];
    if ( Dimension > 2 ) z[labelcount]=comvec[ labelcount ][2];
    if ( Dimension > 3 ) t[labelcount]=comvec[ labelcount ][3];
    }

  py::dict labelStats = py::dict( "LabelValue"_a=labelvals,
                                  "Mean"_a=means,
                                  "Min"_a=mins,
                                  "Max"_a=maxes,
                                  "Variance"_a=variances,
                                  "Count"_a=counts,
                                  "Volume"_a=volumes,
                                  "Mass"_a=mass,
                                  "x"_a=x,
                                  "y"_a=y,
                                  "z"_a=z,
                                  "t"_a=t );
  return (labelStats);
}

template <unsigned int Dimension>
py::dict labelStats(py::capsule py_image,
                    py::capsule py_labelImage)
{ 
  typedef itk::Image<float, Dimension> FloatImageType;
  typedef itk::Image<unsigned int, Dimension> IntImageType;
  typedef typename FloatImageType::Pointer FloatImagePointerType;
  typedef typename IntImageType::Pointer IntImagePointerType;

  FloatImagePointerType myimage = as<itk::Image<float, Dimension>>( py_image );
  IntImagePointerType mylabelimage = as<itk::Image<unsigned int, Dimension>>( py_labelImage );

  return labelStatsHelper<Dimension>( myimage, mylabelimage );
}


PYBIND11_MODULE(labelStats, m)
{
  m.def("labelStats2D", &labelStats<2>);
  m.def("labelStats3D", &labelStats<3>);
  m.def("labelStats4D", &labelStats<4>);
}
