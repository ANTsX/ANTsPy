/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkMGHImageIO.h"
#include "itkByteSwapper.h"
#include "itkMetaDataObject.h"
#include "itksys/SystemTools.hxx"

namespace itk
{

//VALID file extensions
static const std::string __MGH_EXT(".mgh");
static const std::string __MGZ_EXT(".mgz");
static const std::string __GZ_EXT(".gz");

typedef itk::Matrix<double,3,3> MatrixType;
typedef itk::Vector<double, 3>  VectorType;

static MatrixType GetRAS2LPS()
  {
  MatrixType RAS2LAS;
  RAS2LAS.SetIdentity();
  RAS2LAS[0][0]=-1.0;
  RAS2LAS[1][1]=-1.0;
  RAS2LAS[2][2]= 1.0;
  return RAS2LAS;
  }

// -------------------------------
//
// Convert to BE
//
// -------------------------------


} // end namespace itk