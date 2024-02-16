#!/bin/bash
CXX_STD=CXX11
JTHREADS=2
CMAKE_BUILD_TYPE=Release
cd $HOME
itkgit=https://github.com/InsightSoftwareConsortium/ITK.git
itkgit=https://github.com/stnava/ITK.git
itktag=20a456c3803d5d8e38b1c0a62df484fd10b8b71b
if [[ ! -d ITK ]] ; then
  git clone $itkgit
fi
if [[ ! -d itkbuild-linux ]] ; then
  mkdir itkbuild-linux
fi
cd itkbuild-linux
compflags=" -fPIC -O2  "
cmake \
    -DCMAKE_BUILD_TYPE:STRING="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DITK_USE_GIT_PROTOCOL:BOOL=OFF \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_EXAMPLES:BOOL=OFF \
    -DITK_LEGACY_REMOVE:BOOL=OFF  \
    -DITK_FUTURE_LEGACY_REMOVE:=BOOL=ON \
    -DITKV3_COMPATIBILITY:BOOL=ON \
    -DITK_BUILD_DEFAULT_MODULES:BOOL=OFF \
    -DKWSYS_USE_MD5:BOOL=ON \
    -DITK_WRAPPING:BOOL=OFF \
    -DModule_MGHIO:BOOL=ON \
    -DModule_ITKDeprecated:BOOL=OFF \
    -DModule_ITKReview:BOOL=ON \
    -DModule_ITKVtkGlue:BOOL=OFF \
    -DModule_GenericLabelInterpolator:BOOL=ON \
    -DModule_AdaptiveDenoising:BOOL=ON \
    -D ITKGroup_Core=ON \
    -D Module_ITKReview=ON \
    -D ITKGroup_Filtering=ON \
    -D ITKGroup_IO=ON \
    -D ITKGroup_Numerics=ON \
    -D ITKGroup_Registration=ON \
    -D ITKGroup_Segmentation=ON \
    -DCMAKE_C_VISIBILITY_PRESET:BOOL=hidden \
    -DCMAKE_CXX_VISIBILITY_PRESET:BOOL=hidden \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON ../ITK/
make -j 3
cd ../
