#!/bin/bash
CXX_STD=CXX11
JTHREADS=2
CMAKE_BUILD_TYPE=Release
itkdir=${TRAVIS_BUILD_DIR}/itkjunk/itkbuild-${TRAVIS_OS_NAME}
if [[ ! -d $itkdir ]] ; then
  mkdir -p $itkdir
fi
cd $itkdir
itkgit=https://github.com/InsightSoftwareConsortium/ITK.git
itktag=ce57f309f4f4142b80367fd89a88f8dac9d6943a
if [[ ! -d ITK ]] ; then
  git clone $itkgit
fi
cd ITK
git pull
git checkout $itktag
cd ..
/usr/local/bin/cmake --version
compflags=" -fPIC -O2  "
/usr/local/bin/cmake \
    -DCMAKE_BUILD_TYPE:STRING="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DITK_USE_GIT_PROTOCOL:BOOL=OFF \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_EXAMPLES:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=${R_PACKAGE_DIR}/libs/  \
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
    -D ITKGroup_Core=ON \
    -D Module_ITKReview=ON \
    -D ITKGroup_Filtering=ON \
    -D ITKGroup_IO=ON \
    -D ITKGroup_Numerics=ON \
    -D ITKGroup_Registration=ON \
    -D ITKGroup_Segmentation=ON \
    -DCMAKE_C_VISIBILITY_PRESET:BOOL=hidden \
    -DCMAKE_CXX_VISIBILITY_PRESET:BOOL=hidden \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON ./ITK/
make -j $JTHREADS
cd ../
