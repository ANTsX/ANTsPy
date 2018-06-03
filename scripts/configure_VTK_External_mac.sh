#!/bin/bash

# make this where you want to build VTK

CXX_STD=CXX11
JTHREADS=2
if [[ `uname` -eq Darwin ]] ; then
  CMAKE_BUILD_TYPE=Release
fi
if [[ $TRAVIS -eq true ]] ; then
  CMAKE_BUILD_TYPE=Release
  JTHREADS=2
fi

#mkdir $HOME/vtkbuild-mac;

#cd /users/travis/
cd $HOME

vtkgit=https://github.com/Kitware/VTK.git
vtktag=acc5f269186e3571fb2a10af4448076ecac75e8e
# if ther is a directory but no git,
# remove it

# if no directory, clone VTK in `vtksource-mac` dir
git clone $vtkgit vtksource-mac;
cd vtksource-mac;
git checkout master;
git checkout $vtktag;
cd ../

mkdir vtkbuild-mac
cd vtkbuild-mac
compflags=" -fPIC -O2  "
cmake \
    -DCMAKE_BUILD_TYPE:STRING="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_EXAMPLES:BOOL=OFF \
    -DVTK_LEGACY_REMOVE:BOOL=OFF \
    -DVTK_WRAP_PYTHON:BOOL=OFF ../vtksource-mac/
make -j 3
#make install
cd ../


