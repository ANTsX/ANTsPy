#!/bin/bash
CXX_STD=CXX11
JTHREADS=2
if [[ `uname` -eq Darwin ]] ; then
  CMAKE_BUILD_TYPE=Release
fi
if [[ $TRAVIS -eq true ]] ; then
  CMAKE_BUILD_TYPE=Release
  JTHREADS=2
fi

#cd ./src
itkgit=https://github.com/InsightSoftwareConsortium/ITK.git
itktag=902c9d0e0a2d8349796b1b2b805fd94648e8a197 # 5.0
itkgit=https://github.com/stnava/ITK.git
itktag=20a456c3803d5d8e38b1c0a62df484fd10b8b71b
# if ther is a directory but no git,
# remove it
if [[ -d itksource ]]; then
    if [[ ! -d itksource/.git ]]; then
        rm -rf itksource/
    fi
fi
# if no directory, clone ITK in `itksource` dir
if [[ ! -d itksource ]]; then
    git clone $itkgit itksource
fi
cd itksource
if [[ -d .git ]]; then
    git checkout master;
    git checkout $itktag
    rm -rf .git
fi
# go back to main dir
cd ../
#if [[ ! -d ../data/ ]] ; then
#  mkdir -p ../data
#fi
