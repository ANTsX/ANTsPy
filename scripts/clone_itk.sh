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

# cd ./src
itkgit=https://github.com/InsightSoftwareConsortium/ITK.git
itktag=be79ceb0a9343c02dba310f5faee371941f6fa40 # 3-15-24
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
