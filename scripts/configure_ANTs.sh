#!/bin/bash

# ---------------------------------------------
# create local ~/.antspy dir and move package data to it

if [[ ! -d ~/.antspy ]]; then
    mkdir ~/.antspy
fi

cp data/* ~/.antspy/

# ---------------------------------------------
# clone ANTs and move all files into library directory

antsgit=https://github.com/ANTsX/ANTs.git
antstag=52bc0ab588102682587303d31c720de94e6ef6c1 # 2025-06-10 v2.6.2
echo "ANTS;${antstag}" >> ./data/softwareVersions.csv

cd src # go to lib dir

# if antscore doesnt exist, create it
if [[ ! -d antscore ]] ; then
    git clone $antsbranch $antsgit antsrepo

    if [[ ! -d antscore ]] ; then
        mkdir antscore
    fi

    cd antsrepo # go to antscore
    # check out right branch
    if [[ -d .git ]]; then
        git checkout master
        git pull
        git checkout $antstag
    fi
    cd ..
    # copy antscore files into library
    cp -r antsrepo/Examples/*  antscore/
    cp -r antsrepo/Examples/include/*  antscore
    cp -r antsrepo/ImageRegistration/*  antscore/
    cp -r antsrepo/ImageSegmentation/*  antscore/
    cp -r antsrepo/Tensor/*  antscore/
    cp -r antsrepo/Temporary/*  antscore/
    cp -r antsrepo/Utilities/*  antscore/


    rm -rf antsrepo # remove directory

    # lil hack bc of stupid angled import bug in actual files
    # cp ReadWriteData.h antscore/ReadWriteData.h
    # lil hack bc ANTsVersionConfig.h is only created if you build ANTs...
    cp ANTsVersionConfig.h antscore/ANTsVersionConfig.h
fi

cd  ../../ # go back to main dir
