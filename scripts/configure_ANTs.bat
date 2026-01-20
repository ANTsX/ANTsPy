:: Converted with the help of https://daniel-sc.github.io/bash-shell-to-bat-converter/
:: @echo off

echo %USERPROFILE%

:: create local ~/.antspy dir and move package data to it
if not exist %USERPROFILE%\.antspy\ (
  mkdir %USERPROFILE%\.antspy
)
COPY data\* %USERPROFILE%\.antspy

:: clone ANTs and move all files into library directory
SET antsgit=https://github.com/ANTsX/ANTs.git
:: 2026-01-19
SET antstag=b53aad349e6767de8e1d99a392f05a82b1bf2373
echo "ANTS;%antstag%" REM UNKNOWN: {"type":"Redirect","op":{"text":">>","type":"dgreat"},"file":{"text":"./data/softwareVersions.csv","type":"Word"}}
cd src
echo "123"
:: if antscore doesnt exist, create it
if not exist "antscore\" (
  echo "2"
  git clone %antsgit% antsrepo
  mkdir antscore
  cd antsrepo

  :: check out right branch
  if exist ".git\" (
    git checkout master
    git pull
    git checkout %antstag%
  )
  cd ..
  COPY  antsrepo\Examples\* antscore\
  COPY  antsrepo\Examples\include\* antscore
  XCOPY  antsrepo\Examples\include\ antscore\include\
  COPY  antsrepo\ImageRegistration\* antscore\
  COPY  antsrepo\ImageSegmentation\* antscore\
  COPY  antsrepo\Tensor\* antscore\
  COPY  antsrepo\Temporary\* antscore\
  COPY  antsrepo\Utilities\* antscore\
  echo "HERE"

  :: lil hack bc of stupid angled import bug in actual files
  :: cp ReadWriteData.h antscore/ReadWriteData.h
  :: lil hack bc ANTsVersionConfig.h is only created if you build ANTs...
  COPY  ANTsVersionConfig.h antscore\ANTsVersionConfig.h
)
cd ..\..
