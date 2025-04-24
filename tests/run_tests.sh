#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}
COVERAGE=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -p|--python) PYCMD=$2; shift 2 ;;
        -c|--coverage) COVERAGE=1; shift 1;;
        --) shift; break ;;
        *) echo "Invalid argument: $1!" ; exit 1 ;;
    esac
done

if [[ $COVERAGE -eq 1 ]]; then
    coverage erase
    PYCMD="coverage run --parallel-mode --source ants "
    echo "coverage flag found. Setting command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"

echo "Running core tests"
$PYCMD test_core_ants_image.py $@
$PYCMD test_core_ants_image_indexing.py $@
$PYCMD test_core_ants_image_io.py $@
$PYCMD test_core_ants_transform.py $@
$PYCMD test_core_ants_transform_io.py $@
$PYCMD test_core_ants_metric.py $@

echo "Running learn tests"
$PYCMD test_learn.py $@

echo "Running registration tests"
$PYCMD test_registration.py $@

echo "Running segmentation tests"
$PYCMD test_segmentation.py $@

echo "Running utils tests"
$PYCMD test_utils.py $@

echo "Running ops tests"
$PYCMD test_ops.py $@

echo "Running plotting tests"
$PYCMD test_plotting.py $@

echo "Running bug tests"
$PYCMD test_bugs.py $@

echo "Running deeplearn tests"
$PYCMD test_deeplearn.py $@

if [[ $COVERAGE -eq 1 ]]; then
    coverage combine
    coverage xml
fi


popd
