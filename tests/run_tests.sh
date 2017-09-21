#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}


if [[ $COVERAGE -eq 1 ]]; then
    coverage erase
    PYCMD="coverage run --parallel-mode --source torch "
    echo "coverage flag found. Setting python command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"


echo "Running core tests"
$PYCMD core/test_ants_image.py $@
$PYCMD core/test_ants_image_io.py $@
$PYCMD core/test_ants_metric.py $@
$PYCMD core/test_ants_metric_io.py $@
$PYCMD core/test_ants_transform.py $@
$PYCMD core/test_ants_transform_io.py $@

echo "Running learn tests"
$PYCMD learn/test_decomposition.py $@

popd
