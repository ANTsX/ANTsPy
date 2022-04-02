SET PYCMD=python
pushd "tests"
echo "Running core tests"
%PYCMD% test_core_ants_image.py %@%
%PYCMD% test_core_ants_image_io.py %@%
%PYCMD% test_core_ants_transform.py %@%
%PYCMD% test_core_ants_transform_io.py %@%
%PYCMD% test_core_ants_metric.py %@%
echo "Running learn tests"
%PYCMD% test_learn.py %@%
echo "Running registation tests"
%PYCMD% test_registation.py %@%
echo "Running segmentation tests"
%PYCMD% test_segmentation.py %@%
echo "Running utils tests"
%PYCMD% test_utils.py %@%
echo "Running bug tests"
%PYCMD% test_bugs.py %@%
popd