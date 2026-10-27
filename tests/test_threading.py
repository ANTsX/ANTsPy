"""Opt-in native concurrency regressions for the GIL release in PR #942.

Run against a rebuilt extension with::

    ANTSPY_TEST_THREADING=1 python tests/test_threading.py

Each scenario runs in a fresh process with one ITK thread, distinct output
paths, and a timeout. The GIL checks time only a direct native call, excluding
Python argument preparation and output conversion. This suite does not test
concurrent mutation of an image, Ctrl+C handling, or all ANTs global state.
"""

import faulthandler
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import ants
import numpy as np
from ants.internal import get_lib_fn, get_pointer_string, process_arguments


_RESULT_PREFIX = 'ANTSPY_THREADING_RESULT='


def _image(size, dimension=3, variant=0):
    """Positive, asymmetric synthetic anatomy; requires no downloaded data."""
    coordinates = np.meshgrid(*([np.linspace(-1, 1, size)] * dimension), indexing='ij')
    radius = sum((axis - 0.12 * (d + variant % 2)) ** 2
                 for d, axis in enumerate(coordinates))
    values = 10 + 60 * np.exp(-5 * radius)
    values += 8 * np.sin(4 * coordinates[0]) ** 2
    values *= np.exp(0.15 * coordinates[-1])
    return ants.from_numpy(values.astype('float32'))


def _native_workload(operation, directory):
    """Keep image owners alive while native arguments contain raw addresses."""
    if operation == 'registration':
        image = _image(48)
        moving = ants.from_numpy(np.roll(image.numpy(), 2, axis=0))
        output = image.clone('float')
        fixed_ptr, moving_ptr = map(get_pointer_string, (image, moving))
        args = ['-d', 3, '-r', 'Identity',
                '-m', f'MeanSquares[{fixed_ptr},{moving_ptr},1,0]',
                '-t', 'SyN[0.1,3,0]', '-c', '[12,0,10]', '-s', 0, '-f', 1,
                '-o', f'[{directory}/registration_,{get_pointer_string(output)}]',
                '--float', 1, '-v', 0]
        owners = [image, moving, output]
        name = 'antsRegistration'
    elif operation == 'denoise':
        image = _image(40)
        output = image.clone('float')
        args = {'d': 3, 'i': image, 'o': output, 'n': 'Gaussian',
                's': 1, 'p': 1, 'r': 2, 'v': 0}
        owners = [image, output]
        name = 'DenoiseImage'
    elif operation == 'n4':
        image = _image(40)
        mask = image * 0 + 1
        output = image.clone('float')
        args = {'d': 3, 'i': image, 'o': output, 'x': mask, 's': 1,
                'c': '[15,0]', 'b': '[4x4x4]', 'v': 0}
        owners = [image, mask, output]
        name = 'N4BiasFieldCorrection'
    elif operation == 'imath':
        image = _image(128)
        output = image.clone('float')
        args = [3, output, 'Grad', image, 1]
        owners = [image, output]
        name = 'iMath'
    else:
        raise ValueError(operation)
    return get_lib_fn(name), process_arguments(args), owners, output


def _gil_progress(operation, directory):
    native, args, owners, output = _native_workload(operation, directory)
    # A long switch interval prevents an incidental Python context switch just
    # before/after the C++ call from masquerading as progress inside that call.
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1.0)
    ready = threading.Event()
    stop = threading.Event()
    samples = []

    def heartbeat():
        ready.set()
        while not stop.is_set():
            samples.append(time.perf_counter())
            time.sleep(0.002)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    if not ready.wait(5):
        raise AssertionError('Heartbeat thread did not start')
    try:
        started = time.perf_counter()
        status = native(args)
        finished = time.perf_counter()
    finally:
        stop.set()
        worker.join(timeout=5)
        sys.setswitchinterval(previous_interval)
    if worker.is_alive():
        raise AssertionError('Heartbeat thread did not stop')
    if status != 0:
        raise AssertionError(f'{operation} failed with exit status {status}')
    elapsed = finished - started
    if elapsed < 0.05:
        raise AssertionError(f'{operation} workload too short for a GIL check: {elapsed:.4f}s')
    # Discard both edges of the interval, where scheduler timing is ambiguous.
    margin = min(0.02, elapsed / 4)
    interior = [t for t in samples if started + margin < t < finished - margin]
    if len(interior) < 2:
        raise AssertionError(f'{operation} held the GIL: {len(interior)} interior heartbeats '
                             f'during a {elapsed:.3f}s native call')
    values = output.numpy()
    if not np.isfinite(values).all() or np.ptp(values) == 0:
        raise AssertionError(f'{operation} produced an invalid or constant image')
    return {'operation': operation, 'seconds': elapsed, 'heartbeats': len(interior)}


def _run_operation(operation, image, directory):
    if operation == 'registration':
        moving = ants.from_numpy(np.roll(image.numpy(), 1, axis=0),
                                 origin=image.origin, spacing=image.spacing,
                                 direction=image.direction)
        result = ants.registration(
            image, moving, type_of_transform='SyNOnly', initial_transform='Identity',
            syn_metric='meansquares', syn_sampling=0, reg_iterations=(8,),
            outprefix=str(directory / 'registration_'), singleprecision=True)
        arrays = [result['warpedmovout'].numpy(), result['warpedfixout'].numpy()]
        paths = set(result['fwdtransforms'] + result['invtransforms'])
        for filename in sorted(paths):
            path = Path(filename)
            if path.parent.resolve() != directory.resolve() or not path.is_file():
                raise AssertionError(f'Registration output escaped its directory: {path}')
            if path.suffix == '.mat':
                arrays.append(np.asarray(ants.read_transform(str(path)).parameters))
            else:
                arrays.append(ants.image_read(str(path)).numpy())
        return arrays
    if operation == 'denoise':
        return [ants.denoise_image(image, noise_model='Gaussian', p=1, r=2).numpy()]
    if operation == 'n4':
        return [ants.n4_bias_field_correction(
            image, shrink_factor=1, convergence={'iters': [6], 'tol': 0},
            spline_param=[3, 3, 3]).numpy()]
    if operation == 'imath':
        return [ants.iMath(image, 'Grad', 1).numpy()]
    if operation == 'apply_transforms':
        return [ants.apply_transforms(image, image, transformlist=[],
                                      interpolator='linear').numpy()]
    raise ValueError(operation)


def _concurrent_results(operation, directory, shared):
    """Compare repeated overlapping calls with sequential reference results."""
    size = 80 if operation in ('imath', 'apply_transforms') else 24
    images = [_image(size, variant=i) for i in range(2)]
    if shared:
        images[1] = images[0]
    before = [(image.numpy(), image.origin, image.spacing, image.direction.copy())
              for image in images]
    expected = []
    for i, image in enumerate(images):
        path = directory / f'sequential_{i}'
        path.mkdir()
        expected.append(_run_operation(operation, image, path))

    # Validate repeatability before attributing any mismatch to concurrency.
    # In particular, regular/random registration sampling can vary even in
    # sequential calls; this case uses a dense MeanSquares SyN metric instead.
    for i, image in enumerate(images):
        path = directory / f'sequential_repeat_{i}'
        path.mkdir()
        repeated = _run_operation(operation, image, path)
        for result, reference in zip(repeated, expected[i]):
            np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5,
                                       err_msg=f'{operation} is not sequentially repeatable')

    modules = {
        'registration': ('ants.registration.registration', 'antsRegistration'),
        'denoise': ('ants.ops.denoise_image', 'DenoiseImage'),
        'n4': ('ants.ops.bias_correction', 'N4BiasFieldCorrection'),
        'imath': ('ants.ops.iMath', 'iMath'),
        'apply_transforms': ('ants.registration.apply_transforms', 'antsApplyTransforms'),
    }
    module_name, command = modules[operation]
    module = importlib.import_module(module_name)
    lookup = module.get_lib_fn
    overlaps = []
    for repeat in range(3):
        barrier = threading.Barrier(2)
        spans = []

        def native_lookup(name):
            function = lookup(name)
            if name != command:
                return function

            def synchronized_call(args):
                # Both Python wrappers must reach the native call before
                # either starts; Python preparation alone is not concurrency.
                barrier.wait(timeout=10)
                started = time.perf_counter()
                status = function(args)
                finished = time.perf_counter()
                spans.append((started, finished))
                return status
            return synchronized_call

        def task(index):
            path = directory / f'parallel_{repeat}_{index}'
            path.mkdir()
            return _run_operation(operation, images[index], path)

        with patch.object(module, 'get_lib_fn', side_effect=native_lookup):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(task, i) for i in range(2)]
                actual = [future.result(timeout=60) for future in futures]
        if len(spans) != 2:
            raise AssertionError(f'Expected two {command} calls, got {len(spans)}')
        overlap = min(end for _, end in spans) - max(start for start, _ in spans)
        if overlap <= 0:
            raise AssertionError(f'{command} calls did not overlap: {spans}')
        overlaps.append(overlap)
        for index in range(2):
            if len(actual[index]) != len(expected[index]):
                raise AssertionError('Concurrent call returned a different number of outputs')
            for result, reference in zip(actual[index], expected[index]):
                if not np.isfinite(result).all() or np.ptp(result) == 0:
                    raise AssertionError(f'{operation} returned invalid or constant data')
                np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5,
                                           err_msg=f'{operation}, repeat {repeat}, worker {index}')
    for image, (pixels, origin, spacing, direction) in zip(images, before):
        np.testing.assert_array_equal(image.numpy(), pixels)
        np.testing.assert_array_equal(image.direction, direction)
        if image.origin != origin or image.spacing != spacing:
            raise AssertionError('Concurrent calls changed input geometry')
    return {'operation': operation, 'shared_input': shared, 'repeats': 3,
            'native_overlap_seconds': overlaps}


def _scenario(name):
    faulthandler.enable()
    faulthandler.dump_traceback_later(110, exit=True)
    try:
        with tempfile.TemporaryDirectory(prefix='antspy-threading-') as directory:
            parts = name.split(':')
            if parts[0] == 'gil':
                result = _gil_progress(parts[1], directory)
            else:
                result = _concurrent_results(parts[1], Path(directory), parts[2] == 'shared')
            print(_RESULT_PREFIX + json.dumps(result), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


@unittest.skipUnless(os.environ.get('ANTSPY_TEST_THREADING') == '1',
                     'Set ANTSPY_TEST_THREADING=1 with the rebuilt GIL-release extension')
class TestNativeThreading(unittest.TestCase):
    def run_scenario(self, name):
        # Load the exact extension under test in the child, including editable
        # source checkouts whose compiled extension lives outside ants/.
        source = str(Path(__file__).resolve())
        native = str(Path(importlib.import_module("ants.lib").__file__).resolve())
        package_root = str(Path(ants.__file__).resolve().parent.parent)
        bootstrap = (
            'import importlib.util, runpy, sys\n'
            f'sys.path.insert(0, {package_root!r})\n'
            f'spec = importlib.util.spec_from_file_location("ants.lib", {native!r})\n'
            'module = importlib.util.module_from_spec(spec)\n'
            'spec.loader.exec_module(module)\n'
            'sys.modules["ants.lib"] = module\n'
            f'sys.argv = [{source!r}, "--threading-case", {name!r}]\n'
            f'runpy.run_path({source!r}, run_name="__main__")\n'
        )
        environment = os.environ.copy()
        environment['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
        environment['OMP_NUM_THREADS'] = '1'
        environment['OPENBLAS_NUM_THREADS'] = '1'
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        with tempfile.TemporaryDirectory(prefix='antspy-threading-cache-') as cache:
            environment.setdefault('MPLCONFIGDIR', cache)
            try:
                process = subprocess.run([sys.executable, '-c', bootstrap],
                                         env=environment, capture_output=True,
                                         text=True, timeout=120)
            except subprocess.TimeoutExpired as error:
                self.fail(f'{name} timed out (possible native deadlock): {error}')
        self.assertEqual(process.returncode, 0,
                         f'{name} failed:\n{process.stdout}\n{process.stderr}')
        reports = [line[len(_RESULT_PREFIX):] for line in process.stdout.splitlines()
                   if line.startswith(_RESULT_PREFIX)]
        self.assertEqual(len(reports), 1, f'Missing scenario report: {process.stdout}')
        return json.loads(reports[0])

    def test_registration_releases_gil(self):
        self.run_scenario('gil:registration')

    def test_denoising_releases_gil(self):
        self.run_scenario('gil:denoise')

    def test_n4_releases_gil(self):
        self.run_scenario('gil:n4')

    def test_imath_releases_gil(self):
        self.run_scenario('gil:imath')

    def test_independent_registrations_match_sequential_results(self):
        self.run_scenario('concurrent:registration:independent')

    def test_independent_denoising_matches_sequential_results(self):
        self.run_scenario('concurrent:denoise:independent')

    def test_independent_n4_matches_sequential_results(self):
        self.run_scenario('concurrent:n4:independent')

    def test_independent_imath_matches_sequential_results(self):
        self.run_scenario('concurrent:imath:independent')

    def test_shared_read_only_input_matches_sequential_results(self):
        for operation in ('registration', 'denoise', 'n4', 'imath', 'apply_transforms'):
            with self.subTest(operation=operation):
                self.run_scenario(f'concurrent:{operation}:shared')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--threading-case':
        _scenario(sys.argv[2])
    else:
        unittest.main(verbosity=2)
