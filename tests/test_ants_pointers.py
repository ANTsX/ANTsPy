"""This module checks that ANTsPy functions calling libfn pass the correct image pixeltypes to the underlying ANTs executables.
The tests do not check the executables themselves, so they are skipped by default.

Some functions calling libfn to use ANTsPy (rather than ANTs) C++ code will fail if given the wrong pixel type input, however
these will fail with exceptions rather than silently using corrupted data or raising access violations.
"""

from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import ants
import numpy as np
import pandas as pd


class _PointersChecked(Exception):
    """Stop before wrappers try to read outputs from a mocked executable."""


class TestANTsPointers(unittest.TestCase):
    def setUp(self):
        self.image = ants.from_numpy(np.ones((8, 8), dtype=np.uint32))
        # Use a different, incorrect type for roles requiring uint32.
        self.float_image = self.image.clone("float")
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.output_prefix = str(Path(temporary_directory.name) / "output")

    def assert_pointers(self, module_name, command, expected_types, call):
        """Keep images visible to libfn, including images inside pointer strings.

        Replace serialization with a flattening pass. Explicit get_pointer_string
        calls get unique tokens that are expanded back to images, so composite
        output/metric arguments receive the same checks as direct image arguments.
        Check the number and order of images too, to prevent vacuous assertions.
        """
        module = import_module(module_name)
        pointers = {}

        def pointer_string(image):
            token = f"__test_image_{len(pointers)}__"
            pointers[token] = image
            return token

        def image_arguments(args):
            if isinstance(args, dict):
                args = args.values()
            for arg in args:
                if isinstance(arg, (dict, list, tuple)):
                    yield from image_arguments(arg)
                elif isinstance(arg, str):
                    tokens = re.findall(r"__test_image_\d+__", arg)
                    if tokens:
                        yield from (pointers[token] for token in tokens)
                    else:
                        yield arg
                else:
                    yield arg

        def libfn(args):
            images = [arg for arg in args if isinstance(arg, ants.ANTsImage)]
            self.assertEqual(len(images), len(expected_types), command)
            for index, (image, expected) in enumerate(zip(images, expected_types)):
                self.assertEqual(image.pixeltype, expected,
                                 f"{command}: image argument {index}")
            raise _PointersChecked

        native_call = Mock(side_effect=libfn)
        with ExitStack() as stack:
            lookup = stack.enter_context(patch.object(module, "get_lib_fn", return_value=native_call))
            stack.enter_context(patch.object(module, "process_arguments",
                                             side_effect=lambda args: list(image_arguments(args))))
            if hasattr(module, "get_pointer_string"):
                stack.enter_context(patch.object(module, "get_pointer_string", side_effect=pointer_string))
            with self.assertRaises(_PointersChecked):
                call()
            lookup.assert_called_once_with(command)
            native_call.assert_called_once()

    def test_threshold_image(self):
        self.assert_pointers("ants.ops.threshold_image", "ThresholdImage",
                             ["float", "float"], lambda: ants.threshold_image(self.image))

    def test_label_clusters(self):
        # Retain threshold_image's real casts, but skip its executable as well.
        with patch("ants.ops.threshold_image.get_lib_fn", return_value=Mock(return_value=0)):
            self.assert_pointers("ants.label.label_clusters", "LabelClustersUniquely",
                                 ["float", "float"], lambda: ants.label_clusters(self.image))

    def test_n3_bias_field_correction(self):
        self.assert_pointers("ants.ops.bias_correction", "N3BiasFieldCorrection",
                             ["float", "float"], lambda: ants.n3_bias_field_correction(self.image))

    def test_n3_bias_field_correction2(self):
        for supplied_mask in (False, True):
            with self.subTest(supplied_mask=supplied_mask):
                self.assert_pointers(
                    "ants.ops.bias_correction", "N3BiasFieldCorrection", ["float"] * 4,
                    lambda: ants.n3_bias_field_correction2(
                        self.image, mask=self.image if supplied_mask else None))

    def test_n3_bias_field_correction2_weight_mask(self):
        self.assert_pointers(
            "ants.ops.bias_correction", "N3BiasFieldCorrection", ["float"] * 5,
            lambda: ants.n3_bias_field_correction2(
                self.image, mask=self.image, weight_mask=self.image))

    def test_n4_bias_field_correction(self):
        for supplied_mask in (False, True):
            with self.subTest(supplied_mask=supplied_mask):
                self.assert_pointers(
                    "ants.ops.bias_correction", "N4BiasFieldCorrection", ["float"] * 4,
                    lambda: ants.n4_bias_field_correction(
                        self.image, mask=self.image if supplied_mask else None))

    def test_n4_bias_field_correction_weight_mask(self):
        self.assert_pointers(
            "ants.ops.bias_correction", "N4BiasFieldCorrection", ["float"] * 5,
            lambda: ants.n4_bias_field_correction(
                self.image, mask=self.image, weight_mask=self.image))

    def test_denoise_image(self):
        for supplied_mask in (False, True):
            with self.subTest(supplied_mask=supplied_mask):
                expected = ["float", "unsigned char", "float"] if supplied_mask else ["float"] * 2
                self.assert_pointers(
                    "ants.ops.denoise_image", "DenoiseImage", expected,
                    lambda: ants.denoise_image(self.image, mask=self.image if supplied_mask else None))

    def test_atropos(self):
        for channels in (self.image, [self.image], [self.image, self.image], (self.image, self.image)):
            count = len(channels) if isinstance(channels, (list, tuple)) else 1
            with self.subTest(channels=type(channels).__name__, count=count):
                if isinstance(channels, (list, tuple)) and len(channels) > 1:
                    expected = ["unsigned int", "unsigned int"] + ["float"] * len(channels)
                else:
                    expected = ["float", "unsigned int", "unsigned int"]
                self.assert_pointers("ants.segmentation.atropos", "Atropos", expected,
                                     lambda: ants.atropos(a=channels, x=self.float_image))

    def test_kelly_kapowski(self):
        self.assert_pointers(
            "ants.segmentation.kelly_kapowski", "KellyKapowski",
            ["unsigned int"] + ["float"] * 4,
            lambda: ants.kelly_kapowski(s=self.float_image, g=self.image, w=self.image, t=self.image))

    def test_joint_label_fusion(self):
        for with_labels in (False, True):
            with self.subTest(with_labels=with_labels):
                expected = (["float", "unsigned int", "unsigned int", "float",
                             "float", "unsigned int", "float", "unsigned int"]
                            if with_labels else ["float", "unsigned int", "float", "float", "float"])
                self.assert_pointers(
                    "ants.segmentation.joint_label_fusion", "antsJointFusion", expected,
                    lambda: ants.joint_label_fusion(
                        self.image, self.float_image, [self.image, self.image],
                        label_list=[self.float_image, self.float_image] if with_labels else None))

    def test_affine_initializer(self):
        self.assert_pointers(
            "ants.registration.affine_initializer", "antsAffineInitializer",
            ["float", "float", "unsigned char"],
            lambda: ants.affine_initializer(self.image, self.image, mask=self.image,
                                            txfn=self.output_prefix + ".mat"))

    def test_label_geometry_measures(self):
        for supplied_intensity in (False, True):
            with self.subTest(supplied_intensity=supplied_intensity):
                self.assert_pointers(
                    "ants.label.label_geometry_measures", "LabelGeometryMeasures",
                    ["unsigned int", "float"],
                    lambda: ants.label_geometry_measures(
                        self.float_image if supplied_intensity else self.image,
                        intensity_image=self.image if supplied_intensity else None))

    def test_imath(self):
        self.assert_pointers("ants.ops.iMath", "iMath", ["float", "float"],
                             lambda: ants.iMath(self.image, "Normalize"))

    def test_imath_propagate_labels_through_mask(self):
        for helper in (False, True):
            with self.subTest(helper=helper):
                self.assert_pointers(
                    "ants.ops.iMath", "iMath", ["float"] * 3,
                    lambda: (ants.iMath_propagate_labels_through_mask(self.image, self.image)
                             if helper else ants.iMath(self.image, "PropagateLabelsThroughMask", self.image)))

    def test_imath_truncate_intensity(self):
        self.assert_pointers(
            "ants.ops.iMath", "iMath", ["float", "float", "unsigned int"],
            lambda: ants.iMath(self.image, "TruncateIntensity", 0.05, 0.95, 64, self.float_image))

    def test_label_image_registration(self):
        # Geometry is tested above; avoid CSV output and linear transform fitting.
        geometry = pd.DataFrame({"Label": [1]})
        with patch.object(ants, "label_geometry_measures", return_value=geometry), \
             patch("ants.ops.threshold_image.get_lib_fn", return_value=Mock(return_value=0)):
            for intensities in (self.image, [self.image, self.image]):
                with self.subTest(multichannel=isinstance(intensities, list)):
                    count = len(intensities) if isinstance(intensities, list) else 1
                    self.assert_pointers(
                        "ants.registration.registration", "antsRegistration",
                        ["float"] * (2 * count + 2) + ["unsigned char"] * 2,
                        lambda: ants.label_image_registration(
                            self.image, self.image,
                            fixed_intensity_images=intensities, moving_intensity_images=intensities,
                            fixed_mask=self.image, moving_mask=self.image,
                            initial_transforms="identity", output_prefix=self.output_prefix))


if __name__ == "__main__":
    unittest.main()
