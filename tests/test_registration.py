"""
Test ants.registration module

nptest.assert_allclose
self.assertEqual
self.assertTrue
"""

import os
import unittest
from common import run_tests
from numpy.linalg import eigh

import math
import numpy as np
import numpy.testing as nptest
import pandas as pd
import tempfile

import ants


class TestModule_affine_initializer(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        # test ANTsPy/ANTsR example
        fi = ants.image_read(ants.get_ants_data("r16"))
        mi = ants.image_read(ants.get_ants_data("r27"))
        txfile = ants.affine_initializer(fi, mi)
        tx = ants.read_transform(txfile)


class TestModule_apply_transforms(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        # test ANTsPy/ANTsR example
        fixed = ants.image_read(ants.get_ants_data("r16"))
        moving = ants.image_read(ants.get_ants_data("r64"))
        fixed = ants.resample_image(fixed, (64, 64), 1, 0)
        moving = ants.resample_image(moving, (128, 128), 1, 0)
        mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")
        mywarpedimage = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=mytx["fwdtransforms"]
        )
        self.assertEqual(mywarpedimage.pixeltype, moving.pixeltype)
        self.assertTrue(ants.image_physical_space_consistency(fixed, mywarpedimage,
                                                              0.0001, datatype = False))

        # Call with float precision for transforms, but should still return input type
        mywarpedimage2 = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=mytx["fwdtransforms"], singleprecision=True
        )
        self.assertEqual(mywarpedimage2.pixeltype, moving.pixeltype)
        self.assertLessEqual(np.sum((mywarpedimage.numpy() - mywarpedimage2.numpy()) ** 2), 0.1)

        # bad interpolator
        with self.assertRaises(Exception):
            mywarpedimage = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=mytx["fwdtransforms"],
                interpolator="unsupported-interp",
            )

        # transform doesnt exist
        with self.assertRaises(Exception):
            mywarpedimage = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=["blah-blah.mat"],
                interpolator="unsupported-interp",
            )


class TestModule_create_jacobian_determinant_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        mi = ants.image_read(ants.get_ants_data("r64"))
        fi = ants.resample_image(fi, (128, 128), 1, 0)
        mi = ants.resample_image(mi, (128, 128), 1, 0)
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform=("SyN"))
        try:
            jac = ants.create_jacobian_determinant_image(
                fi, mytx["fwdtransforms"][0], 1
            )
        except:
            pass


class TestModule_create_warped_grid(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        mi = ants.image_read(ants.get_ants_data("r64"))
        mygr = ants.create_warped_grid(mi)

        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform=("SyN"))
        mywarpedgrid = ants.create_warped_grid(
            mi,
            grid_directions=(False, True),
            transform=mytx["fwdtransforms"],
            fixed_reference_image=fi,
        )


class TestModule_fsl2antstransform(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fslmat = np.zeros((4, 4))
        np.fill_diagonal(fslmat, 1)
        img = ants.image_read(ants.get_ants_data("ch2"))
        tx = ants.fsl2antstransform(fslmat, img, img)


class TestModule_interface(unittest.TestCase):
    def setUp(self):
        self.transform_types = {
            "SyNBold",
            "SyNBoldAff",
            "ElasticSyN",
            "SyN",
            "SyNRA",
            "SyNOnly",
            "SyNAggro",
            "SyNCC",
            "TRSAA",
            "SyNabp",
            "SyNLessAggro",
            "TVMSQ",
            "TVMSQC",
            "Rigid",
            "Similarity",
            "Translation",
            "Affine",
            "AffineFast",
            "BOLDAffine",
            "QuickRigid",
            "DenseRigid",
            "BOLDRigid",
            "antsRegistrationSyNQuick[b,32,26]",
            "antsRegistrationSyNQuick[s]",
            "antsRegistrationSyNRepro[s]",
            "antsRegistrationSyN[s]"
        }

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        mi = ants.image_read(ants.get_ants_data("r64"))
        fi = ants.resample_image(fi, (60, 60), 1, 0)
        mi = ants.resample_image(mi, (60, 60), 1, 0)
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform="SyN")

    def test_affine_interface(self):
        print("Starting affine interface registration test")
        fi = ants.image_read(ants.get_ants_data("r16"))
        mi = ants.image_read(ants.get_ants_data("r64"))
        with self.assertRaises(ValueError):
            ants.registration(
                fixed=fi,
                moving=mi,
                type_of_transform="Translation",
                aff_iterations=4,
                aff_shrink_factors=4,
                aff_smoothing_sigmas=(4, 4),
            )

        mytx = ants.registration(
            fixed=fi,
            moving=mi,
            type_of_transform="Affine",
            aff_iterations=(4, 4),
            aff_shrink_factors=(4, 4),
            aff_smoothing_sigmas=(4, 4),
        )
        mytx = ants.registration(
            fixed=fi,
            moving=mi,
            type_of_transform="Translation",
            aff_iterations=4,
            aff_shrink_factors=4,
            aff_smoothing_sigmas=4,
        )

    def test_registration_types(self):
        print("Starting long registration interface test")
        fi = ants.image_read(ants.get_ants_data("r16"))
        mi = ants.image_read(ants.get_ants_data("r64"))
        fi = ants.resample_image(fi, (60, 60), 1, 0)
        mi = ants.resample_image(mi, (60, 60), 1, 0)

        for ttype in self.transform_types:
            print(ttype)
            mytx = ants.registration(fixed=fi, moving=mi, type_of_transform=ttype)

            # with mask
            fimask = fi > fi.mean()
            mytx = ants.registration(
                fixed=fi, moving=mi, mask=fimask, type_of_transform=ttype
            )
        print("Finished long registration interface test")

    def test_reg_precision_option(self):
        # Check that registration works with float and double precision
        fi = ants.image_read(ants.get_ants_data("r16"))
        mi = ants.image_read(ants.get_ants_data("r64"))
        fi = ants.resample_image(fi, (60, 60), 1, 0)
        mi = ants.resample_image(mi, (60, 60), 1, 0)
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform="SyN") # should be float precision
        info = ants.image_header_info(mytx["fwdtransforms"][0])
        self.assertEqual(info['pixeltype'], 'float')
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform="SyN", singleprecision=False)
        info = ants.image_header_info(mytx["fwdtransforms"][0])
        self.assertEqual(info['pixeltype'], 'double')


class TestModule_metrics(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data("r16")).clone("float")
        mi = ants.image_read(ants.get_ants_data("r64")).clone("float")
        mival = ants.image_mutual_information(fi, mi)  # -0.1796141


class TestModule_reflect_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        axis = 2
        asym = ants.reflect_image(fi, axis, "Affine")["warpedmovout"]
        asym = asym - fi


class TestModule_reorient_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_reorient_image(self):
        mni = ants.image_read(ants.get_data('mni'))
        mni2 = mni.reorient_image2()

    def test_get_center_of_mass(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        com = ants.get_center_of_mass(fi)

        self.assertEqual(len(com), fi.dimension)

        fi = ants.image_read(ants.get_ants_data("r64"))
        com = ants.get_center_of_mass(fi)
        self.assertEqual(len(com), fi.dimension)

        fi = fi.clone("unsigned int")
        com = ants.get_center_of_mass(fi)
        self.assertEqual(len(com), fi.dimension)

        # 3d
        img = ants.image_read(ants.get_ants_data("mni"))
        com = ants.get_center_of_mass(img)
        self.assertEqual(len(com), img.dimension)


class TestModule_resample_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_resample_image_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        finn = ants.resample_image(fi, (50, 60), True, 0)
        filin = ants.resample_image(fi, (1.5, 1.5), False, 1)

    def test_resample_channels(self):
        img = ants.image_read( ants.get_ants_data("r16"))
        img = ants.merge_channels([img, img])
        outimg = ants.resample_image(img, (128,128), True)
        self.assertEqual(outimg.shape, (128, 128))
        self.assertEqual(outimg.components, 2)

    def test_resample_image_to_target_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        fi2mm = ants.resample_image(fi, (2, 2), use_voxels=0, interp_type=1)
        resampled = ants.resample_image_to_target(fi2mm, fi, verbose=True)
        self.assertTrue(ants.image_physical_space_consistency(fi, resampled, 0.0001, datatype=True))


class TestModule_symmetrize_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        simage = ants.symmetrize_image(image)


class TestModule_build_template(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image2 = ants.image_read(ants.get_ants_data("r27"))
        timage = ants.build_template(image_list=(image, image2))

    def test_type_of_transform(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image2 = ants.image_read(ants.get_ants_data("r27"))
        timage = ants.build_template(image_list=(image, image2))
        timage = ants.build_template(
            image_list=(image, image2), type_of_transform="SyNCC"
        )


class TestModule_multivar(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image2 = ants.image_read(ants.get_ants_data("r27"))
        demonsMetric = ["demons", image, image2, 1, 1]
        ccMetric = ["CC", image, image2, 2, 1]
        metrics = list()
        metrics.append(demonsMetric)
        reg3 = ants.registration(image, image2, "SyNOnly", multivariate_extras=metrics)
        metrics.append(ccMetric)
        reg2 = ants.registration(
            image, image2, "SyNOnly", multivariate_extras=metrics, verbose=True
        )

class TestModule_random(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_landmark_transforms(self):
        fixed = np.array([[50.0,50.0],[200.0,50.0],[200.0,200.0]])
        moving = np.array([[50.0,50.0],[50.0,200.0],[200.0,200.0]])
        xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="syn",
                                            domain_image=ants.image_read(ants.get_data('r16')),
                                            verbose=True)
        xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="tv",
                                            domain_image=ants.image_read(ants.get_data('r16')))
        xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="affine")
        xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="rigid")
        xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="similarity")
        domain_image = ants.image_read(ants.get_ants_data("r16"))
        xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="bspline", domain_image=domain_image, number_of_fitting_levels=5)
        xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="diffeo", domain_image=domain_image, number_of_fitting_levels=6)

        res = ants.fit_time_varying_transform_to_point_sets([fixed, moving, moving],
                                                            domain_image=ants.image_read(ants.get_data('r16')),
                                                            verbose=True)

    def test_deformation_gradient(self):
        fi = ants.image_read( ants.get_ants_data('r16'))
        mi = ants.image_read( ants.get_ants_data('r64'))
        fi = ants.resample_image(fi,(128,128),1,0)
        mi = ants.resample_image(mi,(128,128),1,0)
        mytx = ants.registration(fixed=fi , moving=mi, type_of_transform = ('SyN') )

        dg = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ) )
        # Expect some differences between these two methods
        dg_py = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ),
                                       py_based=True)

        rot = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ),
                                       to_rotation=True)
        rot_py = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ),
                                       to_rotation=True, py_based=True)

        def rotation_angle_diff_field(rot_1, rot_2, degrees=False):
            """
            Compute angular difference between two rotation matrix images.
            Inputs:
                rot_1: antsImage with dim*dim components representing the matrix
                rot_2: antsImage with same shape and dimension as rot_1
            Returns:
                angle_diff: numpy array with angle in radians (or degrees if degrees=True)
            """
            # cast and reshape
            dim = rot_1.dimension
            if rot_2.dimension != dim:
                raise ValueError("Rotation images must have the same dimension.")
            if rot_1.shape != rot_2.shape:
                raise ValueError("Rotation images must have the same shape.")
            rot_py1 = rot_1.numpy().reshape(rot_1.shape + (dim, dim))
            rot_py2 = rot_2.numpy().reshape(rot_2.shape + (dim, dim))
            # Compute relative rotation: R_rel = R_py @ R.T
            R_rel = rot_py1 @ np.swapaxes(rot_py2, -2, -1)
            trace = np.trace(R_rel, axis1=-2, axis2=-1)
            if dim == 2:
                angle = np.arccos(np.clip((trace) / 2, -1.0, 1.0))
            elif dim == 3:
                angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
            else:
                raise ValueError("Only 2D or 3D rotation matrices are supported.")

            return np.degrees(angle) if degrees else angle

        rot_angle_diff = rotation_angle_diff_field(rot_py, rot, degrees=True)
        self.assertTrue(np.all(rot_angle_diff < 1))

        rot_inv = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ),
                                       to_inverse_rotation=True)
        rot_py_inv = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ),
                                       to_inverse_rotation=True, py_based=True)

        rot_angle_diff = rotation_angle_diff_field(rot_inv, rot_py_inv, degrees=True)
        self.assertTrue(np.all(rot_angle_diff < 1))

        # Check it's actually the inverse
        self.assertTrue(np.allclose (rot_py.numpy(), rot_py_inv.numpy()[..., [0, 2, 1, 3]]))

    def test_jacobian(self):
        fi = ants.image_read( ants.get_ants_data('r16'))
        mi = ants.image_read( ants.get_ants_data('r64'))
        fi = ants.resample_image(fi,(128,128),1,0)
        mi = ants.resample_image(mi,(128,128),1,0)
        mytx = ants.registration(fixed=fi , moving=mi, type_of_transform = ('SyN') )
        jac = ants.create_jacobian_determinant_image(fi,mytx['fwdtransforms'][0],1)

    def test_apply_transforms(self):
        fixed = ants.image_read( ants.get_ants_data('r16') )
        moving = ants.image_read( ants.get_ants_data('r64') )
        fixed = ants.resample_image(fixed, (64,64), 1, 0)
        moving = ants.resample_image(moving, (64,64), 1, 0)
        mytx = ants.registration(fixed=fixed , moving=moving ,
                                type_of_transform = 'SyN' )
        mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving,
                                            transformlist=mytx['fwdtransforms'] )

    def test_apply_transforms_to_points(self):
        fixed = ants.image_read( ants.get_ants_data('r16') )
        moving = ants.image_read( ants.get_ants_data('r27') )
        reg = ants.registration( fixed, moving, 'Affine' )
        d = {'x': [128, 127], 'y': [101, 111]}
        pts = pd.DataFrame(data=d)
        ptsw = ants.apply_transforms_to_points( 2, pts, reg['fwdtransforms'])

    def test_warped_grid(self):
        fi = ants.image_read( ants.get_ants_data( 'r16' ) )
        mi = ants.image_read( ants.get_ants_data( 'r64' ) )
        mygr = ants.create_warped_grid( mi )
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
        mywarpedgrid = ants.create_warped_grid( mi, grid_directions=(False,True),
                            transform=mytx['fwdtransforms'], fixed_reference_image=fi )

    def test_more_registration(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        mi = ants.image_read(ants.get_ants_data('r64'))
        fi = ants.resample_image(fi, (60,60), 1, 0)
        mi = ants.resample_image(mi, (60,60), 1, 0)
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN' )
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'antsRegistrationSyN[t]' )
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'antsRegistrationSyN[b]' )
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'antsRegistrationSyN[s]' )

    def test_motion_correction(self):
        fi = ants.image_read(ants.get_ants_data('ch2'))
        mytx = ants.motion_correction( fi )

    def test_label_image_registration(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        mi = ants.image_read(ants.get_ants_data('r64'))
        fi = ants.resample_image(fi, (60,60), 1, 0)
        mi = ants.resample_image(mi, (60,60), 1, 0)
        fi_seg = ants.threshold_image(fi, "Kmeans", 3)-1
        mi_seg = ants.threshold_image(mi, "Kmeans", 3)-1
        mytx = ants.label_image_registration([fi_seg],
                                             [mi_seg],
                                             fixed_intensity_images=fi,
                                             moving_intensity_images=mi)


    def test_tensor_reorient(self):
        fa = 0.75
        trace = 2.1E-3
        l_1 = trace * (1 + (2 * fa) / np.sqrt(3 - 2 * fa**2))
        l_2 = (trace - l_1) / 2

        tensor = np.zeros((3, 3))
        tensor[0, 0] = l_1
        tensor[1, 1] = l_2
        tensor[2, 2] = l_2

        img_shape = (10,10,10)

        # Fill image with this tensor (SymmetricSecondRankTensor has 6 unique values)
        image_data = np.zeros(img_shape + (6,), dtype=np.float32)
        image_data[..., 0] = tensor[0, 0]  # xx
        image_data[..., 1] = tensor[0, 1]  # xy
        image_data[..., 2] = tensor[0, 2]  # xz
        image_data[..., 3] = tensor[1, 1]  # yy
        image_data[..., 4] = tensor[1, 2]  # yz
        image_data[..., 5] = tensor[2, 2]  # zz

        tensor_image = ants.from_numpy(image_data, has_components=True)
        tensor_image.set_direction(np.eye(3))
        tensor_image.set_spacing((1.0, 1.0, 1.0))
        tensor_image.set_origin((0.0, 0.0, 0.0))

        ref_image = ants.from_numpy(image_data[..., 0], has_components=False)
        ants.copy_image_info(tensor_image, ref_image)

        theta = math.radians(20)
        tx_params = np.array([math.cos(theta), -math.sin(theta), 0, math.sin(theta), math.cos(theta), 0, 0, 0, 1, 0, 0, 0])

        affine_tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3, parameters=tx_params)

        fd, tx_fn = tempfile.mkstemp(suffix=".mat")
        os.close(fd)

        ants.write_transform(affine_tx, tx_fn)

        reoriented_tensor = ants.apply_transforms(ref_image, tensor_image, tx_fn, imagetype=2, verbose=True).numpy()

        os.remove(tx_fn)

        mid = reoriented_tensor.shape[0] // 2

        dtUpper = reoriented_tensor[mid, mid, mid]

        # Convert upper-triangular to full tensor
        T = np.array([
            [dtUpper[0], dtUpper[1], dtUpper[2]],
            [dtUpper[1], dtUpper[3], dtUpper[4]],
            [dtUpper[2], dtUpper[4], dtUpper[5]],
        ])

        # Check tensor has been reoriented by theta
        # Compute eigenvectors and eigenvalues
        eigvals, eigvecs = eigh(T)
        principal_ev = eigvecs[:, np.argmax(eigvals)]

        # The expected reorientation is the opposite to the forward transform
        expected_ev = np.array([math.cos(-theta), math.sin(-theta), 0])

        dot = np.abs(np.dot(principal_ev, expected_ev))
        assert math.isclose(dot, 1.0, abs_tol=1e-3), f"Principal eigenvector {principal_ev}, expected {expected_ev}"


if __name__ == "__main__":
    run_tests()
