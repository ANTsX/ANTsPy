

__all__ = ['RigidRegistration']

from ...registration import interface, apply_transforms


class RigidRegistration(object):
    """
    Rigid Registration as a Scikit-Learn compatible transform class

    Example
    -------
    >>> import ants
    >>> import ants.extra as extrants
    >>> fi = ants.image_read(ants.get_data('r16'))
    >>> mi = ants.image_read(ants.get_data('r64'))
    >>> regtx = extrants.RigidRegistration()
    >>> regtx.fit(fi, mi)
    >>> mi_r = regtx.transform(mi)
    >>> ants.plot(fi, mi_r.iMath_Canny(1, 2, 4).iMath('MD',1))
    """
    def __init__(self, fixed_image=None):
        self.type_of_transform = 'Rigid'
        self.fixed_image = fixed_image

    def fit(self, moving_image, fixed_image=None):
        if fixed_image is None:
            if self.fixed_image is None:
                raise ValueError('must give fixed_image in fit() or set it in __init__')
            fixed_image = self.fixed_image

        fit_result = interface.registration(fixed_image,
                                            moving_image,
                                            type_of_transform=self.type_of_transform,
                                            initial_transform=None,
                                            outprefix='',
                                            mask=None,
                                            grad_step=0.2,
                                            flow_sigma=3,
                                            total_sigma=0,
                                            aff_metric='mattes',
                                            aff_sampling=32,
                                            syn_metric='mattes',
                                            syn_sampling=32,
                                            reg_iterations=(40,20,0),
                                            verbose=False)
        self._fit_result = fit_result
        self.fwdtransforms_ = fit_result['fwdtransforms']
        self.invtransforms_ = fit_result['invtransforms']
        self.warpedmovout_ = fit_result['warpedmovout']
        self.warpedfiout_ = fit_result['warpedfixout']

    def transform(self, moving_image, fixed_image=None):
        result = apply_transforms(fixed=fixed_image, moving=moving_image,
                                  transformlist=self.fwdtransforms)
        return result

    def inverse_transform(self, moving_image, fixed_image=None):
        result = apply_transforms(fixed=fixed_image, moving=moving_image,
                                  transformlist=self.invtransforms)
        return result


