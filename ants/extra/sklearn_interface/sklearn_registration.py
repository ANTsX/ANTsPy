

__all__ = ['RigidRegistration']

from ...registration import interface, apply_transforms


class Registration(object):
    """
    How would it work:

    # Co-registration within-visit
    reg = Registration('Rigid', fixed_image=t1_template, 
                       save_dir=save_dir, save_suffix='_coreg')
    for img in other_imgs:
        reg.fit(img)
    
    for img in [flair_img, t2_img]:
        reg = Registration('Rigid', fixed_image=t1_template,
                           save_dir=save_dir, save_suffix='_coreg')
        reg.fit(img)
    
    # Co-registration across-visit
    reg = Registration('Rigid', fixed_image=t1)
    reg.fit(moving=t1_followup)
    
    # now align all followups with first visit
    for img in [flair_follwup, t2_followup]:
        img_reg = reg.transform(img)
    
    # conversly, align all first visits with followups
    for img in [flair, t2]:
        img_reg = reg.inverse_transform(img)
    """

    def __init__(self, type_of_transform, fixed_image):
        """
        Properties:
            type_of_transform
            fixed_image (template)
            save_dir (where to save outputs)
            save_suffix (what to append to saved outputs)
            save_prefix (what to preppend to saved outputs)
        """
        self.type_of_transform = type_of_transform
        self.fixed_image = fixed_image

    def fit(self, X, y=None):
        """
        X : ANTsImage | string | list of ANTsImage types | list of strings
            images to register to fixed image

        y : string | list of strings
            labels for images
        """
        moving_images = X if isinstance(X, (list,tuple)) else [X]
        moving_labels = y if y is not None else [i for i in range(len(moving_images))]
        fixed_image = self.fixed_image

        self.fwdtransforms_ = {}
        self.invtransforms_ = {}
        self.warpedmovout_ = {}
        self.warpedfixout_ = {}
        
        for moving_image, moving_label in zip(moving_images, moving_labels):
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

            self.fwdtransforms_[moving_label] = fit_result['fwdtransforms']
            self.invtransforms_[moving_label] = fit_result['invtransforms']
            self.warpedmovout_[moving_label]  = fit_result['warpedmovout']
            self.warpedfixout_[moving_label]  = fit_result['warpedfixout']

        return self

    def transform(self, X, y=None):
        pass


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


