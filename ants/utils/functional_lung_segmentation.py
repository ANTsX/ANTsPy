
__all__ = ['functional_lung_segmentation']

from .iMath import iMath
from .. import core
from .. import utils
from .. import segmentation

def functional_lung_segmentation(image,
                                 mask=None,
                                 number_of_iterations=1,
                                 number_of_atropos_iterations=0,
                                 mrf_parameters="[0.3,2x2x2]",
                                 verbose=True):

    """
    Ventilation-based segmentation of hyperpolarized gas lung MRI.

    Lung segmentation into four classes based on ventilation as described in
    this paper:

         https://pubmed.ncbi.nlm.nih.gov/21837781/

    Arguments
    ---------
    image : ANTs image
        Input proton-weighted MRI.

    mask : ANTs image
        Mask image designating the region to segment.  0/1 = background/foreground.

    number_of_iterations : integer
        Number of Atropos <--> N4 iterations (outer loop).

    number_of_atropos_iterations : integer
        Number of Atropos iterations (inner loop).  If number_of_atropos_iterations = 0,
        this is equivalent to K-means with no MRF priors.

    mrf_parameters : string
        Parameters for MRF in Atropos.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary with segmentation image, probability images, and
    processed image.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read("lung_image.nii.gz")
    >>> mask = ants.image_read("lung_mask.nii.gz")
    >>> seg = functional_lung_segmentation(image, mask, verbose=True)
    """

    if image.dimension != 3:
        raise ValueError("Function only works for 3-D images.")

    if mask is None:
        raise ValueError("Mask is missing.")

    if number_of_iterations < 1:
        raise ValueError("number_of_iterations must be >= 1.")

    def generate_pure_tissue_n4_weight_mask(probability_images):
        number_of_probability_images = len(probability_images)

        pure_tissue_mask = probability_images[0] * 0
        for i in range(number_of_probability_images):
            negation_image = probability_images[0] * 0 + 1
            for j in range(number_of_probability_images):
                if i != j:
                    negation_image = negation_image * (probability_images[j] * -1.0 + 1.0)
                pure_tissue_mask = pure_tissue_mask + negation_image * probability_images[i]
        return(pure_tissue_mask)

    dilated_mask = iMath(mask, 'MD', 5)
    weight_mask = None

    number_of_atropos_n4_iterations = number_of_iterations
    for i in range(number_of_atropos_n4_iterations):
        if verbose == True:
            print("Atropos/N4 iteration: ", i, " out of ", number_of_atropos_n4_iterations)

        preprocessed_image = core.image_clone(image)

        quantiles = (preprocessed_image.quantile(0.0), preprocessed_image.quantile(0.995))
        preprocessed_image[preprocessed_image < quantiles[0]] = quantiles[0]
        preprocessed_image[preprocessed_image > quantiles[1]] = quantiles[1]

        if verbose == True:
            print("Atropos/N4: initial N4 bias correction.")
        preprocessed_image = utils.n4_bias_field_correction(preprocessed_image,
            mask=dilated_mask, shrink_factor=2, convergence={'iters': [50, 50, 50, 50], 'tol': 0.0000000001},
            spline_param=200, return_bias_field=False, weight_mask=weight_mask, verbose=verbose)
        preprocessed_image = (preprocessed_image - preprocessed_image.min())/(preprocessed_image.max() - preprocessed_image.min())
        preprocessed_image *= 1000

        if verbose == True:
            print("Atropos/N4: Atropos segmentation.")
        atropos_initialization = "kmeans[4]"
        posterior_formulation = "Socrates[0]"
        if i > 0:
            atropos_initialization = atropos_output['probabilityimages']
            posterior_formulation = "Socrates[1]"

        iterations = "[" + str(number_of_atropos_iterations) + ",0]"
        atropos_verbose = 0
        if verbose == True:
            atropos_verbose = 1
        atropos_output = segmentation.atropos(preprocessed_image, x=dilated_mask, i=atropos_initialization,
            m=mrf_parameters, c=iterations, priorweight=0.0, v=atropos_verbose, p=posterior_formulation)

        weight_mask = generate_pure_tissue_n4_weight_mask(atropos_output['probabilityimages'][1:4])

    masked_segmentation_image = atropos_output['segmentation'] * mask
    masked_probability_images = list()
    for i in range(len(atropos_output['probabilityimages'])):
        masked_probability_images.append(atropos_output['probabilityimages'][i] * mask)

    return_dict = {'segmentation_image' : masked_segmentation_image,
                   'probability_images' : masked_probability_images,
                   'processed_image' : preprocessed_image}
    return(return_dict)
