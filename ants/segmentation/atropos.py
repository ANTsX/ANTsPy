"""
Atropos segmentation
"""

 

__all__ = ['atropos']

import os
import glob
from tempfile import mktemp

from ..core import image_io
from .. import lib
from .. import utils


def atropos(a, x, i='Kmeans[3]', m='[0.2,1x1]', c='[5,0]', 
            priorweight=0.25, **kwargs):
    """
    Atropos segmentation method on ANTsImages

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> img = ants.resample_image(img, (64,64), 1, 0)
    >>> mask = ants.get_mask(img)
    >>> ants.atropos( a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )

    """
    probs = mktemp(prefix='antsr', suffix='prob%02d.nii.gz')
    tdir = probs.replace(os.path.basename(probs),'')
    probsbase = os.path.basename(probs)
    searchpattern = probsbase.replace('%02d', '*')

    ct = 0
    if isinstance(i, (list,tuple)) and (len(i) > 1):
        while ct <= len(i):
            probchar = str(ct)
            if ct < 10:
                probcar = '0%s' % probchar
            tempfn = probs.replace('%02d', 'probchar')
            ants.image_write(i[ct], tempfn)
            ct += 1
        i = 'PriorProbabilityImages[%s, %s, %s]' % (str(len(i)), probs, str(priorweight))

    if isinstance(a, list):
        outimg = a[0].clone('unsigned int')
    else:
        outimg = a.clone('unsigned int')

    mydim = outimg.dimension
    outs = '[%s,%s]' % (utils._ptrstr(outimg.pointer), probs)
    mymask = x.clone('unsigned int')

    if (not isinstance(a, (list,tuple))) or (len(a) == 1):
        myargs = {
            'd': mydim,
            'a': a,
            'm-MULTINAME-0': m,
            'o': outs,
            'c': c,
            'm-MULTINAME-1': m,
            'i': i,
            'x': mymask
        }
        #for k, v in kwargs.items():
        #    myargs[k] = v

    elif isinstance(a, (list, tuple)):
        if len(a) > 6:
            print('more than 6 input images not really supported, using first 6')
            a = a[:6]
        myargs = {
            'd': mydim,
            'm-MULTINAME-0': m,
            'o': outs,
            'c': c,
            'm-MULTINAME-1': m,
            'i': i,
            'x': mymask
        }
        for aa_idx, aa in enumerate(a):
            myargs['a-MULTINAME-%i'%aa_idx] = aa

    processed_args = utils._int_antsProcessArguments(myargs)
    lib.Atropos(processed_args)
    
    probsout = glob.glob(os.path.join(tdir,'*'+searchpattern))
    probimgs = [image_io.image_read(probsout[0])]
    for idx in range(1, len(probsout)):
        probimgs.append(image_io.image_read(probsout[idx]))

    outimg = outimg.clone('float')
    return {'segmentation': outimg,
            'probabilityimages': probimgs}

    
