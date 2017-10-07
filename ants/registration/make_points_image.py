

def make_points_image(pts, mask, radius=5):
    """
    Create label image from physical space points
    
    Creates spherical points in the coordinate space of the target image based
    on the n-dimensional matrix of points that the user supplies. The image
    defines the dimensionality of the data so if the input image is 3D then
    the input points should be 2D or 3D.
    
    ANTsR function: `makePointsImage`

    Arguments
    ---------
    pts : numpy.ndarray
        input powers points
    mask : ANTsImage
        mask defining target space
    radius : integer
        radius for the points
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni')).get_mask()
    >>> powers_pts = ants.get_data('powers_areal_mni_itk')
    >>> powers_labels= ants.make_points_image(powers_pts[:,:2], mni, radius=3)

    makePointsImage <- function( pts, mask, radius = 5 )
    {
      powersLabels = mask * 0
      nPts = dim(pts)[1]
      rad  = radius
      n = ceiling( rad / antsGetSpacing( mask ) )
      dim = mask@dimension
      if ( ncol( pts ) < dim )
        stop( "points dimensionality should match that of images" )
      for ( r in 1:nPts) {
        pt = as.numeric(c(pts[r,1:dim]))
        idx = antsTransformPhysicalPointToIndex(mask,pt)
        for ( i in seq(-n[1],n[1],by=0.5) ) {
          for (j in seq(-n[2],n[2],by=0.5) )  {
            if ( dim == 3 )
              {
              for (k in seq(-n[3],n[3],by=0.5)) {
                local = idx + c(i,j,k)
                localpt = antsTransformIndexToPhysicalPoint(mask,local)
                dist = sqrt( sum( (localpt-pt)*(localpt-pt) ))
                inImage = ( prod(idx <= dim(mask))==1) && ( length(which(idx<1)) == 0 )
                if ( (dist <= rad) && ( inImage == TRUE ) ) {
                  if ( powersLabels[ local[1], local[2], local[3] ] < 0.5 )
                    powersLabels[ local[1], local[2], local[3] ] = r
                 }
                }
              } # if dim == 3
            if ( dim == 2 )
              {
              local = idx + c(i,j)
              localpt = antsTransformIndexToPhysicalPoint(mask,local)
              dist = sqrt( sum( (localpt-pt)*(localpt-pt) ))
              inImage = ( prod(idx <= dim(mask))==1) && ( length(which(idx<1)) == 0 )
              if ( (dist <= rad) && ( inImage == TRUE ) ) {
                  if ( powersLabels[ local[1], local[2] ] < 0.5 )
                    powersLabels[ local[1], local[2] ] = r
                  }
              } # if dim == 2
            }
          }
        }
      return( powersLabels )
    }
"""