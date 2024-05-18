Usage Information 
 ImageDimension: 2 or 3 (for 2 or 3 dimensional operations).
 ImageDimension: 4 (for operations on 4D file, e.g. time-series data).
 Operator: See list of valid operators below.
 The last two arguments can be an image or float value 
 NB: Some options output text files

Mathematical Operations:
  m            : Multiply ---  use vm for vector multiply 
  +             : Add ---  use v+ for vector add 
  -             : Subtract ---  use v- for vector subtract 
  /             : Divide
  ^            : Power
  max            : voxelwise max
  exp            : Take exponent exp(imagevalue*value)
  addtozero        : add image-b to image-a only over points where image-a has zero values
  overadd        : replace image-a pixel with image-b pixel if image-b pixel is non-zero
  abs            : absolute value 
  total            : Sums up values in an image or in image1*image2 (img2 is the probability mask)
  mean            :  Average of values in an image or in image1*image2 (img2 is the probability mask)
  vtotal            : Sums up volumetrically weighted values in an image or in image1*image2 (img2 is the probability mask)
  Decision        : Computes result=1./(1.+exp(-1.0*( pix1-0.25)/pix2))
  Neg            : Produce image negative

Spatial Filtering:
  Project Image1.ext axis-a which-projection   : Project an image along axis a, which-projection=0(sum, 1=max, 2=min)
  G Image1.ext s    : Smooth with Gaussian of sigma = s
  MD Image1.ext s    : Morphological Dilation with radius s
  ME Image1.ext s    : Morphological Erosion with radius s
  MO Image1.ext s    : Morphological Opening with radius s
  MC Image1.ext s    : Morphological Closing with radius s
  GD Image1.ext s    : Grayscale Dilation with radius s
  GE Image1.ext s    : Grayscale Erosion with radius s
  GO Image1.ext s    : Grayscale Opening with radius s
  GC Image1.ext s    : Grayscale Closing with radius s
  BlobDetector Image1.ext NumberOfBlobsToExtract  Optional-Input-Image2 Blob-2-out.nii.gz N-Blobs-To-Match  :  blob detection by searching for local extrema of the Laplacian of the Gassian (LoG) 
    Example matching 6 best blobs from 2 images: 
    ImageMath 2 blob.nii.gz BlobDetector image1.nii.gz 1000  image2.nii.gz blob2.nii.gz 6 
  MatchBlobs Image1.ext Image1LM.ext Image2.ext


Transform Image: 
Translate InImage.ext x [ y z ] 

Time Series Operations:
 CompCorrAuto : Outputs a csv file containing global signal vector and N comp-corr eigenvectors determined from PCA of the high-variance voxels.  Also outputs a comp-corr + global signal corrected 4D image as well as a 3D image measuring the time series variance.  Requires a label image with label 1 identifying voxels in the brain.
   ImageMath 4 ${out}compcorr.nii.gz ThreeTissueConfounds ${out}.nii.gz  ${out}seg.nii.gz 1 3   : Outputs average global, CSF and WM signals.  Requires a label image with 3 labels , csf, gm , wm .
    Usage        : ThreeTissueConfounds 4D_TimeSeries.nii.gz LabeLimage.nii.gz  csf-label wm-label 
 TimeSeriesSubset : Outputs n 3D image sub-volumes extracted uniformly from the input time-series 4D image.
    Usage        : TimeSeriesSubset 4D_TimeSeries.nii.gz n 
 TimeSeriesDisassemble : Outputs n 3D image volumes for each time-point in time-series 4D image.
    Usage        : TimeSeriesDisassemble 4D_TimeSeries.nii.gz 

 TimeSeriesAssemble : Outputs a 4D time-series image from a list of 3D volumes.
    Usage        : TimeSeriesAssemble time_spacing time_origin *images.nii.gz 
 TimeSeriesToMatrix : Converts a 4D image + mask to matrix (stored as csv file) where rows are time and columns are space .
    Usage        : TimeSeriesToMatrix 4D_TimeSeries.nii.gz mask 
 TimeSeriesSimpleSubtraction : Outputs a 3D mean pair-wise difference list of 3D volumes.
    Usage        : TimeSeriesSimpleSubtraction image.nii.gz 
 TimeSeriesSurroundSubtraction : Outputs a 3D mean pair-wise difference list of 3D volumes.
    Usage        : TimeSeriesSurroundSubtraction image.nii.gz 
 TimeSeriesSincSubtraction : Outputs a 3D mean pair-wise difference list of 3D volumes.
    Usage        : TimeSeriesSincSubtraction image.nii.gz 
 SplitAlternatingTimeSeries : Outputs 2 3D time series
    Usage        : SplitAlternatingTimeSeries image.nii.gz 
 ComputeTimeSeriesLeverage : Outputs a csv file that identifies the raw leverage and normalized leverage for each time point in the 4D image.  leverage, here, is the difference of the time-point image from the average of the n images.  the normalized leverage is =  average( sum_k abs(Leverage(t)-Leverage(k)) )/Leverage(t). 
    Usage        : ComputeTimeSeriesLeverage 4D_TimeSeries.nii.gz k_neighbors 
 SliceTimingCorrection : Outputs a slice-timing corrected 4D time series
    Usage        : SliceTimingCorrection image.nii.gz sliceTiming [sinc / bspline] [sincRadius=4 / bsplineOrder=3]
 PASL : computes the PASL model of CBF  
f =  
     rac{      lambda DeltaM        } 
 {     2 alpha M_0 TI_1 exp( - TI_2 / T_{1a} )  } 
    Usage        : PASL 3D/4D_TimeSeries.nii.gz BoolFirstImageIsControl M0Image parameter_list.txt 
 pCASL : computes the pCASL model of CBF  
 f =  
      rac{      lambda DeltaM R_{1a}        }  
  {     2 alpha M_0 [ exp( - w R_{1a} ) - exp( -w (     au + w ) R_{1a}) ]     } 
    Usage        : pCASL 3D/4D_TimeSeries.nii.gz parameter_list.txt 
 PASLQuantifyCBF : Outputs a 3D CBF image in ml/100g/min from a magnetization ratio image
    Usage        : PASLQuantifyCBF mag_raants.nii.gz [TI1=700] [TI2=1900] [T1blood=1664] [Lambda=0.9] [Alpha=0.95] [SliceDelay-45] 

Tensor Operations:
  4DTensorTo3DTensor    : Outputs a 3D_DT_Image with the same information. 
    Usage        : 4DTensorTo3DTensor 4D_DTImage.ext
  ComponentTo3DTensor    : Outputs a 3D_DT_Image with the same information as component images. 
    Usage        : ComponentTo3DTensor component_image_prefix[xx,xy,xz,yy,yz,zz] extension
  ExtractComponentFrom3DTensor    : Outputs a component images. 
    Usage        : ExtractComponentFrom3DTensor dtImage.ext which={xx,xy,xz,yy,yz,zz}
  ExtractVectorComponent: Produces the WhichVec component of the vector 
    Usage        : ExtractVectorComponent VecImage WhichVec
  TensorColor        : Produces RGB values identifying principal directions 
    Usage        : TensorColor DTImage.ext
  TensorFA        : 
    Usage        : TensorFA DTImage.ext
  TensorFADenominator    : 
    Usage        : TensorFADenominator DTImage.ext
  TensorFANumerator    : 
    Usage        : TensorFANumerator DTImage.ext
  TensorIOTest    : Will write the DT image back out ... tests I/O processes for consistency. 
    Usage        : TensorIOTest DTImage.ext
  TensorMeanDiffusion      : Mean of the eigenvalues
    Usage        : TensorMeanDiffusion DTImage.ext
  TensorRadialDiffusion    : Mean of the two smallest eigenvalues
    Usage        : TensorRadialDiffusion DTImage.ext
  TensorAxialDiffusion     : Largest eigenvalue, equivalent to TensorEigenvalue DTImage.ext 2
    Usage        : TensorAxialDiffusion DTImage.ext
  TensorEigenvalue         : Gets a single eigenvalue 0-2, where 0 = smallest, 2 = largest
    Usage        : TensorEigenvalue DTImage.ext WhichInd
  TensorToVector    : Produces vector field identifying one of the principal directions, 2 = largest eigenvalue
    Usage        : TensorToVector DTImage.ext WhichVec
  TensorToVectorComponent: 0 => 2 produces component of the principal vector field (largest eigenvalue). 3 = 8 => gets values from the tensor 
    Usage        : TensorToVectorComponent DTImage.ext WhichVec
  TensorMask     : Mask a tensor image, sets background tensors to zero or to isotropic tensors with specified mean diffusivity 
    Usage        : TensorMask DTImage.ext mask.ext [ backgroundMD = 0 ] 
  FuseNImagesIntoNDVectorField     : Create ND field from N input scalar images
    Usage        : FuseNImagesIntoNDVectorField imagex imagey imagez

Label Fusion:
  MajorityVoting : Select label with most votes from candidates
    Usage: MajorityVoting LabelImage1.nii.gz .. LabelImageN.nii.gz
  CorrelationVoting : Select label with local correlation weights
    Usage: CorrelationVoting Template.ext IntenistyImages* LabelImages* {Optional-Radius=5}
  STAPLE : Select label using STAPLE method
    Usage: STAPLE confidence-weighting LabelImages*
    Note:  Gives probabilistic output (float)
  MostLikely : Select label from from maximum probabilistic segmentations
    Usage: MostLikely probabilityThreshold ProbabilityImages*
  AverageLabels : Select label using STAPLE method
    Usage: AverageLabels LabelImages*
    Note:  Gives probabilistic output (float)

Image Metrics & Info:
  PearsonCorrelation: r-value from intesities of two images
    Usage: PearsonCorrelation image1.ext image2.ext {Optional-mask.ext}
  NeighborhoodCorrelation: local correlations
    Usage: NeighborhoodCorrelation image1.ext image2.ext {Optional-radius=5} {Optional-image-mask}
  NormalizedCorrelation: r-value from intesities of two images
    Usage: NormalizedCorrelation image1.ext image2.ext {Optional-image-mask}
  Demons: 
    Usage: Demons image1.ext image2.ext
  Mattes: mutual information
    Usage: Mattes image1.ext image2.ext {Optional-number-bins=32} {Optional-image-mask}

Unclassified Operators:
  ReflectionMatrix : Create a reflection matrix about an axis
 out.mat ReflectionMatrix image_in axis 

  MakeAffineTransform : Create an itk affine transform matrix 
  ClosestSimplifiedHeaderMatrix : does what it says ... image-in, image-out
  Byte            : Convert to Byte image in [0,255]

  CompareHeadersAndImages: Tries to find and fix header errors. Outputs a repaired image with new header. 
                Never use this if you trust your header information. 
      Usage        : CompareHeadersAndImages Image1 Image2

  ConvertImageSetToMatrix: Each row/column contains image content extracted from mask applied to images in *img.nii 
      Usage        : ConvertImageSetToMatrix rowcoloption Mask.nii *images.nii
 ConvertImageSetToMatrix output can be an image type or csv file type.

  RandomlySampleImageSetToCSV: N random samples are selected from each image in a list 
      Usage        : RandomlySampleImageSetToCSV N_samples *images.nii
 RandomlySampleImageSetToCSV outputs a csv file type.

  FrobeniusNormOfMatrixDifference: take the difference between two itk-transform matrices and then compute the frobenius norm
      Usage        : FrobeniusNormOfMatrixDifference mat1 mat2 

  ConvertImageSetToEigenvectors: Each row/column contains image content extracted from mask applied to images in *img.nii 
      Usage        : ConvertImageSetToEigenvectors N_Evecs Mask.nii *images.nii
 ConvertImageSetToEigenvectors output will be a csv file for each label value > 0 in the mask.

  ConvertImageToFile    : Writes voxel values to a file  
      Usage        : ConvertImageToFile imagevalues.nii {Optional-ImageMask.nii}

  ConvertLandmarkFile    : Converts landmark file between formats. See ANTS.pdf for description of formats.
      Usage        : ConvertLandmarkFile InFile.txt
      Example 1        : ImageMath 3  outfile.vtk  ConvertLandmarkFile  infile.txt

  ConvertToGaussian    : 
      Usage        : ConvertToGaussian  TValueImage  sigma-float

  ConvertVectorToImage    : The vector contains image content extracted from a mask. Here the vector is returned to its spatial origins as image content 
      Usage        : ConvertVectorToImage Mask.nii vector.nii

  CorrelationUpdate    : In voxels, compute update that makes Image2 more like Image1.
      Usage        : CorrelationUpdate Image1.ext Image2.ext RegionRadius

  CountVoxelDifference    : The where function from IDL 
      Usage        : CountVoxelDifference Image1 Image2 Mask

  CorruptImage        : 
      Usage        : CorruptImage Image NoiseLevel Smoothing

  D             : Danielson Distance Transform

  MaurerDistance : Maurer distance transform (much faster than Danielson)
      Usage        : MaurerDistance inputImage {foreground=1}

  DiceAndMinDistSum    : Outputs DiceAndMinDistSum and Dice Overlap to text log file + optional distance image
      Usage        : DiceAndMinDistSum LabelImage1.ext LabelImage2.ext OptionalDistImage

  EnumerateLabelInterfaces: 
      Usage        : EnumerateLabelInterfaces ImageIn ColoredImageOutname NeighborFractionToIgnore

  ClusterThresholdVariate        :  for sparse estimation 
      Usage        : ClusterThresholdVariate image mask  MinClusterSize

  ExtractSlice        : Extracts slice number from last dimension of volume (2,3,4) dimensions 
      Usage        : ExtractSlice volume.nii.gz slicetoextract

  FastMarchingSegmentation: final output is the propagated label image. Optional stopping value: higher values allow more distant propagation 
      Usage        : FastMarchingSegmentation speed/binaryimagemask.ext initiallabelimage.ext Optional-Stopping-Value

  FillHoles        : Parameter = ratio of edge at object to edge at background;  --  
                Parameter = 1 is a definite hole bounded by object only, 0.99 is close
                Default of parameter > 1 will fill all holes
      Usage        : FillHoles Image.ext parameter

  InPaint        : very simple inpainting --- assumes zero values should be inpainted  
      Usage        : InPaint #iterations

  PeronaMalik       : anisotropic diffusion w/varying conductance param (0.25 in example below)
      Usage        : PeronaMalik image #iterations conductance 

  Convolve       : convolve input image with kernel image
      Usage        : Convolve inputImage kernelImage {normalize=1} 
  Finite            : replace non-finite values with finite-value (default = 0)
      Usage        : Finite Image.exdt {replace-value=0}

  LabelSurfaceArea        : 
      Usage        : LabelSurfaceArea ImageIn {MaxRad-Default=1}

  FlattenImage        : Replaces values greater than %ofMax*Max to the value %ofMax*Max 
      Usage        : FlattenImage Image %ofMax

  GetLargestComponent    : Get the largest object in an image
      Usage        : GetLargestComponent InputImage {MinObjectSize}

  Grad            : Gradient magnitude with sigma s (if normalize, then output in range [0, 1])
      Usage        : Grad Image.ext s normalize?

  HistogramMatch    : 
      Usage        : HistogramMatch SourceImage ReferenceImage {NumberBins-Default=255} {NumberPoints-Default=64} {useThresholdAtMeanIntensity=false}

  RescaleImage    : 
      Usage        : RescaleImage InputImage min max

  WindowImage    : 
      Usage        : WindowImage InputImage windowMinimum windowMaximum outputMinimum outputMaximum

  NeighborhoodStats    : 
      Usage        : NeighborhoodStats inputImage whichStat radius             whichStat:  1 = min, 2 = max, 3 = variance, 4 = sigma, 5 = skewness, 6 = kurtosis, 7 = entropy

  InvId            : computes the inverse-consistency of two deformations and write the inverse consistency error image 
      Usage        : InvId VectorFieldName VectorFieldName

  ReplicateDisplacement            : replicate a ND displacement to a ND+1 image
      Usage        : ReplicateDisplacement VectorFieldName TimeDims TimeSpacing TimeOrigin

  ReplicateImage            : replicate a ND image to a ND+1 image
      Usage        : ReplicateImage ImageName TimeDims TimeSpacing TimeOrigin

  ShiftImageSlicesInTime            : shift image slices by one 
      Usage        : ShiftImageSlicesInTime ImageName shift-amount-default-1 shift-dim-default-last-dim

  LabelStats        : Compute volumes / masses of objects in a label image. Writes to text file
      Usage        : LabelStats labelimage.ext valueimage.nii

  Laplacian        : Laplacian computed with sigma s (if normalize, then output in range [0, 1])
      Usage        : Laplacian Image.ext s normalize?

  Canny        : Canny edge detector
      Usage        : Canny Image.ext sigma lowerThresh upperThresh

  Lipschitz        : Computes the Lipschitz norm of a vector field 
      Usage        : Lipschitz VectorFieldName

  MakeImage        : 
      Usage        : MakeImage SizeX  SizeY {SizeZ};

  MTR        : Computes the magnetization transfer ratio ( (M0-M1)/M0 ) and truncates values to [0,1]
      Usage        : MTR M0Image M1Image [MaskImage];

  Normalize        : Normalize to [0,1]. Option instead divides by average value.  If opt is a mask image, then we normalize by mean intensity in the mask ROI.
      Usage        : Normalize Image.ext opt

  PadImage       : If Pad-Number is negative, de-Padding occurs
      Usage        : PadImage ImageIn PaddingSize [PaddingVoxelValue=0]

  SigmoidImage   : 
      Usage        : SigmoidImage ImageIn [alpha=1.0] [beta=0.0]

  Sharpen   : 
      Usage        : Sharpen ImageIn

  CenterImage2inImage1        : 
      Usage       : ReferenceImageSpace ImageToCenter 

  PH            : Print Header

  PoissonDiffusion        : Solves Poisson's equation in a designated region using non-zero sources
      Usage        : PoissonDiffusion inputImage labelImage [sigma=1.0] [regionLabel=1] [numberOfIterations=500] [convergenceThreshold=1e-10]

  PropagateLabelsThroughMask: Final output is the propagated label image. Optional stopping value: higher values allow more distant propagation
      Usage        : PropagateLabelsThroughMask speed/binaryimagemask.nii.gz initiallabelimage.nii.gz Optional-Stopping-Value  0/1/2
      0/1/2  =>  0, no topology constraint, 1 - strict topology constraint, 2 - no handles 

  PValueImage        : 
      Usage        : PValueImage TValueImage dof

  RemoveLabelInterfaces: 
      Usage        : RemoveLabelInterfaces ImageIn

  ReplaceVoxelValue: replace voxels in the range [a,b] in the input image with c
      Usage        : ReplaceVoxelValue inputImage a b c

  ROIStatistics        : computes anatomical locations, cluster size and mass of a stat image which should be in the same physical space (but not nec same resolution) as the label image.
      Usage        : ROIStatistics LabelNames.txt labelimage.ext valueimage.nii

  SetOrGetPixel    : 
      Usage        : SetOrGetPixel ImageIn Get/Set-Value IndexX IndexY {IndexZ}
      Example 1        : ImageMath 2 outimage.nii SetOrGetPixel Image Get 24 34; Gets the value at 24, 34
      Example 2        : ImageMath 2 outimage.nii SetOrGetPixel Image 1.e9 24 34; This sets 1.e9 as the value at 23 34
                You can also pass a boolean at the end to force the physical space to be used

  SetTimeSpacing            : sets spacing for last dimension
      Usage        : SetTimeSpacing Image.ext tspacing

  SetTimeSpacingWarp            : sets spacing for last dimension
      Usage        : SetTimeSpacingWarp Warp.ext tspacing

  stack            : Will put 2 images in the same volume
      Usage        : Stack Image1.ext Image2.ext

  ThresholdAtMean    : See the code
      Usage        : ThresholdAtMean Image %ofMean

  TileImages    : 
      Usage        : TileImages NumColumns ImageList*

  TriPlanarView    : 
      Usage        : TriPlanarView  ImageIn.nii.gz PercentageToClampLowIntensity PercentageToClampHiIntensity x-slice y-slice z-slice

  TruncateImageIntensity: 
      Usage        : TruncateImageIntensity InputImage.ext {lowerQuantile=0.05} {upperQuantile=0.95} {numberOfBins=65} {binary-maskImage}

  Where            : The where function from IDL
      Usage        : Where Image ValueToLookFor maskImage-option tolerance