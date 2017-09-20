
```R
library(ANTsR)
img <- antsImageRead( getANTsRData("r16") , 2 )
img <- resampleImage( img, c(64,64), 1, 0 )
mask <- getMask(img)
segs1 <- atropos( a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
```

```python     
import ants
img = ants.image_read(ants.get_ants_data('r16'))
img = ants.resample_image(img, (64,64), 1, 0)
mask = ants.get_mask(img)
ants.atropos(a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
```