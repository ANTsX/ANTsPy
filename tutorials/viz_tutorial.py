"""
This tutorial demonstrates some of the visualization
functionality of ANTsPy
"""

import os
import numpy as np

import ants


## 2D
img = ants.image_read(ants.get_data('r16'))
