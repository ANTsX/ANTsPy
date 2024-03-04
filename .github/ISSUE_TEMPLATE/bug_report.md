---
name: Bug report
about: Report errors or unexpected behavior when installing or using ANTsPy

---

<!--
Text in these brackets are comments, and won't be visible when you submit your
issue. Please read before submitting.
--->

**Describe the bug**
<!--
In a sentence or two, describe the problem
--->

**To reproduce**
<!--
Steps that enable others to reproduce the problem. For example

import ants
mnipath = ants.get_ants_data('mni')
# This produces the error
result = ant.some_function('mnipath', option=value)

Please try to reproduce the problem with public data, or share example data.

Please try to reproduce the problem efficiently, eg if you have an error in a long
optimization process, does it still happen if you reduce the number of iterations?
--->


**Expected behavior**
<!-- A brief description of what you expected to happen. --->

**Screenshots**
<!-- If applicable, add screenshots to help explain your problem. --->

**ANTsPy installation (please complete the following information):**
<!-- Please add information in the square brackets below --->
<!-- PC, Mac, cloud, HPC cluster, other --->
 - Hardware [ ]
<!-- Mac OS, Windows, Ubuntu --->
 - OS: [ ]
 <!--
   To get system details on Linux, open a terminal and run

     uname -mrs

   On Mac, open a terminal and run

     uname -mrs
     sw_vers

   On Windows, open a cmd window and run

     systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
 -->
 - System details [  ]
 <!-- None if running directly, or virtual machine, WSL, Docker, Singularity  --->
 - Sub-system: [ ]
 <!-- Find your version with `pip show antspyx`. If you built from source, include the git hash or tag -->
 - ANTsPy version: [  ]
<!-- pip, downloaded wheel from (URL), built from source, other --->
 - Installation type: [  ]


**Additional context**
Add any other context about the problem here. Many issues are specific to particular data
so please include example data if possible.