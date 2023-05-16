# Pretty Plot

A utility for easily converting VNIR multispectral data from Mastcam-Z into a standardized, pretty plot for quick distribution of results. The tool accepts `marslab`-formatted files.

### Installation / setup
git clone git@github.com:MillionConcepts/pretty-plot.git

cd pretty-plot

conda env create -n pretty-plot --file environment.yml 

conda activate pretty-plot

### Tutorial
The Pretty Plot.ipynb file in this repo is a jupyter notebook tutorial for how to use pretty-plot. To access it make sure you are in your pretty-plot and your pretty-plot conda environment has been activated (see above) and then type:

jupyter notebook

Now, follow the instructions output on the command line to open a juptyer notebook session in your browser (crtl+click on the url) and click on Pretty Plot.ipynb.

---
The contents of this repo are provided by the Western Washington University Reflectance Lab (PI: M. Rice) and Million Concepts (C. Million, M. St. Clair, S.V. Brown) under a BSD 3-Clause License. You may do nearly anything that you want with this code. If you have any questions, leave us a Github Issue.

`MERTools`/`MERspect` is proprietary software (Arizona State University) for rover tactical operations made available on an as-needed basis.
