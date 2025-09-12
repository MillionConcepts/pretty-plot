# Pretty Plot

A utility for easily converting VNIR multispectral data from Mastcam-Z into a standardized, pretty plot for quick distribution of results. The tool accepts `marslab`-formatted files.

### Installation / setup
`git clone git@github.com:MillionConcepts/pretty-plot.git`

`cd pretty-plot`

`conda env create -n pretty-plot --file environment.yml`

`conda activate pretty-plot`

`pip install .`

### Tutorial
The Pretty Plot.ipynb file in this repo is a jupyter notebook tutorial for how to use pretty-plot. To access it make sure your pretty-plot conda environment has been activated (see above) and then type:

`jupyter notebook`

Now, follow the instructions output on the command line to open a jupyter notebook session in your browser (crtl+click on the url) and click on Pretty Plot.ipynb. For a streamlined description of other advanced options, click on brief_examples.ipynb.

### From the command line
You can also run pretty plot directly from the command line. To do that, make sure your pretty-plot conda environment has been activated, then run the `pplot` command. For example:

`pplot /Users/username/Documents/spectra/marslab_file.csv`

### Updating pretty plot
Make sure the conda environment is active:  
`conda activate pretty-plot`

Navigate to your pretty-plot directory. The exact path depends on where you initally installed pretty plot, for example:  
`cd ~/Documents/GitHub/pretty-plot`

Optional: run the command `ls`. If you see pplot.py in the output, then you are in the correct directory.

Download any updates from GitHub:  
`git pull`

Install the updates in your pretty-plot conda environment:  
`pip install .`

### Troubleshooting
- Any time something is not working as expected, a good first step is to check if the pretty-plot conda environment is active.

- Normally you can run pretty plot from any location, but during initial setup and updates the `pip install .` command must be run from within the pretty-plot directory. 
    - If you are getting errors, try running the `ls` command. If pplot.py is listed in the output, then you are in the right directory. If you don't see that file, navigate to the correct directory with `cd` and try the pip install again.

- Sometimes a fresh conda environment will solve the problem. First navigate to your pretty-plot directory with `cd`. After that, the steps to remove and recreate the pretty-plot environment are:
    
    `conda activate`
    
    `conda env remove --name pretty-plot`
    
    `conda env create --name pretty-plot --file environment.yml`
    
    `conda activate pretty-plot`
    
    `pip install .`

---
The contents of this repo are provided by the Western Washington University Reflectance Lab (PI: M. Rice) and Million Concepts (C. Million, M. St. Clair, S. Curtis, S.V. Brown) under a BSD 3-Clause License. You may do nearly anything that you want with this code. If you have any questions, leave us a Github Issue.

`MERTools`/`MERspect` is proprietary software (Arizona State University) for rover tactical operations made available on an as-needed basis.
