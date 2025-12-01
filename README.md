# Pretty Plot

A utility for easily converting VNIR multispectral data from Mastcam-Z into a standardized, pretty plot for quick distribution of results. The tool accepts `marslab`-formatted files.

### Installation / setup
#### Step 0: clone the repository to your computer:

If you have `git` installed on your computer, navigate in a terminal emulator to wherever you'd like to place the software and run `git clone git@github.com:MillionConcepts/pretty-plot.git`.

Alternatively, you can use [GitHub Desktop](https://desktop.github.com/) to clone the repository. Install that program, run it, log in to your account, choose "Clone Repository...", click the "URL" tab, paste `https://github.com/MillionConcepts/pretty-plot.git` into the 'Repository URL' field, and click "Clone".

#### Step 1: install conda

*Note: If you already have Anaconda or Miniconda installed on your computer, you can skip this step. If it's very old or not working well, you should uninstall it first. We **strongly** advise against installing multiple versions of `conda` unless you a very specific reason to do so.*

We recommend using [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install). Follow the instructions on that website to download the installer script and set up your `conda` installation.

#### Step 2: create a pretty-plot conda environment:

Once you have `conda` installed, you can set up a Python environment to use pretty-plot. Open a terminal window: Anaconda Prompt on Windows, Terminal on macOS, or your terminal emulator of choice on Linux. (Windows might name the prompt "Miniconda Prompt" or something else instead; just search for "prompt" in the Start Menu and don't pick Windows Command Prompt.)

Now, navigate to the directory where you downloaded the repository and run the command:  
`conda env create -n pretty-plot --file environment.yml`

Say yes at the prompts and let the installation finish. Then run `conda env list`. You should see `pretty-plot` in the list of environments.

#### Step 3: activate the conda environment and install `pretty-plot`:

Next, run `conda activate pretty-plot` to activate the Python environment that contains the packages pretty-plot needs.

To install the pretty-plot application into the environment, run `pip install -e .` You will never need to run this again unless you delete and recreate the `pretty-plot` environment.

**Important:** now that you've created this environment, you should always have it active whenever you work with pretty-plot. You can do this simply by running `conda activate pretty-plot`.


### Tutorial
The Pretty Plot.ipynb file in this repo is a jupyter notebook tutorial for how to use pretty-plot. To access it make sure your pretty-plot conda environment has been activated (see above) and then type:

`jupyter notebook`

Now, follow the instructions output on the command line to open a jupyter notebook session in your browser (crtl+click on the url) and click on Pretty Plot.ipynb. For a streamlined description of other advanced options, click on brief_examples.ipynb.

### From the command line
You can also run pretty-plot directly from the command line. To do that, make sure your pretty-plot conda environment has been activated, then run the `pplot` command. For example:

`pplot /Users/username/Documents/spectra/marslab_file.csv`

### Updating Pretty Plot
Make sure the conda environment is active:  
`conda activate pretty-plot`

Navigate to your pretty-plot directory. The exact path depends on where you initally installed pretty-plot, for example:  
`cd ~/Documents/GitHub/pretty-plot`

Optional: run the command `ls`. If you see pplot.py in the output, then you are in the correct directory.

Download any updates from GitHub by running the `git pull` command. (Or by clicking the "Pull" button in GitHub Desktop.)


### Troubleshooting
- Any time something is not working as expected, a good first step is to check if the pretty-plot conda environment is active.

- Normally you can run pretty-plot from any directory, but during initial setup the `pip install -e .` command must be run from within the pretty-plot directory. 
    - If you are getting errors, try running the `ls` command. If pplot.py is listed in the output, then you are in the right directory. If you don't see that file, navigate to the correct directory with `cd` and try the pip install again.

- Sometimes a fresh conda environment will solve the problem. First navigate to your pretty-plot directory with `cd`. After that, the steps to remove and recreate the pretty-plot environment are:
    
    `conda activate`
    
    `conda env remove --name pretty-plot`
    
    `conda env create --name pretty-plot --file environment.yml`
    
    `conda activate pretty-plot`
    
    `pip install -e .`

---
The contents of this repo are provided by the Western Washington University Reflectance Lab (PI: M. Rice) and Million Concepts (C. Million, M. St. Clair, S. Curtis, S.V. Brown) under a BSD 3-Clause License. You may do nearly anything that you want with this code. If you have any questions, leave us a Github Issue.

`MERTools`/`MERspect` is proprietary software (Arizona State University) for rover tactical operations made available on an as-needed basis.
