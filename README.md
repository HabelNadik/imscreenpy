# imscreenpy

This is a (more or less) general-purpose repository to postprocess data from image-based screening experiments that were analysed with CellProfiler.

The overall goal is to quantify differential drug responses for different sub populations in heterogenous cell pools.

It was developed during my time at the Boztug laboratory at the St. Anna Children's Cancer Research Institute (CCRI).

Specifically, it provides functionalities for QC on the image- and single-cell level, functionality to predict different cell properties such as cell types or cell viability, and functionality to enumerate different sub-populations for the quantification of sub-population specific drug inhibition effects.

The full workflow is executed by `main.py`. It can execute the following steps:

1. Initial image-qc from cellprofiler output databases
2. Morphology QC on single cells from
3. Aggregating data and adding addtional metadata
4. (Optional): Predicting celltypes or cell states from image patches using `patchpy` or other methods.
5. (Optional): Predicting viability from image patches using `patchpy`
6. (Optional): Generating an inhibition report

Given the differences in analysis goals for different image-based screens, you may want to implement your own version of `main.py`, but we do believe that our implementation is at the very least useful as a template. 

## Setup and Installation

The code in this repository has been tested with Python 3.8.2 and Python 3.11.6. You can install imscreenpy from the git repo, or download it and install it from your local folder:

    # install from git
    python -m pip install git+https://github.com/HabelNadik/imscreenpy
    # install from your local copy
    cd /path/to/local/imscreenpy
    python -m pip install .

If you want to execute the full workflow that we used in our work, you will need a few more things:

You will need patchpy, which you can now find under https://github.com/HabelNadik/patchpy.

You will have to set up a config folder that contains the config files describing the CellProfiler columns that you want to use and the locations of scripts for the predicting cell types and cell states. Defaults for the analyses that we performed - minus the paths to our local scripts - are located in `src/imscreenpy/config_files`.

To make the workflow more efficient on a cohort level, you can set up an annotation file that can then be read by `main.py`, such that you do not have to specify all parameters of an individual experiment via command line arguments. `patchpy` also needs such a file to effectively load raw data from different experiments for model training, so I would recommmend setting it up this way. An example file that lists the typical included columns is located in `pbmc_example_analyses/example_data/sample_annotations.xls`.


## Using imscreenpy

We have used imscreenpy to analze data from peripheral blood mononuclear cells (PBMCs) and bone marrow mononuclear cells (BMMCs) as well as cell lines from other non-hematopoietic tissues. The full workflow as we use it in our work is executed by `main.py`, but we also provide notebooks that you may find instructive in `pbmc_example_analyses` and that demonstrate how you may integrate `imscreenpy` functions into your own analyses workflows.

### Analysis on PBMCs or BMMCs (*Haladik et al.*)
The folder `pbmc_example_analyses` contains scripts and notebooks that give an overview of imscreenpy's functionalities for the analysis of PBMCs or BMMNCs. We have applied these analyses in a manuscript that should be published soon. If you want to learn more about how to analyse this type of screening data, this folder is a good place to start and we will make another repository that demonstrates using `imscreenpy` on a cohort level available soon.

### Analysis on neuroblastoma (*Strohmenger et al.*)
If you want to reproduce the analysis from *Strohmenger et al.* and run `main.py` locally with pre-trained autoencoders from `patchpy`, you will have to change the paths in `imscreenpy/config_files/nb_paths_and_patterns.txt` to match your local system. The full analysis from that manuscript will be publicly available soon and we will then provide an update here.