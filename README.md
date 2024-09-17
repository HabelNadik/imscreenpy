# imscreenpy

This is a (more or less) general-purpose repository to postprocess data from image-based screening experiments that were analysed with CellProfiler.
The desired output are tables that contain numbers of cells with mutually exclusive properties for each treatment condition.

The full workflow is executed by `main.py`. It can execute the following steps:

1. Initial image-qc from cellprofiler output databases
2. Morphology QC on single cells from
3. Aggregating data and adding addtional metadata
4. (Optional): Predicting celltypes or cell states from image patches using `patchpy`
5. (Optional): Predicting viability from image patches using `patchpy`
6. (Optional): Generating an inhibition report

## Setup and Installation

The requirements to run `imscreenpy` are the same as for running `patchpy`. As for `patchpy`, we recommend to install via `pip` or `mamba`, because `conda` installations for these environments can take quite long, and `conda` may not be able to resolve the environments. The required packages are listed in `requirements.txt` and `imscreenpy_env.yml`. If you have an installation for `patchpy` already. You can just use it here. The code in this repository has been tested with Python 3.8.2 and Python 3.11.6.

With pip, do:

    python -m pip install -r requirements.txt

With conda, do

    conda env create -n patchscreen_env --file imscreenpy_env.yml
    conda activate patchscreen_env
    python -m pip install tensorflow_probability==0.12.1


## Running the code as described in *Strohmenger et al.*

If you want to run `main.py` locally with pre-trained autoencoders from `patchpy`, you will have to change the paths in `imscreenpy/config_files/nb_paths_and_patterns.txt` to match your local system. Please note that full instructions on how to run this are only located in the `neuroblastoma_screening` repository at https://github.com/HabelNadik/neuroblastoma_screening.