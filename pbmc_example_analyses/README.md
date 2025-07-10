# Using imscreenpy for analysis of image-based screening data from PBMCs

This folder contains notebooks to showcase  imscreenpy's functionalities on peripheral blood mononuclear cells, particularly to demonstrate how the QC functionality works and how drug responses can be deconvoluted with imscreenpy.

## Demonstration of a screening analysis workflow using imscreenpy

A full workflow would first perform the analysis in `qc_and_aggreate.ipynb` to get from an sqlite database with single-cell measurements to a viability dataframe, ie a dataframe containing the numbers of viable cells per well for each cell type of interest. This notebook performs image-based QC and morphology-based QC, adds predicted cell types and viabilities and subsequently generates this viability dataframe - a dataframe of viable cell numbers per well that can be used for furhter downstream analysis.

The notebook `analyse_drug_responses.ipynb` demonstrates the functionality for subsequent drug response scoring and how `imscreenpy` enables the quantification of differential responses for different sub-populations. 

## Downloading the required data

To run the notebooks, you will need to download the input data. At the moment, they assume that all their input data are present in a sub-folder called `example_data`. These files are a bit too large to be put on github. We are working on generating a zenodo record such that you can download the data and perform the full workflows yourself.
