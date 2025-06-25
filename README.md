[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)
![Latest release](https://img.shields.io/github/v/release/labsyspharm/cylinter-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Computational Notebooks for "Morphology-Aware Profiling of Highly Multiplexed Tissue Images using Variational Autoencoders"

<h5>Gregory J. Baker<sup>1,2,3,&,*,#</sup>,    
Edward Novikov<sup>1,4,*</sup>,
Shannon Coy<sup>1,2,5</sup>,
Yu-An Chen<sup>1,2</sup>,
Clemens B. Hug<sup>1</sup>,
Zergham Ahmed<sup>1,4</sup>, 
Sebastián A. Cajas Ordóñez<sup>4</sup>,
Siyu Huang<sup>4,%</sup>,
Clarence Yapp<sup>1</sup>,
Artem Sokolov<sup>1</sup>,
Hanspeter Pfister<sup>4</sup>,
Peter K. Sorger<sup>1,2,3,#</sup></h5>

<sup>1</sup>Laboratory of Systems Pharmacology, Harvard Medical School, Boston, MA
<sup>2</sup>Ludwig Center for Cancer Research at Harvard, Harvard Medical School, Boston, MA
<sup>3</sup>Department of Systems Biology, Harvard Medical School, Boston, MA
<sup>4</sup>Harvard John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge, MA
<sup>5</sup>Department of Pathology, Brigham and Women’s Hospital, Harvard Medical School, Boston, MA

<sup>&</sup> Current affiliation: Division of Oncological Sciences, Knight Cancer Institute, Oregon Health & Science University, Portland, OR  
<sup>%</sup> Current affiliation: Visual Computing Division, School of Computing, Clemson University, Clemson, SC

\*Co-first Authors: G.J.B., E.N.<br>
\#Corresponding Authors: gbak7696@gmail.com (G.J.B.), peter_sorger@hms.harvard.edu (P.K.S.)<br>

<!-- *Nature Cancer (2023). DOI: [10.1038/s43018-023-00576-1](https://doi.org/10.1038/s43018-023-00576-1)* -->

## Abstract

Spatial proteomics (highly multiplexed tissue imaging) provides unprecedented insight into the types, states, and spatial organization of cells within preserved tissue environments. To enable single-cell analysis, high-plex images are typically segmented using algorithms that assign marker signals to individual cells. However, conventional segmentation is often imprecise and susceptible to signal spillover between adjacent cells, interfering with accurate cell type identification. Segmentation-based methods also fail to capture the morphological detail that histopathologists rely on for disease diagnosis and staging. Here, we present a method that combines unsupervised, pixel-level machine learning using autoencoders with traditional segmentation to generate single-cell data that captures information on protein abundance, morphology, and local neighborhood in a manner analogous to human experts while overcoming the problem of signal spillover. The result is a more accurate and nuanced characterization of cell types and states than segmentation-based analysis alone.

<!-- [Click to read preprint](https://doi.org/10.1101/2023.11.01.565120) [[1]](#1) -->

## Running the computational notebooks 

The Python code in this GitHub repository is organized into Jupyter Notebooks and used to generate the figures shown in the paper. To run the code, first clone this repo onto your computer by opening a terminal window and pasting the following command:
```bash
git clone https://github.com/labsyspharm/vae-paper.git

```

Next, change directories into the top level of the cloned repo and create and activate a dedicated Conda environment containing the necessary Python libraries for running the code using the following commands:

```bash
cd <path/to/cloned/repo>
conda env create -f environment.yml
conda activate morphaeus-paper

```
Note: If conda is not already installed, you can download it by following the instructions provided [here](https://docs.anaconda.com/miniconda/).

To browse the Jupyter Notebooks, change directories to the `src` folder and enter the following command:
```bash
jupyter lab

```

 To re-run any of the Jupyter Notebooks, you must first download the required [input data](s3://lsp-public-data/baker-2025-vae/) from our public Amazon S3 bucket by running the `download.py` script located the `src` folder. This will also download a folder containing precomputed notebook output files:
```bash
python src/download.py

```
 Note: XXXGB of storage space is required to download the complete file set.

 To re-run any of the Notebooks, double click on a .ipynb notebook file at the left and the Notebook will appear at the right. Next click the `double-arrow` button at the top of the Notebook to restart the kernel and run all cells. 

---

## MORPHӔUS Source Code and Demo

Source code for the MORPHÆUS data analysis pipeline is freely available for academic re-use under the MIT license, available on [GitHub](https://github.com/labsyspharm/vae), and archived on Zenodo.


To test the pipeline, first ensure that the input data have been downloaded from S3 as specified above, then navigate to the `demo` directory and run the following command:
```bash
vae config.yml
```
This will launch the pipeline on a small subsample of data from the CyCIF-1A image presented in the paper, demonstrating all major modules ranging from CSV subsampling and image patch generation, to VAE model training, plot visualization, and concept saliency analysis.

Note: demo results will differ from those shown in the paper due to the use of a smaller training dataset and fewer training epochs. Epochs in the demo are estimated to complete in about 30sec - 1min running locally on CPUs. For this subsample, ~100 epochs are required before learned reconstructions begin to resemble cells and data start to cluster. Lightly pre-trained encoder and decoder networks are provided so that the training step can be skipped.

---

## Data Availability

Image files associated with this paper were first generated as part of the Human Tumor Atlas Network (HTAN) project and are available at the [HTAN Data Portal](https://data.humantumoratlas.org).

---

## Funding

This work was supported by Ludwig Cancer Research and the Ludwig Center at Harvard (P.K.S., S.S.), the Gray Foundation, and by NIH NCI grants U01-CA284207, and U2C-CA233262. S.S. is supported by the BWH President’s Scholars Award. Results shown in this study are in part based upon data generated by the Human Tumor Atlas Network (HTAN, https://humantumoratlas.org/).  

---

## Zenodo Archive

The Python code (i.e., Jupyter Notebooks) in this GitHub repository is archived on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)

---

## References

<a id="1">[1]</a> Baker GJ., Novikov E. et al. Morphology-Aware Profiling of Highly Multiplexed Tissue Images using Variational Autoencoders. **bioRxiv** (2025) https://doi.org/10.1101/2023.11.01.565120
