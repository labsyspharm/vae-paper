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
Gaurav N. Joshi<sup>6</sup>,
Fumiki Yanagawa<sup>6</sup>,
Artem Sokolov<sup>1</sup>,
Hanspeter Pfister<sup>4</sup>,
Peter K. Sorger<sup>1,2,3,#</sup></h5>

<sup>1</sup>Laboratory of Systems Pharmacology, Harvard Medical School, Boston, MA
<sup>2</sup>Ludwig Center for Cancer Research at Harvard, Harvard Medical School, Boston, MA
<sup>3</sup>Department of Systems Biology, Harvard Medical School, Boston, MA
<sup>4</sup>Harvard John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge, MA
<sup>5</sup>Department of Pathology, Brigham and Women’s Hospital, Harvard Medical School, Boston, MA
<sup>6</sup>Nikon Instruments, Lexington, MA

<sup>&</sup> Current affiliation: Division of Oncological Sciences, Knight Cancer Institute, Oregon Health & Science University, Portland, OR  
<sup>%</sup> Current affiliation: Visual Computing Division, School of Computing, Clemson University, Clemson, SC

\*Co-first Authors: G.J.B., E.N.<br>
\#Corresponding Authors: bakergr@ohsu.edu (G.J.B.), peter_sorger@hms.harvard.edu (P.K.S.)<br>

<!-- *Nature Cancer (2023). DOI: [10.1038/s43018-023-00576-1](https://doi.org/10.1038/s43018-023-00576-1)* -->

## Abstract

Spatial proteomics (highly multiplexed tissue imaging) provides unprecedented insight into the types, states, and spatial organization of cells within preserved tissue environments. To enable single-cell analysis, high-plex images are typically segmented using algorithms that assign marker signals to individual cells. However, conventional segmentation is often imprecise and susceptible to signal spillover between adjacent cells, interfering with accurate cell type identification. Segmentation-based methods also fail to capture the morphological detail that histopathologists rely on for disease diagnosis and staging. Here, we present a method that combines unsupervised, pixel-level machine learning using autoencoders with traditional segmentation to generate single-cell data that captures information on protein abundance, morphology, and local neighborhood in a manner analogous to human experts while overcoming signal spillover. We demonstrate the generality of this technique by applying it to CyCIF, Lunaphore COMET, and Akoya PhenoCycler data, and show that it can learn histological features across multiple spatial scales.

<!-- [Click to read preprint](https://doi.org/10.1101/2023.11.01.565120) [[1]](#1) -->

## Running the computational notebooks 

Python code in this GitHub repository is organized into Jupyter notebooks used to generate the figures shown in the paper. To run the code, first clone this repository onto your computer by opening a terminal window and entering the following command:
```bash
git clone https://github.com/labsyspharm/vae-paper.git

```

Next, change directories into the top level directory of the cloned repository and create and activate a dedicated Conda environment containing the necessary Python libraries for running the code. If conda is not already installed, you can download it by following the instructions provided [here](https://docs.anaconda.com/miniconda/).

```bash
cd <path/to/cloned/repo>
conda env create -f environment.yml
conda activate morphaeus-paper

```


To browse the Jupyter notebooks, change directories to the `src` folder and activate Jupyter Lab with the following command:
```bash
jupyter lab

```
Notebooks are pre-populated with output cells for ease of review. To re-run the notebooks or explore multiplex images displayed in the Napari image viewer by some notebooks, you must first download the input data from our public Amazon S3 bucket (instructions are provided in the section below).

---

## Downloading input data files 
 To re-run the Jupyter notebooks, [input data](s3://lsp-public-data/baker-2025-vae/) must first be downloaded from our public Amazon S3 bucket into the the top-level directory of the cloned vae-paper GitHub repository. This can be acheived by running the `download.py` script located in the `src` folder from the top-level of the vae-paper repository. In addition to the required input data, this script will also download a folder containing precomputed output files as a reference (`output_reference`):
```bash
# from the top-level directory of the cloned vae-paper GitHub repository
python src/download.py

```
 Note: ~335GB of storage space is required to download the complete file set.

 To re-run any of the notebooks in Jupyter Lab, first double click on the name of an .ipynb file at the left of the screen, the corresponding notebook will open at the right. Then click the double-arrow button at the top of the notebook to restart the kernel and run all the code cells. Notebook output will be saved to a folder called `output` in the top-level directory of the repository.  

---

## MORPHӔUS source code and demo

[MORPHÆUS source code](https://github.com/labsyspharm/vae) is freely available for academic re-use under the MIT license on GitHub.


To run a demonstration of the MORPHÆUS analysis pipeline, be sure that the input data files have first been downloaded as described above, then change directories to the `demo` directory in the cloned vae-paper GitHub repository and run the following command:
```bash
vae config.yml
```
This will execute the pipeline on 9x9um image patches from the CyCIF-1A image presented in the paper, demonstrating all major modules ranging from single-cell CSV sampling and image patch generation, to VAE model training, plot visualization, and concept saliency analysis. Analysis output will be saved in `demo/VAE9_VIG7/`

Note that demo results will differ from those presented in the paper due to a smaller training dataset and fewer training epochs. For convenience, lightly pre-trained encoder and decoder networks are provided so that the pipeline can skip the VAE training step. For those interested in training a model from scratch, simply add a `#` to the beginning of the encoder.hdf5 and decoder.hdf5 filenames in the `demo/VAE9_VIG7/6_train_vae/` directory before running the pipeline. Do the same for the TRAIN_VAE.txt checkpoint file in `demo/VAE9_VIG7/checkpoints/`.

When training the model locally on CPUs, each epoch is estimated to complete in approximately 5 minutes; however, training may be significantly faster using GPU resources. In this example, roughly 30 epochs are needed before learned reconstructions begin to resemble their respective input image patches and for patch embeddings to begin to form distinct clusters in feature space. 

---

## Zenodo archive

This GitHub repository will be archived on Zenodo following publication of the manuscript.

<!-- the link at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212) -->

---

## Funding

This work was supported by NCI grant U01-CA284207, the Harvard Ludwig Center (P.K.S., S.S.), an ASPIRE Award from The Mark Foundation for Cancer Research, and the David Liposarcoma Research Initiative, and was initiated as part of the computational toolbox for the Human Tissue Atlas Network (HTAN).

---

## References

Baker GJ., Novikov E. et al. Morphology-Aware Profiling of Highly Multiplexed Tissue Images using Variational Autoencoders. **bioRxiv** (2025) https://doi.org/10.1101/2025.06.23.661064

<!-- <a id="1">[1]</a> Baker GJ., Novikov E. et al. Morphology-Aware Profiling of Highly Multiplexed Tissue Images using Variational Autoencoders. **bioRxiv** (2025) https://doi.org/10.1101/2025.06.23.661064 -->
