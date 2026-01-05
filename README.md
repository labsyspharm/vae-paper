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

Python code in this GitHub repository is organized as Jupyter notebooks for generating the figures shown in the paper. To view the notebooks, first clone the repository onto your computer by opening a terminal window and entering the following command below. If git is not already installed, you can download it by following the instructions provided [here](https://git-scm.com/install/).
```bash
git clone https://github.com/labsyspharm/vae-paper.git

```

Next, change directories into the top-level directory of the cloned repository and create and activate a dedicated Conda environment containing the necessary Python libraries for running the code. If conda is not already installed, it can be downloaded by following the instructions provided [here](https://www.anaconda.com/download/success).

```bash
cd <path/to/cloned/repo>

# macOS
conda env create -f environment_macOS.yml
conda activate morphaeus

# PC
conda env create -f environment_PC.yml
conda activate morphaeus
pip install git+https://github.com/labsyspharm/vae.git@v0.0.7

```


To browse the notebooks, change directories to the `src` folder and activate Jupyter Lab:
```bash
jupyter lab

```
Notebooks are pre-populated with output cells for ease of review. To re-run notebooks or explore multiplex images displayed in the Napari image viewer by some notebooks the `input` data must first be downloaded from our public Amazon S3 bucket (instructions are provided in the section below).

---

## Downloading input data files 
 To re-run the Jupyter notebooks, [`input` data](s3://lsp-public-data/baker-2025-vae/) must first be downloaded from our public Amazon S3 bucket into the the top-level directory of the cloned repository by running the `download.py` script located in the `src` folder from the top-level of the repository. In addition to the required data, this script will also download a folder containing precomputed output files for at-a-glance ease of reference (`output_reference`):
```bash
# from the top-level directory of the cloned vae-paper GitHub repository
python src/download.py

```
 Note: ~335GB of storage space is required to download the complete file set.

 To re-run any of the Jupyter notebooks, double click on a notebook filename at the left of the screen to open the corresponding notebook at the right. Next click the double-arrow button at the top of the notebook interface to restart the kernel and run all of the code cells. Notebook output is saved to a folder called `output` in the top-level directory of the repository.  

---

## MORPHӔUS source code and demo

[MORPHÆUS source code](https://github.com/labsyspharm/vae) is freely available for academic re-use under the MIT license on GitHub.


To run the MORPHÆUS pipeline demonstration, you must first download the `input` data files whose names begin with `CyCIF-1A`, as described above. These correspond to the first ten files downloaded when running the `src/download.py` script; the remaining files are not required for the MORPHÆUS demo, so the download process can be stopped after these are obtained. Once the `CyCIF-1A` files have been downloaded, navigate to the demo directory in the cloned repository and run the following command:
```bash
# from the demo directory
vae config.yml
```
This will execute the pipeline on 13x13um image patches from the CyCIF-1A image presented in the paper, demonstrating all major modules ranging from single-cell sampling and image patch cropping, to VAE model training, plot visualization, and concept saliency analysis. Depending on the size of images, the cutting and storage of image patches generated in the `RUN_CELLCUTTER` module can be memory limiting; a minimum of 32GB RAM is required to run this demo without having to alter the `cache_size_cellcutter` and `cells_per_chunk` parameters in the MORPHÆUS configuration file (`config.yml`). If sufficient memory is available, the `cache_size_cellcutter` parameter can be increased beyond 32000 MB (the size of the `CyCIF-1A` image) to load the entire image file into RAM. This will allow the `RUN_CELLCUTTER` module to execute significantly faster by avoiding repeat reads from disk. Demo output is saved to `demo/VAE13/`. 

For convenience, lightly pre-trained encoder and decoder networks are provided such that the pipeline skips the VAE training module. For those interested in training a model from scratch, simply add a `#` to the beginning of the `encoder.hdf5` and `decoder.hdf5` filenames in `demo/VAE13/6_train_vae/` before running the pipeline; do the same for the TRAIN_VAE.txt checkpoint file in `demo/VAE13/checkpoints/`. When training on CPUs using relatively modern machines, epochs are estimated to complete in about 5 minutes each; training may be accelerated greatly using GPU resources. 

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
