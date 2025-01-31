[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)
![Latest release](https://img.shields.io/github/v/release/labsyspharm/cylinter-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Computational Notebooks for "MORPHӔUS: Generative AI for Morphology-Aware Profiling of Human Cancer"

<h5>Gregory J. Baker<sup>1,2,3,*,#</sup>,    
Edward Novikov<sup>1,4,*</sup>,
Yu-An Chen<sup>1,2</sup>,
Clemens B. Hug<sup>1</sup>,
Zergham Ahmed<sup>1,4</sup>, 
Sebastián A. Cajas Ordóñez<sup>4</sup>,
Siyu Huang<sup>4</sup>,
Clarence Yapp<sup>1,5</sup>,
Shannon Coy<sup>1,2,6</sup>,
Hanspeter Pfister<sup>4</sup>,
Artem Sokolov<sup>1,7</sup>,
Peter K. Sorger<sup>1,2,3,#</sup></h5>

<sup>1</sup>Laboratory of Systems Pharmacology, Harvard Medical School, Boston, MA
<sup>2</sup>Ludwig Center for Cancer Research at Harvard, Harvard Medical School, Boston, MA
<sup>3</sup>Department of Systems Biology, Harvard Medical School, Boston, MA
<sup>4</sup>Harvard John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge, MA
<sup>5</sup>Department of Pathology, Brigham and Women’s Hospital, Harvard Medical School, Boston, MA

\*Co-first Authors: G.J.B., E.N.<br>
\#Corresponding Authors: gregory_baker2@hms.harvard.edu (G.J.B.), peter_sorger@hms.harvard.edu (P.K.S.)<br>

<!-- *Nature Cancer (2023). DOI: [10.1038/s43018-023-00576-1](https://doi.org/10.1038/s43018-023-00576-1)* -->

## Abstract

Alterations in tissue organization and morphology are critical biomarkers of disease progression and therapeutic response. While immunofluorescence images provide information on protein abundance and spatial distribution within tissues, segmentation-based analysis methods fail to extract morphological detail, suffer from signal spillover across cell boundaries, and rely on custom algorithms to infer spatial relationships among segmented cells. Here we introduce MORPHӔUS, a spatial biology framework that classifies multiplex histology images at the pixel-level across arbitrary length scales using generative modeling with variational autoencoders (VAEs). When applied to human colorectal cancer, MORPHӔUS identifies biologically meaningful cell states, morphologies, cell-cell interactions, and composite tissue structures with greater accuracy than segmentation-based approaches while avoiding the problem of signal spillover. This fully unsupervised method requires no ground truth annotations and is agnostic to the number and nature of immunomarkers used, making it broadly applicable to a wide range of bioimaging applications.

<!-- [Click to read preprint](https://doi.org/10.1101/2023.11.01.565120) [[1]](#1) -->

## Running the computational notebooks
If not already installed, download `conda` following the instructions provided [here](https://docs.anaconda.com/miniconda/). 

The Python code in this GitHub repository is organized in Jupyter Notebooks and used to generate figures shown in the paper. To run the code, first clone this repo onto your computer. Then download the required [input](https://www.synapse.org/#!Synapse:syn24193163/files/) data folder from the Sage Bionetworks Synpase data repository dedicated to the MORPHӔUS project into the `src` folder of the cloned repo. This folder also contains the full images and image patches used in the paper. Change directories into the top level of the cloned repo and create and activate a dedicated Conda environment with the necessary Python libraries for running the code by entering the following commands:

```bash
cd <path/to/cloned/repo>
conda env create -f environment.yml
conda activate morphaeus-paper

```

Next, change directories to the `src` folder and open the computational notebooks in JupyterLab with the following command:
```bash
jupyter lab

```

---


## MORPHӔUS Source Code

MORPHӔUS source code will be made freely-available upon the release of the paper and will be archived on [GitHub](https://github.com/labsyspharm/vae) and Zenodo.

---


## Data Availability

Image files associated with this paper were first generated as part of the Human Tumor Atlas Network (HTAN) project and are available at the [HTAN Data Portal](https://data.humantumoratlas.org). Input images required to run the source code found here is also freely-available at [Sage Synapse](https://www.synapse.org/#!Synapse:syn53216852/files/)


---


## Funding and Acknowledgments

This work was supported by the Ludwig Cancer Research and the Ludwig Center at Harvard (P.K.S., S.S.) and by NIH NCI grants U54-CA225088, U2C-CA233280, and U2C-CA233262 (P.K.S., S.S.). S.S. is supported by the BWH President’s Scholars Award.

---

## Zenodo Archive

The Python code (i.e., Jupyter Notebooks) in this GitHub repository is archived on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)

---


## References

<!-- <a id="1">[1]</a>
Baker GJ. et al. Quality Control for Single Cell Analysis of High-plex Tissue Profiles using CyLinter. **bioRxiv** (2023) https://doi.org/10.1101/2023.11.01.565120 -->


