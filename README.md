[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)
![Latest release](https://img.shields.io/github/v/release/labsyspharm/cylinter-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Computational Notebook for "Segmentation-free Characterization of Cell Types and Microscale Tissue Assemblies in Human Colorectal Cancer with Variational Autoencoders"

<h5>Gregory J. Baker<sup>1,2,3,*,#</sup>,    
Edward Novikov<sup>1,4,*</sup>,
Yu-An Chen<sup>1,2</sup>,
Clemens B. Hug<sup>1</sup>,
Sebastián A. Cajas Ordóñez<sup>4</sup>,
Siyu Huang<sup>4</sup>,
Clarence Yapp<sup>1,5</sup>,
Shannon Coy<sup>1,2,6</sup>,
Hanspeter Pfister<sup>4</sup>,
Artem Sokolov<sup>1,7</sup>,
Peter K. Sorger<sup>1,2,3,#</sup></h5>

<sup>1</sup>Laboratory of Systems Pharmacology, Program in Therapeutic Science, Harvard Medical School, Boston, MA<br>
<sup>2</sup>Ludwig Center for Cancer Research at Harvard, Harvard Medical School, Boston, MA<br>
<sup>3</sup>Department of Systems Biology, Harvard Medical School, Boston, MA<br>
<sup>4</sup>Harvard John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge, MA<br>
<sup>5</sup>Image and Data Analysis Core, Harvard Medical School, Boston, MA<br>
<sup>6</sup>Department of Pathology, Brigham and Women’s Hospital, Harvard Medical School, Boston, MA<br>
\*Co-first Authors: G.J.B., E.N.<br>
\*Corresponding Authors: gregory_baker2@hms.harvard.edu (G.J.B.), peter_sorger@hms.harvard.edu (P.K.S)<br>

<!-- *Nature Cancer (2023). DOI: [10.1038/s43018-023-00576-1](https://doi.org/10.1038/s43018-023-00576-1)* -->

## Abstract

A detailed characterization of human tissue organization and understanding of how multi-scale histological structures differ in response to disease and therapy can serve as important biomarkers of disease progression and therapeutic response. Although highly multiplex images of tissue contain detailed information on the abundance and distribution of proteins within and across cells, their analysis via segmentation-based methods captures little morphological information, suffers from signal contamination across segmentation boundaries, and requires custom algorithms to study multi-cellular tissue organization. Here we classify individual cell states and recurrent microscale tissue architectures in human colorectal adenocarcinoma by training variational autoencoder (VAE) deep learning networks on multi-scale image patches and demonstrate how this fully unsupervised generative computer vision approach can achieve detailed information on cell lineage, morphology, and multi-cellular neighborhood context while overcoming intrinsic limitations of image segmentation.

The Python code (i.e., Jupyter Notebooks) in this GitHub repository was used to generate the figures in the aforementioned study.

<!-- [Click to read preprint](https://doi.org/10.1101/2023.11.01.565120) [[1]](#1) -->

---


## VAE Source Code

![](./docs/cylinter-logo.svg)

CyLinter software is written in Python3, archived on the Anaconda package repository, versioned controlled on [Git/GitHub](https://github.com/labsyspharm/cylinter), instantiated as a configurable Python Class object, and validated for Mac and PC operating systems. Information on how to install and run the program is available at the [CyLinter website](https://labsyspharm.github.io/cylinter/). 

---


## Data Availability

New data associated with this paper is available at the [HTAN Data Portal](https://data.humantumoratlas.org). Previously published data is through public repositories. See Supplementary Table 1 for a complete list of datasets and their associated identifiers and repositories. Online Supplementary Figures 1-4 and the CyLinter demonstration dataset can be accessed at [Sage Synapse](https://www.synapse.org/#!Synapse:syn24193163/files)


---


## Image Processing

The whole-slide and tissue microarray images described in this study were processed using [MCMICRO](https://mcmicro.org/) [[2]](#2) image assembly and feature extraction pipeline.

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


