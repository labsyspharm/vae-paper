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

A detailed characterization of human tissue organization and understanding of how multiscale histological structures differ in response to disease and therapy can serve as important biomarkers of disease progression and therapeutic response. Although highly multiplex images of tissue contain detailed information on the abundance and distribution of proteins within and across cells, their analysis via segmentation-based methods captures little morphological information, is influenced by signal contamination across segmentation boundaries, and requires custom algorithms to study multi-cellular tissue architectures. Here we classify individual cell states and recurrent microscale tissue motifs in human colorectal adenocarcinoma by training a class of generative neural networks (variational autoencoders, VAEs) on multi-scale image patches derived from whole-slide imaging data. Our work demonstrates how this unsupervised computer vision approach can be used to characterize cells and their higher-order structural assemblies in a manner that simultanously accounts for protein abundance and spatial distribution while overcoming many of the limitations intrinsic to segmentation-based analysis.

The Python code (i.e., Jupyter Notebooks) in this GitHub repository was used to generate the figures in the aforementioned study.

<!-- [Click to read preprint](https://doi.org/10.1101/2023.11.01.565120) [[1]](#1) -->

---


## VAE Source Code

Source code for the VAE analysis pipeline used in this study is freely-available and archived on [GitHub](https://github.com/labsyspharm/vae). 

---


## Data Availability

New data associated with this paper is available at the [HTAN Data Portal](https://data.humantumoratlas.org). Input data required to run the source code found here is freely-available at [Sage Synapse](https://www.synapse.org/#!Synapse:syn24193163/files)


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


