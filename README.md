# COSNet
COSNet for composition-structure bimodal learning

This repo is for the paper entitled "Multimodal machine learning for materials science: composition-structure bimodal learning for experimentally measured properties".

The environment to run this code is the same as ROOST: https://github.com/CompRhys/roost

Simply run: python train.py, then the training will start (given correct path to the cif files). 

In the data directory, we provide the band gap dataset, the refractive index dataset, and the dielectric constant dataset. 

The Li conductivity dataset will be released with our new paper about discovery of new Li conductors by machine learning. 

If you use the datasets, please cite the dataset paper mentioned in our work as well as this work.

If you use the code, please cite this paper as well as the following three papers used as the composition and structure network:

Goodall, R.E.A., Lee, A.A. Predicting materials properties without crystal structure: deep representation learning from stoichiometry. Nat Commun 11, 6280 (2020).

arXiv:2208.05039v3 

Tian Xie and Jeffrey C. Grossman, Phys. Rev. Lett. 120, 145301 – Published 6 April 2018
