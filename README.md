# mlcis

Investigating Translation Regulation at the 5'UTR with Machine Learning Interpretation

A recent convolutional neural network (CNN) model accurately quantifies the relationship between 5' untranslated region (5'UTR) sequence and translation phenotype, but the underlying biological determinants remain elusive. Applying model interpretation, we extract representations of regulatory logic, revealing a complex interplay of regulatory sequence elements. Guided by insights from model interpretation, we train a model with superior performance on human reporter data, which will promote applications in synthetic biology and precision medicine.

This github repository contains all code to reproduce findings from the study. To replicate the main results, clone this repository and run the jupyter notebook of your choice in the 'mlcis' environment, which you can create using the .yml file. Plots can be replicated by running the 'fig1', 'fig2' and 'fig3' jupyter notebooks. This repo already contains the datasets used from Sample et al. (2019) and those created in this study in the 'data' folder, but if you want to download it separately, you can do so via this link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002. All trained models can be found in the 'models' folder. The 'utils' folder contains separate python scripts used for analysis.

If you want to use the optimized MRL model, copy the 'OptMRL_model.hdf5' file from this repository.