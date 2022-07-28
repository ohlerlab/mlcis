# mlcis

Machine Learning Interpretation for Identification of Cis-Regulatory Elements from a Deep Learning Model of Translation Regulation.

A Deep Learning Model trained on 5'UTR Reporter Sequence from a Massively Parallel Reporter Assay (Sample et. al 2019) will be used for identification of 5'UTR sequence elements that determine ribosome load, and hence, translation outcome. For this purpose, we applied Integrated Gradients (Sundararajan et. al 2017), a gradient-based attribution method that assesses the importance of individual features for prediction outcome.

Preprocessing of all raw data was conducted in 'data' and training, validation and prediction of and with models in 'models'. To recapitulate the model interpretation process, each figure in the manuscript corresponds to one script in the directory. Each jupyter notebook incorporates the code to generate a figure from preprocessed data. Data preprocessing and model training/validation occurs in seperate dedicated jupyter notebooks. 
