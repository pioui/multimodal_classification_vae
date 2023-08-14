# Multimodal Data Classification using Variational Autoencoder 

This repository provides a framework for multimodal data classification using Variational Autoencoder (VAE). The goal is to classify multimodal data and analyze uncertainty in the classification process. It is based on the work done by Pierre Boyeau in his repository decision-making-vaes.


## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Description

This repository provides a framework for multimodal data classification using Variational Autoencoder (VAE). The goal is to classify multimodal data and analyze uncertainty in the classification process. It is based on the work done by Pierre Boyeau in his repository decision-making-vaes.

## Installation
Create conda envirioment
```
conda env create -f environment.yml
conda activate mcvae
```

Build 
```
python3 -m build 
```


## Usage
### Configuration
Edit the data directories and output paths in the configuration files:
```
mcvae
│   
└───scripts
    │   trento_config.py
    │   houston_config.py
    │   simu_M1M2.py  
    |   simu_multi-M1M2.py  
```


### Classification

Run classification for different classification methods. 
1. SVM and RF 
```
python3 scripts/simu_SVM_RF.py -d <DATASET NAME>
```
2. M1+M2 
```
python3 scripts/simu_vae.py -d <DATASET NAME>
```
3. multi-M1+M2
```
python3 scripts/simu_mvae.py -d <DATASET NAME>
```

### Uncertainty

To calculate uncertainty please use [this repository](https://github.com/pioui/uncertainty).
Save the calculated uncertainties for each classification senario in the following structure:

```
mcvae
│   
└───outputs
    │   
    └───trento
    │   │   trento.logs
    │   │   trento_classification.npy
    │   └───uncertainties
    │       │   
    |       └── trento_uncertainty.npy
```



### Results and Analysis

Metrics, classification, and uncertainty maps for all the output/*.npy files
```
python3 scripts/results_analysis.py
```
Generate plots to compare data distributions for the trento and houston datasets
```
python3 scripts/data_distributions_analysis.py
```
The output files, including .npy, .png, logs, and uncertainty images, are saved in the following directory structure:
```
mcvae
│   
└───outputs
    │   
    └───trento
    │   │   trento.logs
    │   │   trento_classification.npy
    │   └───images
    │       │   
    |       └── trento.png
    |
    └───houston
    │   │   houston.logs
    │   │   houston_classification.npy
    │   └───images
    │       │   
    |       └── houston.png
```

## License

This project is licensed under the MIT License.
## To dos:
 - Logging and documentation
 - Detailed project description
