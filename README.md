# Multimodal Data Classification using Variational Autoencoder - Under Construction


# Uncertainty 
Based on https://github.com/PierreBoyeau/decision-making-vaes I develop a framework for multimodal data classification with VAE.
by Pigi Lozou

## Installation:
#### Create conda envirioment
```
conda env create -f environment.yml
conda activate mcvae
```

#### Install
```
python3 setup.py build install
```
#### Install - editable version
```
pip install -e .
```


## Configuration
#### Edit data directories and output at the configurations files :

```
mcvae
│   
└───scripts
    │   trento_config.py
    │   bcss_config.py
    │  

```


## Classification
#### SVM and RF 
```
python3 scripts/simu_SVM_RF.py -d <DATASET NAME>

```

#### M1+M2 
```
python3 scripts/simu_vae.py -d <DATASET NAME>

```
#### multi-M1+M2
```
python3 scripts/simu_mvae.py -d <DATASET NAME>

```

## Results and Analysis

#### Metrics and classification, and uncertainty maps for all the output/*.npy files
```
python3 scripts/outputs_analysis.py
```

#### Generate plots to compare data distributions for trento and houston dataset
```
python3 scripts/data_distributions.py
```
### Output .npys, pngs, logs and uncertainty images files are saved at :

```
uncertainty
│   
└───outputs
    │   
    └───trento
    │   │   trento.logs
    │   │   trento.npy
    │   └───images
    │       │   
    |       └── trento.png
    |
    └───houston
    │   │   houston.logs
    │   │   houston.npy
    │   └───images
    │       │   
    |       └── houston.png

```

#### To do
 - Logging and documentation
