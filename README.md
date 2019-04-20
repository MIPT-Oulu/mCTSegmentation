# Deep Learning-based 3D Segmentation of Calcified Cartilage Interface in Human Osteochondral Samples Imaged with Micro-Computed Tomography

*Aleksei Tiulpin, Tuomas Frondelius, Heikki J. Nieminen, Petri Lehenkari, Simo Saarakkala*

## Data

The data acompanying this paper can be downloaded as follows: `cd scripts && download_data.sh`. 
By default, it will create a folder in `DATA_DIR` that can be changed.

## Installation
```
pip install git+https://github.com/MIPT-Oulu/KVS.git
pip install -e .

```

## TODO

### Reproducibility
- [ ] Dockerfile

### Model evaluation
- [ ] Test set evaluation metrics
- [ ] Fill the tables below
- [ ] Produce figures

## Experiments

| Model     | IoU@25 microM [95% CI]     | IoU@75 microM [95% CI]      | IoU@105 microM [95% CI]  | IoU@155 microM [95% CI] |
|:---------|:--------------------------:|:---------------------------:|:------------------------:|:-----------------------:|
|  UNet (baseline)     |    # [#, #]                |     # [#, #]                | # [#, #]                 | # [#, #]                |
|  UNet-IN  |    # [#, #]                |     # [#, #]                | # [#, #]                 | # [#, #]                |
|  VGG11-UNet  |    # [#, #]                |     # [#, #]                | # [#, #]                 | # [#, #]                |
|  VGG11-UNet-IN  |    # [#, #]                |     # [#, #]                | # [#, #]                 | # [#, #]                |