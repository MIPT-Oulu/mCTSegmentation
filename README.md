# Deep Learning-based 3D Segmentation of Calcified Cartilage Interface in Human Osteochondral Samples Imaged with Micro-Computed Tomography

*Aleksei Tiulpin, Tuomas Frondelius, Heikki J. Nieminen, Petri Lehenkari, Simo Saarakkala*

## Installation
```python
pip install -e .
```

## TODO

### Reproducibility
- [ ] Dockerfile

### Data pre-processing
- [x] Data generation algorithm
- [x] Co-registration artifacts removal (image and mask must be the same size)
- [x] Crop width bug fix: some samples can't be cropped. Needs thorough check
- [ ] Train / test split integration

### Model evaluation
- [ ] Test set evaluation metrics
- [ ] Fill the tables below
- [ ] Produce figures

## Experiments (Cross-validation)

|        Loss         | Dice c. @25 microM [95% CI]| Dice c. @75 microM [95% CI] | Dice c.  @105 microM [95% CI]   | Dice c.@155 microM [95% CI] |
|:-------------------:|:------------------:|:-------------------:|:------------------------:|:--------------------:|
|    BCE              |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    -log(Jaccard)    |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    BCE-log(Jaccard) |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Focal loss       |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |

## Experiments (Test set)

|        Loss         | Dice c. @25 microM [95% CI]| Dice c. @75 microM [95% CI] | Dice c.  @105 microM [95% CI]   | Dice c.@155 microM [95% CI] |
|:-------------------:|:------------------:|:-------------------:|:------------------------:|:--------------------:|
|    BCE              |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    -log(Jaccard)    |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    BCE-log(Jaccard) |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Focal loss       |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
