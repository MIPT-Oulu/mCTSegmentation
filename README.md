# Deep Learning-based 3D Segmentation of Calcified Cartilage Interface in Human Osteochondral Samples Imaged with Micro-Computed Tomography

*Aleksei Tiulpin, Tuomas Frondelius, Heikki J. Nieminen, Petri Lehenkari, Simo Saarakkala*

## TODO

### Data pre-processing
- [ ] Check sample localization algorithm
- [ ] Cropped dataset to hdf5
- [ ] Co-registration artifacts removal (image and mask must be the same size)
- [ ] Crop width bug fix: some samples can't be cropped. Needs thorough check
- [ ] Train / test split integration

### Model evaluation
- [ ] Test set evaluation metrics (on the assembled back volumes)
- [ ] VNet experiments
- [ ] RefinementNet experiments
- [ ] Fill the tables below
- [ ] Produce figures

## UNet results

|   Metric     | @25 microM [95% CI]| @75 microM [95% CI] |   @105 microM [95% CI]   | @155 microM [95% CI] |
|:------------:|:------------------:|:-------------------:|:------------------------:|:--------------------:|
|      VD      |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Dice      |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Surf.D.   |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |

## VNet results

|   Metric     | @25 microM [95% CI]| @75 microM [95% CI] |   @105 microM [95% CI]   | @155 microM [95% CI] |
|:------------:|:------------------:|:-------------------:|:------------------------:|:--------------------:|
|      VD      |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Dice      |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Surf.D.   |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |

## UNet+Refinement results

|   Metric     | @25 microM [95% CI]| @75 microM [95% CI] |   @105 microM [95% CI]   | @155 microM [95% CI] |
|:------------:|:------------------:|:-------------------:|:------------------------:|:--------------------:|
|      VD      |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Dice      |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |
|    Surf.D.   |    # [#, #]        |     # [#, #]        | # [#, #]                 | # [#, #]             |